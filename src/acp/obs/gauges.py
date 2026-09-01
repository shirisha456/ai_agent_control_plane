"""Refresh DB-derived gauges on a timer.

Why a timer rather than a Prometheus custom collector that queries on scrape:

  * Collectors are synchronous, and every query in this system is async.
  * A scrape would then hold the DB in its critical path. Add a second
    Prometheus replica, or a curious human with `watch curl`, and the
    monitoring system starts contending with the claim path for connections.
    Monitoring must not be able to cause the outage it is meant to observe.

So one process (the API) refreshes on a slow timer and every scrape reads
memory. The cost is that gauges are up to `interval` seconds stale, which is
fine: these are backlog levels, not alarms measured in milliseconds.

Exactly one process does this, deliberately. If every worker refreshed the
same fleet-wide gauges they would overwrite each other with slightly
different values and the graph would flicker between them.
"""

from __future__ import annotations

import asyncio
import contextlib

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from acp.obs import metrics
from acp.obs.logging import get_logger

log = get_logger(__name__)

#: Last observed global runnable depth, refreshed on the same timer as the
#: gauges. Admission reads THIS rather than counting the table per submit.
#:
#: Approximate on purpose. Per-tenant backlog is counted exactly because it
#: gates one tenant's correctness; global depth is a coarse survival signal
#: for a slow-moving condition, and counting the whole table on every submit
#: would make the overload check part of the overload.
_global_queued: int = 0


def cached_global_queued() -> int:
    return _global_queued


# Grouped by tenant NAME, not id: a uuid is unreadable on a dashboard, and the
# name is already unique per tenant.
_QUEUE_SQL = sa.text("""
SELECT te.name AS tenant,
       count(*) FILTER (WHERE t.state = 'QUEUED' AND t.available_at <= now()) AS runnable,
       count(*) FILTER (WHERE t.state = 'RUNNING')                            AS running
  FROM tenants te
  LEFT JOIN tasks t ON t.tenant_id = te.id
 GROUP BY te.name
""")

_BACKLOG_SQL = sa.text("SELECT count(*) FROM tasks WHERE state = 'QUEUED' AND available_at > now()")

_WORKERS_SQL = sa.text("SELECT status::text AS status, count(*) AS n FROM workers GROUP BY status")

# The best alert in the system: expired leases nobody has reclaimed. Zero when
# the reaper is healthy, climbing when it is not.
_PENDING_SQL = sa.text(
    "SELECT count(*) FROM tasks WHERE state = 'RUNNING' AND lease_expires_at < now()"
)


async def refresh_once(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        rows = (await conn.execute(_QUEUE_SQL)).mappings().all()
        backlog = (await conn.execute(_BACKLOG_SQL)).scalar_one()
        worker_rows = (await conn.execute(_WORKERS_SQL)).mappings().all()
        pending = (await conn.execute(_PENDING_SQL)).scalar_one()

    # Clear first: a tenant that drained to zero, or a status with no workers
    # left, would otherwise keep reporting its last non-zero value forever.
    # A gauge that never goes back down is worse than no gauge.
    metrics.queue_depth.clear()
    metrics.tasks_running.clear()
    metrics.workers_by_status.clear()

    for row in rows:
        metrics.queue_depth.labels(tenant=row["tenant"]).set(row["runnable"])
        metrics.tasks_running.labels(tenant=row["tenant"]).set(row["running"])
    for row in worker_rows:
        metrics.workers_by_status.labels(status=row["status"]).set(row["n"])

    global _global_queued
    _global_queued = sum(row["runnable"] for row in rows)

    metrics.tasks_backlogged.set(backlog)
    metrics.leases_expired_pending.set(pending)


async def run_refresher(engine: AsyncEngine, *, interval_s: float, stop: asyncio.Event) -> None:
    """Refresh until `stop` is set.

    Failures are logged and swallowed: a broken gauge refresh must never take
    down the API that serves task submissions.
    """
    while not stop.is_set():
        try:
            await refresh_once(engine)
        except Exception:  # noqa: BLE001 - monitoring must not break the service
            log.warning("gauges.refresh_failed", exc_info=True)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
