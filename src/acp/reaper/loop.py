"""The reaper process: find expired leases, recover or fail their tasks, mark dead workers.

Runs standalone from any worker -- a worker cannot detect its own death, so
this has to be an independent process on its own clock. Recovery latency for
a crashed worker is bounded by lease_ttl_s + reaper_period_s (see
acp.config.Settings), because a task cannot even become a reap candidate
until its lease expires.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

from acp.config import Settings
from acp.db.queries.reap import reap_expired_leases, reap_hung_tasks
from acp.db.queries.workers import mark_dead_workers
from acp.db.session import transaction
from acp.obs import metrics
from acp.obs.logging import get_logger

logger = get_logger("acp.reaper")


class Reaper:
    def __init__(
        self, *, settings: Settings, batch_size: int = 100, max_batches_per_sweep: int = 50
    ) -> None:
        self.settings = settings
        self.batch_size = batch_size
        self.max_batches_per_sweep = max_batches_per_sweep
        self._stopping = asyncio.Event()

    def stop(self) -> None:
        self._stopping.set()

    async def run_forever(self) -> None:
        while not self._stopping.is_set():
            await self._sweep()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=self.settings.reaper_period_s)

    async def _sweep(self) -> None:
        # Drain the whole backlog, not just one batch: recovery latency is
        # documented as lease_ttl_s + reaper_period_s, which only holds if a
        # burst of simultaneous failures doesn't get throttled to
        # batch_size-per-period.
        started = time.monotonic()
        requeued = failed = 0
        # Bounded: without a cap, a retry storm that produces expired leases
        # faster than we reclaim them keeps this loop inside a single sweep
        # forever, so the worker-liveness pass below never runs and the
        # reaper stops doing half its job while looking perfectly busy.
        for _ in range(self.max_batches_per_sweep):
            if self._stopping.is_set():
                break
            async with transaction() as conn:
                result = await reap_expired_leases(conn, limit=self.batch_size)
            requeued += result.requeued
            failed += result.failed_exhausted
            for overdue in result.overdue_s:
                metrics.recovery_latency.observe(overdue)
            if result.reaped:
                metrics.lease_expirations.inc(result.reaped)
            if result.reaped < self.batch_size:
                break

        if requeued:
            metrics.task_recoveries.labels(disposition="requeued").inc(requeued)
        if failed:
            metrics.task_recoveries.labels(disposition="failed_exhausted").inc(failed)
        if requeued or failed:
            logger.warning("reaper.leases_reclaimed", requeued=requeued, failed_exhausted=failed)

        # SECOND sweep, same bounded-rounds shape as the lease sweep above but
        # a DIFFERENT candidate set: RUNNING tasks whose lease is still valid
        # (still being renewed) but whose wall-clock execution time has
        # exceeded the cap they were pinned with at submit. Lease expiry
        # cannot catch this -- a worker stuck in a loop keeps renewing
        # normally right up until something else notices.
        hung_requeued = hung_failed = 0
        for _ in range(self.max_batches_per_sweep):
            if self._stopping.is_set():
                break
            async with transaction() as conn:
                hung_result = await reap_hung_tasks(conn, limit=self.batch_size)
            hung_requeued += hung_result.requeued
            hung_failed += hung_result.failed_exhausted
            if hung_result.reaped < self.batch_size:
                break

        if hung_requeued:
            metrics.hung_tasks_detected.labels(disposition="requeued").inc(hung_requeued)
        if hung_failed:
            metrics.hung_tasks_detected.labels(disposition="failed_exhausted").inc(hung_failed)
        if hung_requeued or hung_failed:
            logger.warning(
                "reaper.hung_tasks_reclaimed", requeued=hung_requeued, failed_exhausted=hung_failed
            )

        async with transaction() as conn:
            dead = await mark_dead_workers(conn, dead_after_s=self.settings.worker_dead_after_s)
        if dead:
            metrics.workers_marked_dead.inc(len(dead))
            logger.warning("reaper.workers_marked_dead", count=len(dead), ids=list(dead))

        metrics.reaper_sweep_duration.observe(time.monotonic() - started)
