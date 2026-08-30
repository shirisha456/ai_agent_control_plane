"""Worker registry: identity, heartbeat, and (Phase 3) liveness.

last_heartbeat_at has been recorded since Phase 2; mark_dead_workers is the
first thing that acts on staleness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from acp.db.models import workers
from acp.db.sqlutil import seconds


async def register_worker(
    conn: AsyncConnection,
    *,
    worker_id: str,
    hostname: str,
    pid: int,
    capacity: int,
    capabilities: Sequence[str] = (),
) -> Mapping[str, Any]:
    """Insert this process's worker row.

    `worker_id` must be generation-unique (see migration 0003) -- callers
    mint a fresh id per process start, never a stable hostname:pid, so a
    restarted worker can never be mistaken for the zombie it replaced.
    """
    row = (
        (
            await conn.execute(
                sa.insert(workers)
                .values(
                    id=worker_id,
                    hostname=hostname,
                    pid=pid,
                    capacity=capacity,
                    capabilities=list(capabilities),
                )
                .returning(*workers.c)
            )
        )
        .mappings()
        .one()
    )
    return dict(row)


async def heartbeat(conn: AsyncConnection, *, worker_id: str) -> bool:
    """Record liveness. Returns False if this worker has been declared DEAD.

    The `status <> 'DEAD'` predicate is what makes death one-way. Without it a
    worker that was marked dead -- because it was partitioned, paused, or just
    slow -- would silently resurrect itself with its next heartbeat, and the
    fleet's view of who is alive would be permanently wrong.

    A False return is the worker's signal to SELF-FENCE: stop claiming, stop
    renewing, and exit so its supervisor restarts it with a fresh,
    generation-unique id. Nothing about task safety depends on this (ownership
    is fenced by lease_worker_id + attempt), but without it a declared-dead
    worker keeps claiming new work forever while every dashboard reports it
    gone.
    """
    result = await conn.execute(
        sa.update(workers)
        .where(workers.c.id == worker_id, workers.c.status != "DEAD")
        .values(last_heartbeat_at=sa.func.now())
    )
    return result.rowcount == 1


async def set_worker_status(conn: AsyncConnection, *, worker_id: str, status: str) -> None:
    """Move a worker between ALIVE / DRAINING / DEAD.

    Used by graceful shutdown so the fleet view distinguishes "stopping on
    purpose" from "stopped answering", which are very different incidents.
    """
    await conn.execute(sa.update(workers).where(workers.c.id == worker_id).values(status=status))


async def mark_dead_workers(conn: AsyncConnection, *, dead_after_s: int) -> Sequence[str]:
    """Flip ALIVE/DRAINING workers whose heartbeat has gone stale to DEAD.

    Not a CAS in the tasks-state-machine sense -- workers have no fencing
    token to protect, `status` is purely informational (task ownership is
    fenced by lease_worker_id + attempt, not by this column). A worker that
    somehow heartbeats again after being marked DEAD stays DEAD; a restarted
    process gets a fresh, generation-unique id instead of resurrecting this
    one (see migration 0003).
    """
    stmt = (
        sa.update(workers)
        .where(
            workers.c.status != "DEAD",
            workers.c.last_heartbeat_at < sa.func.now() - seconds(dead_after_s),
        )
        .values(status="DEAD")
        .returning(workers.c.id)
    )
    return (await conn.execute(stmt)).scalars().all()
