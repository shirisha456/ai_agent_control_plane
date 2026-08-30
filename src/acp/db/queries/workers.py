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


async def heartbeat(conn: AsyncConnection, *, worker_id: str) -> None:
    await conn.execute(
        sa.update(workers).where(workers.c.id == worker_id).values(last_heartbeat_at=sa.func.now())
    )


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
            workers.c.last_heartbeat_at
            < sa.func.now() - sa.text(f"interval '{dead_after_s} seconds'"),
        )
        .values(status="DEAD")
        .returning(workers.c.id)
    )
    return (await conn.execute(stmt)).scalars().all()
