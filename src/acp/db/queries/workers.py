"""Worker registry: identity and heartbeat, nothing else.

Phase 2 deliberately has no failure detection (see migration 0003's docstring)
-- last_heartbeat_at is recorded here so Phase 3's reaper has real data to
read, but nothing yet acts on staleness.
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
        sa.update(workers)
        .where(workers.c.id == worker_id)
        .values(last_heartbeat_at=sa.func.now())
    )
