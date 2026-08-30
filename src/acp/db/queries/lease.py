"""Lease renewal: the heartbeat that keeps a RUNNING task owned.

Not modelled through acp.db.queries.transitions.transition, deliberately --
renewal does not change `state` (RUNNING stays RUNNING), and the state
machine's ALLOWED graph has no self-loop, on purpose: a self-loop would let a
bug write RUNNING->RUNNING through the transition path and bypass the fencing
predicate transition() insists on. Renewal is a narrower operation with its
own fencing, so it gets its own statement.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from acp.db.models import task_events, tasks
from acp.domain.states import EventType, State


async def renew_lease(
    conn: AsyncConnection,
    task_id: UUID,
    *,
    worker_id: str,
    expect_attempt: int,
    lease_ttl_s: int,
) -> Mapping[str, Any] | None:
    """Extend the lease, or return None if this worker no longer owns it.

    None means the same thing as a lost CAS race elsewhere: the caller's
    attempt number or worker id no longer matches, so either the lease
    already expired and was reclaimed, or the task already completed. Either
    way the caller must stop treating the task as its own.
    """
    stmt = (
        sa.update(tasks)
        .where(
            tasks.c.id == task_id,
            tasks.c.state == State.RUNNING.value,
            tasks.c.attempt == expect_attempt,
            tasks.c.lease_worker_id == worker_id,
        )
        .values(
            lease_expires_at=sa.func.now() + sa.text(f"interval '{lease_ttl_s} seconds'"),
            updated_at=sa.func.now(),
        )
        .returning(tasks.c.cancel_requested, tasks.c.lease_expires_at)
    )
    row = (await conn.execute(stmt)).mappings().first()
    if row is None:
        return None

    await conn.execute(
        sa.insert(task_events).values(
            task_id=task_id,
            attempt=expect_attempt,
            event_type=EventType.LEASE_RENEWED.value,
            worker_id=worker_id,
            data={},
        )
    )
    return dict(row)
