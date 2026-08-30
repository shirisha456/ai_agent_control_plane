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

from acp.db.models import tasks
from acp.db.sqlutil import seconds
from acp.domain.states import State


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
            # You cannot renew a lease you have already lost. Without this
            # clause a worker returning from a long pause (GC, VM freeze,
            # partition) silently re-extends a lease that had already
            # expired -- the very operation meant to MAINTAIN ownership would
            # be the one that defeats the fence. The attempt check still
            # catches the case where someone else reclaimed the task, but
            # only this makes "expired" mean "lost" unconditionally, instead
            # of "lost, if the reaper happened to get there first".
            tasks.c.lease_expires_at > sa.func.now(),
        )
        .values(
            lease_expires_at=sa.func.now() + seconds(lease_ttl_s),
            updated_at=sa.func.now(),
        )
        .returning(tasks.c.cancel_requested, tasks.c.lease_expires_at)
    )
    row = (await conn.execute(stmt)).mappings().first()
    if row is None:
        return None

    # Deliberately NO event on a successful renewal. A renewal per task per
    # renew-interval is the highest-frequency operation in the system: at
    # 1,000 concurrent tasks and a 7s interval that is ~143 task_events rows
    # per second of pure noise, in the table whose retention we already have
    # to manage. Renewal success is derivable from lease_expires_at anyway.
    # Renewal FAILURE is the interesting event, and the caller records it.
    return dict(row)
