"""Reclaim RUNNING tasks whose lease expired because their worker went dark.

Phase 2 shipped with no failure detection at all: a crashed or paused worker
left its tasks RUNNING forever. This is the fix. It is the ONLY code path,
other than the worker itself, that ever writes to a RUNNING task -- and it
goes through the same `transition()` CAS everything else does, fenced on
`expect_lease_expired` rather than on attempt/worker, since the whole point is
that we do not know (and do not need to know) who the presumed-dead owner was.

Same "attempt is not bumped here" rule as the worker's own retry path: the
fencing token is allocated at CLAIM time, not at recovery time, so a stale
completion from the presumed-dead worker is rejected by the state check alone
(RUNNING -> QUEUED already moved the row out from under it) without needing a
second guard.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from acp.db.models import tasks
from acp.db.queries.completion import finish_attempt
from acp.db.queries.transitions import record_event, transition
from acp.domain.states import EventType, State


async def find_expired_leases(conn: AsyncConnection, *, limit: int) -> list[Mapping[str, Any]]:
    """Lock a batch of RUNNING tasks whose lease has already expired.

    FOR UPDATE SKIP LOCKED for the same reason claim.py uses it: if more than
    one reaper replica ever runs at once, they partition the work instead of
    both trying to reclaim the same task.
    """
    stmt = (
        sa.select(tasks.c.id, tasks.c.attempt, tasks.c.max_attempts, tasks.c.lease_worker_id)
        .where(tasks.c.state == State.RUNNING.value, tasks.c.lease_expires_at < sa.func.now())
        .order_by(tasks.c.lease_expires_at)
        .limit(limit)
        .with_for_update(skip_locked=True)
    )
    return (await conn.execute(stmt)).mappings().all()


async def reap_task(
    conn: AsyncConnection,
    task_id: UUID,
    *,
    attempt: int,
    max_attempts: int,
    lease_worker_id: str | None,
) -> bool:
    """Recover one expired-lease task: back to QUEUED, or FAILED if attempts are exhausted.

    Returns whether the CAS applied. A `False` here means the presumed-dead
    worker renewed (or finished) between find_expired_leases's SELECT and this
    UPDATE -- routine, expected, not an error.
    """
    error_message = f"lease expired; worker {lease_worker_id!r} presumed dead"
    if attempt < max_attempts:
        res = await transition(
            conn,
            task_id,
            expect_state=State.RUNNING,
            to_state=State.QUEUED,
            event_type=EventType.TASK_RECOVERED,
            expect_lease_expired=True,
            set_fields={
                "available_at": sa.func.now(),
                "lease_worker_id": None,
                "lease_expires_at": None,
                "error_class": "LeaseExpired",
                "error_message": error_message,
            },
            event_data={"lease_worker_id": lease_worker_id},
        )
        outcome = "LOST"
    else:
        res = await transition(
            conn,
            task_id,
            expect_state=State.RUNNING,
            to_state=State.FAILED,
            event_type=EventType.TASK_ABANDONED,
            expect_lease_expired=True,
            set_fields={
                "finished_at": sa.func.now(),
                "lease_worker_id": None,
                "lease_expires_at": None,
                "error_class": "LeaseExpired",
                "error_message": error_message,
            },
            event_data={"lease_worker_id": lease_worker_id},
        )
        outcome = "ABANDONED"

    if not res.applied:
        return False

    await record_event(
        conn,
        task_id,
        EventType.WORKER_LOST,
        attempt=attempt,
        worker_id=lease_worker_id,
        data={},
    )
    await finish_attempt(
        conn,
        task_id,
        attempt,
        outcome=outcome,
        error_class="LeaseExpired",
        error_message=error_message,
    )
    return True


async def reap_expired_leases(conn: AsyncConnection, *, limit: int) -> int:
    """Find and recover up to `limit` expired-lease tasks. Returns how many were reaped."""
    candidates = await find_expired_leases(conn, limit=limit)
    reaped = 0
    for row in candidates:
        if await reap_task(
            conn,
            row["id"],
            attempt=row["attempt"],
            max_attempts=row["max_attempts"],
            lease_worker_id=row["lease_worker_id"],
        ):
            reaped += 1
    return reaped
