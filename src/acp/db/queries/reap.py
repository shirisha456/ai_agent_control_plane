"""Reclaim RUNNING tasks the normal path cannot: expired leases, and hung workers.

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

HUNG TASKS ARE A DIFFERENT FAILURE FROM AN EXPIRED LEASE
---------------------------------------------------------
Lease expiry catches a worker that stopped answering. It cannot catch a
worker that is still faithfully renewing its lease every few seconds but is
actually stuck -- an infinite loop, a call with no timeout -- because the
lease itself is perfectly valid the whole time. That failure is caught here
too, fenced on `expect_execution_time_exceeded` instead: wall-clock time since
the attempt started, compared against that task's own pinned
`max_execution_time_s`, independent of lease validity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from acp.db.models import tasks
from acp.db.queries.completion import finish_attempt
from acp.db.queries.transitions import record_event, transition
from acp.db.sqlutil import seconds
from acp.domain.states import EventType, State


async def find_expired_leases(conn: AsyncConnection, *, limit: int) -> list[Mapping[str, Any]]:
    """Lock a batch of RUNNING tasks whose lease has already expired.

    FOR UPDATE SKIP LOCKED for the same reason claim.py uses it: if more than
    one reaper replica ever runs at once, they partition the work instead of
    both trying to reclaim the same task.
    """
    stmt = (
        sa.select(
            tasks.c.id,
            tasks.c.attempt,
            tasks.c.max_attempts,
            tasks.c.lease_worker_id,
            # How long the lease had ALREADY been expired when we found it.
            # Computed by PostgreSQL, not by subtracting a Python clock from a
            # database timestamp -- the recovery-latency histogram this feeds
            # is the number we claim as `lease_ttl_s + reaper_period_s`, so it
            # must not be measured against a second, drifting clock.
            sa.extract("epoch", sa.func.now() - tasks.c.lease_expires_at).label("overdue_s"),
        )
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
    else:
        res = await transition(
            conn,
            task_id,
            expect_state=State.RUNNING,
            to_state=State.FAILED,
            # TASK_FAILED, not TASK_ABANDONED: the task is ending because its
            # attempts are exhausted. TASK_ABANDONED means a worker handed
            # work back on purpose, which is the opposite of what happened
            # here -- the worker stopped answering.
            event_type=EventType.TASK_FAILED,
            expect_lease_expired=True,
            set_fields={
                "finished_at": sa.func.now(),
                "lease_worker_id": None,
                "lease_expires_at": None,
                "error_class": "LeaseExpired",
                "error_message": error_message,
            },
            event_data={"lease_worker_id": lease_worker_id, "reason": "attempts_exhausted"},
        )

    # LOST on BOTH paths. The attempt's outcome records what happened to the
    # ATTEMPT -- its worker stopped answering and the work was taken back --
    # which is true whether the task then requeued or ran out of attempts.
    # The task-level outcome is already recorded in tasks.state; duplicating
    # it here would lose the only signal that says "a worker died", and the
    # chaos demo's verification counts exactly that:
    #     SELECT count(*) FROM task_attempts WHERE outcome = 'LOST'
    # Marking the exhausted case ABANDONED both undercounts recoveries and
    # collides with the graceful-drain outcome, which genuinely is ABANDONED.
    outcome = "LOST"

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


@dataclass(slots=True)
class ReapResult:
    """What one reaper pass actually did.

    Richer than a count because the interesting questions are "how overdue
    were these?" (is the reaper keeping up with its own SLA) and "how many
    ran out of attempts?" (is something poisoning workers) -- neither of
    which a single integer can answer.
    """

    requeued: int = 0
    failed_exhausted: int = 0
    #: Seconds each reclaimed lease had been expired before we got to it.
    overdue_s: list[float] = field(default_factory=list)

    @property
    def reaped(self) -> int:
        return self.requeued + self.failed_exhausted

    def __int__(self) -> int:
        return self.reaped


async def reap_expired_leases(conn: AsyncConnection, *, limit: int) -> ReapResult:
    """Find and recover up to `limit` expired-lease tasks."""
    candidates = await find_expired_leases(conn, limit=limit)
    result = ReapResult()
    for row in candidates:
        exhausted = row["attempt"] >= row["max_attempts"]
        if await reap_task(
            conn,
            row["id"],
            attempt=row["attempt"],
            max_attempts=row["max_attempts"],
            lease_worker_id=row["lease_worker_id"],
        ):
            if exhausted:
                result.failed_exhausted += 1
            else:
                result.requeued += 1
            result.overdue_s.append(float(row["overdue_s"]))
    return result


async def find_hung_tasks(conn: AsyncConnection, *, limit: int) -> list[Mapping[str, Any]]:
    """Lock a batch of RUNNING tasks that have exceeded their own execution-time cap.

    Deliberately NOT restricted to leases that are also expired -- a hung
    worker's lease is, by definition, still valid (it is still renewing). The
    only signal available is wall-clock time since the attempt started versus
    the cap the task was pinned with at submit.

    No dedicated index: the candidate set is already `state = 'RUNNING'`,
    which is inherently small (bounded by total fleet concurrency, not queue
    depth), so a plain scan filtered to that partial condition costs little --
    the same reasoning that lets find_expired_leases's index also serve as an
    adequate access path here if the planner chooses it.
    """
    stmt = (
        sa.select(
            tasks.c.id,
            tasks.c.attempt,
            tasks.c.max_attempts,
            tasks.c.lease_worker_id,
            sa.extract(
                "epoch",
                sa.func.now() - tasks.c.first_started_at - seconds(tasks.c.max_execution_time_s),
            ).label("overdue_s"),
        )
        .where(
            tasks.c.state == State.RUNNING.value,
            tasks.c.first_started_at.is_not(None),
            tasks.c.first_started_at + seconds(tasks.c.max_execution_time_s) < sa.func.now(),
        )
        .order_by(tasks.c.first_started_at)
        .limit(limit)
        .with_for_update(skip_locked=True)
    )
    return (await conn.execute(stmt)).mappings().all()


async def reap_hung_task(
    conn: AsyncConnection,
    task_id: UUID,
    *,
    attempt: int,
    max_attempts: int,
    lease_worker_id: str | None,
) -> bool:
    """Force-end one hung attempt: back to QUEUED, or FAILED if attempts are exhausted.

    Structurally identical to `reap_task` -- same two-way branch on the
    attempt budget, same "attempt is not bumped here" rule -- fenced on
    `expect_execution_time_exceeded` instead of `expect_lease_expired`. A
    `False` return means the attempt legitimately completed (or the lease
    already separately expired and was reclaimed) between the candidate scan
    and this write -- routine, not an error.
    """
    error_message = f"execution time exceeded; worker {lease_worker_id!r} presumed stuck"
    if attempt < max_attempts:
        res = await transition(
            conn,
            task_id,
            expect_state=State.RUNNING,
            to_state=State.QUEUED,
            event_type=EventType.TASK_RECOVERED,
            expect_execution_time_exceeded=True,
            set_fields={
                "available_at": sa.func.now(),
                "lease_worker_id": None,
                "lease_expires_at": None,
                "error_class": "ExecutionTimeExceeded",
                "error_message": error_message,
            },
            event_data={"lease_worker_id": lease_worker_id},
        )
    else:
        res = await transition(
            conn,
            task_id,
            expect_state=State.RUNNING,
            to_state=State.FAILED,
            event_type=EventType.TASK_FAILED,
            expect_execution_time_exceeded=True,
            set_fields={
                "finished_at": sa.func.now(),
                "lease_worker_id": None,
                "lease_expires_at": None,
                "error_class": "ExecutionTimeExceeded",
                "error_message": error_message,
            },
            event_data={"lease_worker_id": lease_worker_id, "reason": "attempts_exhausted"},
        )

    if not res.applied:
        return False

    await record_event(
        conn,
        task_id,
        EventType.WORKER_LOST,
        attempt=attempt,
        worker_id=lease_worker_id,
        data={"reason": "execution_time_exceeded"},
    )
    await finish_attempt(
        conn,
        task_id,
        attempt,
        outcome="LOST",
        error_class="ExecutionTimeExceeded",
        error_message=error_message,
    )
    return True


async def reap_hung_tasks(conn: AsyncConnection, *, limit: int) -> ReapResult:
    """Find and force-end up to `limit` tasks stuck past their execution-time cap."""
    candidates = await find_hung_tasks(conn, limit=limit)
    result = ReapResult()
    for row in candidates:
        exhausted = row["attempt"] >= row["max_attempts"]
        if await reap_hung_task(
            conn,
            row["id"],
            attempt=row["attempt"],
            max_attempts=row["max_attempts"],
            lease_worker_id=row["lease_worker_id"],
        ):
            if exhausted:
                result.failed_exhausted += 1
            else:
                result.requeued += 1
            result.overdue_s.append(float(row["overdue_s"]))
    return result
