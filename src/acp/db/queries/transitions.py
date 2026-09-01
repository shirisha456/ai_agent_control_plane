"""The one function allowed to mutate tasks.state.

Every later component -- claim, lease renewal, completion, retry, recovery,
cancellation -- is a call to `transition()` with a different compare-and-set
predicate. Getting its contract right is the whole point of Phase 1.

CONTRACT
--------
`applied=False` is a RETURN VALUE, never an exception. Losing a CAS race is a
normal, expected outcome in a distributed system: it is how a worker learns
its lease expired while it was working. If this raised, every call site would
grow a try/except that eventually swallows a genuine lost-ownership signal and
lets a zombie worker overwrite live state.

An *illegal* transition (QUEUED -> SUCCEEDED) is a different thing entirely --
that is a programmer error and raises IllegalTransition.

CONCURRENCY
-----------
Isolation level is READ COMMITTED (PostgreSQL's default), not SERIALIZABLE.
Justification: every operation here is a SINGLE-STATEMENT update, and
PostgreSQL takes a row lock and re-evaluates the WHERE clause against the
latest committed row version (EvalPlanQual) before applying it. So when N
transactions race for one row, they serialize on the row lock and exactly one
sees its predicate still satisfied. SERIALIZABLE would add serialization
failures we must retry, plus predicate-lock overhead, to buy a guarantee we
already have for free.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from acp.db.models import task_events, tasks
from acp.db.sqlutil import seconds
from acp.domain.states import EventType, State, assert_legal


class Rejection(StrEnum):
    """Why a compare-and-set did not apply.

    Worth distinguishing rather than returning a bare False: STATE_MISMATCH
    during a claim is routine contention, while ATTEMPT_MISMATCH on a
    completion is a stale worker being fenced out -- an event you want on a
    dashboard, because a non-zero rate is the proof that fencing works.
    """

    NOT_FOUND = "not_found"
    STATE_MISMATCH = "state_mismatch"
    ATTEMPT_MISMATCH = "attempt_mismatch"
    WORKER_MISMATCH = "worker_mismatch"
    LEASE_NOT_EXPIRED = "lease_not_expired"
    EXECUTION_TIME_NOT_EXCEEDED = "execution_time_not_exceeded"


@dataclass(frozen=True, slots=True)
class TransitionResult:
    applied: bool
    task: Mapping[str, Any] | None = None
    rejection: Rejection | None = None
    observed: Mapping[str, Any] | None = None  # row as it actually was, when rejected

    def __bool__(self) -> bool:
        return self.applied


async def transition(
    conn: AsyncConnection,
    task_id: UUID,
    *,
    expect_state: State | Sequence[State],
    to_state: State,
    event_type: EventType,
    expect_attempt: int | None = None,
    expect_worker: str | None = None,
    expect_lease_expired: bool = False,
    expect_execution_time_exceeded: bool = False,
    set_fields: Mapping[str, Any] | None = None,
    event_data: Mapping[str, Any] | None = None,
) -> TransitionResult:
    """Atomically move a task between states, or report why we could not.

    `conn` must already be inside a transaction. The caller owns the
    transaction so that a claim can write the task row, its task_attempts row,
    and its event in one atomic unit.

    The fence: pass `expect_attempt` and `expect_worker` for any operation
    scoped to lease ownership (renew, complete, fail, checkpoint). A worker
    that lost its lease while paused will present a stale attempt number, the
    predicate will match zero rows, and its result is discarded.

    `expect_lease_expired=True` is the reaper's fence: it does not know (or
    care) who currently holds the lease, only that `lease_expires_at` is in
    the past. Comparing against `now()` in the UPDATE's WHERE clause -- not a
    value read earlier in Python -- means EvalPlanQual re-checks it against
    whatever the row's latest committed version says: if the real owner
    renews between the reaper's candidate SELECT and this UPDATE, the
    predicate now fails and the reaper's write is discarded, exactly like any
    other lost CAS race.

    `expect_execution_time_exceeded=True` is the SAME idea for a different
    failure: a worker that is still faithfully renewing its lease but is
    actually stuck (an infinite loop, a hung network call with no timeout)
    never trips `expect_lease_expired` -- its lease stays valid forever. This
    fences on wall-clock elapsed time since the attempt started
    (`first_started_at`) against that task's own `max_execution_time_s`,
    regardless of lease validity. Like the lease-expiry fence, the comparison
    against `now()` happens IN the UPDATE, so a task that legitimately
    finishes between the reaper's candidate scan and this write loses the
    race cleanly instead of being force-failed out from under a worker that
    was about to complete it.
    """
    expected: tuple[State, ...] = (
        (expect_state,) if isinstance(expect_state, State) else tuple(expect_state)
    )
    if not expected:
        raise ValueError("expect_state is required; it is the CAS anchor, not an optimisation")
    for src in expected:
        assert_legal(src, to_state)

    values: dict[str, Any] = {"state": to_state.value, "updated_at": sa.func.now()}
    values.update(set_fields or {})

    stmt = (
        sa.update(tasks)
        .where(tasks.c.id == task_id)
        .where(tasks.c.state.in_([s.value for s in expected]))
    )
    if expect_attempt is not None:
        stmt = stmt.where(tasks.c.attempt == expect_attempt)
    if expect_worker is not None:
        stmt = stmt.where(tasks.c.lease_worker_id == expect_worker)
    if expect_lease_expired:
        stmt = stmt.where(tasks.c.lease_expires_at < sa.func.now())
    if expect_execution_time_exceeded:
        stmt = stmt.where(
            tasks.c.first_started_at.is_not(None),
            tasks.c.first_started_at + seconds(tasks.c.max_execution_time_s) < sa.func.now(),
        )

    row = (await conn.execute(stmt.values(**values).returning(*tasks.c))).mappings().first()

    if row is None:
        return await _explain(
            conn,
            task_id,
            expected,
            expect_attempt,
            expect_worker,
            expect_lease_expired,
            expect_execution_time_exceeded,
        )

    await conn.execute(
        sa.insert(task_events).values(
            task_id=task_id,
            attempt=row["attempt"],
            event_type=event_type.value,
            worker_id=row["lease_worker_id"] or expect_worker,
            data=dict(event_data or {}),
        )
    )
    return TransitionResult(applied=True, task=dict(row))


async def _explain(
    conn: AsyncConnection,
    task_id: UUID,
    expected: tuple[State, ...],
    expect_attempt: int | None,
    expect_worker: str | None,
    expect_lease_expired: bool,
    expect_execution_time_exceeded: bool = False,
) -> TransitionResult:
    """Re-read the row to classify the rejection.

    Only runs on the rare path, and the cost buys a great deal: without it
    every lost race looks identical, and you cannot tell routine claim
    contention from a stale worker being fenced.
    """
    observed = (
        (
            await conn.execute(
                sa.select(
                    tasks.c.state,
                    tasks.c.attempt,
                    tasks.c.lease_worker_id,
                    tasks.c.lease_expires_at,
                    (tasks.c.lease_expires_at < sa.func.now()).label("lease_expired"),
                    tasks.c.first_started_at,
                    tasks.c.max_execution_time_s,
                    (
                        tasks.c.first_started_at.is_not(None)
                        & (
                            tasks.c.first_started_at + seconds(tasks.c.max_execution_time_s)
                            < sa.func.now()
                        )
                    ).label("execution_time_exceeded"),
                ).where(tasks.c.id == task_id)
            )
        )
        .mappings()
        .first()
    )
    if observed is None:
        return TransitionResult(applied=False, rejection=Rejection.NOT_FOUND)

    obs = dict(observed)
    if State(obs["state"]) not in expected:
        reason = Rejection.STATE_MISMATCH
    elif expect_attempt is not None and obs["attempt"] != expect_attempt:
        reason = Rejection.ATTEMPT_MISMATCH
    elif expect_worker is not None and obs["lease_worker_id"] != expect_worker:
        reason = Rejection.WORKER_MISMATCH
    elif expect_lease_expired and not obs["lease_expired"]:
        reason = Rejection.LEASE_NOT_EXPIRED
    elif expect_execution_time_exceeded and not obs["execution_time_exceeded"]:
        reason = Rejection.EXECUTION_TIME_NOT_EXCEEDED
    else:
        # The row matched on re-read but not during the UPDATE: another
        # transaction committed in between. Treat as contention.
        reason = Rejection.STATE_MISMATCH
    del obs["lease_expired"]
    del obs["execution_time_exceeded"]
    return TransitionResult(applied=False, rejection=reason, observed=obs)


async def record_event(
    conn: AsyncConnection,
    task_id: UUID,
    event_type: EventType,
    *,
    attempt: int | None = None,
    worker_id: str | None = None,
    data: Mapping[str, Any] | None = None,
) -> None:
    """Append an event with no accompanying state change.

    Used for things that are not transitions: a rejected stale write, a lease
    renewal, a cancellation request landing on an already-running task.
    """
    await conn.execute(
        sa.insert(task_events).values(
            task_id=task_id,
            attempt=attempt,
            event_type=event_type.value,
            worker_id=worker_id,
            data=dict(data or {}),
        )
    )
