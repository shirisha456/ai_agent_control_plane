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

    row = (await conn.execute(stmt.values(**values).returning(*tasks.c))).mappings().first()

    if row is None:
        return await _explain(conn, task_id, expected, expect_attempt, expect_worker)

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
    else:
        # The row matched on re-read but not during the UPDATE: another
        # transaction committed in between. Treat as contention.
        reason = Rejection.STATE_MISMATCH
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
