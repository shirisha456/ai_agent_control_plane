"""The first real piece of evidence in this repository.

If `transition()` is correct under contention, every later component -- claim,
lease renewal, completion, retry, recovery -- inherits that correctness,
because each is the same compare-and-set with a different predicate. If it is
not, no amount of chaos testing later will save the design.

These tests use real concurrent PostgreSQL sessions (NullPool, see conftest),
so they exercise actual row locks, not an imagined model of them.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from acp.db.models import task_events, tasks
from acp.db.queries.transitions import Rejection, transition
from acp.domain.states import EventType, State

pytestmark = [pytest.mark.db, pytest.mark.concurrency]

LEASE = sa.func.now() + sa.text("interval '30 seconds'")
CONTENDERS = 50


async def _claim(engine: AsyncEngine, task_id: uuid.UUID, worker: str):
    async with engine.connect() as conn, conn.begin():
        return await transition(
            conn,
            task_id,
            expect_state=State.QUEUED,
            to_state=State.RUNNING,
            event_type=EventType.TASK_CLAIMED,
            set_fields={
                "attempt": tasks.c.attempt + 1,
                "lease_worker_id": worker,
                "lease_expires_at": LEASE,
                "first_started_at": sa.func.coalesce(tasks.c.first_started_at, sa.func.now()),
            },
        )


async def test_only_one_of_fifty_concurrent_claims_wins(engine, make_task) -> None:
    """The core invariant: a task has at most one owner.

    Fifty transactions issue `UPDATE ... WHERE id=X AND state='QUEUED'`
    simultaneously. They serialise on the row lock; PostgreSQL re-evaluates
    each predicate against the latest committed row (EvalPlanQual), so the
    forty-nine that arrive after the winner commits see state='RUNNING' and
    match zero rows. No application-level locking is involved.
    """
    task_id = await make_task()

    results = await asyncio.gather(
        *(_claim(engine, task_id, f"worker-{i}") for i in range(CONTENDERS))
    )

    winners = [r for r in results if r.applied]
    losers = [r for r in results if not r.applied]

    assert len(winners) == 1, f"expected exactly one owner, got {len(winners)}"
    assert len(losers) == CONTENDERS - 1
    assert all(r.rejection is Rejection.STATE_MISMATCH for r in losers)

    # The fencing token advanced exactly once, not fifty times. If a losing
    # UPDATE had applied its `attempt + 1` before failing the predicate, this
    # would be > 1 and every downstream fence would be unsound.
    assert winners[0].task["attempt"] == 1


async def test_losers_write_no_events(engine, make_task) -> None:
    """A rejected transition must leave no trace in the history.

    The event insert lives inside the same transaction as the UPDATE, so a
    losing attempt rolls it back. If losers logged, the timeline for one task
    would show fifty claims and the audit trail would be worthless.
    """
    task_id = await make_task()
    await asyncio.gather(*(_claim(engine, task_id, f"worker-{i}") for i in range(CONTENDERS)))

    async with engine.connect() as conn:
        events = (
            (
                await conn.execute(
                    sa.select(task_events.c.event_type).where(task_events.c.task_id == task_id)
                )
            )
            .scalars()
            .all()
        )
    assert events == [EventType.TASK_CLAIMED.value]


async def test_concurrent_claims_across_many_tasks_lose_nothing(engine, make_task) -> None:
    """Contention must not cost throughput correctness.

    Fifty tasks, two contenders each. Every task must end up claimed exactly
    once -- neither double-claimed nor silently skipped.
    """
    task_ids = [await make_task() for _ in range(CONTENDERS)]

    results = await asyncio.gather(
        *[_claim(engine, t, f"worker-a-{i}") for i, t in enumerate(task_ids)],
        *[_claim(engine, t, f"worker-b-{i}") for i, t in enumerate(task_ids)],
    )

    assert sum(1 for r in results if r.applied) == CONTENDERS

    async with engine.connect() as conn:
        states = (
            (await conn.execute(sa.select(tasks.c.state).where(tasks.c.id.in_(task_ids))))
            .scalars()
            .all()
        )
    assert set(states) == {State.RUNNING.value}


async def test_terminal_write_races_lose_cleanly(engine, make_task) -> None:
    """Two outcomes racing on one running task: exactly one commits.

    This is the shape of the stale-worker race that Phase 3 will exercise for
    real. Here it is proven at the primitive level: an attempt to complete and
    an attempt to fail cannot both land.
    """
    task_id = await make_task()
    claim = await _claim(engine, task_id, "worker-a")
    assert claim.applied
    attempt = claim.task["attempt"]

    async def finish(to_state: State, event: EventType):
        async with engine.connect() as conn, conn.begin():
            return await transition(
                conn,
                task_id,
                expect_state=State.RUNNING,
                to_state=to_state,
                event_type=event,
                expect_attempt=attempt,
                expect_worker="worker-a",
                set_fields={
                    "lease_worker_id": None,
                    "lease_expires_at": None,
                    "finished_at": sa.func.now(),
                },
            )

    results = await asyncio.gather(
        *(
            finish(State.SUCCEEDED, EventType.TASK_SUCCEEDED)
            if i % 2 == 0
            else finish(State.FAILED, EventType.TASK_FAILED)
            for i in range(20)
        )
    )
    assert sum(1 for r in results if r.applied) == 1
