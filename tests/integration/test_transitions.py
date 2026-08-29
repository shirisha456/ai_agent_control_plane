"""Single-threaded behaviour of `transition()`, including the fence.

Phase 3 will build the reaper on top of these guarantees; proving them now
means the chaos tests later fail for interesting reasons, not for this.
"""

from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa

from acp.db.models import task_events, tasks
from acp.db.queries.transitions import Rejection, transition
from acp.domain.states import EventType, IllegalTransition, State

pytestmark = pytest.mark.db

LEASE = sa.func.now() + sa.text("interval '30 seconds'")


async def _claim(conn, task_id, worker="worker-a"):
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
        },
    )


async def test_claim_applies_and_records_history(engine, make_task) -> None:
    task_id = await make_task()
    async with engine.connect() as conn, conn.begin():
        res = await _claim(conn, task_id)

    assert res.applied
    assert res.task["state"] == State.RUNNING
    assert res.task["attempt"] == 1
    assert res.task["lease_worker_id"] == "worker-a"

    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    sa.select(task_events.c.event_type, task_events.c.attempt)
                    .where(task_events.c.task_id == task_id)
                    .order_by(task_events.c.id)
                )
            )
            .mappings()
            .all()
        )
    assert [r["event_type"] for r in rows] == [EventType.TASK_CLAIMED.value]
    assert rows[0]["attempt"] == 1


async def test_illegal_transition_raises_not_returns(engine, make_task) -> None:
    """QUEUED -> SUCCEEDED is a bug in the caller, not a lost race.

    Programmer error raises; lost ownership returns applied=False. Conflating
    them is how lost-lease signals get swallowed by a broad except.
    """
    task_id = await make_task()
    async with engine.connect() as conn, conn.begin():
        with pytest.raises(IllegalTransition):
            await transition(
                conn,
                task_id,
                expect_state=State.QUEUED,
                to_state=State.SUCCEEDED,
                event_type=EventType.TASK_SUCCEEDED,
            )


async def test_stale_attempt_is_fenced_out(engine, make_task) -> None:
    """THE central safety property, at the primitive level.

    Worker A holds attempt 1. Its lease expires and the task is reclaimed by
    worker B as attempt 2. A then tries to commit its result carrying attempt
    1: the CAS predicate matches zero rows and A's work is discarded.
    """
    task_id = await make_task()
    async with engine.connect() as conn, conn.begin():
        first = await _claim(conn, task_id, "worker-a")
    assert first.task["attempt"] == 1

    # Reaper: lease expired, task goes back to the queue. `attempt` is NOT
    # bumped here -- the next claim bumps it, so the token is allocated by the
    # same statement that grants ownership and the two can never disagree.
    async with engine.connect() as conn, conn.begin():
        recovered = await transition(
            conn,
            task_id,
            expect_state=State.RUNNING,
            to_state=State.QUEUED,
            event_type=EventType.TASK_RECOVERED,
            set_fields={"lease_worker_id": None, "lease_expires_at": None},
        )
    assert recovered.applied

    async with engine.connect() as conn, conn.begin():
        second = await _claim(conn, task_id, "worker-b")
    assert second.task["attempt"] == 2

    # Worker A wakes up and tries to finish work it no longer owns.
    async with engine.connect() as conn, conn.begin():
        stale = await transition(
            conn,
            task_id,
            expect_state=State.RUNNING,
            to_state=State.SUCCEEDED,
            event_type=EventType.TASK_SUCCEEDED,
            expect_attempt=1,
            expect_worker="worker-a",
            set_fields={"lease_worker_id": None, "lease_expires_at": None},
        )

    assert not stale.applied
    assert stale.rejection is Rejection.ATTEMPT_MISMATCH
    assert stale.observed["attempt"] == 2
    assert stale.observed["lease_worker_id"] == "worker-b"


async def test_wrong_worker_same_attempt_is_rejected(engine, make_task) -> None:
    """Defence in depth: even with the right token, the wrong owner is refused."""
    task_id = await make_task()
    async with engine.connect() as conn, conn.begin():
        await _claim(conn, task_id, "worker-a")

    async with engine.connect() as conn, conn.begin():
        res = await transition(
            conn,
            task_id,
            expect_state=State.RUNNING,
            to_state=State.SUCCEEDED,
            event_type=EventType.TASK_SUCCEEDED,
            expect_attempt=1,
            expect_worker="worker-impostor",
            set_fields={"lease_worker_id": None, "lease_expires_at": None},
        )
    assert not res.applied
    assert res.rejection is Rejection.WORKER_MISMATCH


async def test_lease_coherence_is_enforced_by_the_database(engine, make_task) -> None:
    """Forgetting to clear the lease on completion must fail loudly.

    A SUCCEEDED task still holding a lease would be invisible to the reaper
    and would permanently consume a tenant's concurrency budget. The CHECK
    constraint makes that unrepresentable rather than merely unlikely.
    """
    task_id = await make_task()
    async with engine.connect() as conn, conn.begin():
        await _claim(conn, task_id)

    with pytest.raises(Exception, match="ck_tasks_lease_coherence"):
        async with engine.connect() as conn, conn.begin():
            await transition(
                conn,
                task_id,
                expect_state=State.RUNNING,
                to_state=State.SUCCEEDED,
                event_type=EventType.TASK_SUCCEEDED,
                expect_attempt=1,
                set_fields={"finished_at": sa.func.now()},  # lease left dangling
            )


async def test_missing_task_reports_not_found(engine) -> None:
    async with engine.connect() as conn, conn.begin():
        res = await transition(
            conn,
            uuid.uuid4(),
            expect_state=State.QUEUED,
            to_state=State.CANCELLED,
            event_type=EventType.TASK_CANCELLED,
        )
    assert not res.applied
    assert res.rejection is Rejection.NOT_FOUND
