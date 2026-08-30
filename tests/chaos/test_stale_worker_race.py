"""The flagship failure demonstration, reproduced deterministically.

Most systems can describe this race. The point of this file is to REPRODUCE
it on demand, every run, with no sleeps and no timing luck:

    Worker A claims task 10          attempt = 1
    Worker A goes dark               (paused, partitioned, GC'd -- not dead)
    lease expires
    reaper reclaims the task         state -> QUEUED
    Worker B claims it               attempt = 2
    Worker A wakes up
    Worker A tries to commit         attempt = 1  -> REJECTED
    Worker B commits                 attempt = 2  -> accepted

Worker A is never dead in this story. It is slow. That is what makes the race
real: a lease system that only worked against genuinely dead workers would not
need fencing at all.

"Going dark" is simulated by moving `lease_expires_at` into the past rather
than by sleeping out a real TTL. The clock is PostgreSQL's either way, so the
code under test cannot tell the difference -- and the test runs in
milliseconds instead of thirty seconds, which is the difference between a
check that runs on every commit and one that never runs at all.
"""

from __future__ import annotations

import asyncio

import pytest
import sqlalchemy as sa

from acp.db.models import task_attempts, task_events, tasks
from acp.db.queries.claim import claim_tasks
from acp.db.queries.completion import complete_success
from acp.db.queries.lease import renew_lease
from acp.db.queries.reap import reap_expired_leases
from acp.db.queries.transitions import Rejection
from acp.domain.states import EventType, State
from acp.scheduling.policy import DEFAULT_POLICY

pytestmark = [pytest.mark.db, pytest.mark.chaos]

LEASE_TTL_S = 30


async def _expire_lease(engine, task_id) -> None:
    """Make the lease look expired. Equivalent to lease_ttl elapsing."""
    async with engine.connect() as conn, conn.begin():
        await conn.execute(
            sa.update(tasks)
            .where(tasks.c.id == task_id)
            .values(lease_expires_at=sa.func.now() - sa.text("interval '1 second'"))
        )


async def _claim_one(engine, worker_id):
    async with engine.connect() as conn, conn.begin():
        rows = await claim_tasks(
            conn, worker_id=worker_id, limit=1, lease_ttl_s=LEASE_TTL_S, policy=DEFAULT_POLICY
        )
    return rows[0] if rows else None


async def test_stale_worker_cannot_overwrite_the_new_owner(engine, make_task, make_worker) -> None:
    """The whole safety argument, end to end, in one test."""
    worker_a = await make_worker()
    worker_b = await make_worker()
    task_id = await make_task()

    # 1. Worker A claims. attempt 1.
    claimed = await _claim_one(engine, worker_a)
    assert claimed["id"] == task_id
    assert claimed["attempt"] == 1

    # 2. Worker A goes dark. Its lease lapses; it does not know that yet.
    await _expire_lease(engine, task_id)

    # 3. The reaper reclaims the task. `attempt` is NOT bumped here -- the
    #    token is allocated by whoever claims next, so ownership and token are
    #    granted by the same statement and can never disagree.
    async with engine.connect() as conn, conn.begin():
        assert await reap_expired_leases(conn, limit=10) == 1

    async with engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    sa.select(tasks.c.state, tasks.c.attempt, tasks.c.lease_worker_id).where(
                        tasks.c.id == task_id
                    )
                )
            )
            .mappings()
            .one()
        )
    assert row["state"] == State.QUEUED
    assert row["attempt"] == 1, "recovery must not allocate a fencing token"
    assert row["lease_worker_id"] is None

    # 4. Worker B picks it up. attempt 2.
    second = await _claim_one(engine, worker_b)
    assert second["id"] == task_id
    assert second["attempt"] == 2

    # 5. Worker A wakes up and tries to finish work it no longer owns.
    async with engine.connect() as conn, conn.begin():
        stale = await complete_success(
            conn, task_id, attempt=1, worker_id=worker_a, result={"from": "worker-a"}
        )
    assert not stale.applied, "a stale worker overwrote the live owner"
    assert stale.rejection is Rejection.ATTEMPT_MISMATCH
    assert stale.observed["attempt"] == 2
    assert stale.observed["lease_worker_id"] == worker_b

    # 6. Worker B commits normally.
    async with engine.connect() as conn, conn.begin():
        good = await complete_success(
            conn, task_id, attempt=2, worker_id=worker_b, result={"from": "worker-b"}
        )
    assert good.applied

    # THE INVARIANT: at-least-once delivery, at-most-once committed outcome.
    # Two attempts existed and both ran; exactly one is recorded as succeeded,
    # and the result stored is the live owner's, not the zombie's.
    async with engine.connect() as conn:
        outcomes = (
            (
                await conn.execute(
                    sa.select(task_attempts.c.attempt, task_attempts.c.outcome)
                    .where(task_attempts.c.task_id == task_id)
                    .order_by(task_attempts.c.attempt)
                )
            )
            .mappings()
            .all()
        )
        result = (
            await conn.execute(sa.select(tasks.c.result).where(tasks.c.id == task_id))
        ).scalar_one()

    assert [(r["attempt"], r["outcome"]) for r in outcomes] == [(1, "LOST"), (2, "SUCCEEDED")]
    assert sum(1 for r in outcomes if r["outcome"] == "SUCCEEDED") == 1
    assert result == {"from": "worker-b"}


async def test_stale_worker_cannot_renew_its_way_back_in(engine, make_task, make_worker) -> None:
    """A revived worker must not be able to re-extend a lapsed lease.

    This is the failure the `lease_expires_at > now()` clause exists for. The
    attempt fence alone would not catch it: before anyone else reclaims the
    task, the zombie still matches on state, attempt AND worker_id. Without
    the expiry check it silently renews and keeps running, and whether it
    keeps the task comes down to a footrace with the reaper.
    """
    worker_a = await make_worker()
    task_id = await make_task()

    claimed = await _claim_one(engine, worker_a)
    assert claimed["attempt"] == 1

    await _expire_lease(engine, task_id)

    # Nobody has reclaimed it yet -- state, attempt and owner all still match.
    async with engine.connect() as conn, conn.begin():
        renewed = await renew_lease(
            conn, task_id, worker_id=worker_a, expect_attempt=1, lease_ttl_s=LEASE_TTL_S
        )
    assert renewed is None, "an expired lease was renewed; 'expired' must mean 'lost'"


async def test_reaper_loses_to_a_worker_that_renews_in_time(engine, make_task, make_worker) -> None:
    """The mirror image: a worker that IS alive keeps its task.

    The reaper's fence is `lease_expires_at < now()` evaluated inside its
    UPDATE, so a renewal that commits first makes the reaper's predicate stop
    matching. A reaper that snapshotted expiry in Python would reclaim a task
    from a healthy worker.
    """
    worker_a = await make_worker()
    task_id = await make_task()
    claimed = await _claim_one(engine, worker_a)
    assert claimed["attempt"] == 1

    await _expire_lease(engine, task_id)

    # Worker A renews just in time -- push the deadline back into the future.
    async with engine.connect() as conn, conn.begin():
        await conn.execute(
            sa.update(tasks)
            .where(tasks.c.id == task_id)
            .values(lease_expires_at=sa.func.now() + sa.text("interval '60 seconds'"))
        )

    async with engine.connect() as conn, conn.begin():
        assert await reap_expired_leases(conn, limit=10) == 0

    async with engine.connect() as conn:
        state = (
            await conn.execute(sa.select(tasks.c.state).where(tasks.c.id == task_id))
        ).scalar_one()
    assert state == State.RUNNING, "a live worker's task was reclaimed out from under it"


async def test_concurrent_reap_and_completion_produce_one_outcome(
    engine, make_task, make_worker
) -> None:
    """Worker finishes at the exact moment its lease expires.

    Both writes target the same row, so PostgreSQL's row lock serialises them
    and whichever commits second finds its predicate no longer satisfied.
    There is no interleaving in which the task both succeeds and is requeued.
    """
    worker_a = await make_worker()
    task_id = await make_task()
    claimed = await _claim_one(engine, worker_a)
    attempt = claimed["attempt"]
    await _expire_lease(engine, task_id)

    async def finish():
        async with engine.connect() as conn, conn.begin():
            return await complete_success(
                conn, task_id, attempt=attempt, worker_id=worker_a, result={}
            )

    async def reap():
        async with engine.connect() as conn, conn.begin():
            return await reap_expired_leases(conn, limit=10)

    finished, reaped = await asyncio.gather(finish(), reap())

    # Exactly one of them took effect.
    assert (1 if finished.applied else 0) + reaped == 1, (
        f"expected exactly one outcome, got completed={finished.applied} reaped={reaped}"
    )

    async with engine.connect() as conn:
        state = (
            await conn.execute(sa.select(tasks.c.state).where(tasks.c.id == task_id))
        ).scalar_one()
    assert state == (State.SUCCEEDED if finished.applied else State.QUEUED)


async def test_recovery_is_visible_in_the_event_timeline(engine, make_task, make_worker) -> None:
    """The execution history has to explain what happened, not just the outcome.

    This timeline is what a broker-backed queue cannot give you: the task's
    own record of being claimed, lost, recovered and reclaimed.
    """
    worker_a = await make_worker()
    worker_b = await make_worker()
    task_id = await make_task()

    await _claim_one(engine, worker_a)
    await _expire_lease(engine, task_id)
    async with engine.connect() as conn, conn.begin():
        await reap_expired_leases(conn, limit=10)
    await _claim_one(engine, worker_b)

    async with engine.connect() as conn:
        events = (
            (
                await conn.execute(
                    sa.select(task_events.c.event_type)
                    .where(task_events.c.task_id == task_id)
                    .order_by(task_events.c.id)
                )
            )
            .scalars()
            .all()
        )

    # No TASK_CREATED: the make_task fixture inserts directly, bypassing the
    # API's submit path. Everything from the claim onward is the real thing.
    #
    # TASK_RECOVERED precedes WORKER_LOST because the reaper only records the
    # loss once its recovery CAS has actually applied -- it will not claim a
    # worker was lost on the strength of a write that then lost a race.
    assert events == [
        EventType.TASK_CLAIMED.value,
        EventType.TASK_RECOVERED.value,
        EventType.WORKER_LOST.value,
        EventType.TASK_CLAIMED.value,
    ]
