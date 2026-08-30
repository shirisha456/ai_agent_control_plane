"""Integration tests for the reaper: expired-lease recovery and dead-worker marking."""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from acp.db.models import task_attempts, task_events, tasks, workers
from acp.db.queries.reap import reap_expired_leases
from acp.db.queries.transitions import Rejection, transition
from acp.db.queries.workers import mark_dead_workers
from acp.domain.states import EventType, State

pytestmark = pytest.mark.db

EXPIRED = sa.text("now() - interval '1 second'")
FUTURE = sa.text("now() + interval '30 seconds'")


async def _insert_attempt_row(engine, task_id, attempt, worker_id) -> None:
    """Mirror what claim_tasks would have inserted, for tests that skip the real claim."""
    async with engine.connect() as conn, conn.begin():
        await conn.execute(
            sa.insert(task_attempts).values(task_id=task_id, attempt=attempt, worker_id=worker_id)
        )


async def test_reap_requeues_task_with_remaining_attempts(engine, make_task, make_worker) -> None:
    worker_id = await make_worker()
    task_id = await make_task(
        state=State.RUNNING.value,
        attempt=1,
        max_attempts=3,
        lease_worker_id=worker_id,
        lease_expires_at=EXPIRED,
    )
    await _insert_attempt_row(engine, task_id, 1, worker_id)

    async with engine.connect() as conn, conn.begin():
        reaped = await reap_expired_leases(conn, limit=10)
    assert reaped == 1

    async with engine.connect() as conn:
        row = (await conn.execute(sa.select(tasks).where(tasks.c.id == task_id))).mappings().one()
    assert row["state"] == State.QUEUED.value
    assert row["attempt"] == 1  # not bumped here -- the next claim bumps it
    assert row["lease_worker_id"] is None
    assert row["lease_expires_at"] is None
    assert row["error_class"] == "LeaseExpired"

    async with engine.connect() as conn:
        attempt_row = (
            (
                await conn.execute(
                    sa.select(task_attempts.c.outcome).where(
                        task_attempts.c.task_id == task_id, task_attempts.c.attempt == 1
                    )
                )
            )
            .mappings()
            .one()
        )
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
    assert attempt_row["outcome"] == "LOST"
    assert events == [EventType.TASK_RECOVERED.value, EventType.WORKER_LOST.value]


async def test_reap_fails_task_with_exhausted_attempts(engine, make_task, make_worker) -> None:
    worker_id = await make_worker()
    task_id = await make_task(
        state=State.RUNNING.value,
        attempt=3,
        max_attempts=3,
        lease_worker_id=worker_id,
        lease_expires_at=EXPIRED,
    )
    await _insert_attempt_row(engine, task_id, 3, worker_id)

    async with engine.connect() as conn, conn.begin():
        reaped = await reap_expired_leases(conn, limit=10)
    assert reaped == 1

    async with engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    sa.select(tasks.c.state, tasks.c.finished_at).where(tasks.c.id == task_id)
                )
            )
            .mappings()
            .one()
        )
        attempt_row = (
            (
                await conn.execute(
                    sa.select(task_attempts.c.outcome).where(
                        task_attempts.c.task_id == task_id, task_attempts.c.attempt == 3
                    )
                )
            )
            .mappings()
            .one()
        )
    assert row["state"] == State.FAILED.value
    assert row["finished_at"] is not None
    assert attempt_row["outcome"] == "ABANDONED"


async def test_reap_ignores_tasks_with_live_leases(engine, make_task, make_worker) -> None:
    worker_id = await make_worker()
    task_id = await make_task(
        state=State.RUNNING.value,
        attempt=1,
        lease_worker_id=worker_id,
        lease_expires_at=FUTURE,
    )

    async with engine.connect() as conn, conn.begin():
        reaped = await reap_expired_leases(conn, limit=10)
    assert reaped == 0

    async with engine.connect() as conn:
        state = (
            await conn.execute(sa.select(tasks.c.state).where(tasks.c.id == task_id))
        ).scalar_one()
    assert state == State.RUNNING.value


async def test_reap_fence_rejects_a_lease_renewed_after_the_candidate_scan(
    engine, make_task, make_worker
) -> None:
    """The race the reaper exists to not lose: renewal wins if it commits first."""
    worker_id = await make_worker()
    task_id = await make_task(
        state=State.RUNNING.value,
        attempt=1,
        lease_worker_id=worker_id,
        lease_expires_at=EXPIRED,
    )

    # Simulate the owner renewing between the reaper's candidate SELECT and
    # its recovery UPDATE.
    async with engine.connect() as conn, conn.begin():
        await conn.execute(
            sa.update(tasks).where(tasks.c.id == task_id).values(lease_expires_at=FUTURE)
        )

    async with engine.connect() as conn, conn.begin():
        res = await transition(
            conn,
            task_id,
            expect_state=State.RUNNING,
            to_state=State.QUEUED,
            event_type=EventType.TASK_RECOVERED,
            expect_lease_expired=True,
            set_fields={"lease_worker_id": None, "lease_expires_at": None},
        )
    assert not res.applied
    assert res.rejection is Rejection.LEASE_NOT_EXPIRED

    async with engine.connect() as conn:
        state = (
            await conn.execute(sa.select(tasks.c.state).where(tasks.c.id == task_id))
        ).scalar_one()
    assert state == State.RUNNING.value


async def test_mark_dead_workers_flips_stale_heartbeats(engine, make_worker) -> None:
    # Other tests in this session-scoped database may already have workers,
    # so scope assertions to just these two rather than the full return set.
    stale = await make_worker(last_heartbeat_at=EXPIRED)
    fresh = await make_worker(last_heartbeat_at=FUTURE)

    async with engine.connect() as conn, conn.begin():
        dead = await mark_dead_workers(conn, dead_after_s=0)
    assert stale in dead
    assert fresh not in dead

    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    sa.select(workers.c.id, workers.c.status).where(
                        workers.c.id.in_([stale, fresh])
                    )
                )
            )
            .mappings()
            .all()
        )
    status_by_id = {r["id"]: r["status"] for r in rows}
    assert status_by_id[stale] == "DEAD"
    assert status_by_id[fresh] == "ALIVE"


async def test_mark_dead_workers_is_idempotent(engine, make_worker) -> None:
    # Other session-scoped rows may also be stale by dead_after_s=0, so check
    # membership for this worker rather than exact equality of the full set.
    worker_id = await make_worker(last_heartbeat_at=EXPIRED)

    async with engine.connect() as conn, conn.begin():
        first = await mark_dead_workers(conn, dead_after_s=0)
    assert worker_id in first

    async with engine.connect() as conn, conn.begin():
        second = await mark_dead_workers(conn, dead_after_s=0)
    assert worker_id not in second
