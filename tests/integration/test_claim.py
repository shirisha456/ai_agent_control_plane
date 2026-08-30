"""Integration tests for the claim mechanism: FOR UPDATE SKIP LOCKED + lease grant."""

from __future__ import annotations

import asyncio

import pytest
import sqlalchemy as sa

from acp.db.models import task_attempts, tasks, tenants
from acp.db.queries.claim import claim_tasks
from acp.domain.states import State
from acp.scheduling.policy import DEFAULT_POLICY

pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
async def _isolated(clean_tasks):
    """These assert WHICH task the claim returned, so no leftovers.

    The claim query is global by design -- it does not filter by test --
    so another suite's QUEUED work would be claimed ahead of this one's
    and the assertions would fail for a reason that has nothing to do
    with claiming.
    """
    yield


async def test_claim_grants_lease_and_bumps_attempt(engine, make_task, make_worker) -> None:
    w1 = await make_worker()
    task_id = await make_task()
    async with engine.connect() as conn, conn.begin():
        claimed = await claim_tasks(
            conn, worker_id=w1, limit=5, lease_ttl_s=30, policy=DEFAULT_POLICY
        )
    assert [c["id"] for c in claimed] == [task_id]
    row = claimed[0]
    assert row["state"] == State.RUNNING
    assert row["attempt"] == 1
    assert row["lease_worker_id"] == w1
    assert row["lease_expires_at"] is not None


async def test_claim_writes_task_attempts_row(engine, make_task, make_worker) -> None:
    w1 = await make_worker()
    task_id = await make_task()
    async with engine.connect() as conn, conn.begin():
        await claim_tasks(conn, worker_id=w1, limit=5, lease_ttl_s=30, policy=DEFAULT_POLICY)

    async with engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    sa.select(task_attempts.c.worker_id, task_attempts.c.attempt).where(
                        task_attempts.c.task_id == task_id
                    )
                )
            )
            .mappings()
            .one()
        )
    assert row["worker_id"] == w1
    assert row["attempt"] == 1


async def test_claim_respects_priority_and_fifo_order(engine, make_task, make_worker) -> None:
    w1 = await make_worker()
    low = await make_task(priority=100)
    high = await make_task(priority=1)
    async with engine.connect() as conn, conn.begin():
        claimed = await claim_tasks(
            conn, worker_id=w1, limit=5, lease_ttl_s=30, policy=DEFAULT_POLICY
        )
    assert [c["id"] for c in claimed] == [high, low]


async def test_claim_skips_future_available_at(engine, make_task, make_worker) -> None:
    w1 = await make_worker()
    await make_task(available_at=sa.func.now() + sa.text("interval '1 hour'"))
    ready = await make_task()
    async with engine.connect() as conn, conn.begin():
        claimed = await claim_tasks(
            conn, worker_id=w1, limit=5, lease_ttl_s=30, policy=DEFAULT_POLICY
        )
    assert [c["id"] for c in claimed] == [ready]


async def test_claim_respects_tenant_concurrency_limit(
    engine, tenant_id, make_task, make_worker
) -> None:
    w1 = await make_worker()
    w2 = await make_worker()
    async with engine.connect() as conn, conn.begin():
        await conn.execute(
            sa.update(tenants).where(tenants.c.id == tenant_id).values(max_concurrent_tasks=1)
        )
    a = await make_task()
    b = await make_task()

    async with engine.connect() as conn, conn.begin():
        first = await claim_tasks(
            conn, worker_id=w1, limit=5, lease_ttl_s=30, policy=DEFAULT_POLICY
        )
    assert [c["id"] for c in first] == [a]

    # Tenant is now at its limit (one RUNNING task): the second task must not
    # be claimable even though it is QUEUED and eligible on every other axis.
    async with engine.connect() as conn, conn.begin():
        second = await claim_tasks(
            conn, worker_id=w2, limit=5, lease_ttl_s=30, policy=DEFAULT_POLICY
        )
    assert second == []
    assert b  # unclaimed, still QUEUED


async def test_concurrent_claims_never_double_claim(engine, make_task, make_worker) -> None:
    """N workers racing for the same small batch of tasks partition it exactly."""
    worker_ids = [await make_worker() for _ in range(4)]
    task_ids = {await make_task() for _ in range(8)}

    async def claim_batch(worker_id: str) -> list:
        async with engine.connect() as conn, conn.begin():
            return await claim_tasks(
                conn, worker_id=worker_id, limit=3, lease_ttl_s=30, policy=DEFAULT_POLICY
            )

    results = await asyncio.gather(*(claim_batch(w) for w in worker_ids))
    claimed_ids = [r["id"] for batch in results for r in batch]

    assert len(claimed_ids) == len(set(claimed_ids)), "the same task was claimed twice"
    assert set(claimed_ids) == task_ids

    async with engine.connect() as conn:
        states = (
            (await conn.execute(sa.select(tasks.c.state).where(tasks.c.id.in_(task_ids))))
            .scalars()
            .all()
        )
    assert all(s == State.RUNNING.value for s in states)
