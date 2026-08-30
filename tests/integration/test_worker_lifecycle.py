"""Worker lifecycle: self-fencing when declared dead, and graceful drain.

Both behaviours exist for the same reason: the difference between a worker
that stopped on purpose and one that stopped answering should be visible in
the data, not inferred from silence.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager

import pytest
import sqlalchemy as sa

from acp.agent.adapters.base import Adapter, AdapterRegistry
from acp.config import Settings
from acp.db.models import task_attempts, task_events, tasks, workers
from acp.db.queries.workers import heartbeat
from acp.domain.states import EventType, State
from acp.obs import metrics
from acp.worker.loop import Worker

pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
async def _isolated(clean_tasks):
    """These tests assert on which task a worker picked up, so no leftovers."""
    yield


def _settings(**overrides) -> Settings:
    defaults = dict(
        database_url="unused",
        lease_ttl_s=30,
        lease_renew_interval_s=7,
        heartbeat_interval_s=0,  # heartbeat on every poll, so tests see it promptly
        poll_interval_ms=20,
        claim_batch_size=5,
        drain_grace_s=0.2,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _bind_transaction(engine):
    @asynccontextmanager
    async def _tx():
        async with engine.connect() as conn, conn.begin():
            yield conn

    return _tx


class _Slow(Adapter):
    """Runs longer than the drain grace period, so shutdown must hand it back."""

    async def run(self, payload, *, is_cancelled):
        await asyncio.sleep(30)
        return {}


class _Echo(Adapter):
    async def run(self, payload, *, is_cancelled):
        return {"ok": True}


# ---------------------------------------------------------------------------
# self-fencing
# ---------------------------------------------------------------------------


async def test_heartbeat_reports_false_once_declared_dead(engine, make_worker) -> None:
    """Death is one-way.

    Without the `status <> 'DEAD'` predicate, a worker that was declared dead
    -- because it was paused, partitioned, or merely slow -- would resurrect
    itself with its next heartbeat, and the fleet's view of who is alive would
    be permanently wrong.
    """
    worker_id = await make_worker()

    async with engine.connect() as conn, conn.begin():
        assert await heartbeat(conn, worker_id=worker_id) is True

    async with engine.connect() as conn, conn.begin():
        await conn.execute(
            sa.update(workers).where(workers.c.id == worker_id).values(status="DEAD")
        )

    async with engine.connect() as conn, conn.begin():
        assert await heartbeat(conn, worker_id=worker_id) is False

    async with engine.connect() as conn:
        status = (
            await conn.execute(sa.select(workers.c.status).where(workers.c.id == worker_id))
        ).scalar_one()
    assert status == "DEAD", "a dead worker resurrected itself by heartbeating"


async def test_declared_dead_worker_stops_claiming(engine, make_task, monkeypatch) -> None:
    """A worker told it is dead must stop, not keep taking new work.

    Task safety never depended on this -- its writes are fenced by
    lease_worker_id + attempt regardless. What it prevents is a declared-dead
    worker quietly continuing to claim, so that `active_workers` and reality
    disagree and a slot's worth of duplicate execution is burned indefinitely.
    """
    import acp.worker.loop as loop_mod

    monkeypatch.setattr(loop_mod, "transaction", _bind_transaction(engine))

    registry = AdapterRegistry()
    registry.register("demo.agent", _Echo)
    worker = Worker(settings=_settings(), registry=registry, capacity=1)

    # run_forever registers the worker itself -- registration is generation-
    # unique, so it happens exactly once per process and cannot be done twice.
    # Wait for that row to appear, then declare it dead out from under it.
    run = asyncio.ensure_future(worker.run_forever())
    async with asyncio.timeout(5):
        while True:
            async with engine.connect() as conn:
                exists = (
                    await conn.execute(
                        sa.select(workers.c.id).where(workers.c.id == worker.worker_id)
                    )
                ).first()
            if exists:
                break
            await asyncio.sleep(0.01)

    async with engine.connect() as conn, conn.begin():
        await conn.execute(
            sa.update(workers).where(workers.c.id == worker.worker_id).values(status="DEAD")
        )

    # Created only now, so "did it claim?" is a real question: the loop
    # heartbeats before it claims, so a fenced worker must never reach this.
    task_id = await make_task()
    await asyncio.wait_for(run, timeout=5.0)

    assert worker._fenced is True

    async with engine.connect() as conn:
        state = (
            await conn.execute(sa.select(tasks.c.state).where(tasks.c.id == task_id))
        ).scalar_one()
    assert state == State.QUEUED, "a fenced worker claimed new work"


# ---------------------------------------------------------------------------
# graceful drain
# ---------------------------------------------------------------------------


async def test_graceful_shutdown_hands_back_unfinished_work(engine, make_task, monkeypatch) -> None:
    """SIGTERM recovers in milliseconds; SIGKILL takes a full lease_ttl.

    A stopping worker returns what it cannot finish with available_at = now(),
    so the next worker starts immediately instead of waiting out the lease.
    The gap between those two recovery latencies is a benchmark result, and it
    only exists because of this path.
    """
    import acp.worker.loop as loop_mod

    monkeypatch.setattr(loop_mod, "transaction", _bind_transaction(engine))

    task_id = await make_task(task_type="demo.slow")
    registry = AdapterRegistry()
    registry.register("demo.slow", _Slow)
    worker = Worker(settings=_settings(), registry=registry, capacity=1)

    run = asyncio.ensure_future(worker.run_forever())

    # Wait until the task is actually running before asking the worker to stop.
    async with asyncio.timeout(5):
        while True:
            async with engine.connect() as conn:
                state = (
                    await conn.execute(sa.select(tasks.c.state).where(tasks.c.id == task_id))
                ).scalar_one()
            if state == State.RUNNING:
                break
            await asyncio.sleep(0.02)

    worker.stop()
    await asyncio.wait_for(run, timeout=10.0)

    async with engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    sa.select(tasks.c.state, tasks.c.lease_worker_id, tasks.c.available_at).where(
                        tasks.c.id == task_id
                    )
                )
            )
            .mappings()
            .one()
        )
        outcome = (
            await conn.execute(
                sa.select(task_attempts.c.outcome).where(task_attempts.c.task_id == task_id)
            )
        ).scalar_one()
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
        status = (
            await conn.execute(sa.select(workers.c.status).where(workers.c.id == worker.worker_id))
        ).scalar_one()

    assert row["state"] == State.QUEUED, "unfinished work was not handed back"
    assert row["lease_worker_id"] is None
    # ABANDONED, not LOST: the worker gave this back on purpose. The reaper's
    # LOST means it was taken away because the worker stopped answering.
    assert outcome == "ABANDONED"
    assert EventType.TASK_ABANDONED.value in events
    assert status == "DEAD"


async def test_drained_task_is_immediately_reclaimable(engine, make_task, monkeypatch) -> None:
    """available_at = now(), so the handback costs no waiting at all."""
    import acp.worker.loop as loop_mod

    monkeypatch.setattr(loop_mod, "transaction", _bind_transaction(engine))

    task_id = await make_task(task_type="demo.slow")
    registry = AdapterRegistry()
    registry.register("demo.slow", _Slow)
    worker = Worker(settings=_settings(), registry=registry, capacity=1)

    run = asyncio.ensure_future(worker.run_forever())
    async with asyncio.timeout(5):
        while True:
            async with engine.connect() as conn:
                state = (
                    await conn.execute(sa.select(tasks.c.state).where(tasks.c.id == task_id))
                ).scalar_one()
            if state == State.RUNNING:
                break
            await asyncio.sleep(0.02)
    worker.stop()
    await asyncio.wait_for(run, timeout=10.0)

    from acp.db.queries.claim import claim_tasks
    from acp.scheduling.policy import DEFAULT_POLICY

    second_worker = f"worker-{uuid.uuid4().hex[:8]}"
    async with engine.connect() as conn, conn.begin():
        await conn.execute(
            sa.insert(workers).values(id=second_worker, hostname="test-host", pid=2, capacity=1)
        )
        claimed = await claim_tasks(
            conn, worker_id=second_worker, limit=5, lease_ttl_s=30, policy=DEFAULT_POLICY
        )

    assert task_id in [r["id"] for r in claimed], (
        "a handed-back task should be claimable immediately, with no backoff"
    )


async def test_execution_duration_is_recorded_for_every_attempt(
    engine, make_task, monkeypatch
) -> None:
    """A regression guard for instrumentation quietly going missing.

    Metrics have no test of their own unless one is written: the worker runs
    correctly whether or not it observes a histogram, so a lost
    `.observe()` call passes every functional test and only shows up as a
    permanently empty Grafana panel weeks later. This one failed for real --
    an edit silently no-matched and took the execution-duration timing with
    it, undetected until the tool-authorization work needed the same code
    path.
    """
    import acp.worker.loop as loop_mod

    monkeypatch.setattr(loop_mod, "transaction", _bind_transaction(engine))

    def _count() -> float:
        return sum(
            sample.value
            for metric in metrics.REGISTRY.collect()
            if metric.name == "acp_execution_duration_seconds"
            for sample in metric.samples
            if sample.name.endswith("_count")
        )

    before = _count()

    task_id = await make_task(task_type="demo.echo")
    registry = AdapterRegistry()
    registry.register("demo.echo", _Echo)
    worker = Worker(settings=_settings(), registry=registry, capacity=1)

    run = asyncio.ensure_future(worker.run_forever())
    try:
        async with asyncio.timeout(8):
            while True:
                async with engine.connect() as conn:
                    state = (
                        await conn.execute(sa.select(tasks.c.state).where(tasks.c.id == task_id))
                    ).scalar_one()
                if state == State.SUCCEEDED:
                    break
                await asyncio.sleep(0.02)
    finally:
        worker.stop()
        await asyncio.wait_for(run, timeout=10.0)

    assert _count() > before, (
        "acp_execution_duration_seconds was not observed -- the timing around "
        "adapter execution has gone missing"
    )
