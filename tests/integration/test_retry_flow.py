"""Retry behaviour end to end: classification, backoff, and non-blocking waits."""

from __future__ import annotations

import asyncio
import random
from contextlib import asynccontextmanager

import pytest
import sqlalchemy as sa

from acp.agent.adapters.base import Adapter, AdapterRegistry
from acp.config import Settings
from acp.db.models import task_attempts, task_events, tasks
from acp.domain.errors import (
    FailureClass,
    InvalidInput,
    PermanentFailure,
    RateLimited,
    Retryable,
)
from acp.domain.states import EventType, State
from acp.worker.loop import Worker

pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
async def _isolated(clean_tasks):
    yield


def _settings(**overrides) -> Settings:
    defaults = dict(
        database_url="unused",
        lease_ttl_s=30,
        lease_renew_interval_s=7,
        heartbeat_interval_s=60,
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


class _Raises(Adapter):
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def run(self, payload, *, is_cancelled):
        raise self.exc


class _Instant(Adapter):
    async def run(self, payload, *, is_cancelled):
        return {"ok": True}


async def _drive(engine, worker: Worker, task_id, timeout=8.0) -> str:
    run = asyncio.ensure_future(worker.run_forever())
    try:
        async with asyncio.timeout(timeout):
            while True:
                async with engine.connect() as conn:
                    state = (
                        await conn.execute(sa.select(tasks.c.state).where(tasks.c.id == task_id))
                    ).scalar_one()
                if state in (State.SUCCEEDED, State.FAILED, State.CANCELLED):
                    return state
                await asyncio.sleep(0.02)
    finally:
        worker.stop()
        await asyncio.wait_for(run, timeout=10.0)


async def _run_with(engine, monkeypatch, exc, *, max_attempts=3, task_type="demo.raise"):
    import acp.worker.loop as loop_mod

    monkeypatch.setattr(loop_mod, "transaction", _bind_transaction(engine))
    registry = AdapterRegistry()
    registry.register(task_type, lambda: _Raises(exc))
    return Worker(settings=_settings(), registry=registry, capacity=1, rng=random.Random(1234))


# ---------------------------------------------------------------------------
# classification decides whether a retry happens at all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected_class"),
    [
        (PermanentFailure("nope"), FailureClass.PERMANENT),
        (InvalidInput("bad payload"), FailureClass.USER_ERROR),
    ],
)
async def test_non_retryable_failures_terminate_on_the_first_attempt(
    engine, make_task, monkeypatch, exc, expected_class
) -> None:
    """max_attempts=5 and it still stops at 1.

    The budget is a ceiling, not a quota to spend: retrying identical input
    that was rejected for being wrong is guaranteed waste.
    """
    task_id = await make_task(task_type="demo.raise", max_attempts=5)
    worker = await _run_with(engine, monkeypatch, exc)

    assert await _drive(engine, worker, task_id) == State.FAILED

    async with engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    sa.select(tasks.c.attempt, tasks.c.error_class).where(tasks.c.id == task_id)
                )
            )
            .mappings()
            .one()
        )
    assert row["attempt"] == 1, "a non-retryable failure consumed extra attempts"
    # error_class stores the FAILURE CLASS, not the Python exception type --
    # it is a metric label and a query filter, so it needs a bounded vocabulary.
    assert row["error_class"] == expected_class.value


async def test_transient_failures_retry_until_the_budget_is_spent(
    engine, make_task, monkeypatch
) -> None:
    task_id = await make_task(task_type="demo.raise", max_attempts=3)
    worker = await _run_with(engine, monkeypatch, Retryable("flaky"))

    assert await _drive(engine, worker, task_id, timeout=20.0) == State.FAILED

    async with engine.connect() as conn:
        attempts = (
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

    assert [a["attempt"] for a in attempts] == [1, 2, 3]
    assert all(a["outcome"] == "FAILED" for a in attempts)
    # Two retries scheduled, then one terminal failure -- not three retries.
    assert events.count(EventType.RETRY_SCHEDULED.value) == 2
    assert events.count(EventType.TASK_FAILED.value) == 1


async def test_unknown_failures_stop_before_the_task_budget(engine, make_task, monkeypatch) -> None:
    """An unclassified error gets a shorter leash than a known-transient one.

    Same task, same max_attempts, different failure class, different number of
    attempts spent -- which is the whole point of classifying.
    """

    class VendorExplosion(Exception):
        pass

    task_id = await make_task(task_type="demo.raise", max_attempts=5)
    worker = await _run_with(engine, monkeypatch, VendorExplosion("???"))

    assert await _drive(engine, worker, task_id, timeout=20.0) == State.FAILED

    async with engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    sa.select(tasks.c.attempt, tasks.c.error_class).where(tasks.c.id == task_id)
                )
            )
            .mappings()
            .one()
        )
    assert row["error_class"] == FailureClass.UNKNOWN.value
    assert row["attempt"] == 2, "UNKNOWN should stop at its attempt cap, not at max_attempts"


# ---------------------------------------------------------------------------
# the backoff is scheduled in the database, not slept in the worker
# ---------------------------------------------------------------------------


async def test_backoff_is_stored_as_a_future_available_at(engine, make_task, monkeypatch) -> None:
    """A retrying task is QUEUED with available_at in the future.

    That is the entire implementation of "retrying": no RETRYING state, and
    the claim query's `available_at <= now()` predicate excludes it for free.
    """
    task_id = await make_task(task_type="demo.raise", max_attempts=3)
    worker = await _run_with(engine, monkeypatch, RateLimited("slow down", retry_after_s=60))

    run = asyncio.ensure_future(worker.run_forever())
    try:
        async with asyncio.timeout(8):
            while True:
                async with engine.connect() as conn:
                    row = (
                        (
                            await conn.execute(
                                sa.select(
                                    tasks.c.state,
                                    tasks.c.attempt,
                                    tasks.c.error_class,
                                    (tasks.c.available_at > sa.func.now()).label("deferred"),
                                ).where(tasks.c.id == task_id)
                            )
                        )
                        .mappings()
                        .one()
                    )
                if row["attempt"] >= 1 and row["state"] == State.QUEUED:
                    break
                await asyncio.sleep(0.02)
    finally:
        worker.stop()
        await asyncio.wait_for(run, timeout=10.0)

    assert row["state"] == State.QUEUED
    assert row["error_class"] == FailureClass.RATE_LIMITED.value
    # Retry-After is a floor: 60s from the server means not before 60s.
    assert row["deferred"] is True


async def test_a_backing_off_task_does_not_block_the_worker(engine, make_task, monkeypatch) -> None:
    """THE property that makes database-side backoff worth having.

    A worker that slept through a backoff would hold a slot doing nothing, so
    one repeatedly-failing task would consume capacity proportional to its own
    backoff curve. Here the failing task is deferred for a minute and the
    worker -- capacity 1 -- must still drain everything else meanwhile.
    """
    import acp.worker.loop as loop_mod

    monkeypatch.setattr(loop_mod, "transaction", _bind_transaction(engine))

    slow_id = await make_task(task_type="demo.raise", max_attempts=3, priority=0)
    others = [await make_task(task_type="demo.ok", priority=50) for _ in range(5)]

    registry = AdapterRegistry()
    registry.register("demo.raise", lambda: _Raises(RateLimited("429", retry_after_s=60)))
    registry.register("demo.ok", _Instant)
    worker = Worker(settings=_settings(), registry=registry, capacity=1, rng=random.Random(7))

    run = asyncio.ensure_future(worker.run_forever())
    try:
        async with asyncio.timeout(10):
            while True:
                async with engine.connect() as conn:
                    done = (
                        await conn.execute(
                            sa.select(sa.func.count())
                            .select_from(tasks)
                            .where(tasks.c.id.in_(others), tasks.c.state == State.SUCCEEDED.value)
                        )
                    ).scalar_one()
                if done == len(others):
                    break
                await asyncio.sleep(0.02)
    finally:
        worker.stop()
        await asyncio.wait_for(run, timeout=10.0)

    async with engine.connect() as conn:
        state = (
            await conn.execute(sa.select(tasks.c.state).where(tasks.c.id == slow_id))
        ).scalar_one()
    assert state == State.QUEUED, "the deferred task should still be waiting"


async def test_retry_storm_is_dispersed_not_synchronised(engine, make_task, monkeypatch) -> None:
    """200 tasks fail at once; their retries must not all land at once.

    This is the thundering-herd property. With deterministic backoff every
    task would come back at the same instant and hit the still-recovering
    dependency together. Asserted on the spread of available_at, which is what
    the claim query actually reads.
    """
    import acp.worker.loop as loop_mod

    monkeypatch.setattr(loop_mod, "transaction", _bind_transaction(engine))

    ids = [await make_task(task_type="demo.raise", max_attempts=3) for _ in range(200)]
    registry = AdapterRegistry()
    registry.register("demo.raise", lambda: _Raises(Retryable("dependency down")))
    worker = Worker(settings=_settings(), registry=registry, capacity=25, rng=random.Random(99))

    run = asyncio.ensure_future(worker.run_forever())
    try:
        async with asyncio.timeout(20):
            while True:
                async with engine.connect() as conn:
                    retried = (
                        await conn.execute(
                            sa.select(sa.func.count())
                            .select_from(tasks)
                            .where(tasks.c.id.in_(ids), tasks.c.attempt >= 1)
                        )
                    ).scalar_one()
                if retried == len(ids):
                    break
                await asyncio.sleep(0.05)
    finally:
        worker.stop()
        await asyncio.wait_for(run, timeout=15.0)

    async with engine.connect() as conn:
        spread = (
            await conn.execute(
                sa.select(
                    sa.extract(
                        "epoch",
                        sa.func.max(tasks.c.available_at) - sa.func.min(tasks.c.available_at),
                    )
                ).where(tasks.c.id.in_(ids))
            )
        ).scalar_one()

    # Base is 1s for TRANSIENT at attempt 1, so full jitter should spread the
    # herd across most of that interval. Deterministic backoff would give ~0.
    assert float(spread) > 0.3, f"retries clustered into {spread}s -- the herd was not dispersed"
