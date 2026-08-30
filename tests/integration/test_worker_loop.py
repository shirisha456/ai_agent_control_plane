"""End-to-end worker behaviour: claim -> execute -> complete, retry, cancel."""

from __future__ import annotations

import asyncio

import pytest
import sqlalchemy as sa

from acp.agent.adapters.base import Adapter, AdapterRegistry
from acp.config import Settings
from acp.db.models import tasks
from acp.domain.errors import AdapterError, FailureClass
from acp.domain.states import State
from acp.worker.loop import Worker

pytestmark = pytest.mark.db


class _ZeroBackoffFailure(AdapterError):
    """WORKER_LOST retries with zero backoff, so the test needs no stub."""

    failure_class = FailureClass.WORKER_LOST


def _settings(**overrides) -> Settings:
    defaults = dict(
        database_url="unused",
        lease_ttl_s=30,
        lease_renew_interval_s=7,
        poll_interval_ms=20,
        claim_batch_size=5,
    )
    defaults.update(overrides)
    return Settings(**defaults)


async def _run_until_terminal(engine, worker: Worker, task_id, timeout: float = 5.0) -> str:
    run_task = asyncio.ensure_future(worker.run_forever())
    try:
        async with asyncio.timeout(timeout):
            while True:
                async with engine.connect() as conn:
                    state = (
                        await conn.execute(sa.select(tasks.c.state).where(tasks.c.id == task_id))
                    ).scalar_one()
                if state in (
                    State.SUCCEEDED.value,
                    State.FAILED.value,
                    State.CANCELLED.value,
                ):
                    return state
                await asyncio.sleep(0.02)
    finally:
        worker.stop()
        await asyncio.wait_for(run_task, timeout=5.0)


class _Echo(Adapter):
    async def run(self, payload, *, is_cancelled):
        return {"got": dict(payload)}


class _AlwaysFail(Adapter):
    """Fails as WORKER_LOST, whose policy is zero backoff.

    Using a real failure class rather than patching the backoff away keeps
    the test exercising the actual retry path: the schedule comes from
    acp.domain.retry, not from a stub.
    """

    async def run(self, payload, *, is_cancelled):
        raise _ZeroBackoffFailure("nope")


async def test_worker_claims_and_succeeds(engine, make_task, monkeypatch) -> None:
    import acp.worker.loop as loop_mod

    monkeypatch.setattr(loop_mod, "transaction", _bind_transaction(engine))

    task_id = await make_task(payload={"x": 1})
    registry = AdapterRegistry()
    registry.register("demo.agent", _Echo)
    worker = Worker(settings=_settings(), registry=registry, capacity=2)

    final_state = await _run_until_terminal(engine, worker, task_id)
    assert final_state == State.SUCCEEDED.value

    async with engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    sa.select(tasks.c.result, tasks.c.attempt).where(tasks.c.id == task_id)
                )
            )
            .mappings()
            .one()
        )
    assert row["result"] == {"got": {"x": 1}}
    assert row["attempt"] == 1


async def test_worker_retries_then_exhausts_to_failed(engine, make_task, monkeypatch) -> None:
    import acp.worker.loop as loop_mod

    monkeypatch.setattr(loop_mod, "transaction", _bind_transaction(engine))

    task_id = await make_task(task_type="demo.fail", max_attempts=2)
    registry = AdapterRegistry()
    registry.register("demo.fail", _AlwaysFail)
    worker = Worker(settings=_settings(), registry=registry, capacity=1)

    final_state = await _run_until_terminal(engine, worker, task_id)
    assert final_state == State.FAILED.value

    async with engine.connect() as conn:
        row = (
            (await conn.execute(sa.select(tasks.c.attempt).where(tasks.c.id == task_id)))
            .mappings()
            .one()
        )
    assert row["attempt"] == 2


def _bind_transaction(engine):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _tx():
        async with engine.connect() as conn, conn.begin():
            yield conn

    return _tx
