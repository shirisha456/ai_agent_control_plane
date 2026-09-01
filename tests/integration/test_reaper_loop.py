"""End-to-end reaper behaviour: run_forever actually recovers a dead worker's task."""

from __future__ import annotations

import asyncio

import pytest
import sqlalchemy as sa

from acp.config import Settings
from acp.db.models import tasks
from acp.domain.states import State
from acp.reaper.loop import Reaper

pytestmark = pytest.mark.db


def _settings(**overrides) -> Settings:
    defaults = dict(database_url="unused", reaper_period_s=1, worker_dead_after_s=1)
    defaults.update(overrides)
    return Settings(**defaults)


async def test_reaper_recovers_a_task_with_an_expired_lease(
    engine, make_task, make_worker, monkeypatch
) -> None:
    import acp.reaper.loop as reaper_mod

    monkeypatch.setattr(reaper_mod, "transaction", _bind_transaction(engine))

    worker_id = await make_worker()
    task_id = await make_task(
        state=State.RUNNING.value,
        attempt=1,
        lease_worker_id=worker_id,
        lease_expires_at=sa.text("now() - interval '1 second'"),
    )

    reaper = Reaper(settings=_settings(), batch_size=10)
    run_task = asyncio.ensure_future(reaper.run_forever())
    try:
        async with asyncio.timeout(5.0):
            while True:
                async with engine.connect() as conn:
                    state = (
                        await conn.execute(sa.select(tasks.c.state).where(tasks.c.id == task_id))
                    ).scalar_one()
                if state == State.QUEUED.value:
                    break
                await asyncio.sleep(0.05)
    finally:
        reaper.stop()
        await asyncio.wait_for(run_task, timeout=5.0)


async def test_reaper_recovers_a_hung_task_despite_a_live_lease(
    engine, make_task, make_worker, monkeypatch
) -> None:
    """The hung-task sweep, not just the lease sweep, runs as part of run_forever."""
    import acp.reaper.loop as reaper_mod

    monkeypatch.setattr(reaper_mod, "transaction", _bind_transaction(engine))

    worker_id = await make_worker()
    task_id = await make_task(
        state=State.RUNNING.value,
        attempt=1,
        lease_worker_id=worker_id,
        lease_expires_at=sa.text("now() + interval '30 seconds'"),
        first_started_at=sa.text("now() - interval '1 hour'"),
        max_execution_time_s=1,
    )

    reaper = Reaper(settings=_settings(), batch_size=10)
    run_task = asyncio.ensure_future(reaper.run_forever())
    try:
        async with asyncio.timeout(5.0):
            while True:
                async with engine.connect() as conn:
                    state = (
                        await conn.execute(sa.select(tasks.c.state).where(tasks.c.id == task_id))
                    ).scalar_one()
                if state == State.QUEUED.value:
                    break
                await asyncio.sleep(0.05)
    finally:
        reaper.stop()
        await asyncio.wait_for(run_task, timeout=5.0)


def _bind_transaction(engine):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _tx():
        async with engine.connect() as conn, conn.begin():
            yield conn

    return _tx
