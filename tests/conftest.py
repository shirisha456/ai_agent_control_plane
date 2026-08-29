"""Test fixtures.

Integration and concurrency tests run against a REAL PostgreSQL, never a
sqlite stand-in or a mock. The behaviour under test -- row locks, EvalPlanQual
re-evaluation of a CAS predicate, partial unique indexes -- is PostgreSQL
behaviour. A test that mocks it tests nothing.

The schema is built by running the actual Alembic migrations, so every test
run also verifies that the migrations produce the schema the code expects.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from acp.db.models import tasks, tenants
from acp.platform import event_loop_policy as _policy

TEST_URL = os.environ.get(
    "ACP_TEST_DATABASE_URL", "postgresql+psycopg://acp:acp@localhost:5434/acp_test"
)


@pytest.fixture(scope="session")
def event_loop_policy():
    """psycopg3 async cannot run on Windows' default ProactorEventLoop.

    pytest-asyncio consumes this fixture to build every test loop, so the
    whole suite gets a compatible loop without any test knowing about it.
    """
    return _policy()


@pytest.fixture(scope="session")
def migrated_db() -> str:
    """Drop and rebuild the test schema once per session via Alembic."""
    eng = sa.create_engine(TEST_URL, poolclass=NullPool)
    with eng.begin() as conn:
        conn.execute(sa.text("DROP SCHEMA public CASCADE"))
        conn.execute(sa.text("CREATE SCHEMA public"))
    eng.dispose()

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", TEST_URL)
    command.upgrade(cfg, "head")
    return TEST_URL


@pytest.fixture
async def engine(migrated_db: str) -> AsyncIterator[AsyncEngine]:
    """NullPool: every connect() opens a real backend connection.

    The concurrency tests need N genuinely concurrent PostgreSQL sessions. A
    pooled engine sized below N would quietly serialise them and the race
    tests would pass for the wrong reason -- the worst possible outcome for a
    test whose entire job is to prove contention is handled.
    """
    eng = create_async_engine(migrated_db, poolclass=NullPool, future=True)
    yield eng
    await eng.dispose()


@pytest.fixture
async def conn(engine: AsyncEngine) -> AsyncIterator[AsyncConnection]:
    async with engine.connect() as c:
        yield c


@pytest.fixture
async def tenant_id(engine: AsyncEngine) -> uuid.UUID:
    async with engine.connect() as c, c.begin():
        row = (
            await c.execute(
                sa.insert(tenants)
                .values(name=f"tenant-{uuid.uuid4().hex[:8]}")
                .returning(tenants.c.id)
            )
        ).scalar_one()
    return row


@pytest.fixture
async def make_task(engine: AsyncEngine, tenant_id: uuid.UUID):
    """Factory inserting a task directly, bypassing the (not yet written) API."""

    async def _make(**overrides) -> uuid.UUID:
        values: dict = {
            "tenant_id": tenant_id,
            "task_type": "demo.agent",
            "payload": {"steps": []},
        }
        values.update(overrides)
        async with engine.connect() as c, c.begin():
            return (
                await c.execute(sa.insert(tasks).values(**values).returning(tasks.c.id))
            ).scalar_one()

    return _make
