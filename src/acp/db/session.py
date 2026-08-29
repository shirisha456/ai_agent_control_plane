"""Async engine and connection helpers.

We use SQLAlchemy Core over psycopg3's async driver. `now()` is used for every
timestamp rather than Python's clock: there is exactly one clock in this
system and it belongs to PostgreSQL. Lease deadlines are computed by the
database and compared by the database, which deletes the entire clock-skew
failure class -- a worker with a wrong clock can only renew early or late,
never wrongly believe it still owns a task.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from acp.config import settings

_engine: AsyncEngine | None = None


def engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        s = settings()
        _engine = create_async_engine(
            s.database_url,
            pool_size=s.db_pool_size,
            max_overflow=s.db_max_overflow,
            pool_pre_ping=True,
            future=True,
        )
    return _engine


async def dispose_engine() -> None:
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


@asynccontextmanager
async def transaction() -> AsyncIterator[AsyncConnection]:
    """One transaction. Every state change in this system happens inside one.

    Callers pass the connection down to acp.db.queries so that a state
    transition, its task_attempts row, and its task_events row commit or roll
    back together. That atomicity is why the event log can never disagree with
    task state.
    """
    async with engine().connect() as conn, conn.begin():
        yield conn
