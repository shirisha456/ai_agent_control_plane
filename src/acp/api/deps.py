"""FastAPI dependencies.

One transaction per request. Every handler that writes gets an open
transaction it shares with acp.db.queries, so a task insert and its
TASK_CREATED event commit together -- which is what makes the event log
incapable of disagreeing with task state.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncConnection

from acp.db.session import engine


async def db_txn() -> AsyncIterator[AsyncConnection]:
    """Open a transaction; commit on success, roll back on any exception.

    Rollback-on-exception is the reason handlers may raise freely: a request
    that fails halfway leaves no partial task behind.
    """
    async with engine().connect() as conn, conn.begin():
        yield conn


async def db_read() -> AsyncIterator[AsyncConnection]:
    """Read-only path: no transaction block, so no write locks are taken."""
    async with engine().connect() as conn:
        yield conn
