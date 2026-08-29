"""The health endpoint must reflect the database, not merely the process.

A control plane that reports healthy while unable to reach its only source of
truth is worse than one that reports unhealthy: load balancers keep routing
submissions to it, and clients see writes fail instead of being retried
against a replica that can actually commit them.
"""

from __future__ import annotations

import httpx
import pytest

from acp.api.app import app
from acp.config import settings
from acp.db.session import dispose_engine

pytestmark = pytest.mark.db


async def test_healthz_reports_database_reachability(migrated_db, monkeypatch) -> None:
    monkeypatch.setenv("ACP_DATABASE_URL", migrated_db)
    settings.cache_clear()
    await dispose_engine()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/healthz")

        assert resp.status_code == 200
        assert resp.json() == {"status": "healthy", "database": "reachable"}
    finally:
        await dispose_engine()
        settings.cache_clear()


async def test_healthz_reports_503_when_database_is_gone(monkeypatch) -> None:
    """The failure path is the one that matters, so it gets its own test.

    It also pins the timeout budget. Before /healthz was bounded, this test
    took 130 seconds -- which is exactly how long a real load balancer would
    have waited on a probe before learning anything.
    """
    import time

    started = time.monotonic()
    monkeypatch.setenv(
        "ACP_DATABASE_URL", "postgresql+psycopg://acp:acp@127.0.0.1:1/definitely_not_here"
    )
    settings.cache_clear()
    await dispose_engine()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/healthz")

        assert resp.status_code == 503
        assert resp.json()["database"] in ("unreachable", "timeout")
        assert time.monotonic() - started < 10, (
            "the probe must fail within its budget; an unbounded health check "
            "makes a slow database indistinguishable from a dead one"
        )
    finally:
        await dispose_engine()
        settings.cache_clear()
