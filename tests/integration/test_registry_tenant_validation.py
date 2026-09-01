"""Regression tests for a bug found reviewing the pushed repo.

`agents.tenant_id` and `tools.tenant_id` both carry a Postgres FK to
`tenants.id`. Neither POST /v1/agents nor POST /v1/tools validated the
tenant existed before inserting -- unlike POST /v1/tasks, which does. A bad
tenant_id therefore raised a ForeignKeyViolation, wrapped by SQLAlchemy as
the SAME IntegrityError the route's `except` clause was written to catch for
a DUPLICATE NAME -- so the API reported "agent name already exists" (409)
for a tenant that does not exist at all. Fixed by checking tenant existence
explicitly first, matching the pattern already used correctly on task
submission.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import httpx
import pytest

from acp.api.app import app
from acp.config import settings
from acp.db.session import dispose_engine

pytestmark = pytest.mark.db


@pytest.fixture
async def client(migrated_db, monkeypatch) -> AsyncIterator[httpx.AsyncClient]:
    monkeypatch.setenv("ACP_DATABASE_URL", migrated_db)
    settings.cache_clear()
    await dispose_engine()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await dispose_engine()
    settings.cache_clear()


async def test_creating_an_agent_for_an_unknown_tenant_is_404_not_409(client) -> None:
    """The bug, reproduced directly.

    Before the fix this returned 409 "agent name already exists" -- true of
    neither the name (never used before) nor, misleadingly, the tenant.
    """
    resp = await client.post(
        "/v1/agents",
        json={"tenant_id": str(uuid.uuid4()), "name": "ghost-agent"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


async def test_creating_a_tool_for_an_unknown_tenant_is_404_not_409(client) -> None:
    resp = await client.post(
        "/v1/tools",
        json={"tenant_id": str(uuid.uuid4()), "name": "ghost-tool"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


async def test_a_genuine_duplicate_name_is_still_409(client) -> None:
    """The fix must not have broken the case it was protecting."""
    tenant = (await client.post("/v1/tenants", json={"name": f"t-{uuid.uuid4().hex[:8]}"})).json()

    first = await client.post("/v1/agents", json={"tenant_id": tenant["id"], "name": "dup-agent"})
    second = await client.post("/v1/agents", json={"tenant_id": tenant["id"], "name": "dup-agent"})

    assert first.status_code == 201
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "conflict"


async def test_a_genuine_duplicate_tool_name_is_still_409(client) -> None:
    tenant = (await client.post("/v1/tenants", json={"name": f"t-{uuid.uuid4().hex[:8]}"})).json()

    first = await client.post("/v1/tools", json={"tenant_id": tenant["id"], "name": "dup-tool"})
    second = await client.post("/v1/tools", json={"tenant_id": tenant["id"], "name": "dup-tool"})

    assert first.status_code == 201
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "conflict"
