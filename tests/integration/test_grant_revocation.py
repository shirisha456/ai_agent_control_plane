"""The DELETE grant endpoint -- previously missing entirely.

db.queries.tools.revoke_grant existed with no API route: a grant added to a
DRAFT version could never be removed short of raw SQL, even though the
design explicitly allows a DRAFT version's grants to change (only an ACTIVE
version's are frozen). This is the endpoint that closes the gap.
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


@pytest.fixture
async def wired(client: httpx.AsyncClient) -> dict:
    tenant = (await client.post("/v1/tenants", json={"name": f"t-{uuid.uuid4().hex[:8]}"})).json()
    agent = (
        await client.post("/v1/agents", json={"tenant_id": tenant["id"], "name": "a"})
    ).json()
    version = (await client.post(f"/v1/agents/{agent['id']}/versions", json={})).json()
    tool = (
        await client.post("/v1/tools", json={"tenant_id": tenant["id"], "name": "web-search"})
    ).json()
    return {"tenant": tenant, "agent": agent, "version": version, "tool": tool}


async def test_a_grant_on_a_draft_version_can_be_revoked(client, wired) -> None:
    v, t = wired["version"], wired["tool"]
    await client.post(f"/v1/agent-versions/{v['id']}/grants", json={"tool_id": t["id"]})

    resp = await client.delete(f"/v1/agent-versions/{v['id']}/grants/{t['id']}")
    assert resp.status_code == 204

    remaining = (await client.get(f"/v1/agent-versions/{v['id']}/grants")).json()
    assert remaining == []


async def test_revoking_a_grant_that_never_existed_is_404(client, wired) -> None:
    v, t = wired["version"], wired["tool"]
    resp = await client.delete(f"/v1/agent-versions/{v['id']}/grants/{t['id']}")
    assert resp.status_code == 404


async def test_a_grant_on_an_active_version_cannot_be_revoked(client, wired) -> None:
    """The same lifecycle guard as granting: an ACTIVE version is a frozen
    capability bundle. To change it, cut a new version."""
    v, t, a = wired["version"], wired["tool"], wired["agent"]
    await client.post(f"/v1/agent-versions/{v['id']}/grants", json={"tool_id": t["id"]})
    await client.post(f"/v1/agents/{a['id']}/activate", json={"version_id": v["id"]})

    resp = await client.delete(f"/v1/agent-versions/{v['id']}/grants/{t['id']}")
    assert resp.status_code == 409

    remaining = (await client.get(f"/v1/agent-versions/{v['id']}/grants")).json()
    assert len(remaining) == 1, "the grant must survive a rejected revocation attempt"
