"""Admission control end to end: 429 for tenant backlog, 503 for global overload.

The distinction under test in every case here: 429 means "you, slow down";
503 means "us, we're in trouble" -- and a client must be able to tell them
apart from the status code and Retry-After alone, without reading prose.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import httpx
import pytest

from acp.api.app import app
from acp.config import settings
from acp.db.session import dispose_engine
from acp.obs import gauges

pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
async def _isolated(clean_tasks):
    yield


@pytest.fixture
async def client(migrated_db, monkeypatch) -> AsyncIterator[httpx.AsyncClient]:
    monkeypatch.setenv("ACP_DATABASE_URL", migrated_db)
    settings.cache_clear()
    await dispose_engine()
    # The cached global-queued gauge is process-global state; reset it so one
    # test's overload scenario cannot leak into the next test's baseline.
    gauges._global_queued = 0
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    gauges._global_queued = 0
    await dispose_engine()
    settings.cache_clear()


async def _submit(client, tenant_id) -> httpx.Response:
    return await client.post(
        "/v1/tasks",
        json={"tenant_id": tenant_id, "task_type": "demo.agent", "payload": {}},
    )


async def test_submissions_are_admitted_under_the_limit(client) -> None:
    tenant = (
        await client.post(
            "/v1/tenants", json={"name": f"t-{uuid.uuid4().hex[:8]}", "max_queued_tasks": 10}
        )
    ).json()

    for _ in range(5):
        resp = await _submit(client, tenant["id"])
        assert resp.status_code == 201


async def test_a_tenant_at_its_backlog_bound_gets_429(client) -> None:
    """The core backpressure case: submissions outpacing execution."""
    tenant = (
        await client.post(
            "/v1/tenants", json={"name": f"t-{uuid.uuid4().hex[:8]}", "max_queued_tasks": 3}
        )
    ).json()

    for _ in range(3):
        assert (await _submit(client, tenant["id"])).status_code == 201

    rejected = await _submit(client, tenant["id"])
    assert rejected.status_code == 429
    body = rejected.json()
    assert body["error"]["code"] == "quota_exceeded"
    assert "Retry-After" in rejected.headers


async def test_a_tenant_at_its_bound_never_sees_503(client) -> None:
    """429, not 503 -- even while the system-wide gauge looks fine.

    A client seeing 503 would (correctly) infer the outage is not their
    fault and back off harder than necessary; 429 tells them precisely what
    to fix.
    """
    tenant = (
        await client.post(
            "/v1/tenants", json={"name": f"t-{uuid.uuid4().hex[:8]}", "max_queued_tasks": 1}
        )
    ).json()
    assert (await _submit(client, tenant["id"])).status_code == 201

    resp = await _submit(client, tenant["id"])
    assert resp.status_code == 429, "a tenant's own backlog must never present as a system outage"


async def test_one_tenants_backlog_does_not_reject_another_tenant(client) -> None:
    """The bound is per-tenant, not shared -- otherwise one busy tenant could
    lock everyone else out, which is exactly the monopolisation this exists
    to prevent."""
    busy = (
        await client.post(
            "/v1/tenants", json={"name": f"busy-{uuid.uuid4().hex[:8]}", "max_queued_tasks": 2}
        )
    ).json()
    quiet = (
        await client.post("/v1/tenants", json={"name": f"quiet-{uuid.uuid4().hex[:8]}"})
    ).json()

    for _ in range(2):
        assert (await _submit(client, busy["id"])).status_code == 201
    assert (await _submit(client, busy["id"])).status_code == 429

    assert (await _submit(client, quiet["id"])).status_code == 201


async def test_backlog_counts_retrying_tasks_not_just_runnable_ones(client, engine) -> None:
    """Backpressure is about STORAGE, not runnability.

    A tenant whose backlog is entirely deferred retries is still
    accumulating rows without bound. Counting only claimable work would let
    that backlog grow forever while admission kept saying yes.
    """
    import sqlalchemy as sa

    from acp.db.models import tasks

    tenant = (
        await client.post(
            "/v1/tenants", json={"name": f"t-{uuid.uuid4().hex[:8]}", "max_queued_tasks": 2}
        )
    ).json()
    assert (await _submit(client, tenant["id"])).status_code == 201

    # Simulate a task deep in retry backoff: QUEUED, but not runnable for an
    # hour. Still counts toward the tenant's stored backlog.
    async with engine.connect() as conn, conn.begin():
        await conn.execute(
            sa.update(tasks)
            .where(tasks.c.tenant_id == uuid.UUID(tenant["id"]))
            .values(available_at=sa.func.now() + sa.text("interval '1 hour'"))
        )
    assert (await _submit(client, tenant["id"])).status_code == 201

    rejected = await _submit(client, tenant["id"])
    assert rejected.status_code == 429, "deferred-retry rows did not count toward the backlog"


async def test_global_overload_sheds_with_503(client, monkeypatch) -> None:
    """The system-wide case: reserved for tenants who did nothing wrong."""
    monkeypatch.setattr(settings(), "global_queue_shed_threshold", 1)
    gauges._global_queued = 5_000  # simulate the cached gauge reporting overload

    tenant = (await client.post("/v1/tenants", json={"name": f"t-{uuid.uuid4().hex[:8]}"})).json()
    resp = await _submit(client, tenant["id"])

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "overloaded"
    assert "Retry-After" in resp.headers


async def test_shedding_disabled_by_default_admits_despite_a_deep_global_queue(
    client, monkeypatch
) -> None:
    monkeypatch.setattr(settings(), "global_queue_shed_threshold", 0)
    gauges._global_queued = 999_999

    tenant = (await client.post("/v1/tenants", json={"name": f"t-{uuid.uuid4().hex[:8]}"})).json()
    assert (await _submit(client, tenant["id"])).status_code == 201


async def test_a_scrape_never_triggers_admission_bookkeeping(client) -> None:
    """Reading /metrics must not itself count as a submission."""
    before = (await client.get("/metrics")).text.count("acp_admissions_total")
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    after = resp.text.count("acp_admissions_total")
    assert before == after
