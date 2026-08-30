"""The scrape endpoint and the DB-derived gauges."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
import sqlalchemy as sa

from acp.api.app import app
from acp.config import settings
from acp.db.models import tasks, workers
from acp.db.session import dispose_engine
from acp.domain.states import State
from acp.obs import gauges, metrics

pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
async def _isolated(clean_tasks):
    """Gauges are fleet-wide counts, so leftovers from other suites would show up."""
    yield


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


async def test_metrics_endpoint_serves_prometheus_text(client) -> None:
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]

    body = resp.text
    # HELP lines prove the metric is registered even before it is incremented,
    # which is what stops a dashboard panel from being permanently blank after
    # a rename.
    assert "# HELP acp_stale_writes_rejected_total" in body
    assert "# HELP acp_recovery_latency_seconds" in body
    assert "# HELP acp_queue_depth" in body


async def test_scraping_does_not_touch_the_database(client, monkeypatch) -> None:
    """A scrape must read memory only.

    Otherwise a second Prometheus replica, or a human with `watch curl`, puts
    the monitoring system into contention with the claim path for connections
    -- letting monitoring cause the incident it exists to observe.
    """
    called = False

    async def _boom(engine):
        nonlocal called
        called = True

    monkeypatch.setattr(gauges, "refresh_once", _boom)
    resp = await client.get("/metrics")

    assert resp.status_code == 200
    assert not called, "the scrape path queried the database"


async def test_submission_is_counted(client) -> None:
    before = metrics.tasks_submitted.labels(
        task_type="metrics.demo", deduplicated="false"
    )._value.get()

    tenant = (await client.post("/v1/tenants", json={"name": f"m-{uuid.uuid4().hex[:8]}"})).json()
    key = f"idem-{uuid.uuid4().hex}"
    body = {"tenant_id": tenant["id"], "task_type": "metrics.demo", "payload": {}}

    assert (
        await client.post("/v1/tasks", json={**body, "idempotency_key": key})
    ).status_code == 201
    assert (
        await client.post("/v1/tasks", json={**body, "idempotency_key": key})
    ).status_code == 200

    after = metrics.tasks_submitted.labels(
        task_type="metrics.demo", deduplicated="false"
    )._value.get()
    deduped = metrics.tasks_submitted.labels(
        task_type="metrics.demo", deduplicated="true"
    )._value.get()

    assert after == before + 1, "a deduplicated submit was counted as a creation"
    assert deduped >= 1


async def test_gauges_reflect_database_state(engine, make_task, make_worker) -> None:
    """Queue depth, running count and worker status come from one query."""
    task_id = await make_task()
    await make_task(available_at=sa.func.now() + sa.text("interval '1 hour'"))
    worker_id = await make_worker()

    async with engine.connect() as conn, conn.begin():
        await conn.execute(
            sa.update(tasks)
            .where(tasks.c.id == task_id)
            .values(
                state=State.RUNNING.value,
                attempt=1,
                lease_worker_id=worker_id,
                lease_expires_at=sa.func.now() + sa.text("interval '30 seconds'"),
            )
        )

    await gauges.refresh_once(engine)

    running = sum(
        s.value
        for m in metrics.REGISTRY.collect()
        if m.name == "acp_tasks_running"
        for s in m.samples
    )
    assert running == 1

    # The delayed task is QUEUED but not runnable, so it belongs in the
    # backlog gauge, not queue depth. Conflating them would make retry backoff
    # look like queue pressure and trigger autoscaling on work nobody can run.
    assert metrics.tasks_backlogged._value.get() == 1
    alive = metrics.workers_by_status.labels(status="ALIVE")._value.get()
    assert alive >= 1


async def test_expired_pending_gauge_detects_a_stalled_reaper(
    engine, make_task, make_worker
) -> None:
    """The single best alert in the system.

    Zero while the reaper is healthy; climbing the moment it is down or
    falling behind. Nothing else distinguishes "a worker died and was
    recovered" from "a worker died and nobody noticed".
    """
    task_id = await make_task()
    worker_id = await make_worker()
    async with engine.connect() as conn, conn.begin():
        await conn.execute(
            sa.update(tasks)
            .where(tasks.c.id == task_id)
            .values(
                state=State.RUNNING.value,
                attempt=1,
                lease_worker_id=worker_id,
                lease_expires_at=sa.func.now() - sa.text("interval '5 seconds'"),
            )
        )

    await gauges.refresh_once(engine)
    assert metrics.leases_expired_pending._value.get() == 1

    async with engine.connect() as conn, conn.begin():
        await conn.execute(
            sa.update(tasks)
            .where(tasks.c.id == task_id)
            .values(state=State.QUEUED.value, lease_worker_id=None, lease_expires_at=None)
        )
    await gauges.refresh_once(engine)
    assert metrics.leases_expired_pending._value.get() == 0


async def test_gauges_drop_back_to_zero_when_work_drains(engine, make_task) -> None:
    """A gauge that never comes back down is worse than no gauge.

    Labelled gauges keep reporting their last value for label sets that have
    since disappeared, so the refresher clears before it sets.
    """
    await make_task()
    await gauges.refresh_once(engine)
    depth = sum(
        s.value
        for m in metrics.REGISTRY.collect()
        if m.name == "acp_queue_depth"
        for s in m.samples
    )
    assert depth >= 1

    async with engine.connect() as conn, conn.begin():
        await conn.execute(sa.delete(tasks))
    await gauges.refresh_once(engine)

    depth = sum(
        s.value
        for m in metrics.REGISTRY.collect()
        if m.name == "acp_queue_depth"
        for s in m.samples
    )
    assert depth == 0, "queue depth stayed stale after the queue drained"


async def test_worker_status_gauge_clears_removed_statuses(engine) -> None:
    async with engine.connect() as conn, conn.begin():
        await conn.execute(sa.delete(workers).where(workers.c.status == "DRAINING"))
        await conn.execute(
            sa.insert(workers).values(
                id=f"w-{uuid.uuid4().hex[:8]}",
                hostname="h",
                pid=1,
                capacity=1,
                status="DRAINING",
            )
        )
    await gauges.refresh_once(engine)
    assert metrics.workers_by_status.labels(status="DRAINING")._value.get() == 1

    async with engine.connect() as conn, conn.begin():
        await conn.execute(sa.delete(workers).where(workers.c.status == "DRAINING"))
    await gauges.refresh_once(engine)

    statuses = {
        s.labels["status"]
        for m in metrics.REGISTRY.collect()
        if m.name == "acp_workers"
        for s in m.samples
    }
    assert "DRAINING" not in statuses
