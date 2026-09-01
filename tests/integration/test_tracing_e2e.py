"""Tracing end to end, through the real database: submit -> queue -> claim -> execute.

This is the property the unit tests in tests/unit/test_tracing.py cannot
prove: that the carrier actually survives being written into a JSONB column,
read back by a different process (a worker, standing in for a worker), and
turned into a real link on the executing span -- not just that the tracing
API behaves correctly in memory.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
import sqlalchemy as sa
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from acp.agent.adapters.base import Adapter, AdapterRegistry
from acp.api.app import app
from acp.config import Settings, settings
from acp.db.models import tasks
from acp.db.session import dispose_engine
from acp.domain.states import State
from acp.obs import tracing
from acp.worker.loop import Worker

pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
async def _isolated(clean_tasks):
    yield


@pytest.fixture
def traced_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    original = tracing._TRACER
    tracing._TRACER = provider.get_tracer("test")
    try:
        yield exporter
    finally:
        tracing._TRACER = original


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


class _Instant(Adapter):
    async def run(self, payload, *, is_cancelled):
        return {"payload_seen": dict(payload)}


def _bind_transaction(engine):
    @asynccontextmanager
    async def _tx():
        async with engine.connect() as conn, conn.begin():
            yield conn

    return _tx


async def test_the_submitting_spans_identity_survives_into_the_stored_task(
    client, engine, traced_exporter
) -> None:
    """The mechanism, checked directly against the database row.

    Before any worker is involved: does the carrier this request captured
    actually make it into tasks.payload as `_trace`, in the shape the worker
    expects to read it back?
    """
    tenant = (await client.post("/v1/tenants", json={"name": f"t-{uuid.uuid4().hex[:8]}"})).json()
    task = (
        await client.post(
            "/v1/tasks",
            json={"tenant_id": tenant["id"], "task_type": "demo.agent", "payload": {"x": 1}},
        )
    ).json()

    (submit_span,) = traced_exporter.get_finished_spans()
    assert submit_span.name == "acp.task.submit"

    async with engine.connect() as conn:
        stored_payload = (
            await conn.execute(
                sa.select(tasks.c.payload).where(tasks.c.id == uuid.UUID(task["id"]))
            )
        ).scalar_one()

    assert stored_payload["_trace"]["trace_id"] == format(submit_span.context.trace_id, "032x")
    assert stored_payload["_trace"]["span_id"] == format(submit_span.context.span_id, "016x")


async def test_the_adapter_never_sees_the_trace_carrier(client, engine, monkeypatch) -> None:
    """`_trace` is transport, not task input. An adapter that echoes its
    payload back must not leak tracing plumbing into its result."""
    tenant = (await client.post("/v1/tenants", json={"name": f"t-{uuid.uuid4().hex[:8]}"})).json()
    task = (
        await client.post(
            "/v1/tasks",
            json={"tenant_id": tenant["id"], "task_type": "demo.echo", "payload": {"x": 1}},
        )
    ).json()

    import acp.worker.loop as loop_mod

    monkeypatch.setattr(loop_mod, "transaction", _bind_transaction(engine))
    registry = AdapterRegistry()
    registry.register("demo.echo", _Instant)
    worker = Worker(
        settings=Settings(database_url="x", poll_interval_ms=20, heartbeat_interval_s=60),
        registry=registry,
        capacity=1,
        rng=random.Random(0),
    )

    run = asyncio.ensure_future(worker.run_forever())
    try:
        async with asyncio.timeout(8):
            while True:
                async with engine.connect() as conn:
                    row = (
                        (
                            await conn.execute(
                                sa.select(tasks.c.state, tasks.c.result).where(
                                    tasks.c.id == uuid.UUID(task["id"])
                                )
                            )
                        )
                        .mappings()
                        .one()
                    )
                if row["state"] == State.SUCCEEDED:
                    break
                await asyncio.sleep(0.02)
    finally:
        worker.stop()
        await asyncio.wait_for(run, timeout=10.0)

    assert "_trace" not in row["result"]["payload_seen"]
    assert row["result"]["payload_seen"] == {"x": 1}


async def test_the_execution_span_links_back_to_the_submitting_span(
    client, engine, monkeypatch, traced_exporter
) -> None:
    """The end-to-end proof: API span -> stored carrier -> worker's execution
    span, linked, across a real claim through PostgreSQL -- not a mock."""
    tenant = (await client.post("/v1/tenants", json={"name": f"t-{uuid.uuid4().hex[:8]}"})).json()
    task = (
        await client.post(
            "/v1/tasks",
            json={"tenant_id": tenant["id"], "task_type": "demo.echo", "payload": {}},
        )
    ).json()
    (submit_span,) = traced_exporter.get_finished_spans()

    import acp.worker.loop as loop_mod

    monkeypatch.setattr(loop_mod, "transaction", _bind_transaction(engine))
    registry = AdapterRegistry()
    registry.register("demo.echo", _Instant)
    worker = Worker(
        settings=Settings(database_url="x", poll_interval_ms=20, heartbeat_interval_s=60),
        registry=registry,
        capacity=1,
        rng=random.Random(0),
    )

    run = asyncio.ensure_future(worker.run_forever())
    try:
        async with asyncio.timeout(8):
            while True:
                async with engine.connect() as conn:
                    state = (
                        await conn.execute(
                            sa.select(tasks.c.state).where(tasks.c.id == uuid.UUID(task["id"]))
                        )
                    ).scalar_one()
                if state == State.SUCCEEDED:
                    break
                await asyncio.sleep(0.02)
    finally:
        worker.stop()
        await asyncio.wait_for(run, timeout=10.0)

    spans = {s.name: s for s in traced_exporter.get_finished_spans()}
    execute_span = spans["acp.task.execute"]

    assert len(execute_span.links) == 1
    linked = execute_span.links[0].context
    assert linked.trace_id == submit_span.context.trace_id
    assert linked.span_id == submit_span.context.span_id
    # Not a parent/child relationship -- see obs/tracing's module docstring
    # on why a child span is wrong once the submitting request may have ended.
    assert execute_span.parent is None
    assert execute_span.context.trace_id != submit_span.context.trace_id
    assert execute_span.attributes["acp.task_id"] == task["id"]
