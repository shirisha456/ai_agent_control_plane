"""Control API behaviour, driven over HTTP against a real database.

These run the actual ASGI app, not the query functions, because the contract
being tested is the one a client sees: which status code tells them their
retry was absorbed, whether a cancel actually cancelled, whether a malformed
payload is rejected before it reaches the hottest table in the system.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
import sqlalchemy as sa

from acp.api.app import app
from acp.config import settings
from acp.db.models import tasks
from acp.db.session import dispose_engine
from acp.domain.states import EventType, State

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
async def tenant(client: httpx.AsyncClient) -> dict:
    resp = await client.post(
        "/v1/tenants", json={"name": f"acme-{uuid.uuid4().hex[:8]}", "max_concurrent_tasks": 5}
    )
    assert resp.status_code == 201
    return resp.json()


async def _submit(client, tenant, **overrides) -> httpx.Response:
    body = {"tenant_id": tenant["id"], "task_type": "research.report", "payload": {"topic": "x"}}
    body.update(overrides)
    return await client.post("/v1/tasks", json=body)


# --------------------------------------------------------------------------
# submission
# --------------------------------------------------------------------------


async def test_submit_creates_a_queued_task_with_history(client, tenant) -> None:
    resp = await _submit(client, tenant)
    assert resp.status_code == 201

    task = resp.json()
    assert task["state"] == State.QUEUED
    assert task["attempt"] == 0
    assert task["lease_worker_id"] is None
    assert task["is_retrying"] is False

    events = (await client.get(f"/v1/tasks/{task['id']}/events")).json()
    assert [e["event_type"] for e in events] == [EventType.TASK_CREATED.value]


async def test_duplicate_submit_returns_200_not_201(client, tenant) -> None:
    """The status code is the only signal that a retry was absorbed.

    A client that cannot tell creation from deduplication cannot tell whether
    its own retry logic is working.
    """
    key = f"idem-{uuid.uuid4().hex}"
    first = await _submit(client, tenant, idempotency_key=key)
    second = await _submit(client, tenant, idempotency_key=key)

    assert first.status_code == 201
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]


async def test_concurrent_duplicate_submits_create_exactly_one_task(client, tenant) -> None:
    """Twenty simultaneous retries of one logical submission.

    This is the API-replica race: several instances receive a client's retries
    at once. Dedup is a unique index, so exactly one INSERT survives no matter
    how they interleave.
    """
    key = f"idem-{uuid.uuid4().hex}"
    responses = await asyncio.gather(
        *(_submit(client, tenant, idempotency_key=key) for _ in range(20))
    )

    assert all(r.status_code in (200, 201) for r in responses)
    assert sum(1 for r in responses if r.status_code == 201) == 1
    assert len({r.json()["id"] for r in responses}) == 1


async def test_reusing_a_key_with_a_different_payload_is_rejected(client, tenant) -> None:
    """A client bug that must not be silently absorbed.

    Returning the original task with 200 would tell the caller their NEW
    request was accepted. It was not, and the difference has side effects.
    """
    key = f"idem-{uuid.uuid4().hex}"
    await _submit(client, tenant, idempotency_key=key, payload={"topic": "a"})
    resp = await _submit(client, tenant, idempotency_key=key, payload={"topic": "b"})

    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "conflict"


async def test_keys_do_not_collide_across_tenants(client) -> None:
    """Idempotency is scoped per tenant, so tenants can neither collide with
    nor probe one another's keys."""
    a = (await client.post("/v1/tenants", json={"name": f"a-{uuid.uuid4().hex[:8]}"})).json()
    b = (await client.post("/v1/tenants", json={"name": f"b-{uuid.uuid4().hex[:8]}"})).json()

    key = "shared-key"
    ra = await _submit(client, a, idempotency_key=key)
    rb = await _submit(client, b, idempotency_key=key)

    assert ra.status_code == 201
    assert rb.status_code == 201
    assert ra.json()["id"] != rb.json()["id"]


async def test_oversized_payload_is_rejected_at_the_boundary(client, tenant) -> None:
    """`tasks` is rewritten on every claim, renewal and completion, and
    PostgreSQL rewrites whole tuples. A megabyte payload would turn every
    lease renewal into a megabyte of WAL."""
    resp = await _submit(client, tenant, payload={"blob": "x" * 100_000})
    assert resp.status_code == 422


async def test_unknown_tenant_is_rejected(client) -> None:
    resp = await _submit(client, {"id": str(uuid.uuid4())})
    assert resp.status_code == 404


# --------------------------------------------------------------------------
# reads
# --------------------------------------------------------------------------


async def test_retrying_is_derived_not_stored(client, tenant, engine) -> None:
    """A task waiting out its backoff is QUEUED with a future available_at.

    The API reports `is_retrying` by computing it. Nothing in the database
    stores a RETRYING state, so there is no second place for it to be wrong.
    """
    task = (await _submit(client, tenant)).json()
    async with engine.connect() as conn, conn.begin():
        await conn.execute(
            sa.update(tasks)
            .where(tasks.c.id == uuid.UUID(task["id"]))
            .values(attempt=1, available_at=sa.func.now() + sa.text("interval '60 seconds'"))
        )

    refreshed = (await client.get(f"/v1/tasks/{task['id']}")).json()
    assert refreshed["state"] == State.QUEUED
    assert refreshed["is_retrying"] is True


async def test_listing_is_keyset_paginated(client, tenant) -> None:
    """Pages must not overlap or skip, and the cursor must terminate."""
    for _ in range(7):
        await _submit(client, tenant)

    seen: list[str] = []
    cursor = None
    pages = 0
    while True:
        url = f"/v1/tasks?tenant_id={tenant['id']}&limit=3"
        if cursor:
            url += f"&cursor={cursor}"
        page = (await client.get(url)).json()
        seen.extend(t["id"] for t in page["tasks"])
        cursor = page["next_cursor"]
        pages += 1
        if cursor is None:
            break
        assert pages < 10, "cursor failed to terminate"

    assert len(seen) == 7
    assert len(set(seen)) == 7, "pages overlapped"


async def test_events_for_unknown_task_is_404(client) -> None:
    resp = await client.get(f"/v1/tasks/{uuid.uuid4()}/events")
    assert resp.status_code == 404


# --------------------------------------------------------------------------
# cancellation
# --------------------------------------------------------------------------


async def test_cancelling_a_queued_task_terminates_it_immediately(client, tenant) -> None:
    task = (await _submit(client, tenant)).json()
    resp = await client.post(f"/v1/tasks/{task['id']}/cancel")

    assert resp.status_code == 200
    assert resp.json()["state"] == State.CANCELLED

    events = (await client.get(f"/v1/tasks/{task['id']}/events")).json()
    assert events[-1]["event_type"] == EventType.TASK_CANCELLED.value


async def test_cancelling_a_running_task_is_accepted_not_completed(client, tenant, engine) -> None:
    """202, because the control plane cannot yank work out of a remote process.

    It records the request; the worker honours it at its next lease renewal,
    and if the worker is already dead the reaper honours it when the lease
    expires. Returning 200 would claim an outcome that has not happened.
    """
    task = (await _submit(client, tenant)).json()
    async with engine.connect() as conn, conn.begin():
        await conn.execute(
            sa.update(tasks)
            .where(tasks.c.id == uuid.UUID(task["id"]))
            .values(
                state=State.RUNNING.value,
                attempt=1,
                lease_worker_id="worker-a",
                lease_expires_at=sa.func.now() + sa.text("interval '30 seconds'"),
            )
        )

    resp = await client.post(f"/v1/tasks/{task['id']}/cancel")
    assert resp.status_code == 202
    assert resp.json()["state"] == State.RUNNING
    assert resp.json()["cancel_requested"] is True

    events = (await client.get(f"/v1/tasks/{task['id']}/events")).json()
    assert events[-1]["event_type"] == EventType.CANCEL_REQUESTED.value


async def test_cancelling_a_finished_task_is_a_conflict(client, tenant, engine) -> None:
    """409, not 200. Silently succeeding would hide a completed task from
    someone who believes they stopped it."""
    task = (await _submit(client, tenant)).json()
    async with engine.connect() as conn, conn.begin():
        await conn.execute(
            sa.update(tasks)
            .where(tasks.c.id == uuid.UUID(task["id"]))
            .values(state=State.SUCCEEDED.value, finished_at=sa.func.now())
        )

    resp = await client.post(f"/v1/tasks/{task['id']}/cancel")
    assert resp.status_code == 409


async def test_repeated_cancel_is_idempotent(client, tenant) -> None:
    task = (await _submit(client, tenant)).json()
    first = await client.post(f"/v1/tasks/{task['id']}/cancel")
    second = await client.post(f"/v1/tasks/{task['id']}/cancel")

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["state"] == State.CANCELLED


async def test_concurrent_cancels_produce_one_cancellation(client, tenant) -> None:
    """The FOR UPDATE row lock makes the read-and-branch atomic, so ten
    simultaneous cancels cannot produce ten TASK_CANCELLED events."""
    task = (await _submit(client, tenant)).json()
    responses = await asyncio.gather(
        *(client.post(f"/v1/tasks/{task['id']}/cancel") for _ in range(10))
    )
    assert all(r.status_code == 200 for r in responses)

    events = (await client.get(f"/v1/tasks/{task['id']}/events")).json()
    cancelled = [e for e in events if e["event_type"] == EventType.TASK_CANCELLED.value]
    assert len(cancelled) == 1
