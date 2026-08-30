"""Agent-routed submission over HTTP: resolution, pinning, and the disable sweep."""

from __future__ import annotations

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
async def wired(client: httpx.AsyncClient) -> dict:
    """A tenant with one agent, one released version, and a route to it."""
    tenant = (await client.post("/v1/tenants", json={"name": f"t-{uuid.uuid4().hex[:8]}"})).json()
    agent = (
        await client.post(
            "/v1/agents",
            json={"tenant_id": tenant["id"], "name": f"research-{uuid.uuid4().hex[:6]}"},
        )
    ).json()
    version = (
        await client.post(
            f"/v1/agents/{agent['id']}/versions",
            json={
                "runtime_spec": {"task_type": "demo.agent", "steps": ["plan", "write"]},
                "required_capabilities": ["internet"],
                "max_attempts": 7,
            },
        )
    ).json()
    await client.post(f"/v1/agents/{agent['id']}/activate", json={"version_id": version["id"]})
    await client.put(
        "/v1/routes",
        json={
            "tenant_id": tenant["id"],
            "request_type": "RESEARCH_REPORT",
            "agent_id": agent["id"],
        },
    )
    return {"tenant": tenant, "agent": agent, "version": version}


async def test_a_routed_submission_pins_its_version(client, wired) -> None:
    """The whole point of the phase.

    A task that sat in the queue for an hour still runs the version that was
    current when it was accepted -- which is what makes "what executed task
    123?" answerable and stops a retry running different code than attempt 1.
    """
    resp = await client.post(
        "/v1/tasks",
        json={
            "tenant_id": wired["tenant"]["id"],
            "request_type": "RESEARCH_REPORT",
            "payload": {"topic": "leases"},
        },
    )
    assert resp.status_code == 201

    task = resp.json()
    assert task["agent_version_id"] == wired["version"]["id"]
    assert task["required_capabilities"] == ["internet"]
    # The VERSION's retry budget wins over the request's default: execution
    # policy travels with the immutable definition, so rolling a version back
    # rolls its limits back too.
    assert task["max_attempts"] == 7


async def test_capability_key_is_derived_and_stored(client, wired, engine) -> None:
    task = (
        await client.post(
            "/v1/tasks",
            json={
                "tenant_id": wired["tenant"]["id"],
                "request_type": "RESEARCH_REPORT",
                "payload": {},
            },
        )
    ).json()

    async with engine.connect() as conn:
        key = (
            await conn.execute(
                sa.select(tasks.c.capability_key).where(tasks.c.id == uuid.UUID(task["id"]))
            )
        ).scalar_one()
    assert key == "internet"


async def test_direct_submission_still_works_and_pins_nothing(client, wired) -> None:
    """The two modes coexist.

    A NULL agent_version_id means "submitted directly", which is information,
    not a missing value -- forcing every task through the registry would break
    the primitive for no benefit.
    """
    resp = await client.post(
        "/v1/tasks",
        json={
            "tenant_id": wired["tenant"]["id"],
            "task_type": "demo.agent",
            "payload": {},
        },
    )
    assert resp.status_code == 201
    assert resp.json()["agent_version_id"] is None
    assert resp.json()["required_capabilities"] == []


async def test_exactly_one_submission_mode_is_required(client, wired) -> None:
    """Both would let the resolved agent and the declared task_type disagree,
    and nothing could say which one actually ran."""
    both = await client.post(
        "/v1/tasks",
        json={
            "tenant_id": wired["tenant"]["id"],
            "task_type": "demo.agent",
            "request_type": "RESEARCH_REPORT",
            "payload": {},
        },
    )
    neither = await client.post(
        "/v1/tasks", json={"tenant_id": wired["tenant"]["id"], "payload": {}}
    )
    assert both.status_code == 422
    assert neither.status_code == 422


async def test_unrouted_request_type_is_404(client, wired) -> None:
    resp = await client.post(
        "/v1/tasks",
        json={"tenant_id": wired["tenant"]["id"], "request_type": "UNKNOWN", "payload": {}},
    )
    assert resp.status_code == 404


async def test_submission_to_a_disabled_agent_is_rejected(client, wired) -> None:
    await client.post(f"/v1/agents/{wired['agent']['id']}/disable")
    resp = await client.post(
        "/v1/tasks",
        json={
            "tenant_id": wired["tenant"]["id"],
            "request_type": "RESEARCH_REPORT",
            "payload": {},
        },
    )
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# disabling an agent stops its live work -- by reusing cancellation
# ---------------------------------------------------------------------------


async def test_disabling_an_agent_cancels_its_queued_tasks(client, wired, engine) -> None:
    """No agent-status predicate was added to the claim query to achieve this.

    The tempting design puts a join against the registry on the hottest query
    in the system to serve an admin action that happens once a month.
    Cancellation already handles "stop this task", including against a worker
    that is already dead, so the sweep reuses it.
    """
    ids = []
    for _ in range(3):
        resp = await client.post(
            "/v1/tasks",
            json={
                "tenant_id": wired["tenant"]["id"],
                "request_type": "RESEARCH_REPORT",
                "payload": {},
            },
        )
        ids.append(uuid.UUID(resp.json()["id"]))

    disable = await client.post(f"/v1/agents/{wired['agent']['id']}/disable")
    assert disable.status_code == 200
    assert disable.json()["tasks_cancelled_or_cancelling"] == 3

    async with engine.connect() as conn:
        states = (
            (await conn.execute(sa.select(tasks.c.state).where(tasks.c.id.in_(ids))))
            .scalars()
            .all()
        )
    assert set(states) == {State.CANCELLED}

    events = (await client.get(f"/v1/tasks/{ids[0]}/events")).json()
    assert events[-1]["event_type"] == EventType.TASK_CANCELLED.value


async def test_disabling_requests_cancellation_for_running_tasks(
    client, wired, engine, make_worker
) -> None:
    """A running task cannot be yanked out of its worker, so it gets the flag.

    The worker honours it at its next lease renewal; if that worker is already
    gone, the reaper honours it when the lease expires. Both paths existed
    before this phase -- which is the point of reusing them.
    """
    task = (
        await client.post(
            "/v1/tasks",
            json={
                "tenant_id": wired["tenant"]["id"],
                "request_type": "RESEARCH_REPORT",
                "payload": {},
            },
        )
    ).json()
    worker_id = await make_worker()

    async with engine.connect() as conn, conn.begin():
        await conn.execute(
            sa.update(tasks)
            .where(tasks.c.id == uuid.UUID(task["id"]))
            .values(
                state=State.RUNNING.value,
                attempt=1,
                lease_worker_id=worker_id,
                lease_expires_at=sa.func.now() + sa.text("interval '30 seconds'"),
            )
        )

    await client.post(f"/v1/agents/{wired['agent']['id']}/disable")

    refreshed = (await client.get(f"/v1/tasks/{task['id']}")).json()
    assert refreshed["state"] == State.RUNNING, "a running task was terminated remotely"
    assert refreshed["cancel_requested"] is True

    events = (await client.get(f"/v1/tasks/{task['id']}/events")).json()
    assert events[-1]["event_type"] == EventType.CANCEL_REQUESTED.value


async def test_routing_to_another_tenants_agent_is_refused(client, wired) -> None:
    """Agents are tenant-scoped precisely to make this impossible."""
    other = (
        await client.post("/v1/tenants", json={"name": f"other-{uuid.uuid4().hex[:8]}"})
    ).json()
    resp = await client.put(
        "/v1/routes",
        json={
            "tenant_id": other["id"],
            "request_type": "RESEARCH_REPORT",
            "agent_id": wired["agent"]["id"],
        },
    )
    assert resp.status_code == 409
