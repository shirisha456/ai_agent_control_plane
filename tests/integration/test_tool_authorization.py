"""Runtime tool authorization end to end: the snapshot, the denial, the audit."""

from __future__ import annotations

import asyncio
import random
import uuid
from contextlib import asynccontextmanager

import pytest
import sqlalchemy as sa

from acp.agent.adapters.base import Adapter, AdapterRegistry
from acp.agent.tools import call_tool
from acp.config import Settings
from acp.db.models import audit_events, task_events, tasks
from acp.db.queries import agents as aq
from acp.db.queries import tools as tq
from acp.domain.agents import VersionStatus
from acp.domain.authz import ToolStatus
from acp.domain.errors import FailureClass
from acp.domain.states import EventType, State
from acp.worker.loop import Worker

pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
async def _isolated(clean_tasks):
    yield


def _settings(**overrides) -> Settings:
    defaults = dict(
        database_url="unused",
        lease_ttl_s=30,
        lease_renew_interval_s=7,
        heartbeat_interval_s=60,
        poll_interval_ms=20,
        claim_batch_size=5,
        drain_grace_s=0.2,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _bind_transaction(engine):
    @asynccontextmanager
    async def _tx():
        async with engine.connect() as conn, conn.begin():
            yield conn

    return _tx


class _CallsTools(Adapter):
    """Takes no `tools` parameter -- access arrives via a contextvar."""

    async def run(self, payload, *, is_cancelled):
        results = [await call_tool(name, {}) for name in payload.get("tools", [])]
        return {"results": results}


@pytest.fixture
async def governed(engine, tenant_id):
    """An agent version granted `web-search` but not `billing-db`."""
    async with engine.connect() as conn, conn.begin():
        agent = await aq.create_agent(
            conn, tenant_id=tenant_id, name=f"research-{uuid.uuid4().hex[:6]}"
        )
        version = await aq.create_version(conn, agent_id=agent["id"])
        web = await tq.create_tool(conn, tenant_id=tenant_id, name="web-search")
        billing = await tq.create_tool(conn, tenant_id=tenant_id, name="billing-db")
        await tq.grant_tool(conn, agent_version_id=version["id"], tool_id=web["id"])
        await aq.activate_version(conn, agent_id=agent["id"], version_id=version["id"])
    return {
        "tenant_id": tenant_id,
        "agent": agent,
        "version": version,
        "web": web,
        "billing": billing,
    }


async def _run_task(engine, monkeypatch, task_id, timeout=8.0) -> str:
    import acp.worker.loop as loop_mod

    monkeypatch.setattr(loop_mod, "transaction", _bind_transaction(engine))
    monkeypatch.setattr(loop_mod, "db_engine", lambda: engine)

    registry = AdapterRegistry()
    registry.register("demo.tools", _CallsTools)
    worker = Worker(settings=_settings(), registry=registry, capacity=1, rng=random.Random(1))

    run = asyncio.ensure_future(worker.run_forever())
    try:
        async with asyncio.timeout(timeout):
            while True:
                async with engine.connect() as conn:
                    state = (
                        await conn.execute(sa.select(tasks.c.state).where(tasks.c.id == task_id))
                    ).scalar_one()
                if state in (State.SUCCEEDED, State.FAILED, State.CANCELLED):
                    return state
                await asyncio.sleep(0.02)
    finally:
        worker.stop()
        await asyncio.wait_for(run, timeout=10.0)


# ---------------------------------------------------------------------------
# the allowed path
# ---------------------------------------------------------------------------


async def test_a_granted_tool_executes_and_lands_in_the_task_timeline(
    engine, make_task, monkeypatch, governed
) -> None:
    """ALLOW is execution history: task_events only, pruned with the task."""
    task_id = await make_task(
        task_type="demo.tools",
        agent_version_id=governed["version"]["id"],
        payload={"tools": ["web-search"]},
    )
    assert await _run_task(engine, monkeypatch, task_id) == State.SUCCEEDED

    async with engine.connect() as conn:
        events = (
            (
                await conn.execute(
                    sa.select(task_events.c.event_type)
                    .where(task_events.c.task_id == task_id)
                    .order_by(task_events.c.id)
                )
            )
            .scalars()
            .all()
        )
        audit = (
            await conn.execute(
                sa.select(sa.func.count())
                .select_from(audit_events)
                .where(audit_events.c.resource_id == task_id)
            )
        ).scalar_one()

    assert EventType.TOOL_ACCESS_ALLOWED.value in events
    assert EventType.TOOL_EXECUTED.value in events
    # An allowed call is one of thousands. Writing it to the audit log would
    # bury the refusals that log exists to preserve.
    assert audit == 0


# ---------------------------------------------------------------------------
# the denied path -- the demo
# ---------------------------------------------------------------------------


async def test_an_ungranted_tool_is_refused_and_audited(
    engine, make_task, monkeypatch, governed
) -> None:
    """The second flagship demo, in one test.

    The agent reaches for a tool it was never granted. The control plane
    refuses at runtime, the task fails without retrying, and the refusal is
    recorded in BOTH the task timeline and the audit log -- because the audit
    log outlives the task's retention.
    """
    task_id = await make_task(
        task_type="demo.tools",
        agent_version_id=governed["version"]["id"],
        payload={"tools": ["billing-db"]},
        max_attempts=5,
    )
    assert await _run_task(engine, monkeypatch, task_id) == State.FAILED

    async with engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    sa.select(tasks.c.attempt, tasks.c.error_class).where(tasks.c.id == task_id)
                )
            )
            .mappings()
            .one()
        )
        events = (
            (
                await conn.execute(
                    sa.select(task_events.c.event_type, task_events.c.data)
                    .where(task_events.c.task_id == task_id)
                    .order_by(task_events.c.id)
                )
            )
            .mappings()
            .all()
        )
        audit = (
            (
                await conn.execute(
                    sa.select(*audit_events.c).where(audit_events.c.resource_id == task_id)
                )
            )
            .mappings()
            .all()
        )

    # PERMISSION_DENIED is non-retryable: the answer will not change, and
    # retrying would write five audit records for one refusal.
    assert row["error_class"] == FailureClass.PERMISSION_DENIED.value
    assert row["attempt"] == 1, "a refusal consumed retry attempts"

    denied = [e for e in events if e["event_type"] == EventType.TOOL_ACCESS_DENIED.value]
    assert len(denied) == 1
    assert denied[0]["data"]["tool"] == "billing-db"
    assert denied[0]["data"]["reason"] == "not_granted"

    assert len(audit) == 1
    assert audit[0]["action"] == "TOOL_ACCESS_DENIED"
    assert audit[0]["outcome"] == "DENIED"
    assert audit[0]["data"]["tool"] == "billing-db"
    # The attempt number is recorded so the timeline stays honest about WHICH
    # attempt was refused.
    assert audit[0]["data"]["attempt"] == 1


async def test_a_disabled_tool_is_refused_despite_a_grant(
    engine, make_task, monkeypatch, governed
) -> None:
    """Grants are immutable; denial is live.

    Revoking a grant would need a new version -- far too slow for an incident.
    The kill switch is what makes versioned grants workable.
    """
    async with engine.connect() as conn, conn.begin():
        await tq.set_tool_status(conn, governed["web"]["id"], ToolStatus.DISABLED)

    task_id = await make_task(
        task_type="demo.tools",
        agent_version_id=governed["version"]["id"],
        payload={"tools": ["web-search"]},
    )
    assert await _run_task(engine, monkeypatch, task_id) == State.FAILED

    async with engine.connect() as conn:
        audit = (
            (
                await conn.execute(
                    sa.select(audit_events.c.data).where(audit_events.c.resource_id == task_id)
                )
            )
            .mappings()
            .one()
        )
    assert audit["data"]["reason"] == "tool_disabled"


async def test_a_directly_submitted_task_may_call_nothing(
    engine, make_task, monkeypatch, governed
) -> None:
    """No pinned version means no governing definition, so nothing is granted.

    Defaulting to permissive here is how authorization gets bypassed by a
    refactor -- "no policy loaded" is not "everything permitted".
    """
    task_id = await make_task(task_type="demo.tools", payload={"tools": ["web-search"]})
    assert await _run_task(engine, monkeypatch, task_id) == State.FAILED

    async with engine.connect() as conn:
        audit = (
            (
                await conn.execute(
                    sa.select(audit_events.c.data).where(audit_events.c.resource_id == task_id)
                )
            )
            .mappings()
            .one()
        )
    assert audit["data"]["reason"] == "no_policy"


# ---------------------------------------------------------------------------
# the snapshot
# ---------------------------------------------------------------------------


async def test_policy_is_frozen_for_the_life_of_an_attempt(engine, governed) -> None:
    """An agent cannot gain a capability halfway through its own execution.

    The snapshot is taken inside the claim transaction, so a grant added after
    the claim is invisible to the running attempt. That is what makes
    revocation latency bounded by attempt duration rather than by a cache TTL.
    """
    async with engine.connect() as conn:
        before = await tq.snapshot_policies(conn, agent_version_ids=[governed["version"]["id"]])
    policy = before[governed["version"]["id"]]
    assert governed["web"]["id"] in policy.granted_tool_ids
    assert governed["billing"]["id"] not in policy.granted_tool_ids

    async with engine.connect() as conn, conn.begin():
        await tq.grant_tool(
            conn,
            agent_version_id=governed["version"]["id"],
            tool_id=governed["billing"]["id"],
        )

    # The already-taken snapshot is unchanged -- it is a value, not a view.
    assert governed["billing"]["id"] not in policy.granted_tool_ids

    async with engine.connect() as conn:
        after = await tq.snapshot_policies(conn, agent_version_ids=[governed["version"]["id"]])
    assert governed["billing"]["id"] in after[governed["version"]["id"]].granted_tool_ids


async def test_snapshot_carries_the_whole_tenant_namespace(engine, governed) -> None:
    """So an ungranted-but-existing tool reports NOT_GRANTED, not UNKNOWN_TOOL."""
    async with engine.connect() as conn:
        policies = await tq.snapshot_policies(conn, agent_version_ids=[governed["version"]["id"]])
    policy = policies[governed["version"]["id"]]
    assert set(policy.tools_by_name) >= {"web-search", "billing-db"}


async def test_snapshot_is_empty_for_ungoverned_tasks(engine) -> None:
    async with engine.connect() as conn:
        assert await tq.snapshot_policies(conn, agent_version_ids=[None, None]) == {}


# ---------------------------------------------------------------------------
# grants belong to the version's construction
# ---------------------------------------------------------------------------


async def test_granting_is_idempotent(engine, governed) -> None:
    async with engine.connect() as conn, conn.begin():
        for _ in range(3):
            await tq.grant_tool(
                conn,
                agent_version_id=governed["version"]["id"],
                tool_id=governed["web"]["id"],
            )
        grants = await tq.list_grants(conn, agent_version_id=governed["version"]["id"])
    assert len([g for g in grants if g["name"] == "web-search"]) == 1


async def test_a_disabled_version_denies_everything_it_holds(
    engine, make_task, monkeypatch, governed
) -> None:
    async with engine.connect() as conn, conn.begin():
        await aq.set_version_status(conn, governed["version"]["id"], VersionStatus.DISABLED)

    task_id = await make_task(
        task_type="demo.tools",
        agent_version_id=governed["version"]["id"],
        payload={"tools": ["web-search"]},
    )
    assert await _run_task(engine, monkeypatch, task_id) == State.FAILED

    async with engine.connect() as conn:
        audit = (
            (
                await conn.execute(
                    sa.select(audit_events.c.data).where(audit_events.c.resource_id == task_id)
                )
            )
            .mappings()
            .one()
        )
    assert audit["data"]["reason"] == "version_disabled"
