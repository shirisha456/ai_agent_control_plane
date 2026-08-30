"""Agent registry: immutability, version allocation, activation, resolution.

The tests that matter here are the three concurrency ones. Everything else in
this phase is CRUD, and CRUD that cannot be raced is not interesting.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import sqlalchemy as sa

from acp.db.models import agent_versions, agents
from acp.db.queries import agents as q
from acp.domain.agents import AgentStatus, VersionStatus, capability_key, worker_satisfies

pytestmark = pytest.mark.db


@pytest.fixture
async def agent_id(engine, tenant_id) -> uuid.UUID:
    async with engine.connect() as conn, conn.begin():
        agent = await q.create_agent(
            conn, tenant_id=tenant_id, name=f"research-{uuid.uuid4().hex[:6]}"
        )
    return agent["id"]


# ---------------------------------------------------------------------------
# immutability -- enforced by the database, not by convention
# ---------------------------------------------------------------------------


async def test_a_released_version_cannot_be_edited(engine, agent_id) -> None:
    """The trigger, not the application, is what makes this true.

    Three properties rest on versions never changing: reproducibility ("v3 is
    still what it was when task 123 ran"), reviewability (widening an agent's
    behaviour requires a diff), and the safety of copying
    required_capabilities onto the task row. An invariant three things depend
    on should be unrepresentable, not merely documented.
    """
    async with engine.connect() as conn, conn.begin():
        version = await q.create_version(conn, agent_id=agent_id, runtime_spec={"steps": ["a"]})

    for field, value in [
        ("runtime_spec", {"steps": ["a", "evil"]}),
        ("required_capabilities", ["gpu"]),
        ("config", {"model": "swapped"}),
        ("max_attempts", 99),
        ("max_execution_time_s", 1),
        ("version", 42),
    ]:
        with pytest.raises(Exception, match="immutable"):
            async with engine.connect() as conn, conn.begin():
                await conn.execute(
                    sa.update(agent_versions)
                    .where(agent_versions.c.id == version["id"])
                    .values(**{field: value})
                )

    async with engine.connect() as conn:
        after = (
            (
                await conn.execute(
                    sa.select(*agent_versions.c).where(agent_versions.c.id == version["id"])
                )
            )
            .mappings()
            .one()
        )
    assert after["runtime_spec"] == {"steps": ["a"]}
    assert after["max_attempts"] == 3


async def test_status_is_the_one_mutable_field(engine, agent_id) -> None:
    """Because status IS the lifecycle.

    Draft, release, deprecate, emergency-stop -- all of that must be possible
    without cutting a new version, or the registry becomes unusable.
    """
    async with engine.connect() as conn, conn.begin():
        version = await q.create_version(conn, agent_id=agent_id)
        assert version["status"] == VersionStatus.DRAFT

    async with engine.connect() as conn, conn.begin():
        updated = await q.set_version_status(conn, version["id"], VersionStatus.ACTIVE)
    assert updated["status"] == VersionStatus.ACTIVE


# ---------------------------------------------------------------------------
# race 1 -- version number allocation
# ---------------------------------------------------------------------------


async def test_concurrent_version_creation_allocates_distinct_numbers(engine, agent_id) -> None:
    """Ten operators cutting versions at once must not collide.

    Without the row lock on the parent agent, all ten read `max(version) = 0`
    and all ten try to insert version 1. The unique constraint would catch it
    -- as nine errors -- but the point is that allocation should not need
    catching.
    """

    async def cut():
        async with engine.connect() as conn, conn.begin():
            return await q.create_version(conn, agent_id=agent_id)

    versions = await asyncio.gather(*(cut() for _ in range(10)))
    numbers = sorted(v["version"] for v in versions)

    assert numbers == list(range(1, 11)), f"expected dense 1..10, got {numbers}"
    assert len({v["id"] for v in versions}) == 10


async def test_version_allocation_is_per_agent_not_global(engine, tenant_id) -> None:
    """Locking the parent serialises one agent, never the registry.

    Two different agents cutting versions concurrently must not block each
    other -- otherwise the lock scope would be the whole table.
    """
    async with engine.connect() as conn, conn.begin():
        a = await q.create_agent(conn, tenant_id=tenant_id, name=f"a-{uuid.uuid4().hex[:6]}")
        b = await q.create_agent(conn, tenant_id=tenant_id, name=f"b-{uuid.uuid4().hex[:6]}")

    async def cut(agent):
        async with engine.connect() as conn, conn.begin():
            return await q.create_version(conn, agent_id=agent["id"])

    results = await asyncio.gather(*(cut(a) for _ in range(3)), *(cut(b) for _ in range(3)))
    assert sorted(r["version"] for r in results) == [1, 1, 2, 2, 3, 3]


# ---------------------------------------------------------------------------
# race 2 -- concurrent activation
# ---------------------------------------------------------------------------


async def test_concurrent_activation_with_cas_has_one_winner(engine, agent_id) -> None:
    """Two operators releasing different versions at the same moment.

    Without the compare-and-set both succeed, one silently overwrites the
    other, and the agent ends up running a version nobody chose. With it, the
    loser is TOLD it raced -- which is the whole difference between a safe
    operation and a coin flip.
    """
    async with engine.connect() as conn, conn.begin():
        v1 = await q.create_version(conn, agent_id=agent_id)
        v2 = await q.create_version(conn, agent_id=agent_id)

    async def activate(version_id):
        async with engine.connect() as conn, conn.begin():
            return await q.activate_version(
                conn,
                agent_id=agent_id,
                version_id=version_id,
                expected_current_version_id=None,
                require_expected=True,
            )

    results = await asyncio.gather(activate(v1["id"]), activate(v2["id"]))
    applied = [r for r in results if r.applied]

    assert len(applied) == 1, "both activations won; the agent's default is now arbitrary"
    loser = next(r for r in results if not r.applied)
    assert loser.reason == "default_version_changed_concurrently"


async def test_activation_without_cas_is_a_deliberate_overwrite(engine, agent_id) -> None:
    """Force-setting is legitimate -- it just has to be asked for."""
    async with engine.connect() as conn, conn.begin():
        v1 = await q.create_version(conn, agent_id=agent_id)
        v2 = await q.create_version(conn, agent_id=agent_id)

    async with engine.connect() as conn, conn.begin():
        assert (await q.activate_version(conn, agent_id=agent_id, version_id=v1["id"])).applied
    async with engine.connect() as conn, conn.begin():
        assert (await q.activate_version(conn, agent_id=agent_id, version_id=v2["id"])).applied

    async with engine.connect() as conn:
        default = (
            await conn.execute(
                sa.select(agents.c.default_version_id).where(agents.c.id == agent_id)
            )
        ).scalar_one()
    assert default == v2["id"]


async def test_a_disabled_version_cannot_be_activated(engine, agent_id) -> None:
    async with engine.connect() as conn, conn.begin():
        version = await q.create_version(conn, agent_id=agent_id)
        await q.set_version_status(conn, version["id"], VersionStatus.DISABLED)

    async with engine.connect() as conn, conn.begin():
        result = await q.activate_version(conn, agent_id=agent_id, version_id=version["id"])
    assert not result.applied
    assert result.reason == "version_disabled"


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


async def test_resolution_returns_everything_the_execution_path_needs(
    engine, tenant_id, agent_id
) -> None:
    """After this, nothing downstream reads the registry again.

    That is the property that keeps the claim query free of joins against
    these tables -- and therefore the reason none of the concurrency
    arguments in the execution engine had to be revisited for this phase.
    """
    async with engine.connect() as conn, conn.begin():
        version = await q.create_version(
            conn,
            agent_id=agent_id,
            required_capabilities=["gpu", "internet"],
            max_attempts=7,
            max_execution_time_s=120,
        )
        await q.activate_version(conn, agent_id=agent_id, version_id=version["id"])
        await q.set_route(
            conn, tenant_id=tenant_id, request_type="RESEARCH_REPORT", agent_id=agent_id
        )

    async with engine.connect() as conn:
        resolved = await q.resolve_route(conn, tenant_id=tenant_id, request_type="RESEARCH_REPORT")

    assert resolved.version_id == version["id"]
    assert resolved.version == 1
    assert sorted(resolved.required_capabilities) == ["gpu", "internet"]
    assert resolved.max_attempts == 7
    assert resolved.max_execution_time_s == 120


async def test_unrouted_request_type_is_rejected(engine, tenant_id) -> None:
    async with engine.connect() as conn:
        with pytest.raises(q.NoRoute):
            await q.resolve_route(conn, tenant_id=tenant_id, request_type="NOPE")


async def test_an_agent_with_no_released_version_is_not_routable(
    engine, tenant_id, agent_id
) -> None:
    """Registering an agent and releasing one are separate acts."""
    async with engine.connect() as conn, conn.begin():
        await q.create_version(conn, agent_id=agent_id)  # DRAFT, never activated
        await q.set_route(conn, tenant_id=tenant_id, request_type="X", agent_id=agent_id)

    async with engine.connect() as conn:
        with pytest.raises(q.NotRoutable, match="no released version"):
            await q.resolve_route(conn, tenant_id=tenant_id, request_type="X")


async def test_a_disabled_agent_is_not_routable(engine, tenant_id, agent_id) -> None:
    async with engine.connect() as conn, conn.begin():
        version = await q.create_version(conn, agent_id=agent_id)
        await q.activate_version(conn, agent_id=agent_id, version_id=version["id"])
        await q.set_route(conn, tenant_id=tenant_id, request_type="X", agent_id=agent_id)
        await q.set_agent_status(conn, agent_id, AgentStatus.DISABLED)

    async with engine.connect() as conn:
        with pytest.raises(q.NotRoutable):
            await q.resolve_route(conn, tenant_id=tenant_id, request_type="X")


async def test_deprecating_a_version_does_not_disturb_pinned_tasks(
    engine, tenant_id, agent_id, make_task
) -> None:
    """Pinning means exactly this, and it is the cost of reproducibility.

    A task already pinned to v1 keeps running v1 after v1 is deprecated. The
    escape hatch is the disable sweep, which reuses cancellation -- not a
    status check on the claim path.
    """
    async with engine.connect() as conn, conn.begin():
        version = await q.create_version(conn, agent_id=agent_id)
        await q.activate_version(conn, agent_id=agent_id, version_id=version["id"])

    task_id = await make_task(agent_version_id=version["id"])

    async with engine.connect() as conn, conn.begin():
        await q.set_version_status(conn, version["id"], VersionStatus.DEPRECATED)

    async with engine.connect() as conn:
        from acp.db.models import tasks

        row = (
            (
                await conn.execute(
                    sa.select(tasks.c.state, tasks.c.agent_version_id).where(tasks.c.id == task_id)
                )
            )
            .mappings()
            .one()
        )
    assert row["state"] == "QUEUED"
    assert row["agent_version_id"] == version["id"]


# ---------------------------------------------------------------------------
# capability keys (pure, but checked against what the DB stores)
# ---------------------------------------------------------------------------


def test_capability_key_is_canonical() -> None:
    """Order and case must not produce two keys for one requirement set."""
    assert capability_key(["GPU", "internet"]) == capability_key(["internet", "gpu"])
    assert capability_key([]) == ""
    assert capability_key([" gpu ", "gpu"]) == "gpu"


def test_empty_requirements_are_satisfied_by_any_worker() -> None:
    """A task stating no requirements must never be unschedulable."""
    assert worker_satisfies([], [])
    assert worker_satisfies([], ["gpu"])
    assert worker_satisfies(["gpu"], ["gpu", "internet"])
    assert not worker_satisfies(["gpu"], ["internet"])
