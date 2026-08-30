"""Agent registry writes and resolution.

These are CONTROL-PLANE operations: admin actions, a handful per day, not per
second. That budget is what makes the concurrency choices here different from
everything in claim.py -- a blocking row lock is exactly right at this rate
and exactly wrong on the claim path, and the reasoning is the rate, not taste.

Three races exist here, all introduced by this phase, all closed cheaply:

  1. Two version creations racing for the same version number.
  2. Two activations racing to set default_version_id.
  3. A version being deprecated between route resolution and task insert.

None of them touch `tasks.state`, so no lease, fencing or CAS argument in
docs/CONCURRENCY.md changes. Adding read-only reference data cannot introduce
a write-write race on a row it never writes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from acp.db.models import agent_routes, agent_versions, agents
from acp.domain.agents import (
    ROUTABLE_AGENT_STATUSES,
    ROUTABLE_VERSION_STATUSES,
    AgentStatus,
    VersionStatus,
)


class AgentNotFound(Exception):
    pass


class NoRoute(Exception):
    """No agent is registered for this request type in this tenant."""


class NotRoutable(Exception):
    """An agent or version exists but is not eligible to receive new work."""


# ---------------------------------------------------------------------------
# agents
# ---------------------------------------------------------------------------


async def create_agent(
    conn: AsyncConnection, *, tenant_id: UUID, name: str, description: str | None = None
) -> Mapping[str, Any]:
    row = (
        (
            await conn.execute(
                sa.insert(agents)
                .values(tenant_id=tenant_id, name=name, description=description)
                .returning(*agents.c)
            )
        )
        .mappings()
        .one()
    )
    return dict(row)


async def get_agent(conn: AsyncConnection, agent_id: UUID) -> Mapping[str, Any] | None:
    row = (
        (await conn.execute(sa.select(*agents.c).where(agents.c.id == agent_id))).mappings().first()
    )
    return dict(row) if row else None


async def list_agents(
    conn: AsyncConnection, *, tenant_id: UUID | None = None
) -> list[Mapping[str, Any]]:
    stmt = sa.select(*agents.c).order_by(agents.c.created_at.desc())
    if tenant_id is not None:
        stmt = stmt.where(agents.c.tenant_id == tenant_id)
    return [dict(r) for r in (await conn.execute(stmt)).mappings().all()]


async def set_agent_status(
    conn: AsyncConnection, agent_id: UUID, status: AgentStatus
) -> Mapping[str, Any] | None:
    row = (
        (
            await conn.execute(
                sa.update(agents)
                .where(agents.c.id == agent_id)
                .values(status=status.value, updated_at=sa.func.now())
                .returning(*agents.c)
            )
        )
        .mappings()
        .first()
    )
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# versions
# ---------------------------------------------------------------------------


async def create_version(
    conn: AsyncConnection,
    *,
    agent_id: UUID,
    runtime_spec: Mapping[str, Any] | None = None,
    required_capabilities: Sequence[str] = (),
    config: Mapping[str, Any] | None = None,
    max_attempts: int = 3,
    max_execution_time_s: int = 300,
    status: VersionStatus = VersionStatus.DRAFT,
) -> Mapping[str, Any]:
    """Cut the next version of an agent.

    RACE 1 -- version number allocation. Two concurrent creations would both
    read `max(version) = 2` and both try to insert version 3. Fixed by locking
    the PARENT agent row first, which serialises allocation per agent while
    leaving different agents fully concurrent.

    A blocking `FOR UPDATE` is the right tool here for the same reason it is
    the wrong tool in claim.py: this runs a few times a day, so serialising it
    costs nothing, whereas serialising claims would cap fleet throughput at
    one worker. The unique constraint on (agent_id, version) remains as the
    backstop that turns a bug in this logic into a loud failure rather than
    two versions quietly sharing a number.
    """
    locked = (
        await conn.execute(sa.select(agents.c.id).where(agents.c.id == agent_id).with_for_update())
    ).first()
    if locked is None:
        raise AgentNotFound(str(agent_id))

    next_version = (
        await conn.execute(
            sa.select(sa.func.coalesce(sa.func.max(agent_versions.c.version), 0) + 1).where(
                agent_versions.c.agent_id == agent_id
            )
        )
    ).scalar_one()

    row = (
        (
            await conn.execute(
                sa.insert(agent_versions)
                .values(
                    agent_id=agent_id,
                    version=next_version,
                    runtime_spec=dict(runtime_spec or {}),
                    required_capabilities=list(required_capabilities),
                    config=dict(config or {}),
                    max_attempts=max_attempts,
                    max_execution_time_s=max_execution_time_s,
                    status=status.value,
                )
                .returning(*agent_versions.c)
            )
        )
        .mappings()
        .one()
    )
    return dict(row)


async def get_version(conn: AsyncConnection, version_id: UUID) -> Mapping[str, Any] | None:
    row = (
        (await conn.execute(sa.select(*agent_versions.c).where(agent_versions.c.id == version_id)))
        .mappings()
        .first()
    )
    return dict(row) if row else None


async def list_versions(conn: AsyncConnection, agent_id: UUID) -> list[Mapping[str, Any]]:
    rows = await conn.execute(
        sa.select(*agent_versions.c)
        .where(agent_versions.c.agent_id == agent_id)
        .order_by(agent_versions.c.version)
    )
    return [dict(r) for r in rows.mappings().all()]


async def set_version_status(
    conn: AsyncConnection, version_id: UUID, status: VersionStatus
) -> Mapping[str, Any] | None:
    """The ONLY field of a version that may change after insert.

    Everything else is rejected by the trigger in migration 0006 -- so a
    caller cannot quietly widen what an already-released agent does.
    """
    row = (
        (
            await conn.execute(
                sa.update(agent_versions)
                .where(agent_versions.c.id == version_id)
                .values(status=status.value)
                .returning(*agent_versions.c)
            )
        )
        .mappings()
        .first()
    )
    return dict(row) if row else None


@dataclass(frozen=True, slots=True)
class ActivationResult:
    applied: bool
    agent: Mapping[str, Any] | None = None
    reason: str | None = None


async def activate_version(
    conn: AsyncConnection,
    *,
    agent_id: UUID,
    version_id: UUID,
    expected_current_version_id: UUID | None = None,
    require_expected: bool = False,
) -> ActivationResult:
    """Release a version: mark it ACTIVE and point the agent's default at it.

    RACE 2 -- two operators activating different versions at once. Without a
    guard, both succeed, one silently overwrites the other, and the agent ends
    up running a version nobody chose. `require_expected` turns the update
    into a compare-and-set on `default_version_id`, so the loser is TOLD it
    raced instead of believing it won.

    Optional rather than mandatory because a first activation has no previous
    value to compare against, and a deliberate force-set is a legitimate
    operation -- it just should be deliberate.
    """
    version = await get_version(conn, version_id)
    if version is None or version["agent_id"] != agent_id:
        return ActivationResult(False, reason="version_not_found_for_agent")
    if VersionStatus(version["status"]) is VersionStatus.DISABLED:
        return ActivationResult(False, reason="version_disabled")

    stmt = sa.update(agents).where(agents.c.id == agent_id)
    if require_expected:
        stmt = stmt.where(
            agents.c.default_version_id.is_not_distinct_from(expected_current_version_id)
        )

    row = (
        (
            await conn.execute(
                stmt.values(default_version_id=version_id, updated_at=sa.func.now()).returning(
                    *agents.c
                )
            )
        )
        .mappings()
        .first()
    )
    if row is None:
        return ActivationResult(False, reason="default_version_changed_concurrently")

    await set_version_status(conn, version_id, VersionStatus.ACTIVE)
    return ActivationResult(True, agent=dict(row))


# ---------------------------------------------------------------------------
# routing and resolution
# ---------------------------------------------------------------------------


async def set_route(
    conn: AsyncConnection, *, tenant_id: UUID, request_type: str, agent_id: UUID
) -> None:
    """Upsert request_type -> agent for one tenant."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    await conn.execute(
        pg_insert(agent_routes)
        .values(tenant_id=tenant_id, request_type=request_type, agent_id=agent_id)
        .on_conflict_do_update(
            index_elements=[agent_routes.c.tenant_id, agent_routes.c.request_type],
            set_={"agent_id": agent_id},
        )
    )


async def list_routes(conn: AsyncConnection, *, tenant_id: UUID) -> list[Mapping[str, Any]]:
    rows = await conn.execute(
        sa.select(*agent_routes.c)
        .where(agent_routes.c.tenant_id == tenant_id)
        .order_by(agent_routes.c.request_type)
    )
    return [dict(r) for r in rows.mappings().all()]


@dataclass(frozen=True, slots=True)
class Resolution:
    """The pinned answer to 'what will execute this task?'."""

    agent_id: UUID
    agent_name: str
    version_id: UUID
    version: int
    required_capabilities: list[str]
    max_attempts: int
    max_execution_time_s: int
    runtime_spec: Mapping[str, Any]


async def resolve_route(conn: AsyncConnection, *, tenant_id: UUID, request_type: str) -> Resolution:
    """request_type -> agent -> its default version, as one query.

    RACE 3 -- a version being deprecated between resolution and the task
    insert. Closed by joining route, agent and version in a single statement
    that the CALLER runs inside the same transaction as the INSERT. Both see
    one snapshot, so a task can never be pinned to a version that was already
    disabled when it was created.

    Everything the execution path needs is returned here and copied onto the
    task row. After this returns, nothing downstream reads the registry again
    -- which is what keeps the claim query free of joins against it.
    """
    stmt = (
        sa.select(
            agents.c.id.label("agent_id"),
            agents.c.name.label("agent_name"),
            agents.c.status.label("agent_status"),
            agents.c.default_version_id,
            agent_versions.c.id.label("version_id"),
            agent_versions.c.version,
            agent_versions.c.status.label("version_status"),
            agent_versions.c.required_capabilities,
            agent_versions.c.max_attempts,
            agent_versions.c.max_execution_time_s,
            agent_versions.c.runtime_spec,
        )
        .select_from(
            agent_routes.join(agents, agents.c.id == agent_routes.c.agent_id).outerjoin(
                agent_versions, agent_versions.c.id == agents.c.default_version_id
            )
        )
        .where(
            agent_routes.c.tenant_id == tenant_id,
            agent_routes.c.request_type == request_type,
        )
    )
    row = (await conn.execute(stmt)).mappings().first()
    if row is None:
        raise NoRoute(f"no agent routed for request_type={request_type!r}")

    if AgentStatus(row["agent_status"]) not in ROUTABLE_AGENT_STATUSES:
        raise NotRoutable(f"agent {row['agent_name']!r} is {row['agent_status']}")
    if row["version_id"] is None:
        raise NotRoutable(f"agent {row['agent_name']!r} has no released version")
    if VersionStatus(row["version_status"]) not in ROUTABLE_VERSION_STATUSES:
        raise NotRoutable(
            f"agent {row['agent_name']!r} default version {row['version']} "
            f"is {row['version_status']}"
        )

    return Resolution(
        agent_id=row["agent_id"],
        agent_name=row["agent_name"],
        version_id=row["version_id"],
        version=row["version"],
        required_capabilities=list(row["required_capabilities"]),
        max_attempts=row["max_attempts"],
        max_execution_time_s=row["max_execution_time_s"],
        runtime_spec=dict(row["runtime_spec"]),
    )
