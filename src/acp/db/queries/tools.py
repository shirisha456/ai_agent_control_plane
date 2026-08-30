"""Tool registry, grants, and the claim-time policy snapshot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncConnection

from acp.db.models import agent_version_tool_grants, agent_versions, agents, tools
from acp.domain.authz import ToolPolicy, ToolRef, ToolStatus, ToolType


async def create_tool(
    conn: AsyncConnection,
    *,
    tenant_id: UUID,
    name: str,
    tool_type: ToolType = ToolType.SIMULATED,
    config: Mapping[str, Any] | None = None,
    description: str | None = None,
) -> Mapping[str, Any]:
    row = (
        (
            await conn.execute(
                sa.insert(tools)
                .values(
                    tenant_id=tenant_id,
                    name=name,
                    tool_type=tool_type.value,
                    config=dict(config or {}),
                    description=description,
                )
                .returning(*tools.c)
            )
        )
        .mappings()
        .one()
    )
    return dict(row)


async def get_tool(conn: AsyncConnection, tool_id: UUID) -> Mapping[str, Any] | None:
    row = (await conn.execute(sa.select(*tools.c).where(tools.c.id == tool_id))).mappings().first()
    return dict(row) if row else None


async def list_tools(conn: AsyncConnection, *, tenant_id: UUID) -> list[Mapping[str, Any]]:
    rows = await conn.execute(
        sa.select(*tools.c).where(tools.c.tenant_id == tenant_id).order_by(tools.c.name)
    )
    return [dict(r) for r in rows.mappings().all()]


async def set_tool_status(
    conn: AsyncConnection, tool_id: UUID, status: ToolStatus
) -> Mapping[str, Any] | None:
    """The live kill switch.

    Takes effect for every task claimed after this commits, regardless of any
    grant. Grants are versioned and immutable; denial is not. Tasks already
    executing finish under the policy they were claimed with, so revocation
    latency is bounded by attempt duration -- a documented SLA rather than a
    cache TTL.
    """
    row = (
        (
            await conn.execute(
                sa.update(tools)
                .where(tools.c.id == tool_id)
                .values(status=status.value)
                .returning(*tools.c)
            )
        )
        .mappings()
        .first()
    )
    return dict(row) if row else None


async def grant_tool(conn: AsyncConnection, *, agent_version_id: UUID, tool_id: UUID) -> None:
    """Attach a tool to an immutable version.

    Idempotent: granting twice is not an error, because the desired end state
    is the same and an operator re-running a provisioning script should not
    have to care.

    Note this is the one write that mutates a version's effective behaviour
    after creation -- which is why it belongs to the version's construction,
    before release, and why the API refuses it on an ACTIVE version.
    """
    await conn.execute(
        pg_insert(agent_version_tool_grants)
        .values(agent_version_id=agent_version_id, tool_id=tool_id)
        .on_conflict_do_nothing(
            index_elements=[
                agent_version_tool_grants.c.agent_version_id,
                agent_version_tool_grants.c.tool_id,
            ]
        )
    )


async def revoke_grant(conn: AsyncConnection, *, agent_version_id: UUID, tool_id: UUID) -> bool:
    result = await conn.execute(
        sa.delete(agent_version_tool_grants).where(
            agent_version_tool_grants.c.agent_version_id == agent_version_id,
            agent_version_tool_grants.c.tool_id == tool_id,
        )
    )
    return result.rowcount > 0


async def list_grants(conn: AsyncConnection, *, agent_version_id: UUID) -> list[Mapping[str, Any]]:
    rows = await conn.execute(
        sa.select(tools.c.id, tools.c.name, tools.c.tool_type, tools.c.status)
        .select_from(
            agent_version_tool_grants.join(tools, tools.c.id == agent_version_tool_grants.c.tool_id)
        )
        .where(agent_version_tool_grants.c.agent_version_id == agent_version_id)
        .order_by(tools.c.name)
    )
    return [dict(r) for r in rows.mappings().all()]


async def snapshot_policies(
    conn: AsyncConnection, *, agent_version_ids: Sequence[UUID]
) -> dict[UUID, ToolPolicy]:
    """Freeze what each claimed version may do, for the life of its attempt.

    Called from inside the CLAIM transaction, on the handful of rows the claim
    actually returned -- so it is a small lookup keyed by primary keys, not
    something the ordering scan pays for.

    Doing it here rather than at tool-call time buys three things:

      * the tool-call path becomes a set lookup in memory: no query, no cache,
        no staleness window to reason about;
      * the policy is CONSISTENT for the whole attempt -- an agent cannot have
        a tool granted halfway through its own execution;
      * revocation latency becomes bounded by attempt duration, which is
        explainable, instead of by a cache TTL, which is a guess.

    The tenant's whole tool namespace is included, not just granted tools, so
    a request for an existing-but-ungranted tool reports NOT_GRANTED rather
    than UNKNOWN_TOOL. To whoever reads the audit log that is the difference
    between a misconfiguration and an agent reaching for something it was
    never meant to touch.
    """
    unique_ids = [i for i in dict.fromkeys(agent_version_ids) if i is not None]
    if not unique_ids:
        return {}

    version_rows = (
        (
            await conn.execute(
                sa.select(agent_versions.c.id, agent_versions.c.status, agents.c.tenant_id)
                .select_from(agent_versions.join(agents, agents.c.id == agent_versions.c.agent_id))
                .where(agent_versions.c.id.in_(unique_ids))
            )
        )
        .mappings()
        .all()
    )
    if not version_rows:
        return {}

    tenant_ids = {r["tenant_id"] for r in version_rows}
    tool_rows = (
        (await conn.execute(sa.select(*tools.c).where(tools.c.tenant_id.in_(tenant_ids))))
        .mappings()
        .all()
    )
    grant_rows = (
        await conn.execute(
            sa.select(
                agent_version_tool_grants.c.agent_version_id,
                agent_version_tool_grants.c.tool_id,
            ).where(agent_version_tool_grants.c.agent_version_id.in_(unique_ids))
        )
    ).all()

    tools_by_tenant: dict[UUID, dict[str, ToolRef]] = {}
    for row in tool_rows:
        tools_by_tenant.setdefault(row["tenant_id"], {})[row["name"]] = ToolRef(
            id=row["id"],
            name=row["name"],
            tool_type=ToolType(row["tool_type"]),
            status=ToolStatus(row["status"]),
            config=dict(row["config"]),
        )

    granted: dict[UUID, set[UUID]] = {}
    for version_id, tool_id in grant_rows:
        granted.setdefault(version_id, set()).add(tool_id)

    return {
        row["id"]: ToolPolicy(
            agent_version_id=row["id"],
            granted_tool_ids=frozenset(granted.get(row["id"], ())),
            tools_by_name=tools_by_tenant.get(row["tenant_id"], {}),
            version_disabled=row["status"] == "DISABLED",
        )
        for row in version_rows
    }
