"""Tool registry, grants, and the audit log — control-plane endpoints."""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection

from acp.api.deps import db_read, db_txn
from acp.api.errors import Conflict, NotFound
from acp.db.queries import agents as aq
from acp.db.queries import audit as auditq
from acp.db.queries import tasks as tq
from acp.db.queries import tools as q
from acp.domain.agents import VersionStatus
from acp.domain.authz import ToolStatus, ToolType

router = APIRouter(prefix="/v1/tools", tags=["tools"])
audit_router = APIRouter(prefix="/v1/audit", tags=["tools"])

Txn = Annotated[AsyncConnection, Depends(db_txn)]
Read = Annotated[AsyncConnection, Depends(db_read)]


class ToolCreate(BaseModel):
    tenant_id: UUID
    name: str = Field(min_length=1, max_length=128)
    tool_type: ToolType = ToolType.SIMULATED
    description: str | None = None
    #: Endpoint and timeouts, plus a REFERENCE to a credential such as
    #: {"secret_ref": "env:GITHUB_TOKEN"}. Never a credential itself -- this
    #: column is readable by anything with database access.
    config: dict[str, Any] = Field(default_factory=dict)


class GrantCreate(BaseModel):
    tool_id: UUID


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_tool(body: ToolCreate, conn: Txn) -> dict:
    # Checked explicitly rather than left to the FK -- see the identical
    # reasoning on POST /v1/agents. tools.tenant_id also references
    # tenants.id, so without this check a bad tenant_id and a duplicate name
    # raise the same IntegrityError and get reported identically, which is
    # wrong for the former.
    if await tq.get_tenant(conn, body.tenant_id) is None:
        raise NotFound("unknown tenant", tenant_id=str(body.tenant_id))

    try:
        return dict(
            await q.create_tool(
                conn,
                tenant_id=body.tenant_id,
                name=body.name,
                tool_type=body.tool_type,
                config=body.config,
                description=body.description,
            )
        )
    except IntegrityError as exc:
        raise Conflict("tool name already exists for this tenant", name=body.name) from exc


@router.get("")
async def list_tools(tenant_id: UUID, conn: Read) -> list[dict]:
    return [dict(t) for t in await q.list_tools(conn, tenant_id=tenant_id)]


@router.post("/{tool_id}/disable")
async def disable_tool(tool_id: UUID, conn: Txn) -> dict:
    """The live kill switch.

    Takes effect for every task claimed after this commits, regardless of any
    grant -- grants are versioned and immutable, denial is not. Tasks already
    executing finish under the policy they were claimed with, so revocation
    latency is bounded by attempt duration rather than by a cache TTL.
    """
    tool = await q.set_tool_status(conn, tool_id, ToolStatus.DISABLED)
    if tool is None:
        raise NotFound("unknown tool", tool_id=str(tool_id))
    await auditq.record_audit(
        conn,
        tenant_id=tool["tenant_id"],
        action="TOOL_DISABLED",
        resource_type="tool",
        resource_id=tool_id,
        outcome="OK",
        data={"name": tool["name"]},
    )
    return dict(tool)


@router.post("/{tool_id}/enable")
async def enable_tool(tool_id: UUID, conn: Txn) -> dict:
    tool = await q.set_tool_status(conn, tool_id, ToolStatus.ACTIVE)
    if tool is None:
        raise NotFound("unknown tool", tool_id=str(tool_id))
    await auditq.record_audit(
        conn,
        tenant_id=tool["tenant_id"],
        action="TOOL_ENABLED",
        resource_type="tool",
        resource_id=tool_id,
        outcome="OK",
        data={"name": tool["name"]},
    )
    return dict(tool)


# --------------------------------------------------------------------------
# grants live on the VERSION
# --------------------------------------------------------------------------

grants_router = APIRouter(prefix="/v1/agent-versions", tags=["tools"])


@grants_router.post("/{version_id}/grants", status_code=status.HTTP_201_CREATED)
async def grant(version_id: UUID, body: GrantCreate, conn: Txn) -> dict:
    """Grant a tool to an agent version.

    Refused once the version is ACTIVE. A version is meant to be a complete,
    immutable capability bundle; letting grants change after release would
    quietly widen what a running agent may reach, which is exactly what
    attaching grants to versions rather than agents exists to prevent. To add
    a tool to a released agent, cut a new version -- that is a reviewable diff.
    """
    version = await aq.get_version(conn, version_id)
    if version is None:
        raise NotFound("unknown agent version", version_id=str(version_id))
    if VersionStatus(version["status"]) is VersionStatus.ACTIVE:
        raise Conflict(
            "cannot change grants on a released version; cut a new version instead",
            version_id=str(version_id),
            status=version["status"],
        )

    tool = await q.get_tool(conn, body.tool_id)
    if tool is None:
        raise NotFound("unknown tool", tool_id=str(body.tool_id))

    agent = await aq.get_agent(conn, version["agent_id"])
    if agent["tenant_id"] != tool["tenant_id"]:
        raise Conflict("tool belongs to a different tenant")

    await q.grant_tool(conn, agent_version_id=version_id, tool_id=body.tool_id)
    await auditq.record_audit(
        conn,
        tenant_id=tool["tenant_id"],
        action="TOOL_GRANTED",
        resource_type="agent_version",
        resource_id=version_id,
        outcome="OK",
        data={"tool": tool["name"], "tool_id": str(body.tool_id)},
    )
    return {"agent_version_id": str(version_id), "tool_id": str(body.tool_id)}


@grants_router.get("/{version_id}/grants")
async def list_grants(version_id: UUID, conn: Read) -> list[dict]:
    if await aq.get_version(conn, version_id) is None:
        raise NotFound("unknown agent version", version_id=str(version_id))
    return [dict(g) for g in await q.list_grants(conn, agent_version_id=version_id)]


@grants_router.delete("/{version_id}/grants/{tool_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_grant(version_id: UUID, tool_id: UUID, conn: Txn) -> None:
    """Remove a tool grant from a version that has not been released yet.

    The counterpart of POST .../grants, with the SAME lifecycle guard: a
    version is meant to be a complete, immutable capability bundle once
    ACTIVE, so its grants may only change while it is still DRAFT. Without
    this endpoint a grant added to a DRAFT version by mistake could be added
    to but never removed from through the API -- the query layer already
    supports it (db.queries.tools.revoke_grant), it was simply never wired up.
    """
    version = await aq.get_version(conn, version_id)
    if version is None:
        raise NotFound("unknown agent version", version_id=str(version_id))
    if VersionStatus(version["status"]) is VersionStatus.ACTIVE:
        raise Conflict(
            "cannot change grants on a released version; cut a new version instead",
            version_id=str(version_id),
            status=version["status"],
        )

    tool = await q.get_tool(conn, tool_id)
    if tool is None:
        raise NotFound("unknown tool", tool_id=str(tool_id))

    removed = await q.revoke_grant(conn, agent_version_id=version_id, tool_id=tool_id)
    if not removed:
        raise NotFound("grant does not exist", version_id=str(version_id), tool_id=str(tool_id))

    await auditq.record_audit(
        conn,
        tenant_id=tool["tenant_id"],
        action="TOOL_GRANT_REVOKED",
        resource_type="agent_version",
        resource_id=version_id,
        outcome="OK",
        data={"tool": tool["name"], "tool_id": str(tool_id)},
    )


@audit_router.get("")
async def list_audit(
    tenant_id: UUID,
    conn: Read,
    action: str | None = None,
    outcome: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict]:
    """Read the audit log.

    Separate from a task's event timeline on purpose: these records outlive
    the tasks they describe, because task_events are pruned and audit records
    are not.
    """
    return [
        dict(e)
        for e in await auditq.list_audit(
            conn, tenant_id=tenant_id, action=action, outcome=outcome, limit=limit
        )
    ]
