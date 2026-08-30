"""Agent registry administration.

Control-plane endpoints: low rate, admin-driven, audited by their effect on
the registry. None of them are on the execution path -- a worker never calls
any of this, which is the boundary that lets the registry be down without
stopping execution.
"""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection

from acp.api.deps import db_read, db_txn
from acp.api.errors import Conflict, NotFound
from acp.db.queries import agents as q
from acp.db.queries import tasks as tq
from acp.domain.agents import AgentStatus, VersionStatus

router = APIRouter(prefix="/v1/agents", tags=["agents"])

Txn = Annotated[AsyncConnection, Depends(db_txn)]
Read = Annotated[AsyncConnection, Depends(db_read)]


class AgentCreate(BaseModel):
    tenant_id: UUID
    name: str = Field(min_length=1, max_length=128)
    description: str | None = None


class VersionCreate(BaseModel):
    runtime_spec: dict[str, Any] = Field(default_factory=dict)
    required_capabilities: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    max_attempts: int = Field(default=3, ge=1, le=20)
    max_execution_time_s: int = Field(default=300, ge=1, le=86_400)


class ActivateRequest(BaseModel):
    version_id: UUID
    #: When set, the update becomes a compare-and-set on the agent's current
    #: default. Two operators activating different versions at once would
    #: otherwise both succeed, one silently overwriting the other.
    expected_current_version_id: UUID | None = None
    require_expected: bool = False


class RouteCreate(BaseModel):
    tenant_id: UUID
    request_type: str = Field(min_length=1, max_length=128)
    agent_id: UUID


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_agent(body: AgentCreate, conn: Txn) -> dict:
    try:
        return dict(
            await q.create_agent(
                conn,
                tenant_id=body.tenant_id,
                name=body.name,
                description=body.description,
            )
        )
    except IntegrityError as exc:
        raise Conflict("agent name already exists for this tenant", name=body.name) from exc


@router.get("")
async def list_agents(conn: Read, tenant_id: UUID | None = None) -> list[dict]:
    return [dict(a) for a in await q.list_agents(conn, tenant_id=tenant_id)]


@router.get("/{agent_id}")
async def get_agent(agent_id: UUID, conn: Read) -> dict:
    agent = await q.get_agent(conn, agent_id)
    if agent is None:
        raise NotFound("unknown agent", agent_id=str(agent_id))
    return dict(agent)


@router.post("/{agent_id}/versions", status_code=status.HTTP_201_CREATED)
async def create_version(agent_id: UUID, body: VersionCreate, conn: Txn) -> dict:
    """Cut a new immutable version.

    Created as DRAFT: cutting a version and releasing it are separate acts, so
    a definition can be reviewed before any traffic reaches it.
    """
    try:
        return dict(
            await q.create_version(
                conn,
                agent_id=agent_id,
                runtime_spec=body.runtime_spec,
                required_capabilities=body.required_capabilities,
                config=body.config,
                max_attempts=body.max_attempts,
                max_execution_time_s=body.max_execution_time_s,
            )
        )
    except q.AgentNotFound as exc:
        raise NotFound("unknown agent", agent_id=str(agent_id)) from exc


@router.get("/{agent_id}/versions")
async def list_versions(agent_id: UUID, conn: Read) -> list[dict]:
    if await q.get_agent(conn, agent_id) is None:
        raise NotFound("unknown agent", agent_id=str(agent_id))
    return [dict(v) for v in await q.list_versions(conn, agent_id)]


@router.post("/{agent_id}/activate")
async def activate(agent_id: UUID, body: ActivateRequest, conn: Txn) -> dict:
    """Release a version: point the agent's default at it and mark it ACTIVE."""
    result = await q.activate_version(
        conn,
        agent_id=agent_id,
        version_id=body.version_id,
        expected_current_version_id=body.expected_current_version_id,
        require_expected=body.require_expected,
    )
    if not result.applied:
        if result.reason == "version_not_found_for_agent":
            raise NotFound("no such version for this agent", version_id=str(body.version_id))
        raise Conflict("activation rejected", reason=result.reason)
    return dict(result.agent)


@router.post("/{agent_id}/disable")
async def disable(agent_id: UUID, conn: Txn) -> dict:
    """Disable an agent and stop its live work.

    The sweep reuses cancellation rather than adding an agent-status check to
    the claim query -- see db/queries/tasks.cancel_agent_tasks. Queued tasks
    end immediately; running ones stop at their next lease renewal, or when
    the reaper notices an expired lease if their worker is already gone.
    """
    agent = await q.set_agent_status(conn, agent_id, AgentStatus.DISABLED)
    if agent is None:
        raise NotFound("unknown agent", agent_id=str(agent_id))

    versions = await q.list_versions(conn, agent_id)
    for version in versions:
        await q.set_version_status(conn, version["id"], VersionStatus.DISABLED)

    affected = await tq.cancel_agent_tasks(
        conn,
        agent_version_ids=[v["id"] for v in versions],
        reason=f"agent {agent['name']!r} disabled",
    )
    return {"agent": dict(agent), "tasks_cancelled_or_cancelling": affected}


routes_router = APIRouter(prefix="/v1/routes", tags=["agents"])


@routes_router.put("", status_code=status.HTTP_200_OK)
async def set_route(body: RouteCreate, conn: Txn) -> dict:
    agent = await q.get_agent(conn, body.agent_id)
    if agent is None:
        raise NotFound("unknown agent", agent_id=str(body.agent_id))
    if agent["tenant_id"] != body.tenant_id:
        # Routing to another tenant's agent would be a cross-tenant execution
        # path, which agents being tenant-scoped exists to prevent.
        raise Conflict("agent belongs to a different tenant")

    await q.set_route(
        conn, tenant_id=body.tenant_id, request_type=body.request_type, agent_id=body.agent_id
    )
    return {
        "tenant_id": str(body.tenant_id),
        "request_type": body.request_type,
        "agent_id": str(body.agent_id),
    }


@routes_router.get("")
async def list_routes(tenant_id: UUID, conn: Read) -> list[dict]:
    return [dict(r) for r in await q.list_routes(conn, tenant_id=tenant_id)]
