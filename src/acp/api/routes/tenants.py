"""Tenant administration.

Minimal on purpose. Tenants exist in V1 so that quotas and fairness have
something to be fair between; the interesting work is in the claim query and
the admission path, not here.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection

from acp.api.deps import db_read, db_txn
from acp.api.errors import Conflict, NotFound
from acp.api.schemas import TenantCreate, TenantOut
from acp.db.queries import tasks as q

router = APIRouter(prefix="/v1/tenants", tags=["tenants"])


@router.post("", status_code=status.HTTP_201_CREATED, response_model=TenantOut)
async def create(
    body: TenantCreate, conn: Annotated[AsyncConnection, Depends(db_txn)]
) -> TenantOut:
    try:
        row = await q.create_tenant(
            conn,
            name=body.name,
            max_concurrent_tasks=body.max_concurrent_tasks,
            max_queued_tasks=body.max_queued_tasks,
        )
    except IntegrityError as exc:
        raise Conflict("tenant name already exists", name=body.name) from exc
    return TenantOut(**row)


@router.get("/{tenant_id}", response_model=TenantOut)
async def get(tenant_id: UUID, conn: Annotated[AsyncConnection, Depends(db_read)]) -> TenantOut:
    row = await q.get_tenant(conn, tenant_id)
    if row is None:
        raise NotFound("unknown tenant", tenant_id=str(tenant_id))
    return TenantOut(**row)
