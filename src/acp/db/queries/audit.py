"""Append-only audit log for control-plane and security-relevant events.

Separate from task_events, and the decisive reason is RETENTION: task_events
are execution noise, pruned after days, growing with throughput; audit records
must be kept, because that is what audit means. One table cannot be both
aggressively pruned and permanently retained.

THE SPLIT
---------
    ALLOW decisions are EXECUTION HISTORY  -> task_events only
    DENY  decisions are AUDIT RECORDS      -> both

An allowed tool call is one of thousands and is only interesting while
debugging that task, so it lives with the task and dies with it. A refusal is
rare by construction and is the thing someone will come looking for months
later, so it goes to both: the task timeline explains what the task did, and
the audit log survives the task's retention.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from acp.db.models import audit_events


async def record_audit(
    conn: AsyncConnection,
    *,
    tenant_id: UUID,
    action: str,
    resource_type: str,
    outcome: str,
    resource_id: UUID | None = None,
    actor: str = "system",
    data: Mapping[str, Any] | None = None,
) -> None:
    """Append one audit record using the caller's transaction.

    Correct for CONTROL-PLANE actions, where the record and the change it
    describes should commit or roll back together: an audit entry saying a
    tool was disabled, when the disable was rolled back, would be a lie.

    NOT correct for a denial raised during execution -- see
    record_audit_independently.
    """
    await conn.execute(
        sa.insert(audit_events).values(
            tenant_id=tenant_id,
            actor=actor,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome=outcome,
            data=dict(data or {}),
        )
    )


async def record_audit_independently(
    engine,
    *,
    tenant_id: UUID,
    action: str,
    resource_type: str,
    outcome: str,
    resource_id: UUID | None = None,
    actor: str = "system",
    data: Mapping[str, Any] | None = None,
) -> None:
    """Append an audit record in its OWN transaction, committed immediately.

    This exists for one specific hazard. A tool denial happens mid-attempt.
    If the record shared the attempt's transaction and the worker then lost
    its lease, the completion CAS would fail, the transaction would roll back,
    and THE RECORD OF THE REFUSAL WOULD VANISH -- silently, exactly in the
    situation where an operator most needs it.

    So security-relevant records are never transactionally coupled to work
    that can be rolled back. The cost is that an audit record can exist for an
    attempt later fenced out, which is not a defect: the denial genuinely
    happened, on a real if doomed attempt, and the record carries the attempt
    number so the timeline stays honest about which one.
    """
    async with engine.connect() as conn, conn.begin():
        await record_audit(
            conn,
            tenant_id=tenant_id,
            action=action,
            resource_type=resource_type,
            outcome=outcome,
            resource_id=resource_id,
            actor=actor,
            data=data,
        )


async def list_audit(
    conn: AsyncConnection,
    *,
    tenant_id: UUID,
    action: str | None = None,
    outcome: str | None = None,
    limit: int = 100,
) -> list[Mapping[str, Any]]:
    stmt = sa.select(*audit_events.c).where(audit_events.c.tenant_id == tenant_id)
    if action is not None:
        stmt = stmt.where(audit_events.c.action == action)
    if outcome is not None:
        stmt = stmt.where(audit_events.c.outcome == outcome)
    rows = await conn.execute(stmt.order_by(audit_events.c.id.desc()).limit(limit))
    return [dict(r) for r in rows.mappings().all()]
