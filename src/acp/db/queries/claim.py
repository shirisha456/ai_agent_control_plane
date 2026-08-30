"""The claim query: how a worker turns QUEUED tasks into RUNNING ones it owns.

This is the mechanism half of scheduling; acp.scheduling.policy is the policy
half. Splitting them means the concurrency argument below never has to be
re-reasoned about when priority ordering changes.

CONCURRENCY
-----------
`FOR UPDATE SKIP LOCKED` is what makes N workers claiming concurrently safe
and non-blocking: each worker's SELECT skips rows another worker's SELECT has
already locked, so no worker ever waits on another's claim, and no two workers
ever select the same row. The lease grant is a single UPDATE ... FROM the
locked id set, so the SELECT and the UPDATE that grants the lease happen in
one statement -- there is no window between "we picked this row" and "we own
this row" for another connection to slip into.

Tenant concurrency (tenants.max_concurrent_tasks) is enforced in two steps.
The SELECT ... FOR UPDATE SKIP LOCKED first locks a policy-ordered batch of
QUEUED candidates -- cheaply over-fetching, since locking a row we end up not
claiming just makes another worker's SELECT skip past it for the (short)
lifetime of this transaction. Then, still inside the same transaction, each
tenant's remaining slack (max_concurrent_tasks - currently RUNNING) is read
once and applied in Python, keeping only the policy-ordered prefix each
tenant has room for. A single-statement correlated subquery cannot do this
correctly: every candidate in one batch would see the SAME pre-batch RUNNING
count, so a tenant at 0/1 with two eligible tasks in the batch would have
BOTH pass the check. Slack has to be consumed as it is spent, not read once
per row.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from acp.db.models import task_attempts, task_events, tasks, tenants
from acp.domain.states import EventType, State
from acp.scheduling.policy import ClaimPolicy


async def claim_tasks(
    conn: AsyncConnection,
    *,
    worker_id: str,
    limit: int,
    lease_ttl_s: int,
    policy: ClaimPolicy,
) -> list[Mapping[str, Any]]:
    """Claim up to `limit` eligible tasks for `worker_id`, granting a lease on each.

    `conn` must already be inside a transaction; the SELECT ... FOR UPDATE SKIP
    LOCKED lock is only meaningful for the lifetime of that transaction.
    """
    order_sql = policy.order_by_sql()

    # Over-fetch: rows locked here but dropped for lacking tenant slack (below)
    # do not count toward `limit`, so fetching exactly `limit` would let a run
    # of at-their-cap tenants at the front of the sort order starve every
    # tenant behind them every single round. The multiplier is a bound on how
    # many consecutive blocked candidates one claim can absorb, not a
    # correctness requirement -- worst case it just costs another poll cycle.
    over_fetch = max(limit * 4, 50)
    locked = (
        sa.select(tasks.c.id, tasks.c.tenant_id)
        .where(tasks.c.state == State.QUEUED.value, tasks.c.available_at <= sa.func.now())
        .order_by(sa.text(order_sql))
        .limit(over_fetch)
        .with_for_update(skip_locked=True)
    )
    candidates = (await conn.execute(locked)).all()
    if not candidates:
        return []

    # FOR UPDATE on the tenant rows themselves: without it, two concurrent
    # claim_tasks transactions each read the RUNNING count before either
    # commits its UPDATE, both see the same slack, and both claim -- letting
    # a tenant exceed max_concurrent_tasks. Locking the tenant row makes the
    # second transaction's SELECT block until the first commits (releasing
    # the lock) and its own RUNNING count reflects that first claim.
    tenant_ids = {row.tenant_id for row in candidates}
    slack_rows = (
        await conn.execute(
            sa.select(
                tenants.c.id,
                tenants.c.max_concurrent_tasks
                - sa.select(sa.func.count())
                .select_from(tasks)
                .where(tasks.c.tenant_id == tenants.c.id, tasks.c.state == State.RUNNING.value)
                .scalar_subquery(),
            )
            .where(tenants.c.id.in_(tenant_ids))
            .with_for_update()
        )
    ).all()
    slack: dict[UUID, int] = {row[0]: row[1] for row in slack_rows}

    claimed_ids: list[UUID] = []
    for row in candidates:
        if len(claimed_ids) >= limit:
            break
        remaining = slack.get(row.tenant_id, 0)
        if remaining <= 0:
            continue
        slack[row.tenant_id] = remaining - 1
        claimed_ids.append(row.id)
    if not claimed_ids:
        return []

    lease_expires_at = sa.func.now() + sa.text(f"interval '{lease_ttl_s} seconds'")
    stmt = (
        sa.update(tasks)
        .where(tasks.c.id.in_(claimed_ids))
        .values(
            state=State.RUNNING.value,
            attempt=tasks.c.attempt + 1,
            lease_worker_id=worker_id,
            lease_expires_at=lease_expires_at,
            first_started_at=sa.func.coalesce(tasks.c.first_started_at, sa.func.now()),
            updated_at=sa.func.now(),
        )
        .returning(*tasks.c)
    )
    rows = (await conn.execute(stmt)).mappings().all()
    # UPDATE ... RETURNING carries no ordering guarantee, so re-impose the
    # policy order the candidate SELECT already established.
    by_id = {row["id"]: dict(row) for row in rows}
    rows = [by_id[task_id] for task_id in claimed_ids if task_id in by_id]

    if rows:
        await conn.execute(
            sa.insert(task_events),
            [
                {
                    "task_id": row["id"],
                    "attempt": row["attempt"],
                    "event_type": EventType.TASK_CLAIMED.value,
                    "worker_id": worker_id,
                    "data": {},
                }
                for row in rows
            ],
        )
        # One task_attempts row per attempt, created here rather than at
        # completion, so the (task_id, attempt) primary key does its job as a
        # duplicate-execution guard: two workers racing to claim the same
        # attempt number is exactly what this insert would catch.
        await conn.execute(
            sa.insert(task_attempts),
            [
                {"task_id": row["id"], "attempt": row["attempt"], "worker_id": worker_id}
                for row in rows
            ],
        )
    return [dict(r) for r in rows]
