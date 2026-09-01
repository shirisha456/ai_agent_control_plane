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

Tenant concurrency (tenants.max_concurrent_tasks) is enforced in two steps,
run in ROUNDS rather than one big over-fetch. Each round locks (SELECT ...
FOR UPDATE SKIP LOCKED) only as many candidates as are still needed to reach
`limit`, then reads each newly-seen tenant's remaining slack
(max_concurrent_tasks - currently RUNNING, itself locked -- see below) and
keeps the policy-ordered prefix that tenant has room for. A candidate
dropped for lack of slack stays locked for the rest of this transaction (Postgres has no
partial-release of a row lock before COMMIT) but is excluded from the next
round's SELECT, so a fresh round can still make progress instead of
re-examining it. Rounds are bounded (MAX_ROUNDS) so a long run of
at-their-cap tenants costs a few extra round-trips, not an unbounded scan.

Locking only what is needed per round, instead of locking a large fixed
over-fetch up front, matters most exactly when it looks like it wouldn't:
under a SHALLOW queue. A worker whose SELECT locks (say) 50 candidates
because the multiplier said so, but only claims 3, holds the other 47 locked
for its whole transaction -- if the queue only has 8 rows total, that is not
"a few wasted locks", it is the entire queue, and every other concurrently
claiming worker sees zero available candidates until the first worker
commits. A single-statement correlated subquery cannot compute slack
correctly either way: every candidate in one batch would see the SAME
pre-batch RUNNING count, so a tenant at 0/1 with two eligible tasks in the
batch would have BOTH pass the check. Slack has to be consumed as it is
spent, not read once per row.

CAPABILITY MATCHING: A KEYED MERGE, NOT A FILTER
-------------------------------------------------
Candidate selection used to be a single ordered scan of idx_tasks_ready with
`required_capabilities <@ ARRAY[worker_capabilities]` applied as a FILTER
after the fact. That degrades badly under one specific, measured load shape:
a specialist worker facing a deep queue of work it cannot run has to walk
past nearly the whole queue in priority order rejecting almost everything --
O(queue depth) instead of O(batch). Measured at ~47ms for a 2,000-row queue
in the version of this file that shipped with only the filter.

The fix uses `domain.agents.satisfiable_capability_keys`: the set of
`capability_key` values a worker with a given capability list can EVER
satisfy is exactly the canonical encoding of every SUBSET of its own
capabilities -- a pure function of the worker, needing no query. Candidate
selection then becomes a K-way merge: one Postgres LATERAL branch per
satisfiable key, each an independently ordered, locked, LIMIT-bounded index
range scan against idx_tasks_ready_by_capability (capability_key, priority,
available_at, id), unioned and re-sorted in Python over at most K * `need`
rows -- bounded by the worker's OWN capability count, never by queue depth.
A generalist worker (no declared capabilities) has exactly one satisfiable
key (`""`), so this reduces to a single ordered range scan identical in shape
to the old idx_tasks_ready scan for the common case.

Bounded by MAX_KEYED_CLAIM_CAPABILITIES: a worker declaring more distinct
capabilities than that falls back to the plain containment-filter query
instead of generating an impractically large number of LATERAL branches --
see domain.agents' docstring for why that bound is generous in practice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncConnection

from acp.db.models import task_attempts, task_events, tasks, tenants
from acp.db.sqlutil import seconds
from acp.domain.agents import MAX_KEYED_CLAIM_CAPABILITIES, satisfiable_capability_keys
from acp.domain.states import EventType, State
from acp.scheduling.policy import ClaimPolicy

#: One Postgres LATERAL branch per key, one ordered+locked+limited range scan
#: per branch, merged and re-sorted in Python. Written as raw SQL rather than
#: assembled from Core: the shape (UNNEST driving a LATERAL join, FOR UPDATE
#: SKIP LOCKED inside the lateral subquery) is exactly the kind of
#: correctness-critical, Postgres-specific construct this project keeps as
#: literal SQL so a reviewer can read precisely what runs, rather than trust
#: a query-builder's rendering of it.
_KEYED_CANDIDATES_SQL = """
    SELECT t.id, t.tenant_id, t.priority, t.available_at
      FROM unnest(CAST(:keys AS text[])) AS k(capability_key)
      CROSS JOIN LATERAL (
          SELECT id, tenant_id, priority, available_at
            FROM tasks
           WHERE state = 'QUEUED'
             AND available_at <= now()
             AND capability_key = k.capability_key
             AND NOT (id = ANY(CAST(:seen_ids AS uuid[])))
           ORDER BY priority ASC, available_at ASC, id ASC
           LIMIT :need
             FOR UPDATE SKIP LOCKED
      ) t
"""

#: How many consecutive tenant-blocked rounds one claim call absorbs before
#: giving up and leaving the rest for the next poll cycle. Not a correctness
#: requirement -- worst case it just costs another poll cycle.
MAX_ROUNDS = 4


async def _fetch_candidates(
    conn: AsyncConnection,
    *,
    worker_capabilities: Sequence[str],
    order_sql: str,
    need: int,
    seen_ids: set[UUID],
) -> list[Any]:
    """One round's worth of lockable candidates, in priority order.

    Two strategies, chosen by how many distinct capabilities the worker
    declares -- see the module docstring for why each is right in its regime.
    """
    normalised_caps = sorted({c.strip().lower() for c in worker_capabilities if c and c.strip()})

    if len(normalised_caps) <= MAX_KEYED_CLAIM_CAPABILITIES:
        keys = satisfiable_capability_keys(normalised_caps)
        stmt = sa.text(_KEYED_CANDIDATES_SQL).bindparams(
            sa.bindparam("keys", value=keys, type_=ARRAY(sa.Text)),
            sa.bindparam("seen_ids", value=list(seen_ids), type_=ARRAY(PG_UUID(as_uuid=True))),
            sa.bindparam("need", value=need),
        )
        rows = (await conn.execute(stmt)).all()
        # The merge is unordered ACROSS keys (each branch is independently
        # ordered, but Postgres does not guarantee an order across UNNEST
        # iterations) -- bounded by (number of satisfiable keys) * need, never
        # by queue depth, so sorting it here is cheap regardless of how deep
        # the underlying queue is.
        rows.sort(key=lambda r: (r.priority, r.available_at, r.id))
        return rows[:need]

    # Fallback for a worker declaring an unusually large capability set: the
    # plain containment filter, correct for any input, just not keyed-index
    # friendly. See MAX_KEYED_CLAIM_CAPABILITIES's docstring for why this path
    # is expected to be rare in practice.
    stmt = sa.select(tasks.c.id, tasks.c.tenant_id, tasks.c.priority, tasks.c.available_at).where(
        tasks.c.state == State.QUEUED.value,
        tasks.c.available_at <= sa.func.now(),
        tasks.c.required_capabilities.contained_by(
            sa.cast(sa.literal(list(worker_capabilities)), ARRAY(sa.Text))
        ),
    )
    if seen_ids:
        stmt = stmt.where(tasks.c.id.notin_(seen_ids))
    stmt = stmt.order_by(sa.text(order_sql)).limit(need).with_for_update(skip_locked=True)
    return (await conn.execute(stmt)).all()


async def claim_tasks(
    conn: AsyncConnection,
    *,
    worker_id: str,
    limit: int,
    lease_ttl_s: int,
    policy: ClaimPolicy,
    worker_capabilities: Sequence[str] = (),
) -> list[Mapping[str, Any]]:
    """Claim up to `limit` eligible tasks for `worker_id`, granting a lease on each.

    `conn` must already be inside a transaction; the SELECT ... FOR UPDATE SKIP
    LOCKED lock is only meaningful for the lifetime of that transaction.
    """
    order_sql = policy.order_by_sql()

    claimed_ids: list[UUID] = []
    seen_ids: set[UUID] = set()
    slack: dict[UUID, int] = {}
    slack_known: set[UUID] = set()

    for _ in range(MAX_ROUNDS):
        need = limit - len(claimed_ids)
        if need <= 0:
            break

        batch = await _fetch_candidates(
            conn,
            worker_capabilities=worker_capabilities,
            order_sql=order_sql,
            need=need,
            seen_ids=seen_ids,
        )
        if not batch:
            break
        seen_ids.update(row.id for row in batch)

        # FOR UPDATE on the tenant rows themselves: without it, two concurrent
        # claim_tasks transactions each read the RUNNING count before either
        # commits its UPDATE, both see the same slack, and both claim --
        # letting a tenant exceed max_concurrent_tasks. Locking the tenant row
        # makes the second transaction's SELECT block until the first commits
        # (releasing the lock) and its own RUNNING count reflects that claim.
        new_tenant_ids = {row.tenant_id for row in batch} - slack_known
        if new_tenant_ids:
            slack_rows = (
                await conn.execute(
                    sa.select(
                        tenants.c.id,
                        tenants.c.max_concurrent_tasks
                        - sa.select(sa.func.count())
                        .select_from(tasks)
                        .where(
                            tasks.c.tenant_id == tenants.c.id, tasks.c.state == State.RUNNING.value
                        )
                        .scalar_subquery(),
                    )
                    .where(tenants.c.id.in_(new_tenant_ids))
                    .with_for_update()
                )
            ).all()
            slack.update({row[0]: row[1] for row in slack_rows})
            slack_known.update(new_tenant_ids)

        for row in batch:
            if len(claimed_ids) >= limit:
                break
            remaining = slack.get(row.tenant_id, 0)
            if remaining <= 0:
                continue
            slack[row.tenant_id] = remaining - 1
            claimed_ids.append(row.id)

    if not claimed_ids:
        return []

    lease_expires_at = sa.func.now() + seconds(lease_ttl_s)
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
