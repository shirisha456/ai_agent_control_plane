"""Task read/write queries backing the Control API.

Everything that mutates state routes through acp.db.queries.transitions so
there is exactly one place where `tasks.state` changes. The functions here add
the submission and query paths around it.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncConnection

from acp.db.models import task_events, tasks, tenants
from acp.db.queries.transitions import record_event, transition
from acp.domain.agents import capability_key
from acp.domain.states import TERMINAL, EventType, State


@dataclass(frozen=True, slots=True)
class SubmitResult:
    created: bool
    task: Mapping[str, Any]


class PayloadMismatch(Exception):
    """An idempotency key was reused with a different payload."""


async def submit_task(
    conn: AsyncConnection,
    *,
    tenant_id: UUID,
    task_type: str,
    payload: dict[str, Any],
    idempotency_key: str | None = None,
    priority: int = 100,
    max_attempts: int = 3,
    available_at: datetime | None = None,
    agent_version_id: UUID | None = None,
    required_capabilities: Sequence[str] = (),
) -> SubmitResult:
    """Insert a task, deduplicating on (tenant_id, idempotency_key).

    Dedup is `ON CONFLICT DO NOTHING` against the partial unique index, NOT a
    read-then-write. Checking "does this key exist?" before inserting is a
    textbook race: two API replicas both read absent, both insert, and the
    tenant's task executes twice.

    Concurrency: PostgreSQL's ON CONFLICT takes a speculative-insertion lock,
    so a losing writer blocks until the winner commits or aborts. If the
    winner committed, the follow-up SELECT (a fresh statement snapshot under
    READ COMMITTED) sees the row. If the winner aborted, our insert proceeds
    normally. Either way exactly one task exists when both calls return.
    """
    values: dict[str, Any] = {
        "tenant_id": tenant_id,
        "task_type": task_type,
        "payload": payload,
        "idempotency_key": idempotency_key,
        "priority": priority,
        "max_attempts": max_attempts,
        "state": State.QUEUED.value,
        # PINNED at submit, never re-resolved. A task that sat in the queue
        # for an hour still runs the version that was current when it was
        # accepted -- which is what makes "what executed task 123?" answerable
        # and keeps a retry from running different code than its first attempt.
        "agent_version_id": agent_version_id,
        # Copied from the (immutable) version rather than joined at claim
        # time. Safe because the source cannot drift; valuable because it
        # keeps the hottest query in the system free of registry joins.
        "required_capabilities": list(required_capabilities),
        "capability_key": capability_key(required_capabilities),
    }
    if available_at is not None:
        values["available_at"] = available_at

    stmt = (
        pg_insert(tasks)
        .values(**values)
        .on_conflict_do_nothing(
            index_elements=[tasks.c.tenant_id, tasks.c.idempotency_key],
            index_where=tasks.c.idempotency_key.isnot(None),
        )
        .returning(*tasks.c)
    )
    row = (await conn.execute(stmt)).mappings().first()

    if row is not None:
        await record_event(
            conn,
            row["id"],
            EventType.TASK_CREATED,
            attempt=0,
            data={"task_type": task_type, "priority": priority},
        )
        return SubmitResult(created=True, task=dict(row))

    existing = (
        (
            await conn.execute(
                sa.select(*tasks.c).where(
                    tasks.c.tenant_id == tenant_id,
                    tasks.c.idempotency_key == idempotency_key,
                )
            )
        )
        .mappings()
        .first()
    )
    if existing is None:
        # The conflicting inserter rolled back after we observed the conflict.
        # Rare, transient, and genuinely retryable -- surfaced as 409 with a
        # retry hint rather than silently returning a task that never existed.
        raise LookupError("insert conflicted but no row is visible; retry the submission")

    if existing["payload"] != payload:
        # Reusing a key with different parameters is a client bug. Returning
        # the original task with a 200 would tell the caller their NEW request
        # was accepted, which is a lie with side effects.
        raise PayloadMismatch(str(existing["id"]))

    return SubmitResult(created=False, task=dict(existing))


async def get_task(conn: AsyncConnection, task_id: UUID) -> Mapping[str, Any] | None:
    row = (await conn.execute(sa.select(*tasks.c).where(tasks.c.id == task_id))).mappings().first()
    return dict(row) if row else None


def _encode_cursor(created_at: datetime, task_id: UUID) -> str:
    return base64.urlsafe_b64encode(f"{created_at.isoformat()}|{task_id}".encode()).decode()


def _decode_cursor(cursor: str) -> tuple[datetime, UUID]:
    raw = base64.urlsafe_b64decode(cursor.encode()).decode()
    ts, tid = raw.split("|", 1)
    return datetime.fromisoformat(ts), UUID(tid)


async def list_tasks(
    conn: AsyncConnection,
    *,
    tenant_id: UUID | None = None,
    state: State | None = None,
    limit: int = 50,
    cursor: str | None = None,
) -> tuple[list[Mapping[str, Any]], str | None]:
    """Keyset pagination on (created_at DESC, id DESC).

    Not OFFSET. Offset pagination re-scans and discards every skipped row, so
    page 500 costs 500 pages of work -- on a table that grows without bound,
    and while the claim path is competing for the same buffers. Keyset seeks
    directly and its cost is independent of page number.
    """
    stmt = sa.select(*tasks.c)
    if tenant_id is not None:
        stmt = stmt.where(tasks.c.tenant_id == tenant_id)
    if state is not None:
        stmt = stmt.where(tasks.c.state == state.value)
    if cursor:
        created_at, task_id = _decode_cursor(cursor)
        stmt = stmt.where(
            sa.tuple_(tasks.c.created_at, tasks.c.id) < sa.tuple_(created_at, task_id)
        )

    rows = [
        dict(r)
        for r in (
            await conn.execute(
                stmt.order_by(tasks.c.created_at.desc(), tasks.c.id.desc()).limit(limit + 1)
            )
        )
        .mappings()
        .all()
    ]

    next_cursor = None
    if len(rows) > limit:
        rows = rows[:limit]
        next_cursor = _encode_cursor(rows[-1]["created_at"], rows[-1]["id"])
    return rows, next_cursor


async def get_task_events(
    conn: AsyncConnection, task_id: UUID, *, limit: int = 200
) -> list[Mapping[str, Any]]:
    """Ordered by id, not created_at.

    Several events written inside one transaction share a created_at, because
    now() is transaction-start time. Ordering by the BIGSERIAL keeps the
    timeline deterministic.
    """
    rows = (
        await conn.execute(
            sa.select(*task_events.c)
            .where(task_events.c.task_id == task_id)
            .order_by(task_events.c.id)
            .limit(limit)
        )
    ).mappings()
    return [dict(r) for r in rows]


class CancelOutcome(StrEnum):
    NOT_FOUND = "not_found"
    CANCELLED = "cancelled"  # was QUEUED, terminated immediately
    REQUESTED = "requested"  # was RUNNING, worker will observe the flag
    ALREADY_REQUESTED = "already_requested"
    ALREADY_CANCELLED = "already_cancelled"
    ALREADY_TERMINAL = "already_terminal"


@dataclass(frozen=True, slots=True)
class CancelResult:
    outcome: CancelOutcome
    task: Mapping[str, Any] | None = None


async def cancel_task(conn: AsyncConnection, task_id: UUID) -> CancelResult:
    """Cancel, or request cancellation of, one task.

    Cancellation is not one operation but two, because a QUEUED task can be
    terminated by the control plane while a RUNNING task can only be ASKED to
    stop -- you cannot yank work out of a remote process. Modelling the
    request as a flag rather than a CANCELLING state keeps that truth in the
    data instead of hiding it in the state machine.

    Concurrency: this takes a blocking row lock (FOR UPDATE) so the read and
    the branch are atomic -- without it, a task could be claimed between
    "observe QUEUED" and "transition to CANCELLED", and the CAS would fail for
    a reason the caller could not distinguish from a missing task. A blocking
    lock is affordable here for the reasons it is NOT affordable on the claim
    path: one row, user-initiated, and rare.
    """
    row = (
        (
            await conn.execute(
                sa.select(tasks.c.id, tasks.c.state, tasks.c.attempt, tasks.c.cancel_requested)
                .where(tasks.c.id == task_id)
                .with_for_update()
            )
        )
        .mappings()
        .first()
    )
    if row is None:
        return CancelResult(CancelOutcome.NOT_FOUND)

    state = State(row["state"])

    if state is State.CANCELLED:
        return CancelResult(CancelOutcome.ALREADY_CANCELLED, await get_task(conn, task_id))

    if state in TERMINAL:
        return CancelResult(CancelOutcome.ALREADY_TERMINAL, await get_task(conn, task_id))

    if state is State.QUEUED:
        res = await transition(
            conn,
            task_id,
            expect_state=State.QUEUED,
            to_state=State.CANCELLED,
            event_type=EventType.TASK_CANCELLED,
            set_fields={"cancel_requested": True, "finished_at": sa.func.now()},
            event_data={"reason": "cancelled_before_start"},
        )
        # The row lock above means this cannot lose the race.
        assert res.applied, f"cancel lost a race it holds the row lock for: {res.rejection}"
        return CancelResult(CancelOutcome.CANCELLED, res.task)

    # RUNNING: cooperative. The worker sees the flag on its next lease renewal
    # (which is already querying this row, so delivery costs no extra query).
    # If the worker is already dead, the reaper honours the flag when the
    # lease expires -- so cancellation works even against a crashed owner.
    if row["cancel_requested"]:
        return CancelResult(CancelOutcome.ALREADY_REQUESTED, await get_task(conn, task_id))

    await conn.execute(
        sa.update(tasks)
        .where(tasks.c.id == task_id)
        .values(cancel_requested=True, updated_at=sa.func.now())
    )
    await record_event(
        conn, task_id, EventType.CANCEL_REQUESTED, attempt=row["attempt"], data={"source": "api"}
    )
    return CancelResult(CancelOutcome.REQUESTED, await get_task(conn, task_id))


async def create_tenant(
    conn: AsyncConnection, *, name: str, max_concurrent_tasks: int, max_queued_tasks: int
) -> Mapping[str, Any]:
    row = (
        (
            await conn.execute(
                sa.insert(tenants)
                .values(
                    name=name,
                    max_concurrent_tasks=max_concurrent_tasks,
                    max_queued_tasks=max_queued_tasks,
                )
                .returning(*tenants.c)
            )
        )
        .mappings()
        .one()
    )
    return dict(row)


async def get_tenant(conn: AsyncConnection, tenant_id: UUID) -> Mapping[str, Any] | None:
    row = (
        (await conn.execute(sa.select(*tenants.c).where(tenants.c.id == tenant_id)))
        .mappings()
        .first()
    )
    return dict(row) if row else None


async def cancel_agent_tasks(
    conn: AsyncConnection, *, agent_version_ids: Sequence[UUID], reason: str
) -> int:
    """Stop an agent's live work by requesting cancellation on it.

    Reuses the cancellation mechanism instead of teaching the claim query to
    check agent status. The tempting design -- `AND agents.status = 'ACTIVE'`
    in the claim predicate -- puts a join against the registry on the hottest
    query in the system to serve an admin action that happens once a month.

    QUEUED tasks are cancelled outright. RUNNING tasks get the flag and stop
    at their worker's next lease renewal; if that worker is already dead, the
    reaper honours the flag when the lease expires. So this works against a
    crashed owner too, for free, because cancellation already had to.
    """
    if not agent_version_ids:
        return 0

    queued = await conn.execute(
        sa.update(tasks)
        .where(
            tasks.c.agent_version_id.in_(agent_version_ids),
            tasks.c.state == State.QUEUED.value,
        )
        .values(
            state=State.CANCELLED.value,
            cancel_requested=True,
            finished_at=sa.func.now(),
            updated_at=sa.func.now(),
            error_class="AgentDisabled",
            error_message=reason,
        )
        .returning(tasks.c.id, tasks.c.attempt)
    )
    queued_rows = queued.mappings().all()

    running = await conn.execute(
        sa.update(tasks)
        .where(
            tasks.c.agent_version_id.in_(agent_version_ids),
            tasks.c.state == State.RUNNING.value,
            tasks.c.cancel_requested.is_(False),
        )
        .values(cancel_requested=True, updated_at=sa.func.now())
        .returning(tasks.c.id, tasks.c.attempt)
    )
    running_rows = running.mappings().all()

    for row in queued_rows:
        await record_event(
            conn,
            row["id"],
            EventType.TASK_CANCELLED,
            attempt=row["attempt"],
            data={"reason": reason},
        )
    for row in running_rows:
        await record_event(
            conn,
            row["id"],
            EventType.CANCEL_REQUESTED,
            attempt=row["attempt"],
            data={"reason": reason, "source": "agent_disabled"},
        )
    return len(queued_rows) + len(running_rows)


async def queued_count_for_tenant(conn: AsyncConnection, tenant_id: UUID) -> int:
    """How many tasks this tenant has waiting.

    Counts every QUEUED row, including those waiting out a retry backoff --
    deliberately, unlike queue_depth in claim.py which counts only runnable
    work. Backpressure is about STORAGE: a tenant whose backlog is entirely
    deferred retries is still accumulating rows without bound, and admitting
    more would be admitting to a queue that is already too long.

    Served by idx_tasks_tenant_queued (migration 0008), so this is an
    index-only scan over the tenant's backlog rather than the table.
    """
    return (
        await conn.execute(
            sa.select(sa.func.count())
            .select_from(tasks)
            .where(tasks.c.tenant_id == tenant_id, tasks.c.state == State.QUEUED.value)
        )
    ).scalar_one()
