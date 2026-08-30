"""The four ways a worker's execution of an attempt can end.

Each function pairs a `tasks` transition (via acp.db.queries.transitions,
fenced on attempt + worker_id so a worker that lost its lease mid-run cannot
overwrite whoever reclaimed it) with the matching update to that attempt's
task_attempts row. Both happen in the caller's transaction, so a completed
attempt and its outcome can never disagree.

`finish_attempt` is also used by acp.db.queries.reap -- a reaper recovery is
a fifth way an attempt ends, just not one a worker chose.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from acp.db.models import task_attempts
from acp.db.queries.transitions import TransitionResult, transition
from acp.domain.states import EventType, State


async def finish_attempt(
    conn: AsyncConnection,
    task_id: UUID,
    attempt: int,
    *,
    outcome: str,
    error_class: str | None = None,
    error_message: str | None = None,
) -> None:
    await conn.execute(
        sa.update(task_attempts)
        .where(task_attempts.c.task_id == task_id, task_attempts.c.attempt == attempt)
        .values(
            finished_at=sa.func.now(),
            outcome=outcome,
            error_class=error_class,
            error_message=error_message,
        )
    )


async def complete_success(
    conn: AsyncConnection,
    task_id: UUID,
    *,
    attempt: int,
    worker_id: str,
    result: Mapping[str, Any],
) -> TransitionResult:
    res = await transition(
        conn,
        task_id,
        expect_state=State.RUNNING,
        to_state=State.SUCCEEDED,
        event_type=EventType.TASK_SUCCEEDED,
        expect_attempt=attempt,
        expect_worker=worker_id,
        set_fields={
            "result": dict(result),
            "finished_at": sa.func.now(),
            "lease_worker_id": None,
            "lease_expires_at": None,
        },
    )
    if res.applied:
        await finish_attempt(conn, task_id, attempt, outcome="SUCCEEDED")
    return res


async def complete_retry(
    conn: AsyncConnection,
    task_id: UUID,
    *,
    attempt: int,
    worker_id: str,
    error_class: str,
    error_message: str,
    backoff_s: float,
) -> TransitionResult:
    res = await transition(
        conn,
        task_id,
        expect_state=State.RUNNING,
        to_state=State.QUEUED,
        event_type=EventType.RETRY_SCHEDULED,
        expect_attempt=attempt,
        expect_worker=worker_id,
        set_fields={
            "available_at": sa.func.now() + sa.text(f"interval '{backoff_s} seconds'"),
            "lease_worker_id": None,
            "lease_expires_at": None,
            "error_class": error_class,
            "error_message": error_message,
        },
        event_data={"error_class": error_class},
    )
    if res.applied:
        await finish_attempt(
            conn, task_id, attempt, outcome="FAILED", error_class=error_class, error_message=error_message
        )
    return res


async def complete_failed(
    conn: AsyncConnection,
    task_id: UUID,
    *,
    attempt: int,
    worker_id: str,
    error_class: str,
    error_message: str,
) -> TransitionResult:
    res = await transition(
        conn,
        task_id,
        expect_state=State.RUNNING,
        to_state=State.FAILED,
        event_type=EventType.TASK_FAILED,
        expect_attempt=attempt,
        expect_worker=worker_id,
        set_fields={
            "finished_at": sa.func.now(),
            "error_class": error_class,
            "error_message": error_message,
            "lease_worker_id": None,
            "lease_expires_at": None,
        },
    )
    if res.applied:
        await finish_attempt(
            conn, task_id, attempt, outcome="FAILED", error_class=error_class, error_message=error_message
        )
    return res


async def complete_cancelled(
    conn: AsyncConnection,
    task_id: UUID,
    *,
    attempt: int,
    worker_id: str,
) -> TransitionResult:
    res = await transition(
        conn,
        task_id,
        expect_state=State.RUNNING,
        to_state=State.CANCELLED,
        event_type=EventType.TASK_CANCELLED,
        expect_attempt=attempt,
        expect_worker=worker_id,
        set_fields={
            "finished_at": sa.func.now(),
            "lease_worker_id": None,
            "lease_expires_at": None,
        },
    )
    if res.applied:
        await finish_attempt(conn, task_id, attempt, outcome="CANCELLED")
    return res
