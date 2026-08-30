"""Task submission and inspection.

The API's job in this system is narrow and important: validate, admit or
shed, and durably record. It never executes anything and holds no state, so
it can be replicated freely and killed without consequence -- a client whose
submission died mid-flight retries with the same idempotency key and the
database dedupes it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncConnection

from acp.api.deps import db_read, db_txn
from acp.api.errors import Conflict, NotFound
from acp.api.schemas import CancelOut, TaskCreate, TaskEventOut, TaskListOut, TaskOut
from acp.db.queries import agents as aq
from acp.db.queries import tasks as q
from acp.domain.states import State
from acp.obs import metrics

router = APIRouter(prefix="/v1/tasks", tags=["tasks"])

Txn = Annotated[AsyncConnection, Depends(db_txn)]
Read = Annotated[AsyncConnection, Depends(db_read)]


def _to_out(row) -> TaskOut:
    task = dict(row)
    # `is_retrying` is computed, never stored: a task waiting out its backoff
    # is QUEUED with a future available_at. Storing a RETRYING state would add
    # a node to the state machine carrying no information these two columns
    # do not already hold.
    now = datetime.now(tz=task["available_at"].tzinfo)
    task["is_retrying"] = (
        task["state"] == State.QUEUED and task["attempt"] > 0 and task["available_at"] > now
    )
    return TaskOut(**task)


@router.post("", status_code=status.HTTP_201_CREATED, response_model=TaskOut)
async def submit(body: TaskCreate, response: Response, conn: Txn) -> TaskOut:
    """Submit a task.

    201 on creation, 200 when an idempotency key deduplicated the request.
    The status code is the only signal the caller gets that their retry was
    absorbed rather than duplicated, so it must be accurate.
    """
    if await q.get_tenant(conn, body.tenant_id) is None:
        raise NotFound("unknown tenant", tenant_id=str(body.tenant_id))

    # Resolution runs in the SAME transaction as the insert below. That is
    # what closes the race where a version is deprecated between "which
    # version should this use?" and "create the task": both statements read
    # one snapshot, so a task can never be pinned to a version that was
    # already withdrawn when it was accepted.
    resolution = None
    task_type = body.task_type
    max_attempts = body.max_attempts
    if body.request_type is not None:
        try:
            resolution = await aq.resolve_route(
                conn, tenant_id=body.tenant_id, request_type=body.request_type
            )
        except aq.NoRoute as exc:
            raise NotFound(str(exc), request_type=body.request_type) from exc
        except aq.NotRoutable as exc:
            raise Conflict(str(exc), request_type=body.request_type) from exc
        task_type = resolution.runtime_spec.get("task_type", resolution.agent_name)
        # The version's retry budget wins over the request's: execution policy
        # travels with the immutable definition, so rolling a version back
        # rolls its limits back too.
        max_attempts = resolution.max_attempts

    try:
        result = await q.submit_task(
            conn,
            tenant_id=body.tenant_id,
            task_type=task_type,
            payload=body.payload,
            idempotency_key=body.idempotency_key,
            priority=body.priority,
            max_attempts=max_attempts,
            available_at=body.available_at,
            agent_version_id=resolution.version_id if resolution else None,
            required_capabilities=resolution.required_capabilities if resolution else (),
        )
    except q.PayloadMismatch as exc:
        raise Conflict(
            "idempotency_key was reused with a different payload",
            existing_task_id=str(exc),
        ) from exc
    except LookupError as exc:
        raise Conflict(str(exc), retry_after_s=1) from exc

    # `deduplicated` as a label rather than two metrics: the ratio of
    # deduplicated to created submissions is exactly the question worth asking
    # (how hard are clients retrying?), and a ratio is easier to read off one
    # metric than off two.
    metrics.tasks_submitted.labels(
        task_type=task_type, deduplicated=str(not result.created).lower()
    ).inc()

    if not result.created:
        response.status_code = status.HTTP_200_OK
    return _to_out(result.task)


@router.get("", response_model=TaskListOut)
async def list_tasks(
    conn: Read,
    tenant_id: UUID | None = None,
    state: State | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = None,
) -> TaskListOut:
    rows, next_cursor = await q.list_tasks(
        conn, tenant_id=tenant_id, state=state, limit=limit, cursor=cursor
    )
    return TaskListOut(tasks=[_to_out(r) for r in rows], next_cursor=next_cursor)


@router.get("/{task_id}", response_model=TaskOut)
async def get_task(task_id: UUID, conn: Read) -> TaskOut:
    row = await q.get_task(conn, task_id)
    if row is None:
        raise NotFound("unknown task", task_id=str(task_id))
    return _to_out(row)


@router.get("/{task_id}/events", response_model=list[TaskEventOut])
async def get_events(task_id: UUID, conn: Read) -> list[TaskEventOut]:
    """The execution timeline: what happened to this task, in order.

    This is the debugging surface that a broker-backed queue cannot offer,
    because there the task's history lives in a broker the application cannot
    query.
    """
    if await q.get_task(conn, task_id) is None:
        raise NotFound("unknown task", task_id=str(task_id))
    return [TaskEventOut(**e) for e in await q.get_task_events(conn, task_id)]


@router.post("/{task_id}/cancel", response_model=CancelOut)
async def cancel(task_id: UUID, response: Response, conn: Txn) -> CancelOut:
    """Cancel a task, or request cancellation if it is already running.

    202 rather than 200 for a running task, because the cancellation has been
    ACCEPTED, not COMPLETED -- the worker will honour it at its next lease
    renewal, and if the worker is already dead the reaper honours it when the
    lease expires. Returning 200 would claim an outcome the control plane
    cannot yet guarantee.
    """
    result = await q.cancel_task(conn, task_id)

    if result.outcome is q.CancelOutcome.NOT_FOUND:
        raise NotFound("unknown task", task_id=str(task_id))

    if result.outcome is q.CancelOutcome.ALREADY_TERMINAL:
        # The caller's intent was not achieved and never can be. Silently
        # returning 200 would hide a completed task from someone who believes
        # they stopped it.
        raise Conflict(
            "task already finished and cannot be cancelled",
            state=result.task["state"],
        )

    if result.outcome in (q.CancelOutcome.REQUESTED, q.CancelOutcome.ALREADY_REQUESTED):
        response.status_code = status.HTTP_202_ACCEPTED

    detail = {
        q.CancelOutcome.CANCELLED: "cancelled before execution started",
        q.CancelOutcome.REQUESTED: "task is running; worker will stop at its next lease renewal",
        q.CancelOutcome.ALREADY_REQUESTED: "cancellation already requested",
        q.CancelOutcome.ALREADY_CANCELLED: "task was already cancelled",
    }[result.outcome]

    return CancelOut(
        task_id=task_id,
        state=result.task["state"],
        cancel_requested=result.task["cancel_requested"],
        detail=detail,
    )
