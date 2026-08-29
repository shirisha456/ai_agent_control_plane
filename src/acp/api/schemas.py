"""Request/response contracts.

Validation happens at the boundary so that nothing downstream has to defend
against a negative priority or a 40MB payload. Every bound below exists
because violating it damages the system, not because a linter asked.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

# Task payloads are PARAMETERS AND REFERENCES, not blobs. `tasks` is the
# hottest table in the system: every claim, renewal and completion rewrites a
# row, and PostgreSQL rewrites the whole tuple. A megabyte payload turns every
# lease renewal into a megabyte of WAL. Large inputs belong in object storage
# with a reference in the payload.
MAX_PAYLOAD_BYTES = 64 * 1024


class TenantCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    max_concurrent_tasks: int = Field(default=10, ge=1, le=10_000)
    max_queued_tasks: int = Field(default=1000, ge=1, le=1_000_000)


class TenantOut(BaseModel):
    id: UUID
    name: str
    max_concurrent_tasks: int
    max_queued_tasks: int
    created_at: datetime


class TaskCreate(BaseModel):
    tenant_id: UUID
    task_type: str = Field(min_length=1, max_length=128)
    payload: dict[str, Any] = Field(default_factory=dict)

    # Optional. When present, submission is idempotent within the tenant.
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=255)

    # Lower is more urgent, matching nice(1) and the ready index's ORDER BY.
    priority: int = Field(default=100, ge=0, le=32_767)

    max_attempts: int = Field(default=3, ge=1, le=20)

    # Delayed submission. Shares the `available_at` column with retry backoff,
    # because "not runnable until T" is one concept, not two.
    available_at: datetime | None = None

    @field_validator("payload")
    @classmethod
    def _bounded_payload(cls, v: dict[str, Any]) -> dict[str, Any]:
        size = len(json.dumps(v).encode())
        if size > MAX_PAYLOAD_BYTES:
            raise ValueError(
                f"payload is {size} bytes; limit is {MAX_PAYLOAD_BYTES}. "
                "Store large inputs externally and pass a reference."
            )
        return v


class TaskOut(BaseModel):
    id: UUID
    tenant_id: UUID
    task_type: str
    payload: dict[str, Any]
    idempotency_key: str | None
    priority: int
    state: str
    attempt: int
    max_attempts: int
    available_at: datetime
    lease_worker_id: str | None
    lease_expires_at: datetime | None
    cancel_requested: bool
    result: dict[str, Any] | None
    error_class: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    first_started_at: datetime | None
    finished_at: datetime | None

    # Derived, never stored. A task waiting out its backoff is QUEUED with a
    # future available_at; materialising a RETRYING state would add a node to
    # the state machine that carries no information the two columns lack.
    is_retrying: bool = False


class TaskListOut(BaseModel):
    tasks: list[TaskOut]
    next_cursor: str | None = None


class TaskEventOut(BaseModel):
    id: int
    task_id: UUID
    attempt: int | None
    event_type: str
    worker_id: str | None
    data: dict[str, Any]
    created_at: datetime


class CancelOut(BaseModel):
    task_id: UUID
    state: str
    cancel_requested: bool
    detail: str
