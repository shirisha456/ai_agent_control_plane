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

from pydantic import BaseModel, Field, field_validator, model_validator

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
    """A submission, in one of two modes.

    DIRECT      task_type + payload. The primitive: the worker looks the type
                up in its adapter registry and runs it. No governance, no
                pinning, no record of which definition ran.

    AGENT-ROUTED
                request_type + payload. The control plane resolves
                request_type -> agent -> its released version, and PINS that
                version onto the task. This is what makes execution
                reproducible and what a tool-authorization decision will later
                hang off.

    Exactly one of the two must be given. Accepting both would leave the
    resolved agent and the declared task_type free to disagree, and nothing
    could say which one actually ran.
    """

    tenant_id: UUID
    task_type: str | None = Field(default=None, min_length=1, max_length=128)
    request_type: str | None = Field(default=None, min_length=1, max_length=128)
    payload: dict[str, Any] = Field(default_factory=dict)

    # Optional. When present, submission is idempotent within the tenant.
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=255)

    # Lower is more urgent, matching nice(1) and the ready index's ORDER BY.
    priority: int = Field(default=100, ge=0, le=32_767)

    max_attempts: int = Field(default=3, ge=1, le=20)

    # Hung-worker cap: the reaper force-fails (or retries) a RUNNING task
    # whose attempt has run longer than this, independent of lease
    # validity -- a worker that keeps renewing but is actually stuck
    # (an infinite loop, a hung call with no timeout) never trips lease
    # expiry, since its lease stays valid forever.
    max_execution_time_s: int = Field(default=300, ge=1, le=86_400)

    # Delayed submission. Shares the `available_at` column with retry backoff,
    # because "not runnable until T" is one concept, not two.
    available_at: datetime | None = None

    @model_validator(mode="after")
    def _exactly_one_submission_mode(self) -> TaskCreate:
        if bool(self.task_type) == bool(self.request_type):
            raise ValueError("provide exactly one of task_type or request_type")
        return self

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
    #: Set only for agent-routed submissions. NULL means "submitted directly",
    #: which is information rather than a missing value.
    agent_version_id: UUID | None = None
    required_capabilities: list[str] = Field(default_factory=list)
    payload: dict[str, Any]
    idempotency_key: str | None
    priority: int
    state: str
    attempt: int
    max_attempts: int
    max_execution_time_s: int
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
