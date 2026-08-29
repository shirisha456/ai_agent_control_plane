"""The task state machine.

PURE MODULE. It must never import from acp.db, acp.api, or anything that does
I/O. That constraint is what lets the entire legality surface be tested in
milliseconds without a database, and it is enforced by tests/unit/test_purity.py.

Design notes (full rationale in docs/adr/0005 and docs/ARCHITECTURE.md):

  * Five states, not nine. `PENDING` and `SCHEDULED` collapse into QUEUED
    because workers pull -- there is no window between "scheduler decided"
    and "worker started" for a state to describe.

  * There is no RETRYING state. A task waiting out its backoff is QUEUED with
    `available_at` in the future; the claim query's `available_at <= now()`
    predicate excludes it for free. "Retrying" is a DERIVED label:
        state = QUEUED AND attempt > 0 AND available_at > now()
    Do not store what you can derive -- every stored state is N more illegal
    transitions to guard.

  * There is no CANCELLING state. You cannot yank a task out of a running
    remote process, so cancellation is a *request* (tasks.cancel_requested),
    orthogonal to execution state. The worker observes the flag at its next
    lease renewal and transitions to CANCELLED itself; if the worker is
    already dead, the reaper does it when the lease expires.

This module defines legality. It does NOT define concurrency safety -- that
lives in the SQL compare-and-set predicates in acp.db.queries.transitions.
Think of this as the linter and the CAS as the guarantee.
"""

from __future__ import annotations

from enum import StrEnum


class State(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL: frozenset[State] = frozenset({State.SUCCEEDED, State.FAILED, State.CANCELLED})
ACTIVE: frozenset[State] = frozenset({State.QUEUED, State.RUNNING})

#: The complete legal transition graph. Anything not listed is illegal.
#:
#:   QUEUED  -> RUNNING     claim (a worker took a lease)
#:   QUEUED  -> CANCELLED   cancel before anyone started it
#:   RUNNING -> SUCCEEDED   the attempt committed a result
#:   RUNNING -> QUEUED      retry after retryable failure, OR lease expiry
#:                          (recovery), OR graceful drain on SIGTERM.
#:                          One transition, three causes -- the cause is
#:                          recorded in task_attempts.outcome and task_events,
#:                          not in the state. One recovery path cannot
#:                          disagree with itself; two paths eventually would.
#:   RUNNING -> FAILED      permanent error, or attempts exhausted
#:   RUNNING -> CANCELLED   worker (or reaper) honoured cancel_requested
ALLOWED: dict[State, frozenset[State]] = {
    State.QUEUED: frozenset({State.RUNNING, State.CANCELLED}),
    State.RUNNING: frozenset({State.QUEUED, State.SUCCEEDED, State.FAILED, State.CANCELLED}),
    State.SUCCEEDED: frozenset(),
    State.FAILED: frozenset(),
    State.CANCELLED: frozenset(),
}


class IllegalTransition(Exception):
    """A transition the state machine forbids.

    This is a PROGRAMMER error and is raised, unlike losing a compare-and-set
    race, which is an expected outcome returned as `applied=False`. Conflating
    the two is how these systems grow bugs: callers wrap everything in
    try/except and then swallow genuine lost-ownership signals.
    """

    def __init__(self, frm: State, to: State) -> None:
        super().__init__(
            f"illegal transition {frm} -> {to}; legal targets are "
            f"{sorted(ALLOWED[frm]) or ['<terminal>']}"
        )
        self.frm = frm
        self.to = to


def is_legal(frm: State, to: State) -> bool:
    return to in ALLOWED[frm]


def assert_legal(frm: State, to: State) -> None:
    if not is_legal(frm, to):
        raise IllegalTransition(frm, to)


def is_terminal(state: State) -> bool:
    return state in TERMINAL


class EventType(StrEnum):
    """Append-only history written in the SAME transaction as the state change.

    Because the event insert shares a transaction with the UPDATE, the log can
    never disagree with `tasks.state`. That is also why this is NOT event
    sourcing: state is the source of truth and events are a transactionally
    consistent audit trail derived from it. Replaying events to rebuild state
    would cost us the CAS predicates the whole design rests on.
    """

    TASK_CREATED = "TASK_CREATED"
    TASK_CLAIMED = "TASK_CLAIMED"
    TASK_STARTED = "TASK_STARTED"
    ATTEMPT_FAILED = "ATTEMPT_FAILED"
    RETRY_SCHEDULED = "RETRY_SCHEDULED"
    LEASE_RENEWED = "LEASE_RENEWED"
    LEASE_EXPIRED = "LEASE_EXPIRED"
    TASK_RECOVERED = "TASK_RECOVERED"
    TASK_ABANDONED = "TASK_ABANDONED"
    WORKER_LOST = "WORKER_LOST"
    STALE_WRITE_REJECTED = "STALE_WRITE_REJECTED"
    TASK_SUCCEEDED = "TASK_SUCCEEDED"
    TASK_FAILED = "TASK_FAILED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    TASK_CANCELLED = "TASK_CANCELLED"
