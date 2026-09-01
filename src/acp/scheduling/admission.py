"""Admission control: whether to accept a submission at all.

PURE MODULE. The decision is a function of four numbers, so every branch is
unit-testable without a database or an HTTP client.

FOUR MECHANISMS THAT GET CONFUSED
---------------------------------
They answer different questions and belong at different layers:

  RATE LIMITING        "too many REQUESTS per second from this tenant?"
                       Shapes ARRIVAL. Not implemented here -- see below.

  QUEUE BOUND          "is this tenant's BACKLOG growing without limit?"
  (backpressure)       Bounds STORAGE. -> 429, this module.

  CONCURRENCY LIMIT    "too many of this tenant's tasks RUNNING at once?"
                       Shapes EXECUTION. Already enforced at claim time
                       (db/queries/claim.py), NOT here -- which is why a busy
                       tenant degrades in latency rather than getting errors.

  LOAD SHEDDING        "is the WHOLE SYSTEM falling over?"
                       Sacrifices availability to preserve liveness. -> 503.

The distinction that matters, and the one worth being able to state:

    429 means "you, slow down."   503 means "us, we're in trouble."

Reporting a system-wide problem as the client's fault sends every client into
a retry pattern tuned for the wrong cause -- and hides the outage behind what
looks like a quota misconfiguration.

WHY TENANT QUOTA IS CHECKED FIRST
---------------------------------
If a tenant is over its own limit, that is true regardless of how the system
is doing, and it is the more actionable message. Checking global overload
first would tell a misbehaving tenant "we're struggling" when the honest
answer is "you're over quota". This ordering reserves 503 for "you did
nothing wrong" -- which is what makes it a trustworthy signal.

WHY RATE LIMITING IS ABSENT
---------------------------
A token bucket needs a counter updated thousands of times per second, shared
across API replicas. PostgreSQL is genuinely bad at that -- it is the one
place in this system where a row would be updated far more often than it is
read. This is the concrete trigger for introducing Redis, and it has not been
reached: nothing here needs it yet, and adding it early would mean a second
source of truth for no measured benefit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Admission(StrEnum):
    ADMIT = "admit"
    #: This tenant's queue is at its bound. 429.
    REJECT_TENANT_BACKLOG = "reject_tenant_backlog"
    #: The system as a whole is over its threshold. 503.
    SHED_OVERLOADED = "shed_overloaded"


@dataclass(frozen=True, slots=True)
class Verdict:
    decision: Admission
    reason: str = ""
    #: Seconds to put in Retry-After. Without it, well-behaved clients guess --
    #: and a fleet guessing in unison is a retry storm aimed at a system that
    #: just told them it was struggling.
    retry_after_s: int = 0

    @property
    def admitted(self) -> bool:
        return self.decision is Admission.ADMIT


def decide(
    *,
    tenant_queued: int,
    tenant_max_queued: int,
    global_queued: int = 0,
    global_shed_threshold: int = 0,
) -> Verdict:
    """Admit, push back on one tenant, or shed load system-wide.

    `tenant_queued` is measured exactly (an indexed count, see migration 0008)
    because it gates one tenant's correctness and must not overshoot by much.
    `global_queued` is deliberately approximate -- it comes from the gauge
    refresher's cached value -- because global overload is a slow-moving
    condition and counting the whole table on every submit would make the
    admission check itself part of the overload.

    A `global_shed_threshold` of 0 disables shedding, which is the right
    default: shedding should be switched on with a number someone chose from
    a measured drain rate, not left at a guess.
    """
    if tenant_queued >= tenant_max_queued:
        return Verdict(
            Admission.REJECT_TENANT_BACKLOG,
            reason=(
                f"tenant has {tenant_queued} queued tasks, at its limit of "
                f"{tenant_max_queued}; submissions are outpacing execution"
            ),
            # Short: this clears as the tenant's own backlog drains, and it is
            # the client's own throughput that decides how fast.
            retry_after_s=5,
        )

    if global_shed_threshold > 0 and global_queued >= global_shed_threshold:
        return Verdict(
            Admission.SHED_OVERLOADED,
            reason=(
                f"system queue depth {global_queued} is at or above the shed "
                f"threshold {global_shed_threshold}"
            ),
            # Longer: a system-wide backlog takes longer to drain than one
            # tenant's, and clients hammering during an overload is what turns
            # a slowdown into an outage.
            retry_after_s=30,
        )

    return Verdict(Admission.ADMIT)
