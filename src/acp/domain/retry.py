"""Retry policy: whether to try again, and when.

PURE MODULE. No I/O, no clock, no global RNG -- `decide()` takes the random
source as an argument, so every test in tests/unit/test_retry.py is
deterministic and the whole policy is verifiable without a database or a
worker.

THE WORKER NEVER SLEEPS FOR A BACKOFF
-------------------------------------
`decide()` returns a delay, and the worker writes it to `tasks.available_at`
and immediately claims something else. The delay lives in the database, not
in a process. A worker that slept through its backoff would hold a slot doing
nothing, so one task failing repeatedly would eat capacity proportional to its
own backoff curve -- and the delay would evaporate if the worker crashed.

WHY FULL JITTER
---------------
Backoff is `uniform(0, min(cap, base * 2**(attempt-1)))`, not
`base * 2**(attempt-1)` with a small random nudge.

The problem being solved is correlation, not fairness. When a dependency goes
down, a thousand tasks fail within the same second and, with deterministic
backoff, retry within the same second -- hitting the recovering dependency
with the identical thundering herd that is still down, then doing it again
1s, 2s, 4s later. Full jitter spreads that herd across the whole interval.
It retries sooner on average than deterministic backoff AND spreads better;
AWS's published comparison of jitter strategies is the standard reference.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from acp.domain.errors import FailureClass


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How one failure class is retried."""

    retryable: bool
    #: First-retry delay, before jitter and doubling.
    base_s: float = 1.0
    #: Ceiling on the pre-jitter interval. Without one, exponential growth
    #: means attempt 20 schedules a retry for next week.
    cap_s: float = 60.0
    #: Hard stop regardless of the task's own max_attempts. Used where the
    #: failure class itself implies we should not spend a full budget --
    #: notably UNKNOWN, which is as likely to be a bug as a blip.
    attempt_cap: int | None = None


POLICIES: dict[FailureClass, RetryPolicy] = {
    FailureClass.TRANSIENT: RetryPolicy(True, base_s=1.0, cap_s=60.0),
    # Slower and with a higher ceiling: retrying a throttle aggressively is how
    # a rate limit becomes an outage. Honours Retry-After when the server sends
    # one (see decide()).
    FailureClass.RATE_LIMITED: RetryPolicy(True, base_s=5.0, cap_s=300.0),
    FailureClass.TIMEOUT: RetryPolicy(True, base_s=2.0, cap_s=120.0),
    # Long cap: a struggling dependency recovers faster if we stop hitting it.
    FailureClass.DEPENDENCY_UNAVAILABLE: RetryPolicy(True, base_s=5.0, cap_s=300.0),
    # No backoff at all. The attempt was taken away, not failed -- the task
    # never misbehaved, so delaying it punishes the workload for the
    # infrastructure's problem. The reaper already requeues with
    # available_at = now(); this keeps the policy consistent with that.
    FailureClass.WORKER_LOST: RetryPolicy(True, base_s=0.0, cap_s=0.0),
    FailureClass.PERMANENT: RetryPolicy(False),
    FailureClass.USER_ERROR: RetryPolicy(False),
    FailureClass.PERMISSION_DENIED: RetryPolicy(False),
    # Retryable, but on a shorter leash: an unclassified error is as likely to
    # be a bug that will fail identically every time as it is to be a blip.
    FailureClass.UNKNOWN: RetryPolicy(True, base_s=2.0, cap_s=60.0, attempt_cap=2),
}


@dataclass(frozen=True, slots=True)
class RetryDecision:
    should_retry: bool
    backoff_s: float
    #: Machine-readable reason, for the RETRY_SCHEDULED / TASK_FAILED event.
    reason: str


def backoff_s(
    failure_class: FailureClass,
    attempt: int,
    *,
    retry_after_s: float | None = None,
    rng: random.Random | None = None,
) -> float:
    """Full-jitter exponential backoff for one failure class.

    `attempt` is the fencing token, already incremented at claim time, so the
    first execution is attempt 1 and its retry uses the un-doubled base.

    A server-supplied `retry_after_s` is treated as a FLOOR, not a
    replacement: we never retry sooner than the server asked, but we still add
    jitter on top. Obeying Retry-After exactly would re-synchronise every
    throttled caller onto the same instant -- reconstructing the thundering
    herd the backoff exists to prevent.
    """
    policy = POLICIES[failure_class]
    rng = rng or random.Random()

    ceiling = min(policy.cap_s, policy.base_s * (2 ** max(0, attempt - 1)))
    jittered = rng.uniform(0.0, ceiling) if ceiling > 0 else 0.0

    if retry_after_s is not None:
        return retry_after_s + rng.uniform(0.0, max(1.0, policy.base_s))
    return jittered


def decide(
    failure_class: FailureClass,
    *,
    attempt: int,
    max_attempts: int,
    retry_after_s: float | None = None,
    rng: random.Random | None = None,
) -> RetryDecision:
    """Whether this attempt earns another one, and after how long."""
    policy = POLICIES[failure_class]

    if not policy.retryable:
        return RetryDecision(False, 0.0, f"{failure_class.value.lower()}_not_retryable")

    if attempt >= max_attempts:
        return RetryDecision(False, 0.0, "max_attempts_exhausted")

    if policy.attempt_cap is not None and attempt >= policy.attempt_cap:
        # Distinct reason from max_attempts_exhausted on purpose: "we stopped
        # early because we could not classify this" is a different operational
        # signal from "this task used its whole budget", and the first one
        # means someone should improve the adapter's error typing.
        return RetryDecision(False, 0.0, f"{failure_class.value.lower()}_attempt_cap_reached")

    delay = backoff_s(failure_class, attempt, retry_after_s=retry_after_s, rng=rng)
    return RetryDecision(True, delay, f"retry_after_{failure_class.value.lower()}")
