"""The retry policy. Pure -- no database, no clock, no real randomness."""

from __future__ import annotations

import random

import pytest

from acp.domain.errors import (
    TERMINAL_CLASSES,
    DependencyUnavailable,
    FailureClass,
    InvalidInput,
    PermanentFailure,
    PermissionDenied,
    RateLimited,
    Retryable,
    UpstreamTimeout,
    classify,
    retry_after_of,
)
from acp.domain.retry import POLICIES, backoff_s, decide

SEEDED = lambda: random.Random(1234)  # noqa: E731 - deterministic per call


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (Retryable("x"), FailureClass.TRANSIENT),
        (RateLimited("x"), FailureClass.RATE_LIMITED),
        (UpstreamTimeout("x"), FailureClass.TIMEOUT),
        (DependencyUnavailable("x"), FailureClass.DEPENDENCY_UNAVAILABLE),
        (PermanentFailure("x"), FailureClass.PERMANENT),
        (InvalidInput("x"), FailureClass.USER_ERROR),
        (PermissionDenied("x"), FailureClass.PERMISSION_DENIED),
        # Well-known builtins an adapter might let escape untyped.
        (TimeoutError("x"), FailureClass.TIMEOUT),
        (ConnectionError("x"), FailureClass.DEPENDENCY_UNAVAILABLE),
        (ValueError("x"), FailureClass.USER_ERROR),
        (KeyError("x"), FailureClass.USER_ERROR),
    ],
)
def test_classification(exc: Exception, expected: FailureClass) -> None:
    assert classify(exc) is expected


def test_unrecognised_exceptions_are_unknown_not_guessed() -> None:
    """UNKNOWN is the honest answer, and it is load-bearing.

    Guessing PERMANENT would silently discard recoverable work; guessing
    TRANSIENT would retry genuine bugs to exhaustion. UNKNOWN carries its own
    conservative policy instead of pretending to knowledge we lack.
    """

    class VendorSpecificExplosion(Exception):
        pass

    assert classify(VendorSpecificExplosion()) is FailureClass.UNKNOWN


def test_every_failure_class_has_a_policy() -> None:
    """A class with no policy would KeyError inside the worker's finalize path."""
    assert set(POLICIES) == set(FailureClass)


def test_terminal_classes_are_exactly_the_non_retryable_ones() -> None:
    """The two ways of saying 'do not retry' must not drift apart."""
    non_retryable = {c for c, p in POLICIES.items() if not p.retryable}
    assert non_retryable == set(TERMINAL_CLASSES)


# ---------------------------------------------------------------------------
# the retry decision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("failure_class", sorted(TERMINAL_CLASSES))
def test_terminal_classes_never_retry(failure_class: FailureClass) -> None:
    """Even on attempt 1 of 10.

    Retrying a malformed payload or a policy denial is guaranteed waste: the
    input is identical, so the answer will be too. A denial retried three
    times also writes three audit records for one refusal.
    """
    decision = decide(failure_class, attempt=1, max_attempts=10, rng=SEEDED())
    assert not decision.should_retry
    assert decision.backoff_s == 0.0
    assert decision.reason.endswith("_not_retryable")


def test_retries_stop_at_max_attempts() -> None:
    assert decide(FailureClass.TRANSIENT, attempt=2, max_attempts=3, rng=SEEDED()).should_retry
    exhausted = decide(FailureClass.TRANSIENT, attempt=3, max_attempts=3, rng=SEEDED())
    assert not exhausted.should_retry
    assert exhausted.reason == "max_attempts_exhausted"


def test_unknown_failures_stop_earlier_than_the_task_budget_allows() -> None:
    """An unclassified error gets fewer attempts than a known-transient one.

    And the reason is distinct from max_attempts_exhausted on purpose: "we
    stopped early because we could not classify this" tells an operator to
    improve the adapter's error typing, which "budget spent" does not.
    """
    assert decide(FailureClass.UNKNOWN, attempt=1, max_attempts=10, rng=SEEDED()).should_retry
    capped = decide(FailureClass.UNKNOWN, attempt=2, max_attempts=10, rng=SEEDED())
    assert not capped.should_retry
    assert capped.reason == "unknown_attempt_cap_reached"

    # A classified transient failure with the same budget keeps going.
    assert decide(FailureClass.TRANSIENT, attempt=2, max_attempts=10, rng=SEEDED()).should_retry


# ---------------------------------------------------------------------------
# backoff shape
# ---------------------------------------------------------------------------


def test_backoff_is_bounded_by_the_doubling_ceiling() -> None:
    """Full jitter means uniform(0, ceiling) -- so the ceiling is the contract."""
    policy = POLICIES[FailureClass.TRANSIENT]
    rng = random.Random(7)
    for attempt in range(1, 12):
        ceiling = min(policy.cap_s, policy.base_s * 2 ** (attempt - 1))
        for _ in range(50):
            delay = backoff_s(FailureClass.TRANSIENT, attempt, rng=rng)
            assert 0.0 <= delay <= ceiling


def test_backoff_ceiling_actually_doubles() -> None:
    """Sampling the max over many draws should track the exponential curve."""
    rng = random.Random(11)
    peaks = [
        max(backoff_s(FailureClass.TRANSIENT, attempt, rng=rng) for _ in range(400))
        for attempt in (1, 2, 3, 4)
    ]
    assert peaks[0] < peaks[1] < peaks[2] < peaks[3]


def test_backoff_is_capped_so_late_attempts_do_not_schedule_next_week() -> None:
    rng = random.Random(3)
    cap = POLICIES[FailureClass.TRANSIENT].cap_s
    assert all(backoff_s(FailureClass.TRANSIENT, 40, rng=rng) <= cap for _ in range(200))


def test_full_jitter_actually_spreads_the_herd() -> None:
    """The property that matters: a thousand simultaneous failures must not
    retry simultaneously.

    Deterministic backoff would give one value a thousand times, recreating
    against the recovering dependency the exact thundering herd that took it
    down. This asserts the spread is wide, not that the values differ.
    """
    rng = random.Random(99)
    delays = [backoff_s(FailureClass.TRANSIENT, 4, rng=rng) for _ in range(1000)]
    ceiling = min(POLICIES[FailureClass.TRANSIENT].cap_s, 1.0 * 2**3)

    assert min(delays) < ceiling * 0.05, "no retries landed early -- not full jitter"
    assert max(delays) > ceiling * 0.95, "no retries landed late -- not full jitter"
    # Buckets across the interval should all be occupied.
    buckets = {int(d / (ceiling / 8)) for d in delays}
    assert len(buckets) >= 7, f"herd clustered into {len(buckets)} of 8 buckets"


def test_rate_limited_backs_off_harder_than_a_generic_transient_failure() -> None:
    """Retrying a throttle aggressively is how a rate limit becomes an outage."""
    assert POLICIES[FailureClass.RATE_LIMITED].base_s > POLICIES[FailureClass.TRANSIENT].base_s
    assert POLICIES[FailureClass.RATE_LIMITED].cap_s > POLICIES[FailureClass.TRANSIENT].cap_s


def test_retry_after_is_a_floor_not_a_replacement() -> None:
    """Never retry before the server said -- but still jitter after it.

    Obeying Retry-After exactly would re-synchronise every throttled caller
    onto one instant, rebuilding the herd the backoff exists to prevent.
    """
    rng = random.Random(5)
    delays = [
        backoff_s(FailureClass.RATE_LIMITED, 1, retry_after_s=30.0, rng=rng) for _ in range(200)
    ]
    assert all(d >= 30.0 for d in delays), "retried sooner than the server asked"
    assert len(set(delays)) > 100, "every caller would return at the same instant"


def test_worker_lost_retries_immediately() -> None:
    """The task never misbehaved -- its worker did.

    Backing off here punishes the workload for the infrastructure's problem,
    and would make crash recovery slower than the lease timeout that caused
    it. Consistent with the reaper, which requeues with available_at = now().
    """
    rng = random.Random(2)
    for attempt in (1, 2, 5, 20):
        assert backoff_s(FailureClass.WORKER_LOST, attempt, rng=rng) == 0.0

    decision = decide(FailureClass.WORKER_LOST, attempt=1, max_attempts=3, rng=rng)
    assert decision.should_retry
    assert decision.backoff_s == 0.0


def test_decisions_are_reproducible_for_a_seeded_rng() -> None:
    """Determinism is why the RNG is an argument rather than module state.

    Without it, no test could assert an exact backoff schedule and benchmark
    runs would not replay.
    """
    a = decide(FailureClass.TRANSIENT, attempt=3, max_attempts=5, rng=random.Random(42))
    b = decide(FailureClass.TRANSIENT, attempt=3, max_attempts=5, rng=random.Random(42))
    assert a == b


def test_retry_after_extraction() -> None:
    assert retry_after_of(RateLimited("x", retry_after_s=12.5)) == 12.5
    assert retry_after_of(RateLimited("x")) is None
    assert retry_after_of(ValueError("x")) is None
    # A negative Retry-After is nonsense; treat it as absent rather than
    # scheduling a retry in the past.
    assert retry_after_of(RateLimited("x", retry_after_s=-5)) is None
