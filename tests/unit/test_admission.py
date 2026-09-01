"""Admission control policy. Pure -- four numbers in, a verdict out."""

from __future__ import annotations

from acp.scheduling.admission import Admission, decide


def test_admits_when_under_every_limit() -> None:
    verdict = decide(tenant_queued=5, tenant_max_queued=100)
    assert verdict.admitted
    assert verdict.decision is Admission.ADMIT


def test_rejects_a_tenant_at_its_own_backlog_bound() -> None:
    verdict = decide(tenant_queued=100, tenant_max_queued=100)
    assert not verdict.admitted
    assert verdict.decision is Admission.REJECT_TENANT_BACKLOG
    assert verdict.retry_after_s > 0


def test_tenant_rejection_is_independent_of_system_health() -> None:
    """429 is true regardless of how the rest of the system is doing.

    A tenant at its own bound is rejected whether the system is idle or
    globally overloaded -- the message is about THEM, not us.
    """
    idle = decide(
        tenant_queued=100, tenant_max_queued=100, global_queued=0, global_shed_threshold=10_000
    )
    busy = decide(
        tenant_queued=100,
        tenant_max_queued=100,
        global_queued=9_999,
        global_shed_threshold=10_000,
    )
    assert idle.decision is busy.decision is Admission.REJECT_TENANT_BACKLOG


def test_tenant_quota_is_checked_before_global_shedding() -> None:
    """Ordering is the whole point, not an implementation detail.

    If global overload were checked first, a tenant over its own quota during
    a system-wide slowdown would be told "we're struggling" -- hiding the
    fact that THEY are also over their own limit, and giving them the wrong
    thing to fix.
    """
    verdict = decide(
        tenant_queued=100,
        tenant_max_queued=100,
        global_queued=999_999,
        global_shed_threshold=10,
    )
    assert verdict.decision is Admission.REJECT_TENANT_BACKLOG


def test_sheds_when_the_system_is_globally_overloaded() -> None:
    """503, and only once the tenant's own quota has already cleared --
    which is what makes this a trustworthy 'us, not you' signal."""
    verdict = decide(
        tenant_queued=5, tenant_max_queued=100, global_queued=10_000, global_shed_threshold=10_000
    )
    assert verdict.decision is Admission.SHED_OVERLOADED
    assert verdict.retry_after_s > 0


def test_shedding_is_disabled_by_a_zero_threshold() -> None:
    """The right default: shedding should be switched on with a number derived
    from a measured drain rate, not left at a guess that fires on a normal
    burst."""
    verdict = decide(
        tenant_queued=5, tenant_max_queued=100, global_queued=999_999_999, global_shed_threshold=0
    )
    assert verdict.admitted


def test_429_and_503_carry_different_retry_after_values() -> None:
    """Distinguishable at the client, not just internally.

    A tenant's own backlog clears at a rate the tenant controls, so a short
    Retry-After is honest. System-wide overload takes longer to drain and
    every client retrying too soon is how a slowdown becomes an outage.
    """
    tenant_reject = decide(tenant_queued=100, tenant_max_queued=100)
    shed = decide(
        tenant_queued=5, tenant_max_queued=100, global_queued=10_000, global_shed_threshold=10_000
    )
    assert shed.retry_after_s > tenant_reject.retry_after_s


def test_admitted_verdict_has_no_reason_or_retry_hint() -> None:
    verdict = decide(tenant_queued=0, tenant_max_queued=100)
    assert verdict.reason == ""
    assert verdict.retry_after_s == 0
