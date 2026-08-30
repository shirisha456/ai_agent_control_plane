"""The authorization matrix. Pure -- no database, no tools, no worker."""

from __future__ import annotations

import uuid

import pytest

from acp.domain.authz import (
    UNGOVERNED,
    DenyReason,
    ToolPolicy,
    ToolRef,
    ToolStatus,
    ToolType,
    authorize,
)


def _tool(name: str, status: ToolStatus = ToolStatus.ACTIVE) -> ToolRef:
    return ToolRef(id=uuid.uuid4(), name=name, tool_type=ToolType.SIMULATED, status=status)


def _policy(*, granted=(), tools=(), version_disabled=False) -> ToolPolicy:
    return ToolPolicy(
        agent_version_id=uuid.uuid4(),
        granted_tool_ids=frozenset(t.id for t in granted),
        tools_by_name={t.name: t for t in tools},
        version_disabled=version_disabled,
    )


def test_a_granted_active_tool_is_allowed() -> None:
    web = _tool("web-search")
    decision = authorize(_policy(granted=[web], tools=[web]), "web-search")
    assert decision.allowed
    assert decision.tool == web


def test_an_ungranted_tool_is_refused() -> None:
    """The demo case: support-agent reaching for billing-db."""
    web, billing = _tool("web-search"), _tool("billing-db")
    decision = authorize(_policy(granted=[web], tools=[web, billing]), "billing-db")
    assert not decision.allowed
    assert decision.reason is DenyReason.NOT_GRANTED
    # The tool is still reported, so the audit record can name what was reached
    # for rather than just that something was.
    assert decision.tool == billing


def test_not_granted_is_distinguished_from_unknown_tool() -> None:
    """A distinction that exists for whoever reads the audit log.

    UNKNOWN_TOOL is a misconfiguration -- someone named a tool that does not
    exist. NOT_GRANTED is an agent reaching for something real it was never
    meant to touch. Collapsing them would hide the second inside the first.
    """
    billing = _tool("billing-db")
    policy = _policy(granted=[], tools=[billing])

    assert authorize(policy, "billing-db").reason is DenyReason.NOT_GRANTED
    assert authorize(policy, "typo-db").reason is DenyReason.UNKNOWN_TOOL


def test_disabling_a_tool_beats_an_existing_grant() -> None:
    """THE property that makes the design workable.

    Grants are versioned and immutable, so revoking one would need a new
    version -- far too slow for an incident. The kill switch is what makes
    that acceptable: static grants are immutable, DENIAL IS ALWAYS LIVE.
    Same shape as certificate revocation.
    """
    billing = _tool("billing-db", status=ToolStatus.DISABLED)
    decision = authorize(_policy(granted=[billing], tools=[billing]), "billing-db")

    assert not decision.allowed, "a disabled tool was usable because a grant existed"
    assert decision.reason is DenyReason.TOOL_DISABLED


def test_live_checks_are_evaluated_before_grants() -> None:
    """Ordering is load-bearing, not stylistic.

    If grants were checked first and returned early, disabling a tool would
    not stop an agent that already held a grant for it -- which is the entire
    point of having a kill switch.
    """
    disabled = _tool("billing-db", status=ToolStatus.DISABLED)
    policy = _policy(granted=[disabled], tools=[disabled])
    assert authorize(policy, "billing-db").reason is DenyReason.TOOL_DISABLED


def test_a_disabled_version_may_use_nothing() -> None:
    web = _tool("web-search")
    policy = _policy(granted=[web], tools=[web], version_disabled=True)
    assert authorize(policy, "web-search").reason is DenyReason.VERSION_DISABLED


def test_a_task_with_no_pinned_version_may_use_nothing() -> None:
    """ "No policy loaded" must never mean "everything permitted".

    A directly-submitted task has no governing definition, so there is no
    basis on which anything could have been granted to it. Defaulting to
    permissive here is how authorization gets bypassed by a refactor.
    """
    assert not authorize(UNGOVERNED, "web-search").allowed
    assert authorize(UNGOVERNED, "web-search").reason is DenyReason.NO_POLICY


@pytest.mark.parametrize(
    ("granted", "tool_status", "version_disabled", "expected"),
    [
        (True, ToolStatus.ACTIVE, False, None),
        (True, ToolStatus.DISABLED, False, DenyReason.TOOL_DISABLED),
        (False, ToolStatus.ACTIVE, False, DenyReason.NOT_GRANTED),
        (False, ToolStatus.DISABLED, False, DenyReason.TOOL_DISABLED),
        (True, ToolStatus.ACTIVE, True, DenyReason.VERSION_DISABLED),
        (False, ToolStatus.DISABLED, True, DenyReason.VERSION_DISABLED),
    ],
)
def test_the_full_matrix(granted, tool_status, version_disabled, expected) -> None:
    """Every combination, so no branch is decided by accident."""
    tool = _tool("t", status=tool_status)
    policy = _policy(
        granted=[tool] if granted else [], tools=[tool], version_disabled=version_disabled
    )
    decision = authorize(policy, "t")

    assert decision.allowed is (expected is None)
    assert decision.reason is expected


def test_decision_is_truthy_only_when_allowed() -> None:
    web = _tool("web-search")
    assert bool(authorize(_policy(granted=[web], tools=[web]), "web-search"))
    assert not bool(authorize(_policy(tools=[web]), "web-search"))
