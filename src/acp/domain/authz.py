"""Agent-to-tool authorization.

PURE MODULE. The decision is a function of a snapshot and a tool name -- no
I/O, no clock -- so the entire authorization matrix is unit-testable and the
worker's hot path never queries anything to make it.

WHERE GRANTS LIVE, AND WHY IT IS THE VERSION
--------------------------------------------
A grant attaches to an AGENT VERSION, not to an agent. That makes a version a
complete, self-describing capability bundle: "what was this allowed to do?"
is answered by the same immutable row that answers "what did it run?", rather
than by reconstructing history from an audit log.

It also means widening an agent's reach requires cutting and releasing a new
version -- a reviewable diff -- instead of an INSERT that silently grants a
running agent access to a new system.

The cost is real and worth stating: revoking a grant would need a new version,
which is far too slow for an incident. So revocation does not go through
grants at all:

    STATIC GRANTS ARE VERSIONED AND IMMUTABLE.
    DENIAL IS ALWAYS LIVE.

`tools.status = 'DISABLED'` denies every use of a tool immediately, whatever
any grant says, and a DISABLED version denies everything it holds. Exactly the
shape of certificate revocation: the certificate is immutable, but validity is
checked at use.

WHEN THE SNAPSHOT IS TAKEN
--------------------------
At CLAIM time, inside the claim transaction (see db/queries/tools.py). The
tool-call path then does a set lookup in memory: no query, no cache, and no
staleness window to reason about. Revocation latency is therefore bounded by
attempt duration -- tasks already executing finish under the policy they were
claimed with -- which is a documented, explainable SLA rather than "we cache
for 5 seconds and hope".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID


class ToolStatus(StrEnum):
    ACTIVE = "ACTIVE"
    #: The live kill switch. Overrides every grant, immediately.
    DISABLED = "DISABLED"


class ToolType(StrEnum):
    """Deliberately two. Integrations teach nothing this project is about."""

    SIMULATED = "SIMULATED"
    HTTP = "HTTP"


class DenyReason(StrEnum):
    UNKNOWN_TOOL = "unknown_tool"
    NOT_GRANTED = "not_granted"
    TOOL_DISABLED = "tool_disabled"
    VERSION_DISABLED = "version_disabled"
    NO_POLICY = "no_policy"


@dataclass(frozen=True, slots=True)
class ToolRef:
    """One registered tool, as the executor sees it."""

    id: UUID
    name: str
    tool_type: ToolType
    status: ToolStatus
    #: Endpoint and timeouts. Holds a REFERENCE to a secret, never a secret --
    #: see migration 0007.
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """What one agent version may do, frozen at claim time.

    Carries the tenant's whole tool namespace rather than only granted tools,
    so a request for a tool that exists but was not granted can be reported as
    NOT_GRANTED instead of UNKNOWN_TOOL. The difference matters to whoever
    reads the audit log: one is a misconfiguration, the other is an agent
    reaching for something it was never meant to touch.
    """

    agent_version_id: UUID | None
    granted_tool_ids: frozenset[UUID] = frozenset()
    tools_by_name: Mapping[str, ToolRef] = field(default_factory=dict)
    version_disabled: bool = False


#: The policy for a task submitted directly, with no agent version pinned.
#: Denies everything: a task with no governing definition has no basis on
#: which anything could be granted to it.
UNGOVERNED = ToolPolicy(agent_version_id=None)


@dataclass(frozen=True, slots=True)
class AuthzDecision:
    allowed: bool
    tool: ToolRef | None = None
    reason: DenyReason | None = None

    def __bool__(self) -> bool:
        return self.allowed


def authorize(policy: ToolPolicy, tool_name: str) -> AuthzDecision:
    """May this agent version use this tool right now?

    Order matters. The live checks (version disabled, tool disabled) are
    evaluated so that a kill switch wins over any grant -- if grants were
    checked first and returned early, disabling a tool would not stop an agent
    that already held a grant for it, which is the entire point of having a
    kill switch.
    """
    if policy.version_disabled:
        return AuthzDecision(False, reason=DenyReason.VERSION_DISABLED)

    if policy.agent_version_id is None:
        return AuthzDecision(False, reason=DenyReason.NO_POLICY)

    tool = policy.tools_by_name.get(tool_name)
    if tool is None:
        return AuthzDecision(False, reason=DenyReason.UNKNOWN_TOOL)

    if tool.status is ToolStatus.DISABLED:
        return AuthzDecision(False, tool=tool, reason=DenyReason.TOOL_DISABLED)

    if tool.id not in policy.granted_tool_ids:
        return AuthzDecision(False, tool=tool, reason=DenyReason.NOT_GRANTED)

    return AuthzDecision(True, tool=tool)
