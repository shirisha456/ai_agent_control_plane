"""Tool invocation from inside an adapter, with authorization enforced.

HOW AN ADAPTER REACHES THIS
---------------------------
Through a contextvar, not through a changed `Adapter.run` signature.

The alternative -- threading a `tools=` argument through `run()` -- would
force every adapter and every test double to accept a parameter most of them
never use, and would make adding the NEXT execution-scoped facility another
breaking change. A contextvar is set by the worker for the duration of one
attempt, and asyncio copies the context when a Task is created, so each
in-flight attempt gets its own binding with no leakage between the tasks a
worker runs concurrently.

Adapters that never call a tool are untouched and need no updating.

WHAT ENFORCEMENT COSTS AT RUNTIME
---------------------------------
A dictionary lookup and a set membership test. The policy was snapshotted
inside the claim transaction (db/queries/tools.snapshot_policies), so there is
no query here, no cache, and no staleness window -- and the policy cannot
change underneath a running attempt, which means an agent can never gain a
capability halfway through its own execution.
"""

from __future__ import annotations

import contextvars
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from acp.domain.authz import AuthzDecision, ToolPolicy, ToolRef
from acp.domain.errors import PermissionDenied

#: Bound by the worker for the duration of one attempt. None outside an
#: attempt, which is why call_tool refuses rather than guessing.
_TOOL_ACCESS: contextvars.ContextVar[ToolAccess | None] = contextvars.ContextVar(
    "acp_tool_access", default=None
)

#: How a tool of a given type is actually invoked once ALLOWED. Authorization
#: and execution are kept apart on purpose: the decision is pure and testable
#: without any tool existing, and a new tool type cannot accidentally change
#: who may call it.
ToolInvoker = Callable[[ToolRef, Mapping[str, Any]], Awaitable[Mapping[str, Any]]]


@dataclass(slots=True)
class ToolAccess:
    """Execution-scoped tool gateway for one attempt."""

    policy: ToolPolicy
    invoker: ToolInvoker
    #: Called with the decision so the worker can write the timeline entry and,
    #: for denials, the independent audit record.
    on_decision: Callable[[str, AuthzDecision], Awaitable[None]]
    on_executed: Callable[[ToolRef, bool, str | None], Awaitable[None]]

    async def call(self, tool_name: str, request: Mapping[str, Any]) -> Mapping[str, Any]:
        from acp.domain.authz import authorize

        decision = authorize(self.policy, tool_name)
        await self.on_decision(tool_name, decision)

        if not decision.allowed:
            # PERMISSION_DENIED is non-retryable (see domain/retry.POLICIES):
            # the answer will not change, and retrying would turn one refusal
            # into max_attempts audit records.
            raise PermissionDenied(
                f"agent is not permitted to use tool {tool_name!r} "
                f"({decision.reason.value if decision.reason else 'denied'})"
            )

        assert decision.tool is not None
        try:
            result = await self.invoker(decision.tool, request)
        except Exception as exc:
            await self.on_executed(decision.tool, False, f"{type(exc).__name__}: {exc}")
            raise
        await self.on_executed(decision.tool, True, None)
        return result


def bind(access: ToolAccess) -> contextvars.Token:
    return _TOOL_ACCESS.set(access)


def unbind(token: contextvars.Token) -> None:
    _TOOL_ACCESS.reset(token)


def current() -> ToolAccess | None:
    return _TOOL_ACCESS.get()


async def call_tool(tool_name: str, request: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    """Call a registered tool from inside an adapter.

    Refuses outside an attempt rather than falling back to unchecked
    execution. A tool call with no governing policy is exactly the case that
    must not silently succeed -- "no policy loaded" is not the same as
    "everything permitted", and treating it that way is how authorization
    gets bypassed by a refactor.
    """
    access = current()
    if access is None:
        raise PermissionDenied(
            "no tool policy is bound; tools may only be called from inside a task attempt"
        )
    return await access.call(tool_name, dict(request or {}))


async def simulated_invoker(tool: ToolRef, request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Default invoker: echoes the request back.

    Enough to demonstrate and test the governance path end to end without a
    network. The engineering interest is that tools are registered resources
    the control plane authorizes -- not in whatever a particular integration
    returns.
    """
    return {"tool": tool.name, "type": tool.tool_type.value, "request": dict(request)}
