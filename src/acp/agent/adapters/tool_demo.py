"""Adapters that call tools, for exercising the authorization path."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from acp.agent.adapters.base import Adapter, AdapterRegistry
from acp.agent.tools import call_tool


class ToolCallingAdapter(Adapter):
    """Calls whatever tools its payload names, in order.

    Note it takes no `tools` parameter: access arrives through a contextvar
    the worker binds for the attempt, so an adapter opts in by calling
    `call_tool` and adapters that never do are untouched.

    A denial raises PermissionDenied, which classifies as PERMISSION_DENIED
    and is never retried -- so a refusal fails the task once instead of
    writing max_attempts audit records for one refusal.
    """

    async def run(
        self, payload: Mapping[str, Any], *, is_cancelled: Callable[[], bool]
    ) -> Mapping[str, Any]:
        results = []
        for name in payload.get("tools", []):
            results.append(await call_tool(name, {"task": payload.get("topic")}))
        return {"tool_results": results}


def register_tool_adapters(registry: AdapterRegistry) -> None:
    registry.register("demo.tools", ToolCallingAdapter)
