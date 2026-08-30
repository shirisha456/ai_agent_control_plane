"""A no-dependency adapter for exercising the worker loop against real tests.

`demo.agent` echoes its payload back as the result. `demo.fail` always raises
Retryable, so tests can drive the retry path without needing a flaky adapter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from acp.agent.adapters.base import Adapter, AdapterRegistry, Retryable


class EchoAdapter(Adapter):
    async def run(
        self, payload: Mapping[str, Any], *, is_cancelled: Callable[[], bool]
    ) -> Mapping[str, Any]:
        return {"echo": dict(payload)}


class AlwaysFailAdapter(Adapter):
    async def run(
        self, payload: Mapping[str, Any], *, is_cancelled: Callable[[], bool]
    ) -> Mapping[str, Any]:
        raise Retryable("demo.fail always fails")


def register_demo_adapters(registry: AdapterRegistry) -> None:
    registry.register("demo.agent", EchoAdapter)
    registry.register("demo.fail", AlwaysFailAdapter)
