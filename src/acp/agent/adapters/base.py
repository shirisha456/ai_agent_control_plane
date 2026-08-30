"""The adapter contract: how a worker turns a claimed task into work done.

An adapter is deliberately NOT given a database connection. It receives a
plain payload and returns a plain result (or raises); the worker owns every
transition, lease renewal, and cancellation check around it. That boundary is
what lets an adapter be tested with no database at all, and what stops
task-type-specific code from ever bypassing the CAS-guarded transition path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol


class Retryable(Exception):
    """Raised by an adapter for a failure the scheduler should retry.

    Anything else an adapter raises is treated as permanent (see
    acp.worker.loop): the task fails outright rather than burning attempts on
    an error that will never succeed.
    """


class CancelledByRequest(Exception):
    """Raised by an adapter (or the worker) when cancel_requested was observed."""


class Adapter(Protocol):
    async def run(
        self, payload: Mapping[str, Any], *, is_cancelled: Callable[[], bool]
    ) -> Mapping[str, Any]:
        """Execute one task attempt.

        `is_cancelled` is polled cooperatively at safe points; an adapter that
        never calls it simply cannot be cancelled mid-run, which is a property
        of that adapter, not a bug in the worker.
        """
        ...


AdapterFactory = Callable[[], Adapter]


class AdapterRegistry:
    """task_type -> Adapter lookup, closed and explicit rather than a plugin scan.

    A worker only ever executes task types it was configured with; an
    unregistered task_type fails fast with a clear error instead of a claimed
    task silently rotting.
    """

    def __init__(self) -> None:
        self._factories: dict[str, AdapterFactory] = {}

    def register(self, task_type: str, factory: AdapterFactory) -> None:
        self._factories[task_type] = factory

    def get(self, task_type: str) -> Adapter:
        try:
            factory = self._factories[task_type]
        except KeyError:
            raise UnknownTaskType(task_type) from None
        return factory()

    def known_types(self) -> frozenset[str]:
        return frozenset(self._factories)


class UnknownTaskType(Exception):
    def __init__(self, task_type: str) -> None:
        super().__init__(f"no adapter registered for task_type={task_type!r}")
        self.task_type = task_type
