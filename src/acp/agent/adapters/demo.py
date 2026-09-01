"""Adapters for exercising the worker loop against real tests and demos.

Deliberately dependency-free: no LLM, no network. A control plane whose test
suite needs an API key cannot run 200 tasks in CI, and a benchmark whose
latency comes from someone else's service measures their system, not ours.

These also serve as worked examples of the failure taxonomy -- each one shows
an adapter declaring what KIND of failure it hit, which is the information
acp.domain.retry needs and which no amount of inspection at the worker can
recover after the fact.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Mapping
from typing import Any

from acp.agent.adapters.base import Adapter, AdapterRegistry
from acp.domain.errors import (
    InvalidInput,
    PermanentFailure,
    RateLimited,
    Retryable,
)


class EchoAdapter(Adapter):
    """Succeeds immediately, echoing its payload."""

    async def run(
        self, payload: Mapping[str, Any], *, is_cancelled: Callable[[], bool]
    ) -> Mapping[str, Any]:
        return {"echo": dict(payload)}


class AlwaysFailAdapter(Adapter):
    """Always transient, so tests can drive the retry path to exhaustion."""

    async def run(
        self, payload: Mapping[str, Any], *, is_cancelled: Callable[[], bool]
    ) -> Mapping[str, Any]:
        raise Retryable("demo.fail always fails")


class SlowAdapter(Adapter):
    """Takes real wall-clock time before succeeding.

    Exists for one reason: proving that recovery works requires a worker to
    actually be killed WHILE holding a task, and every other demo adapter
    completes near-instantly. A benchmark or chaos demo submitting only
    instant tasks will drain the whole batch before a `docker kill` lands on
    anything, and "0 tasks running" the moment you check.

    `duration_s` is drawn per-attempt from the payload's
    `[min_s, max_s]` range (default 1-4s) using a seeded RNG when one is
    given, so a benchmark run can be replayed exactly.
    """

    async def run(
        self, payload: Mapping[str, Any], *, is_cancelled: Callable[[], bool]
    ) -> Mapping[str, Any]:
        low, high = payload.get("duration_s", [1.0, 4.0])
        duration = random.uniform(low, high)
        # Sleep in short slices so a cooperative cancel (cancel_requested)
        # is honoured within a slice rather than only after the full sleep.
        elapsed = 0.0
        slice_s = 0.1
        while elapsed < duration and not is_cancelled():
            await asyncio.sleep(min(slice_s, duration - elapsed))
            elapsed += slice_s
        return {"slept_s": round(duration, 2)}


class RateLimitedAdapter(Adapter):
    """Refuses with a server-supplied Retry-After.

    The retry policy treats that value as a floor and jitters on top, rather
    than obeying it exactly -- otherwise every throttled caller returns at the
    same instant and rebuilds the herd the backoff exists to prevent.
    """

    async def run(
        self, payload: Mapping[str, Any], *, is_cancelled: Callable[[], bool]
    ) -> Mapping[str, Any]:
        raise RateLimited("demo.rate_limited", retry_after_s=float(payload.get("retry_after", 5)))


class PermanentFailAdapter(Adapter):
    """Never retried, whatever max_attempts says."""

    async def run(
        self, payload: Mapping[str, Any], *, is_cancelled: Callable[[], bool]
    ) -> Mapping[str, Any]:
        raise PermanentFailure("demo.permanent will never succeed")


class ValidatingAdapter(Adapter):
    """Rejects bad input as USER_ERROR rather than letting it look transient.

    Retrying a malformed payload three times produces three identical
    failures, three log entries and no progress -- and points the operator at
    our service when the bug is the submitter's.
    """

    async def run(
        self, payload: Mapping[str, Any], *, is_cancelled: Callable[[], bool]
    ) -> Mapping[str, Any]:
        if "n" not in payload:
            raise InvalidInput("payload requires an 'n' field")
        return {"n": payload["n"], "doubled": payload["n"] * 2}


def register_demo_adapters(registry: AdapterRegistry) -> None:
    registry.register("demo.agent", EchoAdapter)
    registry.register("demo.slow", SlowAdapter)
    registry.register("demo.fail", AlwaysFailAdapter)
    registry.register("demo.rate_limited", RateLimitedAdapter)
    registry.register("demo.permanent", PermanentFailAdapter)
    registry.register("demo.validate", ValidatingAdapter)
