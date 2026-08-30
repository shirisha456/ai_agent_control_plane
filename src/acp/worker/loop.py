"""The worker process: claim, execute, renew, complete.

One `Worker` runs up to `capacity` attempts concurrently. Each attempt gets
its own asyncio task pairing adapter execution with a lease-renewal loop, so
a slow adapter cannot starve renewal and a stalled renewal cannot block the
adapter -- the two only communicate through a shared cancellation flag.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
import uuid
from collections.abc import Mapping
from typing import Any

from acp.agent.adapters.base import Adapter, AdapterRegistry, Retryable, UnknownTaskType
from acp.config import Settings
from acp.db.queries.claim import claim_tasks
from acp.db.queries.completion import complete_cancelled, complete_failed, complete_retry, complete_success
from acp.db.queries.lease import renew_lease
from acp.db.queries.workers import heartbeat, register_worker
from acp.db.session import transaction
from acp.scheduling.policy import DEFAULT_POLICY, ClaimPolicy

logger = logging.getLogger("acp.worker")


def new_worker_id() -> str:
    """Generation-unique: fresh per process start. See migration 0003."""
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _retry_backoff_s(attempt: int) -> float:
    """Full-jitter exponential backoff, capped at 60s.

    Attempt is the fencing token, already incremented at claim time, so
    attempt=1 is the first try -- backoff only grows from the second attempt.
    """
    import random

    base = min(60.0, 2.0 ** max(0, attempt - 1))
    return random.uniform(0, base)


class Worker:
    def __init__(
        self,
        *,
        settings: Settings,
        registry: AdapterRegistry,
        capacity: int = 5,
        policy: ClaimPolicy = DEFAULT_POLICY,
        worker_id: str | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.capacity = capacity
        self.policy = policy
        self.worker_id = worker_id or new_worker_id()
        self._stopping = asyncio.Event()
        self._inflight: set[asyncio.Task[None]] = set()
        self._last_heartbeat_at: float = float("-inf")

    async def register(self) -> None:
        async with transaction() as conn:
            await register_worker(
                conn,
                worker_id=self.worker_id,
                hostname=socket.gethostname(),
                pid=os.getpid(),
                capacity=self.capacity,
                capabilities=tuple(self.registry.known_types()),
            )

    def stop(self) -> None:
        """Request a graceful stop; in-flight attempts are allowed to finish."""
        self._stopping.set()

    async def run_forever(self) -> None:
        await self.register()
        try:
            while not self._stopping.is_set():
                await self._maybe_heartbeat()
                free = self.capacity - len(self._inflight)
                if free > 0:
                    await self._claim_and_dispatch(free)
                self._inflight = {t for t in self._inflight if not t.done()}
                await asyncio.wait(
                    [asyncio.ensure_future(self._sleep_poll())] + list(self._inflight),
                    return_when=asyncio.FIRST_COMPLETED,
                )
        finally:
            if self._inflight:
                await asyncio.gather(*self._inflight, return_exceptions=True)

    async def _sleep_poll(self) -> None:
        await asyncio.sleep(self.settings.poll_interval_ms / 1000)

    async def _maybe_heartbeat(self) -> None:
        """Throttled to heartbeat_interval_s, not the (much shorter) poll interval."""
        now = time.monotonic()
        if now - self._last_heartbeat_at < self.settings.heartbeat_interval_s:
            return
        self._last_heartbeat_at = now
        async with transaction() as conn:
            await heartbeat(conn, worker_id=self.worker_id)

    async def _claim_and_dispatch(self, limit: int) -> None:
        async with transaction() as conn:
            claimed = await claim_tasks(
                conn,
                worker_id=self.worker_id,
                limit=min(limit, self.settings.claim_batch_size),
                lease_ttl_s=self.settings.lease_ttl_s,
                policy=self.policy,
            )
        for row in claimed:
            self._inflight.add(asyncio.ensure_future(self._run_attempt(row)))

    async def _run_attempt(self, row: Mapping[str, Any]) -> None:
        task_id = row["id"]
        attempt = row["attempt"]
        task_type = row["task_type"]
        cancelled = asyncio.Event()
        stop_renewal = asyncio.Event()

        renewal = asyncio.ensure_future(
            self._renew_until_stopped(task_id, attempt, cancelled, stop_renewal)
        )
        try:
            adapter = self.registry.get(task_type)
            outcome, detail = await self._execute(adapter, row, cancelled)
        except UnknownTaskType as exc:
            outcome, detail = "failed", exc
        finally:
            stop_renewal.set()
            await renewal

        await self._finalize(task_id, attempt, row["max_attempts"], outcome, detail, cancelled.is_set())

    async def _execute(
        self, adapter: Adapter, row: Mapping[str, Any], cancelled: asyncio.Event
    ) -> tuple[str, Any]:
        try:
            result = await adapter.run(row["payload"], is_cancelled=cancelled.is_set)
        except Retryable as exc:
            return "retryable", exc
        except Exception as exc:  # noqa: BLE001 - adapter errors are data, not our bug
            return "failed", exc
        return "succeeded", result

    async def _renew_until_stopped(
        self, task_id: Any, attempt: int, cancelled: asyncio.Event, stop: asyncio.Event
    ) -> None:
        interval = self.settings.lease_renew_interval_s
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
            if stop.is_set():
                return
            async with transaction() as conn:
                row = await renew_lease(
                    conn,
                    task_id,
                    worker_id=self.worker_id,
                    expect_attempt=attempt,
                    lease_ttl_s=self.settings.lease_ttl_s,
                )
            if row is None:
                # Lease already lost -- stop renewing, let the adapter's
                # result (if it ever returns) fail the fenced completion CAS.
                cancelled.set()
                return
            if row["cancel_requested"]:
                cancelled.set()

    async def _finalize(
        self,
        task_id: Any,
        attempt: int,
        max_attempts: int,
        outcome: str,
        detail: Any,
        cancel_requested: bool,
    ) -> None:
        async with transaction() as conn:
            if outcome == "succeeded":
                if cancel_requested:
                    res = await complete_cancelled(conn, task_id, attempt=attempt, worker_id=self.worker_id)
                else:
                    res = await complete_success(
                        conn, task_id, attempt=attempt, worker_id=self.worker_id, result=dict(detail)
                    )
            elif outcome == "retryable" and not cancel_requested and attempt < max_attempts:
                res = await complete_retry(
                    conn,
                    task_id,
                    attempt=attempt,
                    worker_id=self.worker_id,
                    error_class=type(detail).__name__,
                    error_message=str(detail),
                    backoff_s=_retry_backoff_s(attempt),
                )
            elif cancel_requested:
                res = await complete_cancelled(conn, task_id, attempt=attempt, worker_id=self.worker_id)
            else:
                res = await complete_failed(
                    conn,
                    task_id,
                    attempt=attempt,
                    worker_id=self.worker_id,
                    error_class=type(detail).__name__ if isinstance(detail, Exception) else "Error",
                    error_message=str(detail),
                )
        if not res.applied:
            logger.warning(
                "completion CAS lost for task=%s attempt=%s: %s (lease reclaimed by another worker)",
                task_id,
                attempt,
                res.rejection,
            )
