"""The worker process: claim, execute, renew, complete.

One `Worker` runs up to `capacity` attempts concurrently. Each attempt gets
its own asyncio task pairing adapter execution with a lease-renewal loop, so
a slow adapter cannot starve renewal and a stalled renewal cannot block the
adapter -- the two only communicate through a shared cancellation flag.
"""

from __future__ import annotations

import asyncio
import contextlib
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
from acp.db.queries.completion import (
    complete_abandoned,
    complete_cancelled,
    complete_failed,
    complete_retry,
    complete_success,
)
from acp.db.queries.lease import renew_lease
from acp.db.queries.transitions import record_event
from acp.db.queries.workers import heartbeat, register_worker, set_worker_status
from acp.db.session import transaction
from acp.domain.states import EventType
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
        #: task_id -> attempt for everything this worker currently owns, so
        #: graceful shutdown can hand back work it will not finish.
        self._owned: dict[Any, int] = {}
        #: Set when the control plane has declared us DEAD. See _maybe_heartbeat.
        self._fenced = False

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
        """Request a graceful stop; in-flight attempts get a grace period."""
        self._stopping.set()

    async def run_forever(self) -> None:
        await self.register()
        try:
            while not self._stopping.is_set():
                await self._maybe_heartbeat()
                # Re-check before claiming: the heartbeat above may have just
                # discovered we were declared DEAD and set _stopping. The
                # `while` condition is not re-evaluated until the next
                # iteration, so without this a fenced worker takes one more
                # batch of work on its way out the door.
                if self._stopping.is_set():
                    break
                free = self.capacity - len(self._inflight)
                if free > 0:
                    await self._claim_and_dispatch(free)
                self._inflight = {t for t in self._inflight if not t.done()}
                await asyncio.wait(
                    [asyncio.ensure_future(self._sleep_poll())] + list(self._inflight),
                    return_when=asyncio.FIRST_COMPLETED,
                )
        finally:
            await self._drain()

    async def _drain(self) -> None:
        """Stop cleanly: finish what we can, hand back what we cannot.

        The grace period is what separates a clean stop from a crash. Work
        that finishes inside it completes normally; work that does not is
        explicitly returned to the queue with available_at = now(), so another
        worker starts it in milliseconds instead of after a full lease_ttl.

        Skipped entirely when we have been fenced -- a worker the control
        plane has declared DEAD must not write to tasks at all, not even to
        be helpful. Its leases will expire and the reaper will reclaim them.
        """
        if self._fenced:
            for task in self._inflight:
                task.cancel()
            await asyncio.gather(*self._inflight, return_exceptions=True)
            return

        try:
            async with transaction() as conn:
                await set_worker_status(conn, worker_id=self.worker_id, status="DRAINING")
        except Exception:  # noqa: BLE001 - shutdown must not fail on bookkeeping
            logger.warning("worker.drain_status_failed id=%s", self.worker_id, exc_info=True)

        if self._inflight:
            done, pending = await asyncio.wait(self._inflight, timeout=self.settings.drain_grace_s)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        # Anything still owned never reached a terminal transition. Hand it
        # back rather than making the next worker wait out the lease.
        for task_id, attempt in list(self._owned.items()):
            try:
                async with transaction() as conn:
                    await complete_abandoned(
                        conn,
                        task_id,
                        attempt=attempt,
                        worker_id=self.worker_id,
                        reason="worker_shutdown",
                    )
            except Exception:  # noqa: BLE001
                logger.warning("worker.abandon_failed task=%s", task_id, exc_info=True)
        self._owned.clear()

        try:
            async with transaction() as conn:
                await set_worker_status(conn, worker_id=self.worker_id, status="DEAD")
        except Exception:  # noqa: BLE001
            logger.warning("worker.final_status_failed id=%s", self.worker_id, exc_info=True)

    async def _sleep_poll(self) -> None:
        await asyncio.sleep(self.settings.poll_interval_ms / 1000)

    async def _maybe_heartbeat(self) -> None:
        """Throttled to heartbeat_interval_s, not the (much shorter) poll interval."""
        now = time.monotonic()
        if now - self._last_heartbeat_at < self.settings.heartbeat_interval_s:
            return
        self._last_heartbeat_at = now
        async with transaction() as conn:
            alive = await heartbeat(conn, worker_id=self.worker_id)
        if not alive:
            # The control plane declared us dead while we were away. Nothing
            # about task safety depends on us noticing -- our writes are
            # fenced by lease_worker_id + attempt regardless -- but a
            # declared-dead worker that keeps claiming makes every dashboard
            # wrong and wastes a slot's worth of duplicate execution. Stop,
            # and let the supervisor restart us with a fresh id.
            logger.error("worker.fenced id=%s declared DEAD by the control plane", self.worker_id)
            self._fenced = True
            self._stopping.set()

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
        self._owned[task_id] = attempt

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

        try:
            await self._finalize(
                task_id, attempt, row["max_attempts"], outcome, detail, cancelled.is_set()
            )
        finally:
            # Ownership ends here whether or not the transition applied: if it
            # did not, we lost the lease and no longer have anything to hand
            # back. Leaving it in _owned would make drain try to abandon a
            # task another worker now owns -- fenced out, but noisy.
            self._owned.pop(task_id, None)

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
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=interval)
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
                    res = await complete_cancelled(
                        conn, task_id, attempt=attempt, worker_id=self.worker_id
                    )
                else:
                    res = await complete_success(
                        conn,
                        task_id,
                        attempt=attempt,
                        worker_id=self.worker_id,
                        result=dict(detail),
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
                res = await complete_cancelled(
                    conn, task_id, attempt=attempt, worker_id=self.worker_id
                )
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
                "worker.stale_write_rejected task=%s attempt=%s rejection=%s worker=%s",
                task_id,
                attempt,
                res.rejection,
                self.worker_id,
            )
            # Record it in the database, not just the log. A non-zero count of
            # these is the PROOF that fencing works, and the chaos demo's
            # verification reads it back with
            #     SELECT count(*) FROM task_events
            #      WHERE event_type = 'STALE_WRITE_REJECTED'
            # A log line cannot be asserted on in a test or charted in Grafana.
            #
            # Written in its own transaction: the one above just failed its
            # CAS, and a security- or correctness-relevant record must not be
            # rolled back along with the work it describes.
            try:
                async with transaction() as conn:
                    await record_event(
                        conn,
                        task_id,
                        EventType.STALE_WRITE_REJECTED,
                        attempt=attempt,
                        worker_id=self.worker_id,
                        data={
                            "operation": outcome,
                            "rejection": res.rejection.value if res.rejection else None,
                        },
                    )
            except Exception:  # noqa: BLE001 - never let bookkeeping mask the real event
                logger.warning("worker.stale_write_event_failed task=%s", task_id, exc_info=True)
