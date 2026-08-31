"""The worker process: claim, execute, renew, complete.

One `Worker` runs up to `capacity` attempts concurrently. Each attempt gets
its own asyncio task pairing adapter execution with a lease-renewal loop, so
a slow adapter cannot starve renewal and a stalled renewal cannot block the
adapter -- the two only communicate through a shared cancellation flag.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import random
import socket
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from acp.agent import tools as tool_runtime
from acp.agent.adapters.base import Adapter, AdapterRegistry, UnknownTaskType
from acp.config import Settings
from acp.db.queries.audit import record_audit_independently
from acp.db.queries.claim import claim_tasks
from acp.db.queries.completion import (
    complete_abandoned,
    complete_cancelled,
    complete_failed,
    complete_retry,
    complete_success,
)
from acp.db.queries.lease import renew_lease
from acp.db.queries.tools import snapshot_policies
from acp.db.queries.transitions import record_event
from acp.db.queries.workers import heartbeat, register_worker, set_worker_status
from acp.db.session import engine as db_engine
from acp.db.session import transaction
from acp.domain.authz import UNGOVERNED, AuthzDecision, ToolPolicy, ToolRef
from acp.domain.errors import FailureClass, classify, retry_after_of
from acp.domain.retry import RetryDecision
from acp.domain.retry import decide as decide_retry
from acp.domain.states import EventType
from acp.obs import metrics
from acp.obs.logging import get_logger
from acp.scheduling.policy import DEFAULT_POLICY, ClaimPolicy

logger = get_logger("acp.worker")


def new_worker_id() -> str:
    """Generation-unique: fresh per process start. See migration 0003."""
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


class Worker:
    def __init__(
        self,
        *,
        settings: Settings,
        registry: AdapterRegistry,
        capacity: int = 5,
        policy: ClaimPolicy = DEFAULT_POLICY,
        worker_id: str | None = None,
        rng: random.Random | None = None,
        tool_invoker: tool_runtime.ToolInvoker | None = None,
        capabilities: Sequence[str] | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.capacity = capacity
        self.policy = policy
        self.worker_id = worker_id or new_worker_id()
        #: Injectable so tests can assert an exact backoff schedule. The
        #: policy takes the RNG as an argument precisely so this is
        #: possible without patching module globals.
        self._rng = rng or random.Random()
        #: How an ALLOWED tool call is actually performed. Separate from
        #: authorization so a new tool type cannot change who may call it.
        self.tool_invoker = tool_invoker or tool_runtime.simulated_invoker
        #: Normalised through the same path as task requirements, so a
        #: worker advertising "GPU " satisfies a task requiring "gpu".
        self.capabilities = (
            tuple(sorted({c.strip().lower() for c in capabilities if c.strip()}))
            if capabilities is not None
            else settings.capabilities()
        )
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
                # NOT the adapter registry's task types. Worker capability
                # is what the MACHINE offers (gpu, internet); task_type is
                # what the software can execute. Conflating them means a
                # task requiring a GPU would be 'satisfied' by any worker
                # that merely knows its task type.
                capabilities=self.capabilities,
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
                # Retrieve results before dropping finished tasks. Without
                # this, an exception raised inside _run_attempt vanishes with
                # the Task object and the worker silently stops finalising
                # attempts while still looking healthy.
                self._inflight = self._harvest(self._inflight)
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

    def _record_outcome(
        self,
        task_type: str,
        outcome: str,
        failure_class: FailureClass,
        cancel_requested: bool,
        decision: RetryDecision | None,
    ) -> None:
        """Count what actually happened, mirroring _finalize's branches.

        Only called when the transition APPLIED. Counting an outcome we were
        fenced out of would inflate throughput with work another worker
        actually did -- the metric would say the task succeeded here when it
        succeeded somewhere else.
        """
        if cancel_requested:
            attempt_outcome = task_state = "CANCELLED"
        elif outcome == "succeeded":
            attempt_outcome = task_state = "SUCCEEDED"
        elif decision is not None and decision.should_retry:
            metrics.tasks_retried.labels(task_type=task_type, error_class=failure_class.value).inc()
            # The ATTEMPT finished; the TASK has not reached a terminal state,
            # so it is deliberately absent from tasks_terminal_total.
            # Conflating the two would make "tasks completed" count retries.
            metrics.task_attempts_finished.labels(task_type=task_type, outcome="FAILED").inc()
            return
        else:
            attempt_outcome = task_state = "FAILED"

        metrics.task_attempts_finished.labels(task_type=task_type, outcome=attempt_outcome).inc()
        metrics.tasks_terminal.labels(task_type=task_type, state=task_state).inc()

    def _harvest(self, tasks_: set[asyncio.Task[None]]) -> set[asyncio.Task[None]]:
        """Drop completed attempt tasks, surfacing anything they raised."""
        still_running = set()
        for task in tasks_:
            if not task.done():
                still_running.add(task)
                continue
            if task.cancelled():
                continue
            exc = task.exception()
            if exc is not None:
                logger.error("worker.attempt_crashed", worker_id=self.worker_id, exc_info=exc)
        return still_running

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
            logger.error("worker.fenced", worker_id=self.worker_id)
            metrics.workers_self_fenced.inc()
            self._fenced = True
            self._stopping.set()

    async def _claim_and_dispatch(self, limit: int) -> None:
        started = time.monotonic()
        async with transaction() as conn:
            claimed = await claim_tasks(
                conn,
                worker_id=self.worker_id,
                limit=min(limit, self.settings.claim_batch_size),
                lease_ttl_s=self.settings.lease_ttl_s,
                policy=self.policy,
                worker_capabilities=self.capabilities,
            )
            # Freeze each claimed version's tool policy in the SAME
            # transaction. A small lookup on the handful of rows the claim
            # returned -- the ordering scan does not pay for it. Doing it here
            # rather than at tool-call time means the policy is consistent for
            # the whole attempt and the tool path needs no query at all.
            policies = await snapshot_policies(
                conn, agent_version_ids=[r["agent_version_id"] for r in claimed]
            )
        metrics.claim_duration.observe(time.monotonic() - started)
        metrics.claim_batch.observe(len(claimed))

        for row in claimed:
            # Queue wait measured entirely from database timestamps:
            # `updated_at` was set to now() by the claim UPDATE and
            # `available_at` by whoever queued it. Subtracting a local clock
            # from a database timestamp would fold this machine's skew into
            # the latency histogram.
            wait_s = (row["updated_at"] - row["available_at"]).total_seconds()
            metrics.queue_wait.labels(task_type=row["task_type"]).observe(max(0.0, wait_s))
            # UNGOVERNED for a directly-submitted task: with no agent version
            # pinned there is no definition on which anything could have been
            # granted, so it may call nothing. "No policy" must never mean
            # "everything permitted".
            policy = policies.get(row["agent_version_id"], UNGOVERNED)
            self._inflight.add(asyncio.ensure_future(self._run_attempt(row, policy)))

    async def _run_attempt(self, row: Mapping[str, Any], policy: ToolPolicy = UNGOVERNED) -> None:
        task_id = row["id"]
        attempt = row["attempt"]
        task_type = row["task_type"]
        cancelled = asyncio.Event()
        stop_renewal = asyncio.Event()
        self._owned[task_id] = attempt

        renewal = asyncio.ensure_future(
            self._renew_until_stopped(task_id, attempt, cancelled, stop_renewal)
        )
        started = time.monotonic()
        # Bound for the duration of this attempt only. asyncio copies the
        # current context when a Task is created, so concurrent attempts on
        # one worker cannot observe each other's policy -- and an adapter that
        # never calls a tool is completely unaffected.
        token = tool_runtime.bind(self._tool_access(row, policy))
        try:
            adapter = self.registry.get(task_type)
            outcome, detail = await self._execute(adapter, row, cancelled)
        except UnknownTaskType as exc:
            outcome, detail = "failed", exc
        finally:
            # Unbound before finalisation: nothing after the adapter returns
            # has any business calling a tool, and leaving it bound would let a
            # refactor smuggle a tool call into the completion path, where the
            # policy snapshot is no longer the right authority.
            tool_runtime.unbind(token)
            stop_renewal.set()
            await renewal

        metrics.execution_duration.labels(task_type=task_type, outcome=outcome).observe(
            time.monotonic() - started
        )

        try:
            await self._finalize(
                task_id,
                attempt,
                row["max_attempts"],
                outcome,
                detail,
                cancelled.is_set(),
                task_type=task_type,
            )
        finally:
            # Ownership ends here whether or not the transition applied: if it
            # did not, we lost the lease and no longer have anything to hand
            # back. Leaving it in _owned would make drain try to abandon a
            # task another worker now owns -- fenced out, but noisy.
            self._owned.pop(task_id, None)

    def _tool_access(self, row: Mapping[str, Any], policy: ToolPolicy) -> tool_runtime.ToolAccess:
        task_id = row["id"]
        attempt = row["attempt"]
        tenant_id = row["tenant_id"]

        async def on_decision(tool_name: str, decision: AuthzDecision) -> None:
            if decision.allowed:
                # ALLOW is execution history: high volume, only interesting
                # while debugging this task, so it lives with the task and is
                # pruned with it.
                metrics.tool_calls.labels(tool=tool_name, decision="allowed").inc()
                async with transaction() as conn:
                    await record_event(
                        conn,
                        task_id,
                        EventType.TOOL_ACCESS_ALLOWED,
                        attempt=attempt,
                        worker_id=self.worker_id,
                        data={"tool": tool_name},
                    )
                return

            reason = decision.reason.value if decision.reason else "denied"
            metrics.tool_calls.labels(tool=tool_name, decision="denied").inc()
            metrics.tool_access_denied.labels(reason=reason).inc()
            logger.warning(
                "worker.tool_access_denied",
                task_id=str(task_id),
                attempt=attempt,
                tool=tool_name,
                reason=reason,
                agent_version_id=str(policy.agent_version_id),
            )
            # DENY goes to BOTH. The task timeline explains what the task did;
            # the audit log outlives the task's retention, because a refusal is
            # what someone comes looking for months later. Denials are rare by
            # construction, so the duplication costs nothing.
            async with transaction() as conn:
                await record_event(
                    conn,
                    task_id,
                    EventType.TOOL_ACCESS_DENIED,
                    attempt=attempt,
                    worker_id=self.worker_id,
                    data={"tool": tool_name, "reason": reason},
                )
            # Its OWN transaction, committed immediately. If this shared the
            # attempt's transaction and the worker then lost its lease, the
            # completion CAS would fail, everything would roll back, and the
            # record of the refusal would vanish -- precisely when it matters
            # most. Security records are never coupled to work that can be
            # rolled back.
            await record_audit_independently(
                db_engine(),
                tenant_id=tenant_id,
                action="TOOL_ACCESS_DENIED",
                resource_type="task",
                resource_id=task_id,
                outcome="DENIED",
                actor=f"agent_version:{policy.agent_version_id}",
                data={"tool": tool_name, "reason": reason, "attempt": attempt},
            )

        async def on_executed(tool: ToolRef, ok: bool, error: str | None) -> None:
            async with transaction() as conn:
                await record_event(
                    conn,
                    task_id,
                    EventType.TOOL_EXECUTED if ok else EventType.TOOL_EXECUTION_FAILED,
                    attempt=attempt,
                    worker_id=self.worker_id,
                    data={"tool": tool.name, "error": error},
                )

        return tool_runtime.ToolAccess(
            policy=policy,
            invoker=self.tool_invoker,
            on_decision=on_decision,
            on_executed=on_executed,
        )

    async def _execute(
        self, adapter: Adapter, row: Mapping[str, Any], cancelled: asyncio.Event
    ) -> tuple[str, Any]:
        try:
            result = await adapter.run(row["payload"], is_cancelled=cancelled.is_set)
        except Exception as exc:  # noqa: BLE001 - adapter errors are data, not our bug
            # One except clause, because the decision is no longer "was this
            # the retryable exception type" but "what class of failure is
            # this" -- which acp.domain.errors answers for typed adapter
            # errors and well-known builtins alike.
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
        task_type: str = "unknown",
    ) -> None:
        failure_class = FailureClass.UNKNOWN
        decision = None
        if outcome != "succeeded":
            failure_class = (
                classify(detail) if isinstance(detail, BaseException) else FailureClass.UNKNOWN
            )
            decision = decide_retry(
                failure_class,
                attempt=attempt,
                max_attempts=max_attempts,
                retry_after_s=(
                    retry_after_of(detail) if isinstance(detail, BaseException) else None
                ),
                rng=self._rng,
            )

        # `error_class` stores the FAILURE CLASS, not the Python exception
        # type. It is a Prometheus label, a query filter and a dashboard
        # dimension, and all three want a bounded vocabulary; an adapter
        # author must not be able to add a value by naming a new exception.
        # The exception type is preserved in error_message, where high
        # cardinality is free.
        error_class = failure_class.value
        error_message = (
            f"{type(detail).__name__}: {detail}"
            if isinstance(detail, BaseException)
            else str(detail)
        )

        async with transaction() as conn:
            if cancel_requested:
                # Checked first: a task told to stop is cancelled, whatever its
                # adapter happened to return on the way out. Ordering this
                # after the success branch would let a task that finished
                # microseconds before the cancel land as SUCCEEDED, and the
                # caller who asked for cancellation would never learn why.
                res = await complete_cancelled(
                    conn, task_id, attempt=attempt, worker_id=self.worker_id
                )
            elif outcome == "succeeded":
                res = await complete_success(
                    conn,
                    task_id,
                    attempt=attempt,
                    worker_id=self.worker_id,
                    result=dict(detail),
                )
            elif decision is not None and decision.should_retry:
                res = await complete_retry(
                    conn,
                    task_id,
                    attempt=attempt,
                    worker_id=self.worker_id,
                    error_class=error_class,
                    error_message=error_message,
                    backoff_s=decision.backoff_s,
                )
            else:
                res = await complete_failed(
                    conn,
                    task_id,
                    attempt=attempt,
                    worker_id=self.worker_id,
                    error_class=error_class,
                    error_message=error_message,
                )
        if res.applied:
            self._record_outcome(task_type, outcome, failure_class, cancel_requested, decision)
            if decision is not None:
                logger.info(
                    "worker.attempt_finished",
                    task_id=str(task_id),
                    attempt=attempt,
                    failure_class=error_class,
                    retrying=decision.should_retry,
                    backoff_s=round(decision.backoff_s, 3),
                    reason=decision.reason,
                )

        if not res.applied:
            rejection = res.rejection.value if res.rejection else "unknown"
            metrics.stale_writes_rejected.labels(rejection=rejection).inc()
            logger.warning(
                "worker.stale_write_rejected",
                task_id=str(task_id),
                attempt=attempt,
                rejection=rejection,
                worker_id=self.worker_id,
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
                            "rejection": rejection,
                        },
                    )
            except Exception:  # noqa: BLE001 - never let bookkeeping mask the real event
                logger.warning("worker.stale_write_event_failed task=%s", task_id, exc_info=True)
