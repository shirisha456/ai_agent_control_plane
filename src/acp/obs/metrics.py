"""Prometheus metrics.

CARDINALITY IS THE WHOLE DESIGN PROBLEM HERE
--------------------------------------------
Every distinct combination of label values is a separate time series that
Prometheus stores forever. So the label set is chosen by asking "how many
values can this take, over the lifetime of the system?" -- not "what would be
useful to see".

Deliberately NOT labels, anywhere in this file:

  task_id            unbounded by construction; one series per task
  idempotency_key    client-supplied, so unbounded AND attacker-controlled
  worker_id          the subtle one. Worker ids are GENERATION-UNIQUE -- a
                     fresh uuid every process start (see migration 0003), so
                     a fleet that redeploys daily mints new series daily,
                     forever. A label that looks low-cardinality at any
                     instant can still be unbounded over time.
  error_message      free text

Those belong in traces and structured logs, which are indexed for high
cardinality and retained for days rather than months. `error_class` IS a
label, but normalised through `normalize_error_class` against a closed list,
because adapters can raise arbitrary exception types and an adapter author
should not be able to blow up the metrics backend by naming a new exception.

`tenant` appears on DB-derived gauges only. Tenant count is bounded by an
admin-controlled table rather than by user traffic, and queue depth is
meaningless without knowing whose queue is deep. It is kept off the hot-path
counters so the worker never has to carry a tenant id into a metric.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from acp.domain.errors import FailureClass

#: A dedicated registry rather than the global default: the default carries
#: process/GC collectors we do not want, and re-importing a module that
#: registers into it raises Duplicated timeseries errors under pytest.
REGISTRY = CollectorRegistry()

CONTENT_TYPE = CONTENT_TYPE_LATEST

# Exception class names allowed as label values. Anything else becomes "other"
# so an adapter cannot mint unbounded series by raising a novel exception.
# The vocabulary IS the FailureClass enum -- a closed set the domain layer
# already maintains, rather than a second list here that would silently drift
# out of sync with it.
KNOWN_ERROR_CLASSES = frozenset(c.value for c in FailureClass)


def normalize_error_class(name: str | None) -> str:
    if not name:
        return "none"
    return name if name in KNOWN_ERROR_CLASSES else "other"


# --------------------------------------------------------------------------
# submission / lifecycle counters
# --------------------------------------------------------------------------

tasks_submitted = Counter(
    "acp_tasks_submitted_total",
    "Tasks accepted by the Control API.",
    ["task_type", "deduplicated"],
    registry=REGISTRY,
)

task_attempts_finished = Counter(
    "acp_task_attempts_finished_total",
    "Execution attempts that reached a terminal outcome.",
    ["task_type", "outcome"],
    registry=REGISTRY,
)

tasks_terminal = Counter(
    "acp_tasks_terminal_total",
    "Tasks that reached a terminal state (the task, not the attempt).",
    ["task_type", "state"],
    registry=REGISTRY,
)

tasks_retried = Counter(
    "acp_tasks_retried_total",
    "Attempts that failed retryably and were rescheduled with backoff.",
    ["task_type", "error_class"],
    registry=REGISTRY,
)

# --------------------------------------------------------------------------
# the fencing evidence
# --------------------------------------------------------------------------

stale_writes_rejected = Counter(
    "acp_stale_writes_rejected_total",
    (
        "Completions rejected because the writer no longer owned the task. "
        "A non-zero rate is the PROOF that fencing works -- it is the metric "
        "the chaos demo points at."
    ),
    ["rejection"],
    registry=REGISTRY,
)

tool_calls = Counter(
    "acp_tool_calls_total",
    "Tool invocations by authorization decision.",
    ["tool", "decision"],
    registry=REGISTRY,
)

tool_access_denied = Counter(
    "acp_tool_access_denied_total",
    (
        "Tool calls refused by policy. The governance counterpart to "
        "stale_writes_rejected: a non-zero rate is the proof that "
        "authorization is actually enforced at runtime."
    ),
    ["reason"],
    registry=REGISTRY,
)

lease_expirations = Counter(
    "acp_lease_expirations_total",
    "Leases found expired by the reaper.",
    registry=REGISTRY,
)

task_recoveries = Counter(
    "acp_task_recoveries_total",
    "Tasks reclaimed from a presumed-dead worker.",
    ["disposition"],  # requeued | failed_exhausted
    registry=REGISTRY,
)

workers_marked_dead = Counter(
    "acp_workers_marked_dead_total",
    "Workers whose heartbeat went stale.",
    registry=REGISTRY,
)

workers_self_fenced = Counter(
    "acp_workers_self_fenced_total",
    "Workers that stopped after discovering they had been declared DEAD.",
    registry=REGISTRY,
)

# --------------------------------------------------------------------------
# latency
# --------------------------------------------------------------------------
# Buckets are chosen per metric, never left at the library default
# (.005 -> 10s), which is wrong at both ends here: claims complete in single
# milliseconds and recovery takes tens of seconds. Default buckets would put
# every claim in the first bucket and every recovery in +Inf, making both p99s
# meaningless.

claim_duration = Histogram(
    "acp_claim_duration_seconds",
    "Time for one claim transaction, including tenant slack accounting.",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=REGISTRY,
)

claim_batch = Histogram(
    "acp_claim_batch_size",
    "Tasks returned per claim. Zeros mean the worker is starved or the queue is empty.",
    buckets=(0, 1, 2, 3, 5, 8, 13, 21),
    registry=REGISTRY,
)

queue_wait = Histogram(
    "acp_queue_wait_seconds",
    "From a task becoming runnable (available_at) to being claimed.",
    labelnames=["task_type"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 300),
    registry=REGISTRY,
)

execution_duration = Histogram(
    "acp_execution_duration_seconds",
    "Adapter execution time for one attempt, excluding queue wait.",
    labelnames=["task_type", "outcome"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 300),
    registry=REGISTRY,
)

recovery_latency = Histogram(
    "acp_recovery_latency_seconds",
    (
        "From lease expiry to the task being reclaimed. Design bound is "
        "lease_ttl_s + reaper_period_s; this is where you check whether "
        "reality agrees."
    ),
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 45, 60, 120, 300),
    registry=REGISTRY,
)

reaper_sweep_duration = Histogram(
    "acp_reaper_sweep_duration_seconds",
    "One full reaper pass.",
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10),
    registry=REGISTRY,
)

# --------------------------------------------------------------------------
# DB-derived gauges (refreshed on a timer, not on scrape -- see obs.gauges)
# --------------------------------------------------------------------------

queue_depth = Gauge(
    "acp_queue_depth",
    "Tasks runnable right now (QUEUED and available_at <= now).",
    ["tenant"],
    registry=REGISTRY,
)

tasks_running = Gauge(
    "acp_tasks_running",
    "Tasks currently leased to a worker.",
    ["tenant"],
    registry=REGISTRY,
)

tasks_backlogged = Gauge(
    "acp_tasks_backlogged",
    "QUEUED tasks not yet runnable -- i.e. waiting out a retry backoff.",
    registry=REGISTRY,
)

workers_by_status = Gauge(
    "acp_workers",
    "Registered workers by status.",
    ["status"],
    registry=REGISTRY,
)

leases_expired_pending = Gauge(
    "acp_leases_expired_pending",
    (
        "RUNNING tasks whose lease has already expired but which have not "
        "been reclaimed yet. Sustained non-zero means the reaper is down or "
        "cannot keep up -- the single best alert in the system."
    ),
    registry=REGISTRY,
)


def render() -> bytes:
    """Serialise the registry in Prometheus text exposition format."""
    return generate_latest(REGISTRY)
