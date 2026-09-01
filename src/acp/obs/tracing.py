"""Distributed tracing: API request -> queue -> worker -> agent step -> tool.

WHY A SPAN LINK ACROSS THE QUEUE HOP, NOT A CHILD SPAN
-------------------------------------------------------
A task can sit QUEUED for seconds, minutes, or (mid-retry-backoff) longer.
By the time a worker claims it, the submitting request's span has already
ENDED. A child span requires a live parent; attaching one to an ended parent
produces a waterfall where the "child" appears to start after its parent
finished, which most trace viewers render as broken or simply drop.

A LINK records the causal relationship ("this execution was caused by that
submission") without requiring the parent to still be open. That is the
correct primitive for exactly this shape: an asynchronous handoff through
durable storage, which is the core of this whole system's execution model.
So the propagation carried in the task payload is a plain
`(trace_id, span_id)` pair, not a parent context.

WHY TRACING IS DISABLED BY DEFAULT
----------------------------------
Every trace is inert (a `contextlib.nullcontext`-shaped no-op) unless
`ACP_OTEL_ENDPOINT` is set. Tracing is instrumentation, not a runtime
dependency: a control plane that could deadlock claiming a task because its
collector is unreachable would have let observability cause the outage it
exists to diagnose. Every exporter call is therefore wrapped to fail silently.

WHAT GETS RECORDED AS SPAN ATTRIBUTES VS. WHERE ELSE
-----------------------------------------------------
task_id, tenant_id, worker_id and attempt are exactly the identifiers that
Prometheus labels must NOT carry (see obs/metrics.py) -- traces and logs are
where high-cardinality identifiers belong, because they are indexed and
retained differently than a metrics backend. This module is where that
"belongs elsewhere" promise is actually kept.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from typing import Any
from uuid import UUID

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Link, SpanContext, TraceFlags

_TRACER: trace.Tracer | None = None


def configure_tracing(
    *, service_name: str, otlp_endpoint: str | None, console: bool = False
) -> None:
    """Set up the process-wide tracer. Safe to call multiple times; only the
    first call takes effect, matching OTel's own SDK guarantee.

    `otlp_endpoint=None` and `console=False` together mean tracing stays a
    no-op: `current_tracer()` still returns something, but nothing is
    exported, so instrumented code pays only the cost of a few no-op context
    managers.
    """
    global _TRACER
    if _TRACER is not None:
        return

    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
    if otlp_endpoint:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    if console:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _TRACER = trace.get_tracer("acp")


def current_tracer() -> trace.Tracer:
    """The tracer, configuring a no-op provider on first use if nobody called
    `configure_tracing`. Instrumented code should never need to check whether
    tracing is enabled -- it should just be cheap when it is not."""
    global _TRACER
    if _TRACER is None:
        configure_tracing(service_name="acp", otlp_endpoint=None)
    assert _TRACER is not None
    return _TRACER


# ---------------------------------------------------------------------------
# propagation across the queue hop
# ---------------------------------------------------------------------------


def carrier_for_current_span() -> dict[str, str] | None:
    """Capture the active span's identity to stash in a task payload.

    Returns None outside a recording span (tracing disabled, or called from
    code with no active span) so callers can skip storing an empty carrier.
    """
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return None
    return {"trace_id": format(ctx.trace_id, "032x"), "span_id": format(ctx.span_id, "016x")}


def _link_from_carrier(carrier: Mapping[str, str] | None) -> list[Link]:
    if not carrier or "trace_id" not in carrier or "span_id" not in carrier:
        return []
    try:
        ctx = SpanContext(
            trace_id=int(carrier["trace_id"], 16),
            span_id=int(carrier["span_id"], 16),
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
    except (ValueError, KeyError):
        return []
    if not ctx.is_valid:
        return []
    return [Link(ctx)]


@contextlib.contextmanager
def span(
    name: str,
    *,
    attributes: Mapping[str, Any] | None = None,
    link_carrier: Mapping[str, str] | None = None,
) -> Iterator[trace.Span]:
    """Start a span, linked to a carrier captured across an async boundary
    if one is given.

    Deliberately NOT wrapped in a broad try/except around the `yield`. A
    generator-based context manager receives the CALLER's exception by having
    it thrown in at the yield point -- so a bare `except Exception` here would
    catch the traced code's own real errors (a 404, a permission denial, a
    genuine bug) and replace them with a second, invalid yield. That is
    exactly backwards: tracing must be transparent to the exceptions of the
    code it observes. The one place OTel work can legitimately fail --
    building a span link out of a malformed carrier -- is already guarded
    inside `_link_from_carrier`, before the `with` block below ever opens.
    """
    with current_tracer().start_as_current_span(name, links=_link_from_carrier(link_carrier)) as s:
        if attributes:
            for key, value in attributes.items():
                if value is not None:
                    s.set_attribute(key, str(value))
        yield s


def task_attributes(
    *, task_id: UUID, tenant_id: UUID | None = None, task_type: str | None = None
) -> dict[str, Any]:
    """The identifiers that belong on a SPAN, not on a Prometheus label --
    see the module docstring and obs/metrics.py's cardinality argument."""
    attrs: dict[str, Any] = {"acp.task_id": str(task_id)}
    if tenant_id is not None:
        attrs["acp.tenant_id"] = str(tenant_id)
    if task_type is not None:
        attrs["acp.task_type"] = task_type
    return attrs
