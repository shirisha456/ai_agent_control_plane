"""Tracing correctness, using an in-memory exporter -- no collector needed.

The one property worth failing a build over: tracing must be TRANSPARENT to
the exceptions of the code it wraps. A tracing bug that swallows a real 404
or a real permission denial is worse than no tracing at all, because it fails
silently in exactly the place a human would go looking for the real error.
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from acp.obs import tracing


@pytest.fixture
def traced():
    """A real SDK tracer backed by an in-memory exporter, isolated per test.

    Bypasses configure_tracing's "first call wins" global-provider guard
    (this project has exactly one process-wide provider by design) by setting
    tracing._TRACER directly, then restores it -- so this test file can prove
    real span behaviour without permanently hijacking the module for every
    other test that happens to run in the same process.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    original = tracing._TRACER
    tracing._TRACER = provider.get_tracer("test")
    try:
        yield exporter
    finally:
        tracing._TRACER = original


def test_a_span_is_recorded_with_its_attributes(traced) -> None:
    with tracing.span("acp.test", attributes={"acp.task_id": "abc", "acp.attempt": 1}):
        pass

    (recorded,) = traced.get_finished_spans()
    assert recorded.name == "acp.test"
    assert recorded.attributes["acp.task_id"] == "abc"
    assert recorded.attributes["acp.attempt"] == "1"  # attributes are stringified


def test_none_valued_attributes_are_omitted_not_stringified(traced) -> None:
    """`str(None)` == "None" would be a worse trace than no attribute at all."""
    with tracing.span("acp.test", attributes={"acp.tenant_id": None, "acp.task_type": "demo"}):
        pass

    (recorded,) = traced.get_finished_spans()
    assert "acp.tenant_id" not in recorded.attributes
    assert recorded.attributes["acp.task_type"] == "demo"


def test_span_is_transparent_to_the_wrapped_codes_own_exception(traced) -> None:
    """THE regression this file exists to prevent.

    A generator-based context manager receives the caller's exception by
    having it thrown in at the yield point. A naive `except Exception` around
    that yield catches the WRAPPED CODE's real error -- a 404, a permission
    denial, a genuine bug -- not a tracing failure, and replacing it with a
    second yield is not just wrong, it is a RuntimeError ("generator didn't
    stop after throw()") that hides the original exception entirely.
    """

    class DomainError(Exception):
        pass

    with pytest.raises(DomainError, match="the real error"), tracing.span("acp.test"):
        raise DomainError("the real error")

    # The span still closes cleanly -- tracing observes the failure, it does
    # not need to swallow it to do so.
    (recorded,) = traced.get_finished_spans()
    assert recorded.name == "acp.test"


def test_span_records_every_exception_type_transparently(traced) -> None:
    """Parametrised the obvious way tests miss regressions: only checking one
    exception type. A future refactor narrowing the except clause to (say)
    ValueError would still break on RuntimeError."""
    for exc_type in (ValueError, KeyError, RuntimeError, LookupError):
        with pytest.raises(exc_type), tracing.span("acp.test"):
            raise exc_type("boom")


def test_carrier_round_trips_into_a_link(traced) -> None:
    """The whole point of the module: propagation across the queue hop.

    A submitting request's span identity, captured as a carrier, must
    reappear as a LINK (not a parent) on the executing span -- because the
    submitting span may have ended long before a worker claims the task.
    """
    with tracing.span("acp.task.submit"):
        carrier = tracing.carrier_for_current_span()
    assert carrier is not None

    with tracing.span("acp.task.execute", link_carrier=carrier):
        pass

    submit_span, execute_span = traced.get_finished_spans()
    assert execute_span.name == "acp.task.execute"
    assert len(execute_span.links) == 1
    linked_ctx = execute_span.links[0].context
    assert linked_ctx.trace_id == submit_span.context.trace_id
    assert linked_ctx.span_id == submit_span.context.span_id


def test_execute_is_not_a_child_of_submit(traced) -> None:
    """Confirms the design choice, not just its mechanism.

    If execute were instead started as a CHILD of submit's context, it would
    share submit's trace_id as a PARENT relationship and typically fail to
    render at all once the parent has ended -- which is the whole reason this
    module uses links. Different trace_ids is the observable proof that no
    parent/child relationship was created.
    """
    with tracing.span("acp.task.submit"):
        carrier = tracing.carrier_for_current_span()

    with tracing.span("acp.task.execute", link_carrier=carrier):
        pass

    submit_span, execute_span = traced.get_finished_spans()
    assert execute_span.parent is None
    assert execute_span.context.trace_id != submit_span.context.trace_id


@pytest.mark.parametrize("carrier", [None, {}, {"trace_id": "not-hex"}, {"span_id": "abc"}])
def test_a_missing_or_malformed_carrier_produces_no_link_and_does_not_raise(
    traced, carrier
) -> None:
    """A task submitted before tracing existed, or with a corrupted payload,
    must still execute normally -- tracing is instrumentation, never a
    dependency of the execution path."""
    with tracing.span("acp.task.execute", link_carrier=carrier):
        pass

    (recorded,) = traced.get_finished_spans()
    assert recorded.links == ()


def test_carrier_is_none_with_no_active_span() -> None:
    """Outside any span (tracing disabled, or simply not inside one),
    capturing a carrier must return None rather than a garbage value that
    would silently produce a bogus link somewhere downstream."""
    tracing._TRACER = None  # force the no-op provider path
    assert tracing.carrier_for_current_span() is None


def test_task_attributes_omits_absent_optional_fields() -> None:
    attrs = tracing.task_attributes(task_id="abc")
    assert attrs == {"acp.task_id": "abc"}

    full = tracing.task_attributes(task_id="abc", tenant_id="t1", task_type="demo.agent")
    assert full == {"acp.task_id": "abc", "acp.tenant_id": "t1", "acp.task_type": "demo.agent"}
