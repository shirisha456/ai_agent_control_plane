"""Guard the metric label set against cardinality explosions.

Cardinality mistakes are not caught by any normal test: the code works, the
dashboard looks great, and three weeks later Prometheus falls over holding
several million time series it can never garbage-collect. The only way to
catch it is to assert the label set directly, which is what this file does.

Pure -- no database, no scrape, runs in milliseconds.
"""

from __future__ import annotations

import pytest
from prometheus_client.metrics import MetricWrapperBase

from acp.obs import metrics

#: Labels that must never appear on a metric, with why.
FORBIDDEN_LABELS = {
    "task_id": "unbounded by construction -- one series per task, forever",
    "attempt": "unbounded; a poison-pill task alone would mint series",
    "idempotency_key": "client-supplied, so unbounded AND attacker-controlled",
    "worker_id": (
        "generation-unique (a fresh uuid per process start, see migration "
        "0003), so a fleet that redeploys daily mints new series daily -- "
        "low cardinality at any instant, unbounded over time"
    ),
    "error_message": "free text",
    "trace_id": "one per request",
    "hostname": "unbounded in any autoscaled or containerised deployment",
    "lease_worker_id": "same problem as worker_id",
}

#: Labels allowed despite being an id-ish dimension, with the reason they are
#: actually bounded. Anything not here and not obviously an enum should be
#: argued for explicitly rather than added quietly.
JUSTIFIED_LABELS = {
    "tenant": "bounded by an admin-controlled table, not by user traffic",
    "task_type": "bounded by the adapter registry, which is a closed map",
    "outcome": "the attempt_outcome enum",
    "state": "the five task states",
    "status": "the worker_status enum",
    "rejection": "the Rejection enum",
    "disposition": "requeued | failed_exhausted",
    "error_class": "normalised against KNOWN_ERROR_CLASSES",
    "deduplicated": "true | false",
}


def _declared_metrics() -> list[tuple[str, MetricWrapperBase]]:
    return [
        (name, obj) for name, obj in vars(metrics).items() if isinstance(obj, MetricWrapperBase)
    ]


def test_there_are_metrics_to_check() -> None:
    """Guard the guard: a rename that empties this list must not pass silently."""
    assert len(_declared_metrics()) >= 15


@pytest.mark.parametrize(
    ("attr", "metric"), _declared_metrics(), ids=lambda v: getattr(v, "_name", v)
)
def test_no_metric_uses_a_high_cardinality_label(attr: str, metric: MetricWrapperBase) -> None:
    offenders = {
        label: FORBIDDEN_LABELS[label] for label in metric._labelnames if label in FORBIDDEN_LABELS
    }
    assert not offenders, (
        f"metric {attr!r} ({metric._name}) uses forbidden label(s): "
        + "; ".join(f"{k} -- {v}" for k, v in offenders.items())
        + ". Put it in a trace or a structured log instead."
    )


@pytest.mark.parametrize(
    ("attr", "metric"), _declared_metrics(), ids=lambda v: getattr(v, "_name", v)
)
def test_every_label_is_a_justified_bounded_dimension(attr: str, metric: MetricWrapperBase) -> None:
    """New labels require a documented reason they are bounded.

    Failing here is not "add it to the dict" -- it is "explain why this
    dimension cannot grow without limit, then add it to the dict".
    """
    unknown = set(metric._labelnames) - set(JUSTIFIED_LABELS)
    assert not unknown, (
        f"metric {attr!r} uses undocumented label(s) {sorted(unknown)}. "
        "Add them to JUSTIFIED_LABELS with the reason their cardinality is bounded."
    )


@pytest.mark.parametrize(
    ("attr", "metric"), _declared_metrics(), ids=lambda v: getattr(v, "_name", v)
)
def test_metric_names_are_namespaced(attr: str, metric: MetricWrapperBase) -> None:
    assert metric._name.startswith("acp_"), f"{attr} is not namespaced under acp_"


def test_error_class_normalisation_bounds_the_label() -> None:
    """Adapters raise arbitrary exceptions; the label set must not follow them.

    Without this, an adapter author who defines a new exception class silently
    adds a time series -- and one that raises a dynamically-named class adds
    unbounded series.
    """
    # The vocabulary is the FailureClass enum, so every member round-trips
    # and nothing outside it can.
    from acp.domain.errors import FailureClass

    for member in FailureClass:
        assert metrics.normalize_error_class(member.value) == member.value
    assert metrics.normalize_error_class("SomeVendorSpecificError_42") == "other"
    assert metrics.normalize_error_class(None) == "none"
    assert metrics.normalize_error_class("") == "none"


def test_histogram_buckets_are_explicit() -> None:
    """Default buckets (.005 -> 10s) are wrong at both ends of this system.

    Claims complete in single milliseconds and recovery takes tens of seconds.
    Left at the default, every claim lands in the first bucket and every
    recovery in +Inf, so both p99s become meaningless -- while still rendering
    a confident-looking line on a dashboard.
    """
    from prometheus_client import Histogram

    default = Histogram.DEFAULT_BUCKETS
    for attr, metric in _declared_metrics():
        if isinstance(metric, Histogram):
            assert metric._upper_bounds != list(default), (
                f"{attr} left the default histogram buckets in place"
            )
