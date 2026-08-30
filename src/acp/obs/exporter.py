"""Standalone Prometheus exporter for the non-HTTP processes.

The API already serves /metrics from its own ASGI app. Workers and the reaper
have no HTTP server, so they start a tiny one.

Each process exports its OWN counters. That is not a limitation to work
around -- it is the only way a counter can be correct without cross-process
coordination. If workers pushed into a shared counter you would need
distributed increments; if one process exported the fleet's totals it would be
guessing about processes it cannot see. Prometheus is built to sum across
scraped instances, so let it.
"""

from __future__ import annotations

from prometheus_client import start_http_server

from acp.obs import metrics
from acp.obs.logging import get_logger

log = get_logger(__name__)


def start_exporter(port: int) -> None:
    """Serve /metrics on `port`. A port of 0 disables the exporter.

    Failure is logged, never raised: a worker that cannot open its metrics
    port should keep executing tasks. Losing observability is bad; losing
    the fleet because observability failed is worse.
    """
    if not port:
        return
    try:
        start_http_server(port, registry=metrics.REGISTRY)
        log.info("metrics.exporter_started", port=port)
    except OSError:
        log.warning("metrics.exporter_failed", port=port, exc_info=True)
