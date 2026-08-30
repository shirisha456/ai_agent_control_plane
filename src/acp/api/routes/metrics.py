"""Prometheus scrape endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Response

from acp.obs import metrics

router = APIRouter(tags=["observability"])


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    """Serve process-local counters plus the DB-derived gauges.

    Reads memory only. The gauges are refreshed on a timer (see obs.gauges),
    so a scrape -- or a hundred scrapes -- never touches PostgreSQL. The
    monitoring system must not be able to cause the incident it exists to
    observe.
    """
    return Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE)
