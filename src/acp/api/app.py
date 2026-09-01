from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager

from fastapi import FastAPI

from acp.api.errors import install_error_handlers
from acp.api.routes import agents, health, metrics, tasks, tenants, tools
from acp.config import settings
from acp.db.session import dispose_engine, engine
from acp.obs.gauges import run_refresher
from acp.obs.logging import configure_logging, get_logger
from acp.obs.tracing import configure_tracing

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = settings()
    configure_logging(s.log_level, s.log_format)
    configure_tracing(
        service_name=f"{s.otel_service_name}-api",
        otlp_endpoint=s.otel_endpoint,
        console=s.otel_console,
    )
    log.info(
        "api.start",
        lease_ttl_s=s.lease_ttl_s,
        heartbeat_interval_s=s.heartbeat_interval_s,
        worker_dead_after_s=s.worker_dead_after_s,
        recovery_bound_s=s.lease_ttl_s + s.reaper_period_s,
    )
    stop = asyncio.Event()
    refresher = asyncio.create_task(
        run_refresher(engine(), interval_s=s.gauge_refresh_s, stop=stop)
    )
    try:
        yield
    finally:
        stop.set()
        refresher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await refresher
        await dispose_engine()
        log.info("api.stop")


app = FastAPI(
    title="AI Agent Control Plane",
    version="0.1.0",
    summary="Distributed scheduling and durable execution for AI agent workloads",
    lifespan=lifespan,
)
install_error_handlers(app)
app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(tenants.router)
app.include_router(agents.router)
app.include_router(agents.routes_router)
app.include_router(tools.router)
app.include_router(tools.grants_router)
app.include_router(tools.audit_router)
app.include_router(tasks.router)


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    return {"service": "ai-agent-control-plane", "phase": "1", "docs": "/docs"}
