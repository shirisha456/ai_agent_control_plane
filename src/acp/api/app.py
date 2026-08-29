from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from acp.api.routes import health
from acp.config import settings
from acp.db.session import dispose_engine
from acp.obs.logging import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = settings()
    configure_logging(s.log_level, s.log_format)
    log.info(
        "api.start",
        lease_ttl_s=s.lease_ttl_s,
        heartbeat_interval_s=s.heartbeat_interval_s,
        worker_dead_after_s=s.worker_dead_after_s,
        recovery_bound_s=s.lease_ttl_s + s.reaper_period_s,
    )
    yield
    await dispose_engine()
    log.info("api.stop")


app = FastAPI(
    title="AI Agent Control Plane",
    version="0.1.0",
    summary="Distributed scheduling and durable execution for AI agent workloads",
    lifespan=lifespan,
)
app.include_router(health.router)


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    return {"service": "ai-agent-control-plane", "phase": "1", "docs": "/docs"}
