"""Liveness and dependency health.

Two design points that are easy to get wrong and expensive to get wrong:

1. The probe touches the database. A control plane that reports healthy while
   unable to reach its only source of truth is worse than one reporting
   unhealthy -- load balancers keep routing submissions to it and clients see
   writes fail instead of being retried against a replica that can commit.

2. The probe is TIME-BOUNDED. A health check without a timeout inherits the
   latency of the thing it checks, so a slow-but-alive database makes /healthz
   hang. The load balancer then cannot distinguish "down" from "slow", probes
   pile up, and the health endpoint becomes the outage. A probe that exceeds
   its budget IS an unhealthy answer, not a missing one.
"""

from __future__ import annotations

import asyncio

import sqlalchemy as sa
from fastapi import APIRouter, Response, status

from acp.config import settings
from acp.db.session import engine

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz(response: Response) -> dict[str, object]:
    budget = settings().health_probe_timeout_s
    try:
        async with asyncio.timeout(budget):
            async with engine().connect() as conn:
                await conn.execute(sa.text("SELECT 1"))
    except TimeoutError:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unhealthy", "database": "timeout", "budget_s": budget}
    except Exception as exc:  # noqa: BLE001 - the reason is the payload
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unhealthy", "database": "unreachable", "detail": str(exc)[:200]}
    return {"status": "healthy", "database": "reachable"}
