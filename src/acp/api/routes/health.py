from __future__ import annotations

import sqlalchemy as sa
from fastapi import APIRouter, Response, status

from acp.db.session import engine

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz(response: Response) -> dict[str, object]:
    """Liveness + dependency check.

    Deliberately touches the database. A control plane that reports healthy
    while unable to reach its only source of truth is worse than one that
    reports unhealthy: load balancers keep sending it work it cannot durably
    accept, and clients see writes silently fail instead of being retried
    elsewhere.
    """
    try:
        async with engine().connect() as conn:
            await conn.execute(sa.text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - the reason is the payload
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unhealthy", "database": "unreachable", "detail": str(exc)[:200]}
    return {"status": "healthy", "database": "reachable"}
