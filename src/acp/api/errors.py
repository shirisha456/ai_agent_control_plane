"""One error shape for the whole API.

Clients retry on the code, not on prose. A submission rejected because the
tenant is over quota (retry later, same key) and one rejected because the
payload is malformed (never retry) must be distinguishable without parsing
English.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class NotFound(ApiError):
    status_code = 404
    code = "not_found"


class Conflict(ApiError):
    """The request was well-formed but conflicts with current state.

    Distinct from 429: a conflict will not resolve by waiting.
    """

    status_code = 409
    code = "conflict"


class QuotaExceeded(ApiError):
    """Backpressure: this TENANT is over its own limit. Retry later.

    Never used for system-wide overload -- that is 503. The distinction
    matters: 429 says "you, slow down", 503 says "us, we are in trouble".
    Reporting a system problem as the client's fault sends every client into
    a retry pattern tuned for the wrong cause.
    """

    status_code = 429
    code = "quota_exceeded"


class Overloaded(ApiError):
    status_code = 503
    code = "overloaded"


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _handle(_: Request, exc: ApiError) -> JSONResponse:
        body: dict[str, Any] = {"error": {"code": exc.code, "message": exc.message}}
        if exc.details:
            body["error"]["details"] = exc.details
        headers = {}
        if isinstance(exc, QuotaExceeded | Overloaded):
            # Without Retry-After, well-behaved clients guess -- and a fleet
            # guessing in unison is a retry storm.
            headers["Retry-After"] = str(exc.details.get("retry_after_s", 5))
        return JSONResponse(status_code=exc.status_code, content=body, headers=headers)
