"""Host-platform quirks, isolated in one place.

Windows defaults asyncio to ProactorEventLoop, which psycopg3's async driver
refuses to run on (it needs the selector-based loop for socket readiness).
Containers run Linux so this never bites in `docker compose up`, but it bites
immediately when running the test suite or a worker natively on Windows --
which is exactly how this project is developed.

Keeping it here rather than sprinkling `if sys.platform` through entrypoints
means there is one line to delete if we ever drop native Windows support.
"""

from __future__ import annotations

import asyncio
import sys


def install_event_loop_policy() -> None:
    """Select an event loop psycopg can actually use. Safe to call repeatedly."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.get_event_loop_policy()
