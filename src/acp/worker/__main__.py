"""Worker process entrypoint: `python -m acp.worker`."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal

from acp.agent.adapters.base import AdapterRegistry
from acp.agent.adapters.demo import register_demo_adapters
from acp.config import settings
from acp.db.session import dispose_engine
from acp.platform import install_event_loop_policy
from acp.worker.loop import Worker


def build_registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    register_demo_adapters(registry)
    return registry


async def main() -> None:
    logging.basicConfig(level=settings().log_level)
    worker = Worker(settings=settings(), registry=build_registry())

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        # Windows has no SIGTERM handler; SIGINT (Ctrl+C) still works.
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, worker.stop)

    try:
        await worker.run_forever()
    finally:
        await dispose_engine()


if __name__ == "__main__":
    install_event_loop_policy()
    asyncio.run(main())
