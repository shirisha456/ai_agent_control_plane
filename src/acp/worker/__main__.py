"""Worker process entrypoint: `python -m acp.worker`."""

from __future__ import annotations

import asyncio
import contextlib
import signal

from acp.agent.adapters.base import AdapterRegistry
from acp.agent.adapters.demo import register_demo_adapters
from acp.agent.adapters.tool_demo import register_tool_adapters
from acp.config import settings
from acp.db.session import dispose_engine
from acp.obs.exporter import start_exporter
from acp.obs.logging import configure_logging
from acp.obs.tracing import configure_tracing
from acp.platform import install_event_loop_policy
from acp.worker.loop import Worker


def build_registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    register_demo_adapters(registry)
    register_tool_adapters(registry)
    return registry


async def main() -> None:
    s = settings()
    configure_logging(s.log_level, s.log_format)
    configure_tracing(
        service_name=f"{s.otel_service_name}-worker",
        otlp_endpoint=s.otel_endpoint,
        console=s.otel_console,
    )
    start_exporter(s.metrics_port)
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
