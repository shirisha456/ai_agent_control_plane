"""Reaper process entrypoint: `python -m acp.reaper`."""

from __future__ import annotations

import asyncio
import contextlib
import signal

from acp.config import settings
from acp.db.session import dispose_engine
from acp.obs.exporter import start_exporter
from acp.obs.logging import configure_logging
from acp.platform import install_event_loop_policy
from acp.reaper.loop import Reaper


async def main() -> None:
    s = settings()
    configure_logging(s.log_level, s.log_format)
    start_exporter(s.metrics_port)
    reaper = Reaper(settings=settings())

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        # Windows has no SIGTERM handler; SIGINT (Ctrl+C) still works.
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, reaper.stop)

    try:
        await reaper.run_forever()
    finally:
        await dispose_engine()


if __name__ == "__main__":
    install_event_loop_policy()
    asyncio.run(main())
