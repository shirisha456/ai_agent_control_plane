"""Reaper process entrypoint: `python -m acp.reaper`."""

from __future__ import annotations

import asyncio
import logging
import signal

from acp.config import settings
from acp.db.session import dispose_engine
from acp.platform import install_event_loop_policy
from acp.reaper.loop import Reaper


async def main() -> None:
    logging.basicConfig(level=settings().log_level)
    reaper = Reaper(settings=settings())

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, reaper.stop)
        except NotImplementedError:
            pass  # Windows has no SIGTERM handler; SIGINT (Ctrl+C) still works.

    try:
        await reaper.run_forever()
    finally:
        await dispose_engine()


if __name__ == "__main__":
    install_event_loop_policy()
    asyncio.run(main())
