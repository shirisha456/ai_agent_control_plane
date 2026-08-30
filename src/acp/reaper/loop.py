"""The reaper process: find expired leases, recover or fail their tasks, mark dead workers.

Runs standalone from any worker -- a worker cannot detect its own death, so
this has to be an independent process on its own clock. Recovery latency for
a crashed worker is bounded by lease_ttl_s + reaper_period_s (see
acp.config.Settings), because a task cannot even become a reap candidate
until its lease expires.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from acp.config import Settings
from acp.db.queries.reap import reap_expired_leases
from acp.db.queries.workers import mark_dead_workers
from acp.db.session import transaction

logger = logging.getLogger("acp.reaper")


class Reaper:
    def __init__(self, *, settings: Settings, batch_size: int = 100) -> None:
        self.settings = settings
        self.batch_size = batch_size
        self._stopping = asyncio.Event()

    def stop(self) -> None:
        self._stopping.set()

    async def run_forever(self) -> None:
        while not self._stopping.is_set():
            await self._sweep()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=self.settings.reaper_period_s)

    async def _sweep(self) -> None:
        # Drain the whole backlog, not just one batch: recovery latency is
        # documented as lease_ttl_s + reaper_period_s, which only holds if a
        # burst of simultaneous failures doesn't get throttled to
        # batch_size-per-period.
        total_reaped = 0
        while not self._stopping.is_set():
            async with transaction() as conn:
                reaped = await reap_expired_leases(conn, limit=self.batch_size)
            total_reaped += reaped
            if reaped < self.batch_size:
                break
        if total_reaped:
            logger.warning("reaper.leases_reclaimed count=%d", total_reaped)

        async with transaction() as conn:
            dead = await mark_dead_workers(conn, dead_after_s=self.settings.worker_dead_after_s)
        if dead:
            logger.warning("reaper.workers_marked_dead ids=%s", list(dead))
