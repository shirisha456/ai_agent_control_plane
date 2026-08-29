"""Idempotency enforced by a partial unique index, not by read-then-write.

The API layer arrives later in Phase 1; this proves the database-level
mechanism it will rely on. Checking "does a task with this key exist?" before
inserting is a textbook race: two API replicas both read absent, both insert,
and the tenant gets its task executed twice.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from acp.db.models import tasks

pytestmark = pytest.mark.db


async def _submit(engine, tenant_id, key):
    async with engine.connect() as conn, conn.begin():
        try:
            return (
                await conn.execute(
                    sa.insert(tasks)
                    .values(
                        tenant_id=tenant_id,
                        task_type="demo.agent",
                        payload={},
                        idempotency_key=key,
                    )
                    .returning(tasks.c.id)
                )
            ).scalar_one()
        except IntegrityError:
            return None


async def test_concurrent_duplicate_submits_create_one_task(engine, tenant_id) -> None:
    key = f"idem-{uuid.uuid4().hex}"
    results = await asyncio.gather(*(_submit(engine, tenant_id, key) for _ in range(25)))

    created = [r for r in results if r is not None]
    assert len(created) == 1, "the unique index must admit exactly one writer"

    async with engine.connect() as conn:
        count = (
            await conn.execute(
                sa.select(sa.func.count()).select_from(tasks).where(tasks.c.idempotency_key == key)
            )
        ).scalar_one()
    assert count == 1


async def test_null_keys_do_not_collide(engine, tenant_id) -> None:
    """Tasks without an idempotency key must never dedupe against each other.

    This is part of why the index is PARTIAL. A plain unique index would
    technically allow it (NULLs compare distinct) but would still carry an
    entry for every keyless row -- paying index maintenance on the write path
    for rows it cannot constrain.
    """
    ids = await asyncio.gather(*(_submit(engine, tenant_id, None) for _ in range(10)))
    assert len(set(ids)) == 10
    assert None not in ids
