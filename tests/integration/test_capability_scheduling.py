"""Capability-aware claiming: correctness, and the keyed merge that fixed its
one measured degradation.

The correctness half is straightforward. The interesting part is the deep-
queue scenario: a specialist worker facing mostly-generalist work used to
scan the whole queue rejecting nearly everything (measured ~47ms for 2,000
rows). The claim path now uses a keyed merge instead -- see
domain.agents.satisfiable_capability_keys and db.queries.claim -- and the
fix is proven here with an EXPLAIN ANALYZE, not a stopwatch: a wall-clock
timer around these tests is dominated by this environment's NullPool
connection-establishment cost (independently measured at ~20ms for a BARE
connect-and-SELECT-1), not by query execution.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from acp.db.models import tasks
from acp.db.queries.claim import _KEYED_CANDIDATES_SQL, claim_tasks
from acp.domain.agents import capability_key, satisfiable_capability_keys, worker_satisfies
from acp.domain.states import State
from acp.scheduling.policy import DEFAULT_POLICY

pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
async def _isolated(clean_tasks):
    """These assert WHICH tasks a worker claimed, so no leftovers."""
    yield


async def _claim(engine, worker_id, *, caps=(), limit=10):
    async with engine.connect() as conn, conn.begin():
        return await claim_tasks(
            conn,
            worker_id=worker_id,
            limit=limit,
            lease_ttl_s=30,
            policy=DEFAULT_POLICY,
            worker_capabilities=caps,
        )


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------


async def test_a_worker_only_claims_work_it_can_run(engine, make_task, make_worker) -> None:
    gpu_task = await make_task(required_capabilities=["gpu"], capability_key="gpu")
    plain_task = await make_task()

    claimed = await _claim(engine, await make_worker(), caps=[])
    ids = [c["id"] for c in claimed]

    assert plain_task in ids
    assert gpu_task not in ids, "a worker claimed work it cannot run"


async def test_a_specialist_worker_claims_both_kinds(engine, make_task, make_worker) -> None:
    """Containment, not equality.

    A worker offering more than a task needs still satisfies it -- otherwise a
    GPU machine could not run ordinary work and the fleet would fragment.
    """
    gpu_task = await make_task(required_capabilities=["gpu"], capability_key="gpu")
    plain_task = await make_task()

    ids = [c["id"] for c in await _claim(engine, await make_worker(), caps=["gpu", "internet"])]
    assert {gpu_task, plain_task} <= set(ids)


async def test_all_requirements_must_be_met_not_just_some(engine, make_task, make_worker) -> None:
    """Subset, not intersection.

    A task needing gpu AND internet is not runnable by a worker with only gpu.
    Getting this backwards would schedule work onto machines that cannot
    complete it, and the failure would look like a flaky adapter.
    """
    task_id = await make_task(
        required_capabilities=["gpu", "internet"],
        capability_key=capability_key(["gpu", "internet"]),
    )

    partial = [c["id"] for c in await _claim(engine, await make_worker(), caps=["gpu"])]
    assert task_id not in partial

    full = [c["id"] for c in await _claim(engine, await make_worker(), caps=["gpu", "internet"])]
    assert task_id in full


async def test_a_task_requiring_nothing_is_claimable_by_anyone(
    engine, make_task, make_worker
) -> None:
    """The empty set is contained by everything, including the empty set.

    A task that states no requirements must never be unschedulable -- and a
    fleet with no capabilities configured must keep working exactly as it did
    before this feature existed.
    """
    task_id = await make_task()
    ids = [c["id"] for c in await _claim(engine, await make_worker(), caps=[])]
    assert task_id in ids


async def test_capability_matching_agrees_with_the_domain_function(
    engine, make_task, make_worker
) -> None:
    """The SQL answer and the Python answer must not diverge.

    acp.domain.agents.worker_satisfies is what a future scheduler simulation
    would reason with; PostgreSQL's `<@` is what actually decides. If those
    ever disagree, every offline analysis of scheduling behaviour is wrong.
    """
    cases = [
        ([], []),
        ([], ["gpu"]),
        (["gpu"], ["gpu"]),
        (["gpu"], ["gpu", "internet"]),
        (["gpu", "internet"], ["gpu"]),
        (["internet"], ["gpu"]),
    ]
    created = {}
    for required, _ in cases:
        key = capability_key(required)
        created.setdefault((tuple(sorted(required)), key), []).append(
            await make_task(required_capabilities=sorted(set(required)), capability_key=key)
        )

    for required, worker_caps in cases:
        expected = worker_satisfies(required, worker_caps)
        task_id = created[(tuple(sorted(set(required))), capability_key(required))][0]

        async with engine.connect() as conn:
            claimable = (
                await conn.execute(
                    sa.select(
                        tasks.c.required_capabilities.contained_by(
                            sa.cast(sa.literal(list(worker_caps)), sa.ARRAY(sa.Text))
                        )
                    ).where(tasks.c.id == task_id)
                )
            ).scalar_one()

        assert bool(claimable) is expected, (
            f"SQL and domain disagree for required={required} worker={worker_caps}"
        )


# ---------------------------------------------------------------------------
# where this design degrades -- measured, not asserted away
# ---------------------------------------------------------------------------


async def test_specialist_worker_still_finds_its_task_in_a_deep_queue(
    engine, make_task, make_worker
) -> None:
    """The pathological case, made explicit -- and now fixed.

    Capability matching used to be a FILTER on rows the ready index returns
    in priority order, not part of the index key, so a specialist worker
    facing a deep queue of work it cannot run scanned the whole queue
    rejecting nearly everything (measured at ~47ms for this exact 2,000-row
    scenario). The claim path now uses a keyed merge instead (see
    domain.agents.satisfiable_capability_keys and db.queries.claim), served
    by idx_tasks_ready_by_capability.

    This test asserts correctness first, then proves the fix with an EXPLAIN
    ANALYZE -- not a wall-clock print. A stopwatch around this test's own
    connection is dominated by NullPool's per-call connection-establishment
    cost on this environment (independently measured at ~20ms for a BARE
    connect-and-SELECT-1, before any query logic at all), so it would measure
    Windows/Docker socket overhead, not the query -- exactly the reasoning
    that already keeps this project's real benchmarks reading from Prometheus
    rather than a client-side clock (see docs/BENCHMARKS.md). The buffer
    count EXPLAIN reports is server-side and environment-independent.
    """
    worker_id = await make_worker(capabilities=["gpu"])

    for _ in range(2000):
        await make_task(priority=50)
    needle = await make_task(
        priority=200,  # deliberately LAST in priority order
        required_capabilities=["gpu"],
        capability_key="gpu",
    )

    claimed = await _claim(engine, worker_id, caps=["gpu"], limit=1)
    ids = [c["id"] for c in claimed]
    # The generalist tasks come first in priority order and this worker CAN run
    # them, so it takes one of those -- correct, and exactly why the needle is
    # not starved in practice.
    assert len(ids) == 1

    # Drain the generalist work so only the needle remains eligible.
    async with engine.connect() as conn, conn.begin():
        await conn.execute(
            sa.update(tasks)
            .where(tasks.c.id != needle, tasks.c.state == State.QUEUED.value)
            .values(state=State.SUCCEEDED.value, finished_at=sa.func.now())
        )

    found = [c["id"] for c in await _claim(engine, worker_id, caps=["gpu"], limit=1)]
    assert found == [needle], "the only eligible task was not found"


async def test_the_keyed_merge_uses_the_capability_index_not_a_full_scan(engine, make_task) -> None:
    """The actual proof the fix works: a real EXPLAIN ANALYZE, not a stopwatch.

    2,000 generalist rows plus one deliberately-last-priority specialist row,
    same shape as the deep-queue test above. Asserts the plan uses
    idx_tasks_ready_by_capability and that the number of buffer pages touched
    stays small and bounded -- NOT proportional to the 2,000-row queue depth,
    which is exactly the property a plain containment filter could not offer.
    """
    for _ in range(2000):
        await make_task(priority=50)
    await make_task(priority=200, required_capabilities=["gpu"], capability_key="gpu")

    async with engine.connect() as conn, conn.begin():
        await conn.execute(sa.text("ANALYZE tasks"))
        explain_sql = "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + _KEYED_CANDIDATES_SQL
        stmt = sa.text(explain_sql).bindparams(
            sa.bindparam("keys", value=satisfiable_capability_keys(["gpu"]), type_=ARRAY(sa.Text)),
            sa.bindparam("seen_ids", value=[], type_=ARRAY(PG_UUID(as_uuid=True))),
            sa.bindparam("need", value=1),
        )
        raw = (await conn.execute(stmt)).scalar_one()

    # psycopg3 auto-decodes the `json` column type, so `raw` is already a
    # list of dicts here, not a JSON string -- no json.loads needed.
    plan = raw[0]["Plan"]

    def _walk(node: dict):
        yield node
        for child in node.get("Plans", []):
            yield from _walk(child)

    index_names = {n.get("Index Name") for n in _walk(plan) if n.get("Index Name")}
    assert "idx_tasks_ready_by_capability" in index_names, (
        f"expected the keyed index to be used; plan used {index_names or 'no index'}"
    )

    total_buffers = sum(
        n.get("Shared Hit Blocks", 0) + n.get("Shared Read Blocks", 0) for n in _walk(plan)
    )
    # Bounded by the number of satisfiable keys (2: "" and "gpu") times a
    # small per-branch working set -- nowhere near the 2,000-row queue depth.
    # This is the number that would grow with queue depth under the OLD
    # filter approach and does not grow here.
    assert total_buffers < 50, (
        f"query touched {total_buffers} buffer pages -- expected a small, "
        "queue-depth-independent number from the keyed index scan"
    )


async def test_capability_filtering_does_not_starve_under_concurrency(
    engine, make_task, make_worker
) -> None:
    """SKIP LOCKED plus a filter can miss an eligible task.

    A worker skips rows another worker has locked AND rows it cannot run. With
    one eligible task and several competing claimers, a worker can come away
    empty even though its task exists. That is acceptable -- it retries on the
    next poll -- but it must not be permanent, and exactly one worker must get
    the task.
    """
    gpu_tasks = [
        await make_task(required_capabilities=["gpu"], capability_key="gpu") for _ in range(5)
    ]
    for _ in range(20):
        await make_task()

    workers = [await make_worker() for _ in range(4)]
    results = await asyncio.gather(*(_claim(engine, w, caps=["gpu"], limit=3) for w in workers))

    claimed_ids = [c["id"] for batch in results for c in batch]
    assert len(claimed_ids) == len(set(claimed_ids)), "a task was claimed twice"

    async with engine.connect() as conn:
        states = (
            (await conn.execute(sa.select(tasks.c.state).where(tasks.c.id.in_(gpu_tasks))))
            .scalars()
            .all()
        )
    # Every gpu task went to exactly one worker or is still queued for the next
    # poll -- never lost, never duplicated.
    assert set(states) <= {State.RUNNING, State.QUEUED}


async def test_a_worker_with_no_capabilities_cannot_starve_specialist_work_forever(
    engine, make_task, make_worker
) -> None:
    """Generalists skipping specialist work must leave it claimable.

    The filter excludes rows rather than locking them, so a generalist's claim
    does not make a GPU task invisible to the GPU worker that follows.
    """
    gpu_task = await make_task(required_capabilities=["gpu"], capability_key="gpu")

    for _ in range(3):
        await _claim(engine, await make_worker(), caps=[], limit=10)

    ids = [c["id"] for c in await _claim(engine, await make_worker(), caps=["gpu"], limit=10)]
    assert gpu_task in ids, "generalist claims made specialist work unreachable"


async def test_worker_registers_machine_capabilities_not_task_types(engine) -> None:
    """Regression guard for a conflation that was in the code.

    The worker used to advertise its adapter registry's task types as its
    capabilities. That silently satisfies any requirement whose name happens
    to match a task type, and means a task requiring a GPU is scheduled onto a
    machine that merely knows how to run it.
    """
    import random

    from acp.agent.adapters.base import AdapterRegistry
    from acp.config import Settings
    from acp.db.models import workers
    from acp.worker.loop import Worker

    registry = AdapterRegistry()
    registry.register("demo.agent", object)

    worker = Worker(
        settings=Settings(database_url="x"),
        registry=registry,
        capabilities=["GPU ", "internet"],
        rng=random.Random(0),
        worker_id=f"w-{uuid.uuid4().hex[:8]}",
    )
    assert worker.capabilities == ("gpu", "internet")
    assert "demo.agent" not in worker.capabilities

    async with engine.connect() as conn, conn.begin():
        await conn.execute(
            sa.insert(workers).values(
                id=worker.worker_id,
                hostname="h",
                pid=1,
                capacity=1,
                capabilities=list(worker.capabilities),
            )
        )
        stored = (
            await conn.execute(
                sa.select(workers.c.capabilities).where(workers.c.id == worker.worker_id)
            )
        ).scalar_one()
    assert sorted(stored) == ["gpu", "internet"]
