"""Capability-aware claiming, and the cost of doing it as a filter.

The correctness half is straightforward. The interesting half is the last two
tests, which measure the case where this design degrades -- because knowing
where a design breaks is worth more than asserting that it works.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest
import sqlalchemy as sa

from acp.db.models import tasks
from acp.db.queries.claim import claim_tasks
from acp.domain.agents import capability_key, worker_satisfies
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
    """The pathological case, made explicit.

    Capability matching is a FILTER on rows the ready index returns in
    priority order, not part of the index key. So a specialist worker facing a
    deep queue of work it cannot run scans the whole queue rejecting nearly
    everything. This asserts correctness (it does find its task) and records
    the cost, which is the number that justifies the keyed-index upgrade when
    it becomes necessary.
    """
    for _ in range(2000):
        await make_task(priority=50)
    needle = await make_task(
        priority=200,  # deliberately LAST in priority order
        required_capabilities=["gpu"],
        capability_key="gpu",
    )

    started = time.perf_counter()
    claimed = await _claim(engine, await make_worker(), caps=["gpu"], limit=1)
    elapsed = time.perf_counter() - started

    ids = [c["id"] for c in claimed]
    # The generalist tasks come first in priority order and this worker CAN run
    # them, so it takes one of those -- correct, and exactly why the needle is
    # not starved in practice.
    assert len(ids) == 1
    print(f"\n  specialist claim over 2000-row queue: {elapsed * 1000:.1f} ms")

    # Drain the generalist work so only the needle remains eligible.
    async with engine.connect() as conn, conn.begin():
        await conn.execute(
            sa.update(tasks)
            .where(tasks.c.id != needle, tasks.c.state == State.QUEUED.value)
            .values(state=State.SUCCEEDED.value, finished_at=sa.func.now())
        )

    started = time.perf_counter()
    found = [c["id"] for c in await _claim(engine, await make_worker(), caps=["gpu"], limit=1)]
    elapsed_needle = time.perf_counter() - started

    assert found == [needle], "the only eligible task was not found"
    print(f"  needle-only claim: {elapsed_needle * 1000:.1f} ms")


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
