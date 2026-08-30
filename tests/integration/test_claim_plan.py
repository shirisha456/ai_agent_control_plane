"""A performance regression test for the claim query's execution plan.

Correctness tests cannot catch this. The claim query returns exactly the same
rows whether PostgreSQL walks idx_tasks_ready in order or sequentially scans
`tasks` and sorts the result -- so every other test in this repo passes either
way. The difference only shows up as a production incident once the table is
large enough, which is the worst possible time to find out.

What goes wrong, concretely: a Sort node cannot return its first row until it
has consumed its ENTIRE input. So a claim that wants 5 rows reads the whole
eligible set first, turning an O(batch) operation into O(queue depth) -- and
queue depth is largest exactly when the system is under pressure and claims
most need to be fast.

`enable_seqscan = off` is set so the test asks the right question. Without it,
on a small test table, a sequential scan is genuinely cheaper and the planner
correctly chooses one -- which would tell us nothing about the plan on a table
with a million rows. Disabling it forces the question we actually care about:
CAN the index serve this ordering, or does the planner still have to sort?
"""

from __future__ import annotations

import json

import pytest
import sqlalchemy as sa

from acp.scheduling.policy import DEFAULT_POLICY, READY_INDEX_COLUMNS

pytestmark = pytest.mark.db

READY_INDEX = "idx_tasks_ready"


def _walk(node: dict):
    yield node
    for child in node.get("Plans", []):
        yield from _walk(child)


async def _explain_claim(conn, policy=DEFAULT_POLICY) -> dict:
    """Plan the candidate SELECT the claim query issues.

    FOR UPDATE SKIP LOCKED is omitted: it adds a LockRows node above the scan
    but changes neither the ordering requirement nor the index choice, and
    EXPLAIN on a locking statement inside a test transaction is needlessly
    awkward.
    """
    await conn.execute(sa.text("SET LOCAL enable_seqscan = off"))
    sql = f"""
        EXPLAIN (FORMAT JSON, COSTS OFF)
        SELECT id, tenant_id
          FROM tasks
         WHERE state = 'QUEUED' AND available_at <= now()
         ORDER BY {policy.order_by_sql()}
         LIMIT 5
    """
    raw = (await conn.execute(sa.text(sql))).scalar_one()
    plan = json.loads(raw) if isinstance(raw, str) else raw
    return plan[0]["Plan"]


@pytest.fixture
async def populated(engine, make_task):
    """Enough rows, and current statistics, for the planner to be making a real choice."""
    for i in range(300):
        await make_task(priority=i % 5)
    async with engine.connect() as conn, conn.begin():
        await conn.execute(sa.text("ANALYZE tasks"))
    return True


async def test_claim_uses_the_ready_index(engine, populated) -> None:
    async with engine.connect() as conn, conn.begin():
        plan = await _explain_claim(conn)

    indexes = {n.get("Index Name") for n in _walk(plan) if n.get("Index Name")}
    assert READY_INDEX in indexes, (
        f"the claim scan did not use {READY_INDEX}; used {indexes or 'no index'}. "
        "The ready queue IS that index -- without it the claim scans the table."
    )


async def test_claim_plan_contains_no_sort(engine, populated) -> None:
    """The ordering must come from the index, not from a Sort node.

    This is the assertion that actually protects the hot path. Adding
    `created_at DESC` to the policy, or reordering its keys, would still
    return correct results and would still use the index -- with a Sort
    stacked on top, silently making every claim O(queue depth).
    """
    async with engine.connect() as conn, conn.begin():
        plan = await _explain_claim(conn)

    sorts = [n["Node Type"] for n in _walk(plan) if "Sort" in n["Node Type"]]
    assert not sorts, (
        f"claim plan contains {sorts}. A Sort must consume its entire input "
        "before returning the first row, so a 5-row claim would read the whole "
        "eligible set. Realign the policy with idx_tasks_ready "
        f"({', '.join(READY_INDEX_COLUMNS)}) or add an index that serves it."
    )


async def test_policy_ordering_matches_the_index_it_relies_on() -> None:
    """The static half of the same check -- no database required.

    Kept alongside the EXPLAIN test rather than replacing it: this one fails
    fast and explains WHY, the EXPLAIN one proves PostgreSQL agrees.
    """
    assert DEFAULT_POLICY.is_index_servable()
    assert tuple(key.value for key, _ in DEFAULT_POLICY.order_by) == READY_INDEX_COLUMNS


async def test_a_policy_the_index_cannot_serve_is_detected(engine, populated) -> None:
    """Prove the test can fail: a bad policy must actually produce a Sort.

    A regression test that cannot demonstrate the regression is decoration.
    """
    from acp.scheduling.policy import ClaimPolicy, Direction, SortKey

    bad = ClaimPolicy(
        name="created-at-first",
        order_by=((SortKey.CREATED_AT, Direction.DESC), (SortKey.PRIORITY, Direction.ASC)),
    )
    assert not bad.is_index_servable()

    async with engine.connect() as conn, conn.begin():
        plan = await _explain_claim(conn, policy=bad)

    sorts = [n["Node Type"] for n in _walk(plan) if "Sort" in n["Node Type"]]
    assert sorts, "expected a policy misaligned with the index to force a Sort node"
