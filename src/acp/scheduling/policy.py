"""Scheduling POLICY, separated from the claim MECHANISM.

The claim query in acp.db.queries.claim is the mechanism: SELECT ... FOR
UPDATE SKIP LOCKED, then an atomic UPDATE that grants a lease. This module is
the policy it executes: which tasks are eligible, and in what order.

Keeping them apart is what makes the scheduler evolvable. Phase 8 adds tenant
concurrency limits and agent capability matching by extending a policy object
here; the mechanism -- and its concurrency argument -- does not change. It is
also what lets fairness be unit-tested without a database or a cluster.

PURE MODULE. No I/O, no SQL strings from outside a closed enum. Ordering is
expressed as (SortKey, Direction) pairs rather than text so that a policy can
never inject SQL, and so that the compiled ORDER BY can be checked against the
index that has to serve it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SortKey(StrEnum):
    """Columns a policy may order by. Closed set, by design."""

    PRIORITY = "priority"
    AVAILABLE_AT = "available_at"
    CREATED_AT = "created_at"
    ID = "id"


class Direction(StrEnum):
    ASC = "ASC"
    DESC = "DESC"


#: Column order of idx_tasks_ready (migration 0003). A policy whose ORDER BY
#: does not prefix-match this list forces PostgreSQL to add a Sort node, which
#: must read the ENTIRE eligible set before returning the first row -- turning
#: an O(batch) claim into O(queue depth). tests/unit/test_policy.py checks the
#: default policy against this, and tests/integration/test_claim_plan.py
#: verifies it against a real query plan.
READY_INDEX_COLUMNS: tuple[str, ...] = ("priority", "available_at", "id")


@dataclass(frozen=True, slots=True)
class ClaimPolicy:
    name: str
    order_by: tuple[tuple[SortKey, Direction], ...]

    def order_by_sql(self) -> str:
        """Compile to a SQL fragment.

        Safe to interpolate: every component comes from a closed enum, so no
        caller-supplied text ever reaches the query.
        """
        return ", ".join(f"{key.value} {direction.value}" for key, direction in self.order_by)

    def is_index_servable(self) -> bool:
        """True when idx_tasks_ready can satisfy this ordering without a Sort."""
        columns = tuple(key.value for key, _ in self.order_by)
        directions = {direction for _, direction in self.order_by}
        return (
            columns == READY_INDEX_COLUMNS[: len(columns)]
            and columns == READY_INDEX_COLUMNS
            and directions == {Direction.ASC}
        )


#: Lowest priority number first, then oldest-available first, then id.
#:
#: `priority` first means an urgent task jumps a deep queue. `available_at`
#: second gives FIFO within a priority band -- and, for free, correct handling
#: of retry backoff, since a task waiting out its backoff simply has a future
#: available_at and is excluded by the eligibility predicate rather than by
#: any special state.
#:
#: `id` last is not decoration: without a total order, two workers scanning
#: concurrently can disagree about row order and interleave badly, and the
#: plan becomes non-deterministic between runs, which makes benchmarks
#: unreproducible.
#:
#: KNOWN LIMITATION, addressed in Phase 8: strict priority ordering can starve
#: low-priority work indefinitely if high-priority work never stops arriving.
#: Fair-share ordering and priority ageing are policy changes here, not
#: mechanism changes.
PRIORITY_FIFO = ClaimPolicy(
    name="priority-fifo",
    order_by=(
        (SortKey.PRIORITY, Direction.ASC),
        (SortKey.AVAILABLE_AT, Direction.ASC),
        (SortKey.ID, Direction.ASC),
    ),
)

DEFAULT_POLICY = PRIORITY_FIFO
