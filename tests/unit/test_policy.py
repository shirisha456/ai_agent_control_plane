"""Pure tests for the claim policy: SQL compilation and index-servability.

No database: this is exactly what acp.scheduling.policy being a pure module
buys.
"""

from __future__ import annotations

from acp.scheduling.policy import (
    PRIORITY_FIFO,
    READY_INDEX_COLUMNS,
    ClaimPolicy,
    Direction,
    SortKey,
)


def test_default_policy_matches_ready_index_column_order() -> None:
    assert tuple(key.value for key, _ in PRIORITY_FIFO.order_by) == READY_INDEX_COLUMNS


def test_default_policy_is_index_servable() -> None:
    assert PRIORITY_FIFO.is_index_servable()


def test_order_by_sql_compiles_in_declared_order() -> None:
    assert PRIORITY_FIFO.order_by_sql() == "priority ASC, available_at ASC, id ASC"


def test_reordered_columns_are_not_index_servable() -> None:
    reordered = ClaimPolicy(
        name="broken",
        order_by=(
            (SortKey.AVAILABLE_AT, Direction.ASC),
            (SortKey.PRIORITY, Direction.ASC),
            (SortKey.ID, Direction.ASC),
        ),
    )
    assert not reordered.is_index_servable()


def test_partial_prefix_is_not_index_servable() -> None:
    partial = ClaimPolicy(name="partial", order_by=((SortKey.PRIORITY, Direction.ASC),))
    assert not partial.is_index_servable()


def test_descending_direction_is_not_index_servable() -> None:
    desc = ClaimPolicy(
        name="desc",
        order_by=(
            (SortKey.PRIORITY, Direction.DESC),
            (SortKey.AVAILABLE_AT, Direction.ASC),
            (SortKey.ID, Direction.ASC),
        ),
    )
    assert not desc.is_index_servable()
