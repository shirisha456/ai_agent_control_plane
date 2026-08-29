"""Legality of the state machine. Pure -- no database, runs in milliseconds."""

from __future__ import annotations

import itertools

import pytest

from acp.domain.states import (
    ALLOWED,
    TERMINAL,
    IllegalTransition,
    State,
    assert_legal,
    is_legal,
    is_terminal,
)

LEGAL: set[tuple[State, State]] = {
    (State.QUEUED, State.RUNNING),
    (State.QUEUED, State.CANCELLED),
    (State.RUNNING, State.QUEUED),
    (State.RUNNING, State.SUCCEEDED),
    (State.RUNNING, State.FAILED),
    (State.RUNNING, State.CANCELLED),
}


@pytest.mark.parametrize(("frm", "to"), sorted(LEGAL))
def test_legal_transitions_are_allowed(frm: State, to: State) -> None:
    assert is_legal(frm, to)
    assert_legal(frm, to)


@pytest.mark.parametrize(
    ("frm", "to"),
    sorted(set(itertools.product(State, State)) - LEGAL),
)
def test_every_other_transition_is_rejected(frm: State, to: State) -> None:
    """Exhaustive: 25 ordered pairs, 6 legal, 19 must raise. No gaps by omission."""
    assert not is_legal(frm, to)
    with pytest.raises(IllegalTransition):
        assert_legal(frm, to)


@pytest.mark.parametrize("state", sorted(TERMINAL))
def test_terminal_states_have_no_exits(state: State) -> None:
    """Terminal means terminal. A completed task is never reopened; a rerun is
    a new task, so that its history stays a straight line."""
    assert ALLOWED[state] == frozenset()
    assert is_terminal(state)


def test_allowed_table_is_total() -> None:
    """Every state has an entry, so is_legal can never KeyError at runtime."""
    assert set(ALLOWED) == set(State)


def test_self_transitions_are_illegal() -> None:
    """RUNNING -> RUNNING would let a lease renewal masquerade as a claim."""
    for state in State:
        assert not is_legal(state, state)


def test_recovery_and_retry_share_one_transition() -> None:
    """Worker death and application failure both land on RUNNING -> QUEUED.

    One recovery path cannot disagree with itself; two eventually would. The
    *cause* is recorded in task_events, not in the state graph.
    """
    assert State.QUEUED in ALLOWED[State.RUNNING]
