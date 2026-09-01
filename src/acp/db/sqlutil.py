"""Small SQL helpers shared across the query layer."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.sql.elements import ColumnElement


def seconds(value: float | int | ColumnElement) -> ColumnElement:
    """A duration, bound as a PARAMETER rather than interpolated into SQL.

    The obvious spelling, `sa.text(f"interval '{n} seconds'")`, works and is
    not exploitable while `n` only ever comes from config. But it is one
    refactor away from being exploitable -- the moment a caller passes a
    client-supplied backoff or TTL -- and the parameterised form costs
    nothing. `<number> * interval '1 second'` keeps the literal fixed and the
    variable bound, which is the property we actually want.

    `value` may also be a column (e.g. a per-row `max_execution_time_s`) --
    it is already a SQL expression, not a Python value, so it must not be
    re-wrapped in `sa.literal()`.
    """
    bound = value if isinstance(value, ColumnElement) else sa.literal(value)
    return bound * sa.text("interval '1 second'")
