"""Recommended-improvements phase: hung-task detection + keyed capability index.

Two independent additions, closing two gaps the project had documented
rather than hidden:

  1. tasks.max_execution_time_s -- pinned at submit exactly like max_attempts
     -- lets the reaper detect a worker that is still faithfully renewing its
     lease but is actually stuck (an infinite loop, a hung call with no
     timeout). Lease expiry alone can never catch this: the lease stays
     valid forever if the worker keeps renewing it.

  2. idx_tasks_ready_by_capability -- lets a specialist worker's claim use an
     ordered index range scan keyed on capability_key instead of a filtered
     walk of the whole priority-ordered queue. Measured need: a specialist
     claim over a 2,000-row generalist queue took ~47ms with the old filter
     approach (tests/integration/test_capability_scheduling.py) because
     capability matching was a FILTER applied after the ready index's scan,
     not part of the index key.

Revision ID: 0009
Revises: 0008
"""

import sqlalchemy as sa
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- hung-task detection -------------------------------------------------
    # Nullable-free with a default: existing rows get the same 300s default
    # agent_versions.max_execution_time_s already uses, so a task submitted
    # before this migration is treated exactly as if it had explicitly asked
    # for the platform default -- not as "no cap", which would silently
    # exempt old rows from the very check this migration adds.
    op.add_column(
        "tasks",
        sa.Column("max_execution_time_s", sa.Integer, nullable=False, server_default="300"),
    )
    op.create_check_constraint(
        "ck_tasks_max_execution_time", "tasks", "max_execution_time_s >= 1"
    )

    # --- keyed capability index ----------------------------------------------
    # A SECOND partial index on state='QUEUED', alongside idx_tasks_ready. Not
    # a replacement: idx_tasks_ready (priority, available_at, id) still serves
    # a worker with no declared capabilities (the common case) with a plain
    # ordered scan; this one serves the keyed-merge claim path used when a
    # worker's satisfiable-key count is small enough to make a per-key LATERAL
    # scan cheaper than the containment filter (see
    # domain.agents.satisfiable_capability_keys and db.queries.claim). Column
    # order matches the merge query's per-key ORDER BY exactly, so each
    # per-key branch is a Sort-free ordered range scan.
    op.create_index(
        "idx_tasks_ready_by_capability",
        "tasks",
        ["capability_key", "priority", "available_at", "id"],
        postgresql_where=sa.text("state = 'QUEUED'"),
    )


def downgrade() -> None:
    op.drop_index("idx_tasks_ready_by_capability", table_name="tasks")
    op.drop_constraint("ck_tasks_max_execution_time", "tasks", type_="check")
    op.drop_column("tasks", "max_execution_time_s")
