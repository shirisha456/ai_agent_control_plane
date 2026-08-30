"""Phase 3 follow-up: index the tenant concurrency check.

The claim path reads each candidate tenant's remaining slack as
`max_concurrent_tasks - count(*) WHERE tenant_id = ? AND state = 'RUNNING'`.
That count ran on EVERY claim -- and on every round within a claim -- with
nothing to serve it, so the hottest query in the system was doing a
sequential scan of `tasks` to answer it. The cost grows with total table
size, so it stays invisible in tests and appears in production.

Revision ID: 0005
Revises: 0004
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Partial on state='RUNNING' for the same reason as the other two hot
    # indexes: it tracks the small in-flight set rather than the whole table,
    # so it stays in memory and shrinks as attempts finish. This makes the
    # slack count an index-only scan over a handful of rows.
    #
    # KNOWN CEILING: counting on the hot path is O(running tasks per tenant)
    # per claim. That is fine at the concurrency this project benchmarks, and
    # the replacement -- a per-tenant counter row updated in the claim
    # transaction -- trades the scan for a per-tenant hot row that serialises
    # that tenant's claims. Measure before switching.
    op.create_index(
        "idx_tasks_tenant_running",
        "tasks",
        ["tenant_id"],
        postgresql_where=sa.text("state = 'RUNNING'"),
    )


def downgrade() -> None:
    op.drop_index("idx_tasks_tenant_running", table_name="tasks")
