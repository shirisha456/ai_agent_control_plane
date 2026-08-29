"""Phase 1: index for the task listing endpoint.

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Serves GET /v1/tasks: a tenant's tasks, newest first, keyset-paginated
    # on (created_at DESC, id DESC). The column order matches the ORDER BY
    # exactly so the planner satisfies it from the index with no sort node.
    #
    # Deliberately ONE index, not two. A second index keyed on
    # (tenant_id, state, created_at) would make the state filter selective,
    # but every index on `tasks` is maintained on every claim, renewal and
    # completion -- the hottest writes in the system -- to speed up an
    # operator endpoint that runs a few times a minute. State stays a filter
    # until a benchmark shows the scan actually costs something.
    op.create_index(
        "idx_tasks_tenant_created",
        "tasks",
        ["tenant_id", sa.text("created_at DESC"), sa.text("id DESC")],
    )


def downgrade() -> None:
    op.drop_index("idx_tasks_tenant_created", table_name="tasks")
