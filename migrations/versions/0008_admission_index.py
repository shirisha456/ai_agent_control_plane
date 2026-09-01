"""Phase 9: index the per-tenant queue-depth check.

Admission counts a tenant's QUEUED tasks on every submit. Without this the
count is a sequential scan of `tasks`, so the check meant to protect the
system under load becomes part of the load -- and it degrades exactly when
the queue is deep, which is when it is most needed.

Revision ID: 0008
Revises: 0007
"""

import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The third partial index on `tasks`, and worth naming the cost: every
    # index is maintained on every claim, completion and retry -- the hottest
    # writes in the system. This one is paid for by making admission O(tenant
    # backlog) instead of O(table), which is the difference between a bound
    # that holds under pressure and one that collapses under it.
    op.create_index(
        "idx_tasks_tenant_queued",
        "tasks",
        ["tenant_id"],
        postgresql_where=sa.text("state = 'QUEUED'"),
    )


def downgrade() -> None:
    op.drop_index("idx_tasks_tenant_queued", table_name="tasks")
