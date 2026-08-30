"""Phase 3: indexes the reaper needs to find expired leases and dead workers.

Migration 0003 deliberately shipped without these -- Phase 2 had no failure
detection, so there was nothing yet to serve. Now there is.

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The reaper's candidate scan: RUNNING tasks whose lease has expired.
    # Partial on state='RUNNING' for the same reason idx_tasks_ready is
    # partial on QUEUED -- this index tracks the (small) set of in-flight
    # tasks, not the whole table, and shrinks as attempts finish.
    op.create_index(
        "idx_tasks_lease_expiry",
        "tasks",
        ["lease_expires_at"],
        postgresql_where=sa.text("state = 'RUNNING'"),
    )

    # The reaper's other candidate scan: ALIVE/DRAINING workers whose
    # heartbeat has gone stale. Partial on status <> 'DEAD' so a fleet's
    # history of dead workers never bloats the index that finds live ones.
    op.create_index(
        "idx_workers_heartbeat",
        "workers",
        ["last_heartbeat_at"],
        postgresql_where=sa.text("status <> 'DEAD'"),
    )


def downgrade() -> None:
    op.drop_index("idx_workers_heartbeat", table_name="workers")
    op.drop_index("idx_tasks_lease_expiry", table_name="tasks")
