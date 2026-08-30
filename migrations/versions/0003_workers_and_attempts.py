"""Phase 2: worker registry, execution attempts, and the ready-queue index.

Three additions, each with a query in this phase that needs it. Notably ABSENT:
the lease-expiry index and the worker-heartbeat index. Phase 2 has no failure
detection at all -- that is deliberate, so Phase 3's reaper has a real,
demonstrated bug to fix rather than a hypothetical one. Their indexes arrive
with the queries that read them.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

TS = sa.TIMESTAMP(timezone=True)


def upgrade() -> None:
    op.execute("CREATE TYPE worker_status AS ENUM ('ALIVE', 'DRAINING', 'DEAD')")
    op.execute(
        "CREATE TYPE attempt_outcome AS ENUM "
        "('SUCCEEDED', 'FAILED', 'LOST', 'ABANDONED', 'CANCELLED')"
    )

    op.create_table(
        "workers",
        # TEXT, not UUID, and GENERATION-UNIQUE: a fresh id per PROCESS START,
        # never per host. If a restarted worker reused hostname:pid or any
        # stable name, a zombie process and its restarted twin would share an
        # identity and the fencing check `lease_worker_id = :worker_id` would
        # accept writes from the zombie. This is the subtle bug that makes an
        # otherwise-correct lease implementation unsound.
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("hostname", sa.Text, nullable=False),
        sa.Column("pid", sa.Integer, nullable=False),
        # How many tasks this process will execute concurrently. The control
        # plane never pushes more than this because the worker never claims
        # more than this -- capacity is enforced where it is known.
        sa.Column("capacity", sa.Integer, nullable=False),
        sa.Column(
            "capabilities",
            pg.ARRAY(sa.Text),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        sa.Column(
            "status",
            pg.ENUM(name="worker_status", create_type=False),
            nullable=False,
            server_default="ALIVE",
        ),
        sa.Column("registered_at", TS, nullable=False, server_default=sa.text("now()")),
        sa.Column("last_heartbeat_at", TS, nullable=False, server_default=sa.text("now()")),
        sa.CheckConstraint("capacity >= 1", name="ck_workers_capacity"),
    )

    op.create_table(
        "task_attempts",
        sa.Column(
            "task_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("attempt", sa.Integer, nullable=False),
        # RESTRICT, not CASCADE: workers are never deleted, only marked DEAD.
        # If one ever were deleted, silently erasing the execution history that
        # references it is the last thing we want.
        sa.Column(
            "worker_id",
            sa.Text,
            sa.ForeignKey("workers.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("started_at", TS, nullable=False, server_default=sa.text("now()")),
        sa.Column("finished_at", TS),
        sa.Column("outcome", pg.ENUM(name="attempt_outcome", create_type=False)),
        sa.Column("error_class", sa.Text),
        sa.Column("error_message", sa.Text),
        # THE PRIMARY KEY IS A CORRECTNESS MECHANISM, not just an identifier.
        # Two workers physically cannot both record attempt 4 of a task. If the
        # claim path ever hands the same attempt number to two workers, this
        # constraint aborts the transaction loudly instead of letting duplicate
        # execution pass silently.
        sa.PrimaryKeyConstraint("task_id", "attempt", name="pk_task_attempts"),
    )

    # THE READY QUEUE. This index IS the queue -- there is no separate broker.
    #
    # Partial on state='QUEUED' because in a mature table the overwhelming
    # majority of rows are terminal. The partial index stays roughly the size
    # of the live queue depth rather than the table, so it stays in memory and
    # SHRINKS as tasks complete.
    #
    # The column order matches the claim query's ORDER BY exactly, so the plan
    # is an index scan with NO SORT NODE. That matters more than it sounds: a
    # sort would have to read the entire eligible set before it could return
    # the first 5 rows, turning an O(batch) claim into O(queue depth).
    # tests/integration/test_claim_plan.py asserts this with EXPLAIN.
    op.create_index(
        "idx_tasks_ready",
        "tasks",
        ["priority", "available_at", "id"],
        postgresql_where=sa.text("state = 'QUEUED'"),
    )


def downgrade() -> None:
    op.drop_index("idx_tasks_ready", table_name="tasks")
    op.drop_table("task_attempts")
    op.drop_table("workers")
    op.execute("DROP TYPE attempt_outcome")
    op.execute("DROP TYPE worker_status")
