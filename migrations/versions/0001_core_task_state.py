"""Phase 1: tenants, tasks, task_events -- durable task state.

Only the indexes Phase 1 actually queries are created here. The ready-queue
index, the lease-expiry index and the tenant-concurrency index arrive in the
migrations for the phases that introduce those queries. An index without a
query that needs it is a guess.

Revision ID: 0001
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

TS = sa.TIMESTAMP(timezone=True)


def upgrade() -> None:
    # gen_random_uuid() -- generating ids in the database means a retried
    # INSERT cannot invent a second id for the same logical submission.
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # A native enum rather than TEXT + CHECK. The state set is deliberately
    # closed (see acp.domain.states), so the cost of ALTER TYPE to extend it
    # is a feature: adding a state should require a migration and a moment's
    # thought, not a string literal.
    op.execute(
        "CREATE TYPE task_state AS ENUM ('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED')"
    )
    task_state = pg.ENUM(name="task_state", create_type=False)

    op.create_table(
        "tenants",
        sa.Column(
            "id",
            pg.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.Text, nullable=False, unique=True),
        sa.Column("max_concurrent_tasks", sa.Integer, nullable=False, server_default="10"),
        sa.Column("max_queued_tasks", sa.Integer, nullable=False, server_default="1000"),
        sa.Column("priority_weight", sa.Integer, nullable=False, server_default="1"),
        sa.Column("created_at", TS, nullable=False, server_default=sa.text("now()")),
    )

    op.create_table(
        "tasks",
        sa.Column(
            "id",
            pg.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "tenant_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("task_type", sa.Text, nullable=False),
        sa.Column("payload", pg.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("idempotency_key", sa.Text),
        sa.Column("priority", sa.SmallInteger, nullable=False, server_default="100"),
        sa.Column("state", task_state, nullable=False, server_default="QUEUED"),
        sa.Column("attempt", sa.Integer, nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer, nullable=False, server_default="3"),
        sa.Column("available_at", TS, nullable=False, server_default=sa.text("now()")),
        sa.Column("lease_worker_id", sa.Text),
        sa.Column("lease_expires_at", TS),
        sa.Column("cancel_requested", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("result", pg.JSONB),
        sa.Column("error_class", sa.Text),
        sa.Column("error_message", sa.Text),
        sa.Column("created_at", TS, nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", TS, nullable=False, server_default=sa.text("now()")),
        sa.Column("first_started_at", TS),
        sa.Column("finished_at", TS),
        sa.CheckConstraint("max_attempts >= 1", name="ck_tasks_max_attempts"),
        sa.CheckConstraint("attempt >= 0", name="ck_tasks_attempt_nonneg"),
        sa.CheckConstraint("priority >= 0", name="ck_tasks_priority_nonneg"),
        sa.CheckConstraint(
            "(state = 'RUNNING') = (lease_worker_id IS NOT NULL AND lease_expires_at IS NOT NULL)",
            name="ck_tasks_lease_coherence",
        ),
    )

    # IDEMPOTENT SUBMIT. Enforced by the database, not by a read-then-write in
    # the API, which is a race. Partial because most tasks carry no key, and a
    # non-partial unique index would treat every NULL as distinct anyway --
    # paying for rows it cannot constrain. Scoped by tenant so tenants can
    # neither collide with nor probe each other's keys.
    op.create_index(
        "idx_tasks_idem",
        "tasks",
        ["tenant_id", "idempotency_key"],
        unique=True,
        postgresql_where=sa.text("idempotency_key IS NOT NULL"),
    )

    op.create_table(
        "task_events",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "task_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("attempt", sa.Integer),
        sa.Column("event_type", sa.Text, nullable=False),
        sa.Column("worker_id", sa.Text),
        sa.Column("data", pg.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", TS, nullable=False, server_default=sa.text("now()")),
    )

    # Timeline read for one task: "what happened to this, in order". Ordering
    # by the BIGSERIAL rather than created_at because several events inside
    # one transaction share a created_at (now() is transaction-start time) and
    # would otherwise sort non-deterministically.
    op.create_index("idx_events_task", "task_events", ["task_id", "id"])


def downgrade() -> None:
    op.drop_table("task_events")
    op.drop_index("idx_tasks_idem", table_name="tasks")
    op.drop_table("tasks")
    op.drop_table("tenants")
    op.execute("DROP TYPE task_state")
