"""SQLAlchemy Core table definitions.

Core, not ORM, deliberately. The hot paths in acp.db.queries need exact
control over FOR UPDATE SKIP LOCKED, CTE shape, and RETURNING; the ORM's
identity map and autoflush semantics are actively hostile to compare-and-set
patterns (a stale identity-mapped object will happily overwrite a row you no
longer own). The ORM may be introduced later for read models only.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

metadata = sa.MetaData()

task_state = pg.ENUM(
    "QUEUED",
    "RUNNING",
    "SUCCEEDED",
    "FAILED",
    "CANCELLED",
    name="task_state",
    create_type=False,
)
worker_status = pg.ENUM("ALIVE", "DRAINING", "DEAD", name="worker_status", create_type=False)
agent_status = pg.ENUM("ACTIVE", "DEPRECATED", "DISABLED", name="agent_status", create_type=False)
version_status = pg.ENUM(
    "DRAFT", "ACTIVE", "DEPRECATED", "DISABLED", name="version_status", create_type=False
)
attempt_outcome = pg.ENUM(
    "SUCCEEDED",
    "FAILED",
    "LOST",
    "ABANDONED",
    "CANCELLED",
    name="attempt_outcome",
    create_type=False,
)

TS = sa.TIMESTAMP(timezone=True)


tenants = sa.Table(
    "tenants",
    metadata,
    sa.Column(
        "id", pg.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")
    ),
    sa.Column("name", sa.Text, nullable=False, unique=True),
    # Execution limit: how many of this tenant's tasks may be RUNNING at once.
    # Enforced at CLAIM time, not at submit time -- exceeding it queues the
    # task rather than rejecting it, so a busy tenant degrades in latency
    # instead of erroring.
    sa.Column("max_concurrent_tasks", sa.Integer, nullable=False, server_default="10"),
    # Backpressure bound: how deep this tenant's queue may get before submits
    # are rejected with 429. This is what stops unbounded queue growth.
    sa.Column("max_queued_tasks", sa.Integer, nullable=False, server_default="1000"),
    sa.Column("priority_weight", sa.Integer, nullable=False, server_default="1"),
    sa.Column("created_at", TS, nullable=False, server_default=sa.text("now()")),
)


tasks = sa.Table(
    "tasks",
    metadata,
    sa.Column(
        "id", pg.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")
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
    sa.Column(
        "priority", sa.SmallInteger, nullable=False, server_default="100"
    ),  # lower = more urgent
    sa.Column("state", task_state, nullable=False, server_default="QUEUED"),
    # FENCING TOKEN. Monotonic, incremented by the same UPDATE that grants a
    # lease, so ownership and token can never disagree. Every ownership-scoped
    # write must present it. See docs/adr/0003.
    sa.Column("attempt", sa.Integer, nullable=False, server_default="0"),
    sa.Column("max_attempts", sa.Integer, nullable=False, server_default="3"),
    # One column serves both delayed submission and retry backoff. A separate
    # next_retry_at would be a second way to say the same thing, and the two
    # would eventually drift.
    sa.Column("available_at", TS, nullable=False, server_default=sa.text("now()")),
    # The lease, inlined rather than in its own table: lease grant and state
    # transition must be atomic anyway, so a separate row buys only a join on
    # the hottest query in the system. See docs/adr/0004.
    # No FOREIGN KEY to workers, deliberately. The lease pointer's integrity
    # comes from the fencing protocol, not from referential integrity: an
    # expired lease is reclaimed whether or not the worker row still exists,
    # so an FK would assert a fact the protocol does not rely on -- while
    # adding a constraint check to the single hottest write in the system.
    sa.Column("lease_worker_id", sa.Text),
    sa.Column("lease_expires_at", TS),
    # Pinned at submit from an immutable agent version, or left empty for a
    # direct task_type submission. Copied rather than joined so the claim
    # query never touches the registry -- safe precisely because the source
    # cannot change (migration 0006 enforces that with a trigger).
    sa.Column("agent_version_id", pg.UUID(as_uuid=True)),
    sa.Column(
        "required_capabilities",
        pg.ARRAY(sa.Text),
        nullable=False,
        server_default=sa.text("'{}'::text[]"),
    ),
    sa.Column("capability_key", sa.Text, nullable=False, server_default=""),
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
    # "A RUNNING task always has an owner and a deadline" is an invariant, so
    # the database enforces it. Any code path that forgets to clear the lease
    # on completion fails loudly here instead of leaving a task that the
    # reaper will never notice.
    sa.CheckConstraint(
        "(state = 'RUNNING') = (lease_worker_id IS NOT NULL AND lease_expires_at IS NOT NULL)",
        name="ck_tasks_lease_coherence",
    ),
)


task_events = sa.Table(
    "task_events",
    metadata,
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


workers = sa.Table(
    "workers",
    metadata,
    # Generation-unique: a fresh id per process start. See migration 0003.
    sa.Column("id", sa.Text, primary_key=True),
    sa.Column("hostname", sa.Text, nullable=False),
    sa.Column("pid", sa.Integer, nullable=False),
    sa.Column("capacity", sa.Integer, nullable=False),
    sa.Column(
        "capabilities", pg.ARRAY(sa.Text), nullable=False, server_default=sa.text("'{}'::text[]")
    ),
    sa.Column("status", worker_status, nullable=False, server_default="ALIVE"),
    sa.Column("registered_at", TS, nullable=False, server_default=sa.text("now()")),
    sa.Column("last_heartbeat_at", TS, nullable=False, server_default=sa.text("now()")),
    sa.CheckConstraint("capacity >= 1", name="ck_workers_capacity"),
)


task_attempts = sa.Table(
    "task_attempts",
    metadata,
    sa.Column(
        "task_id",
        pg.UUID(as_uuid=True),
        sa.ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("attempt", sa.Integer, nullable=False),
    sa.Column(
        "worker_id", sa.Text, sa.ForeignKey("workers.id", ondelete="RESTRICT"), nullable=False
    ),
    sa.Column("started_at", TS, nullable=False, server_default=sa.text("now()")),
    sa.Column("finished_at", TS),
    sa.Column("outcome", attempt_outcome),
    sa.Column("error_class", sa.Text),
    sa.Column("error_message", sa.Text),
    # (task_id, attempt) is a duplicate-execution guard, not merely a key.
    sa.PrimaryKeyConstraint("task_id", "attempt", name="pk_task_attempts"),
)


agents = sa.Table(
    "agents",
    metadata,
    sa.Column(
        "id", pg.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")
    ),
    sa.Column("tenant_id", pg.UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False),
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("description", sa.Text),
    sa.Column("status", agent_status, nullable=False, server_default="ACTIVE"),
    sa.Column("default_version_id", pg.UUID(as_uuid=True)),
    sa.Column("created_at", TS, nullable=False, server_default=sa.text("now()")),
    sa.Column("updated_at", TS, nullable=False, server_default=sa.text("now()")),
    sa.UniqueConstraint("tenant_id", "name", name="uq_agents_tenant_name"),
)


agent_versions = sa.Table(
    "agent_versions",
    metadata,
    sa.Column(
        "id", pg.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")
    ),
    sa.Column("agent_id", pg.UUID(as_uuid=True), sa.ForeignKey("agents.id"), nullable=False),
    sa.Column("version", sa.Integer, nullable=False),
    # Everything below except `status` is IMMUTABLE after insert, enforced by
    # a BEFORE UPDATE trigger in migration 0006.
    sa.Column("runtime_spec", pg.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column(
        "required_capabilities",
        pg.ARRAY(sa.Text),
        nullable=False,
        server_default=sa.text("'{}'::text[]"),
    ),
    sa.Column("config", pg.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    sa.Column("max_attempts", sa.Integer, nullable=False, server_default="3"),
    sa.Column("max_execution_time_s", sa.Integer, nullable=False, server_default="300"),
    sa.Column("status", version_status, nullable=False, server_default="DRAFT"),
    sa.Column("created_at", TS, nullable=False, server_default=sa.text("now()")),
    sa.UniqueConstraint("agent_id", "version", name="uq_agent_versions_agent_version"),
)


agent_routes = sa.Table(
    "agent_routes",
    metadata,
    sa.Column(
        "tenant_id",
        pg.UUID(as_uuid=True),
        sa.ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("request_type", sa.Text, nullable=False),
    sa.Column("agent_id", pg.UUID(as_uuid=True), sa.ForeignKey("agents.id"), nullable=False),
    sa.Column("created_at", TS, nullable=False, server_default=sa.text("now()")),
    sa.PrimaryKeyConstraint("tenant_id", "request_type", name="pk_agent_routes"),
)
