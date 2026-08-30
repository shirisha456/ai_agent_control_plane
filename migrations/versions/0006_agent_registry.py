"""Phase 6: agent registry, immutable agent versions, and request routing.

The execution engine does not change. Everything added here is READ-ONLY on
the execution path: resolution happens once at submit and its results are
copied onto the task row. Nothing in the claim, lease, completion or reaper
paths gains a join against these tables, which is why none of the concurrency
arguments in docs/ need revisiting.

Revision ID: 0006
Revises: 0005
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

TS = sa.TIMESTAMP(timezone=True)


def upgrade() -> None:
    op.execute("CREATE TYPE agent_status AS ENUM ('ACTIVE', 'DEPRECATED', 'DISABLED')")
    op.execute("CREATE TYPE version_status AS ENUM ('DRAFT', 'ACTIVE', 'DEPRECATED', 'DISABLED')")

    op.create_table(
        "agents",
        sa.Column(
            "id",
            pg.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        # Tenant-scoped, not global. This gives "tenant A may only run its own
        # agents" for free -- they can reference nothing else -- and makes
        # cross-tenant agent sharing an explicit future decision rather than an
        # accidental capability.
        sa.Column(
            "tenant_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("description", sa.Text),
        sa.Column(
            "status",
            pg.ENUM(name="agent_status", create_type=False),
            nullable=False,
            server_default="ACTIVE",
        ),
        # Which version new tasks resolve to. Nullable because an agent exists
        # before its first version does -- registering an agent and releasing
        # one are separate acts.
        sa.Column("default_version_id", pg.UUID(as_uuid=True)),
        sa.Column("created_at", TS, nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", TS, nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("tenant_id", "name", name="uq_agents_tenant_name"),
    )

    op.create_table(
        "agent_versions",
        sa.Column(
            "id",
            pg.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "agent_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer, nullable=False),
        # The executable definition. Kept as one JSONB document rather than a
        # step table: it is written once, read whole, and never queried by
        # part -- so a child table would add joins and a second immutability
        # surface for nothing.
        sa.Column("runtime_spec", pg.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column(
            "required_capabilities",
            pg.ARRAY(sa.Text),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        sa.Column("config", pg.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        # Execution policy travels WITH the version, so rolling back a version
        # rolls back its limits too. A task pinned to v3 keeps v3's retry
        # budget even after v4 changes it.
        sa.Column("max_attempts", sa.Integer, nullable=False, server_default="3"),
        sa.Column("max_execution_time_s", sa.Integer, nullable=False, server_default="300"),
        sa.Column(
            "status",
            pg.ENUM(name="version_status", create_type=False),
            nullable=False,
            server_default="DRAFT",
        ),
        sa.Column("created_at", TS, nullable=False, server_default=sa.text("now()")),
        # Version numbers are per agent and dense. Allocation takes a row lock
        # on the parent agent (see db/queries/agents.py); this constraint is
        # the backstop that turns a bug in that logic into a loud failure
        # instead of two versions silently sharing a number.
        sa.UniqueConstraint("agent_id", "version", name="uq_agent_versions_agent_version"),
        sa.CheckConstraint("version >= 1", name="ck_agent_versions_version"),
        sa.CheckConstraint("max_attempts >= 1", name="ck_agent_versions_max_attempts"),
        sa.CheckConstraint(
            "max_execution_time_s >= 1", name="ck_agent_versions_max_execution_time"
        ),
    )

    # Circular reference, so the FK is added after both tables exist.
    op.create_foreign_key(
        "fk_agents_default_version",
        "agents",
        "agent_versions",
        ["default_version_id"],
        ["id"],
        ondelete="RESTRICT",
    )

    # IMMUTABILITY, ENFORCED BY THE DATABASE.
    #
    # Reproducibility, reviewability and the safety of denormalising
    # required_capabilities onto `tasks` all rest on versions never changing.
    # An invariant three other things depend on should be unrepresentable, not
    # merely documented -- the same reasoning as ck_tasks_lease_coherence.
    #
    # `status` is deliberately mutable: that is the lifecycle (draft, release,
    # deprecate, emergency-stop), and it is exactly the part that must be
    # changeable without cutting a new version.
    op.execute("""
        CREATE FUNCTION acp_agent_versions_immutable() RETURNS trigger AS $$
        BEGIN
            IF NEW.agent_id             IS DISTINCT FROM OLD.agent_id
            OR NEW.version              IS DISTINCT FROM OLD.version
            OR NEW.runtime_spec         IS DISTINCT FROM OLD.runtime_spec
            OR NEW.required_capabilities IS DISTINCT FROM OLD.required_capabilities
            OR NEW.config               IS DISTINCT FROM OLD.config
            OR NEW.max_attempts         IS DISTINCT FROM OLD.max_attempts
            OR NEW.max_execution_time_s IS DISTINCT FROM OLD.max_execution_time_s
            THEN
                RAISE EXCEPTION
                    'agent_versions are immutable; only status may change (version id %)',
                    OLD.id
                    USING ERRCODE = 'restrict_violation';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_agent_versions_immutable
        BEFORE UPDATE ON agent_versions
        FOR EACH ROW EXECUTE FUNCTION acp_agent_versions_immutable();
    """)

    # Routing: request_type -> agent. This is the entire routing subsystem, and
    # it is a primary-key lookup on purpose. The interesting scheduling
    # decision is which WORKER runs a task; which AGENT handles a request type
    # is configuration, and dressing it up as more than that would be
    # pretending.
    op.create_table(
        "agent_routes",
        sa.Column(
            "tenant_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("request_type", sa.Text, nullable=False),
        sa.Column(
            "agent_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("created_at", TS, nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("tenant_id", "request_type", name="pk_agent_routes"),
    )

    # --- pinning on the task row -------------------------------------------
    #
    # NULLABLE, deliberately. Two submission modes coexist: direct
    # (task_type + payload, the primitive) and agent-routed (request_type
    # resolved to a pinned version, governance on top). Forcing every task
    # through the registry would break the direct path for no benefit -- and
    # a NULL here means exactly "this was submitted directly", which is
    # information, not a missing value.
    op.add_column("tasks", sa.Column("agent_version_id", pg.UUID(as_uuid=True)))
    op.create_foreign_key(
        "fk_tasks_agent_version",
        "tasks",
        "agent_versions",
        ["agent_version_id"],
        ["id"],
        ondelete="RESTRICT",
    )

    # COPIED from the pinned version at submit. Normally denormalisation risks
    # drift; here the source is immutable, so drift is impossible by
    # construction. The payoff is that Phase 8's capability matching never has
    # to join the registry from the claim query.
    op.add_column(
        "tasks",
        sa.Column(
            "required_capabilities",
            pg.ARRAY(sa.Text),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
    )
    op.add_column("tasks", sa.Column("capability_key", sa.Text, nullable=False, server_default=""))

    # Operator query: "what is agent X running right now?". Partial on the
    # active states so it tracks in-flight work rather than all history.
    op.create_index(
        "idx_tasks_agent_version_active",
        "tasks",
        ["agent_version_id"],
        postgresql_where=sa.text("state IN ('QUEUED', 'RUNNING')"),
    )


def downgrade() -> None:
    op.drop_index("idx_tasks_agent_version_active", table_name="tasks")
    op.drop_constraint("fk_tasks_agent_version", "tasks", type_="foreignkey")
    op.drop_column("tasks", "capability_key")
    op.drop_column("tasks", "required_capabilities")
    op.drop_column("tasks", "agent_version_id")
    op.drop_table("agent_routes")
    op.execute("DROP TRIGGER trg_agent_versions_immutable ON agent_versions")
    op.execute("DROP FUNCTION acp_agent_versions_immutable")
    op.drop_constraint("fk_agents_default_version", "agents", type_="foreignkey")
    op.drop_table("agent_versions")
    op.drop_table("agents")
    op.execute("DROP TYPE version_status")
    op.execute("DROP TYPE agent_status")
