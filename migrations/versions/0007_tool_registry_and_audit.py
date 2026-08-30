"""Phase 7: tool registry, versioned grants, and an audit log.

Still read-only on the execution path: the tool policy is snapshotted inside
the claim transaction and consulted from memory thereafter, so no lease,
fencing or CAS argument changes.

Revision ID: 0007
Revises: 0006
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None

TS = sa.TIMESTAMP(timezone=True)


def upgrade() -> None:
    op.execute("CREATE TYPE tool_status AS ENUM ('ACTIVE', 'DISABLED')")

    op.create_table(
        "tools",
        sa.Column(
            "id", pg.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")
        ),
        sa.Column(
            "tenant_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("description", sa.Text),
        # SIMULATED | HTTP, and that is the whole list. Building integrations
        # teaches nothing this project is about; what matters is that tools
        # are REGISTERED RESOURCES the control plane governs.
        sa.Column("tool_type", sa.Text, nullable=False, server_default="SIMULATED"),
        # Endpoint, timeouts, and a REFERENCE to a credential -- e.g.
        # {"secret_ref": "env:GITHUB_TOKEN"}. Never the credential itself.
        # A homegrown secret store would be worse than none: this column is
        # readable by anything with database access, and pretending otherwise
        # is how secrets end up in backups and logs.
        sa.Column("config", pg.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column(
            "status",
            pg.ENUM(name="tool_status", create_type=False),
            nullable=False,
            server_default="ACTIVE",
        ),
        sa.Column("created_at", TS, nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("tenant_id", "name", name="uq_tools_tenant_name"),
        sa.CheckConstraint("tool_type IN ('SIMULATED', 'HTTP')", name="ck_tools_type"),
    )

    # GRANTS ATTACH TO A VERSION, NOT AN AGENT.
    #
    # That makes a version a complete, self-describing capability bundle:
    # "what was this allowed to do?" is answered by the same immutable row
    # that answers "what did it run?". Widening an agent's reach then requires
    # cutting and releasing a new version -- a reviewable diff -- instead of an
    # INSERT that silently grants a running agent access to a new system.
    #
    # The cost is that revoking would need a new version, which is far too slow
    # for an incident. So revocation does not go through grants at all:
    # tools.status is the live kill switch. Static grants are versioned and
    # immutable; DENIAL IS ALWAYS LIVE. Same shape as certificate revocation.
    op.create_table(
        "agent_version_tool_grants",
        sa.Column(
            "agent_version_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("agent_versions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "tool_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("tools.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("created_at", TS, nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("agent_version_id", "tool_id", name="pk_agent_version_tool_grants"),
    )
    # The claim-time snapshot reads grants for a handful of version ids at
    # once; the PK's leading column already serves that. This one serves the
    # opposite question -- "which agents can reach this tool?" -- which is what
    # you ask during an incident, when it needs to be fast.
    op.create_index(
        "idx_grants_by_tool", "agent_version_tool_grants", ["tool_id", "agent_version_id"]
    )

    # AUDIT EVENTS, SEPARATE FROM task_events.
    #
    # The decisive argument is RETENTION. task_events are execution noise,
    # pruned after days, and they grow with throughput. Audit records must be
    # kept -- that is what audit means. One table cannot be both aggressively
    # pruned and permanently retained.
    #
    # Two supporting arguments: task_events.task_id is NOT NULL with an FK and
    # is half its index, but AGENT_REGISTERED has no task; and task_events are
    # written by the data plane inside hot transactions while these are
    # written by the control plane on rare admin actions -- different lock
    # profiles on one table for no benefit.
    op.create_table(
        "audit_events",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("tenant_id", pg.UUID(as_uuid=True), nullable=False),
        # Who did it. A principal name today; an authenticated identity when
        # there is an auth layer to supply one.
        sa.Column("actor", sa.Text, nullable=False, server_default="system"),
        sa.Column("action", sa.Text, nullable=False),
        sa.Column("resource_type", sa.Text, nullable=False),
        # No FK: audit records must outlive the rows they describe. An FK would
        # either block deleting a task or cascade away the record of what it
        # did -- both of which defeat the point.
        sa.Column("resource_id", pg.UUID(as_uuid=True)),
        sa.Column("outcome", sa.Text, nullable=False),
        sa.Column("data", pg.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", TS, nullable=False, server_default=sa.text("now()")),
    )
    # "What happened in this tenant, most recent first" -- the query an
    # operator actually runs.
    op.create_index(
        "idx_audit_tenant_time", "audit_events", ["tenant_id", sa.text("created_at DESC")]
    )
    # "Everything that touched this resource, in order."
    op.create_index("idx_audit_resource", "audit_events", ["resource_type", "resource_id", "id"])


def downgrade() -> None:
    op.drop_index("idx_audit_resource", table_name="audit_events")
    op.drop_index("idx_audit_tenant_time", table_name="audit_events")
    op.drop_table("audit_events")
    op.drop_index("idx_grants_by_tool", table_name="agent_version_tool_grants")
    op.drop_table("agent_version_tool_grants")
    op.drop_table("tools")
    op.execute("DROP TYPE tool_status")
