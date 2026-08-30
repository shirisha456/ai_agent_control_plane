"""Agent definitions: what a task is, as opposed to how it runs.

PURE MODULE. No I/O.

THE DISTINCTION THIS DRAWS
--------------------------
Until now a task carried a `task_type` string and a payload, and the worker
looked the type up in its adapter registry. That is enough to *execute* work.
It is not enough to *govern* it, because nothing records WHICH definition ran:
change the step list, and every task that already ran becomes unexplainable.

An agent version is an immutable, addressable answer to "what exactly
executed this?". Tasks pin one at submit time and never re-resolve.

WHY IMMUTABLE
-------------
Three properties fall out of it, and none survive without it:

  reproducible   task 123 ran research-agent v3, and v3 is still exactly what
                 it was when task 123 ran.
  reviewable     changing what an agent does means cutting a new version,
                 which is a diff someone can look at -- rather than an UPDATE
                 that silently widens what a running agent may do.
  cheap on the   because the definition cannot drift, its fields can be COPIED
  hot path       onto the task row at submit and trusted forever. That is what
                 keeps the claim query free of joins against the registry.
                 Denormalisation is only dangerous when the source can change.

Enforced by a database trigger, not by convention -- see migration 0006. Only
`status` may change after insert.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum


class AgentStatus(StrEnum):
    """Lifecycle of the agent as a whole."""

    ACTIVE = "ACTIVE"
    #: Still runnable for tasks already pinned to it, but not routable to.
    DEPRECATED = "DEPRECATED"
    #: Disabling triggers a cancellation sweep over the agent's live tasks.
    #: The claim query is deliberately NOT taught to check agent status: that
    #: would put a join against the registry on the hottest query in the
    #: system to serve an admin action that happens once a month. Reusing
    #: cancellation costs nothing and already handles a dead owner.
    DISABLED = "DISABLED"


class VersionStatus(StrEnum):
    """Lifecycle of one immutable version."""

    #: Created but not routable. Lets a version be reviewed before traffic
    #: reaches it.
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    #: Not routable to; existing pinned tasks still run. This is what makes
    #: pinning safe to live with -- deprecating never breaks in-flight work.
    DEPRECATED = "DEPRECATED"
    #: Emergency stop. Unlike DEPRECATED, this is checked when resolving and
    #: is the lever for "stop using this version right now".
    DISABLED = "DISABLED"


#: Versions a route may resolve to. DRAFT is excluded so creating a version is
#: not the same act as releasing it.
ROUTABLE_VERSION_STATUSES: frozenset[VersionStatus] = frozenset({VersionStatus.ACTIVE})

#: Agent states a route may resolve to.
ROUTABLE_AGENT_STATUSES: frozenset[AgentStatus] = frozenset({AgentStatus.ACTIVE})


def capability_key(capabilities: Iterable[str]) -> str:
    """Canonical, readable digest of a required-capability set.

    Sorted and comma-joined rather than hashed. A hash would be opaque in
    `psql` and on a dashboard for no benefit: the value's job is to be an
    equality key that groups tasks with identical requirements, and the number
    of distinct sets is bounded by the number of agent versions -- a handful,
    not a namespace needing collision resistance.

    Computed at submit and stored on the task, so the claim query can one day
    filter and ORDER BY it from a single index instead of evaluating array
    containment per candidate row. (Phase 8; the column exists now so a hot
    table is not rewritten twice.)

    An empty set means "any worker can run this", which is the right default:
    a task with no stated requirements should never be unschedulable.
    """
    normalised = sorted({c.strip().lower() for c in capabilities if c and c.strip()})
    return ",".join(normalised)


def worker_satisfies(required: Iterable[str], worker_capabilities: Iterable[str]) -> bool:
    """Whether a worker may run a task requiring `required`.

    Subset containment, in the domain layer so Phase 8's scheduling policy can
    be unit-tested without a database -- and so the Python answer and the SQL
    answer are checkable against each other.
    """
    have = {c.strip().lower() for c in worker_capabilities if c and c.strip()}
    need = {c.strip().lower() for c in required if c and c.strip()}
    return need.issubset(have)
