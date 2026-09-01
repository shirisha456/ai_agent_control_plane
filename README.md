# AI Agent Control Plane

Distributed infrastructure for scheduling and reliably executing AI agent workloads across a dynamic fleet of worker processes.

The project focuses particularly deeply on **execution correctness under concurrency and partial failure** — lease-based ownership, fencing tokens, compare-and-set state transitions, and bounded failure recovery. AI agents are the workload; distributed systems are the project.

> **Status:** all 11 phases are implemented and benchmarked. See [BENCHMARKS.md](docs/BENCHMARKS.md) for methodology and real numbers -- nothing below is estimated.

---

## About

A portfolio project built to demonstrate backend and distributed-systems engineering — scheduling, concurrency, fault tolerance, and observability — using AI agent execution as the workload, not the point. It is built in phases (see the git history and [Roadmap](#14-roadmap)), each landing with working code, tests, and — where the phase touches a live behavior — a real run against the actual stack rather than a description of one.

**If you only look at three things:** [`src/acp/db/queries/transitions.py`](src/acp/db/queries/transitions.py) for the compare-and-set core everything else is built on; [`tests/chaos/test_stale_worker_race.py`](tests/chaos/test_stale_worker_race.py) for the fencing-token race reproduced deterministically with no sleeps; and [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for measured numbers, including two benchmarking bugs the harness caught in itself and left in the writeup rather than quietly fixing.

Skills this project is meant to speak to: distributed coordination without a consensus library, optimistic concurrency and race-condition reasoning, database-backed queueing, lease/heartbeat failure detection, retry and backoff design, multi-tenant admission control, runtime authorization, and treating a benchmark or demo script's own bugs as findings worth keeping rather than embarrassments to hide.

---

## 1. The problem

An agent task runs for seconds to minutes across several external calls — an LLM, a database, a third-party API, a tool. That has three consequences an ordinary request/response backend never faces:

1. **State outlives the process.** Progress must survive the death of the machine executing it, so task state lives outside the executor.
2. **Ownership must be revocable without cooperation.** A crashed worker cannot hand its work back. Someone else must take it — safely, without the possibility that the "crashed" worker was merely slow and is still running.
3. **Capacity is finite and contended.** Submission rate is unrelated to execution rate, and tenants share one fleet.

## 2. Why not just use Celery

| | Celery / RQ | This |
|---|---|---|
| Where task state lives | In the broker; the app can't query it | In PostgreSQL — queryable, joinable, constrained |
| Ownership model | Broker visibility timeout the app can't observe | Explicit lease + fencing token the app owns |
| Stale-worker writes | Acks aren't fenced — a revived worker's result can land | Rejected by CAS on `(state, attempt, worker)`, and **counted** |
| Recovery latency | Emergent, unmeasured | A design parameter: `lease_ttl + reaper_period` |
| "Why is this task slow?" | Guess | `task_events` timeline + metrics |

## 3. Architecture

```
                       CLIENT
                          │ HTTP
              ┌───────────▼────────────┐
              │  Control API (FastAPI) │  stateless · replicable · disposable
              │  idempotent submit     │
              │  cancel · read models  │
              └───────────┬────────────┘
                          │
   ═══════════════════════▼═══════════════════════════════
   ║              P O S T G R E S Q L                    ║
   ║        single source of truth · ONE CLOCK           ║
   ║  tenants │ tasks(+lease) │ task_attempts │ events   ║
   ║  idx_tasks_ready   ← the queue (partial index)      ║
   ║  idx_tasks_lease_expiry ← the reaper's scan         ║
   ═══▲══════════════▲═══════════════════▲═══════════════
      │ claim        │ renew / heartbeat │ reap
      │ SKIP LOCKED  │ CAS(attempt)      │ CAS(expired)
   ┌──┴───────┐  ┌───┴──────┐        ┌───┴──────┐
   │ Worker 1 │  │ Worker N │   ...  │  Reaper  │
   └──┬───────┘  └──────────┘        └──────────┘
      ▼
   Agent runtime — adapters (task_type → Adapter)

   Prometheus ← /metrics (API: DB gauges; worker/reaper: own counters)
```

**Three deployables, not ten.** API, worker, reaper — the boundary is drawn where failure domains actually differ. Everything inside them shares a PostgreSQL transaction, and splitting transactional code across a network turns one ACID commit into a saga.

## 4. Distributed systems concepts

**Leases with fencing tokens.** A worker owns a task only until `lease_expires_at`. `tasks.attempt` is a monotonic fencing token allocated by the *same* `UPDATE` that grants the lease, so ownership and token can never disagree. Every ownership-scoped write presents it.

**One clock.** `lease_expires_at` is computed by PostgreSQL and compared by PostgreSQL. Workers never compare a local clock against a lease deadline for correctness — only to decide when to renew. This deletes the entire clock-skew failure class: a worker with a wrong clock renews early or late, never wrongly believes it still owns a task.

**Compare-and-set, not locks.** Every state change is a single-statement `UPDATE ... WHERE <predicate>`. Under `READ COMMITTED`, PostgreSQL takes a row lock and re-evaluates the predicate against the latest committed row (EvalPlanQual), so N racing transactions produce exactly one winner. `SERIALIZABLE` would add serialization failures to buy a guarantee we already have.

**`applied=False` is a return value, not an exception.** Losing a CAS is how a worker *learns* its lease expired. Making it throw would grow a `try/except` at every call site that eventually swallows a genuine lost-ownership signal.

**Failure classification drives retry.** "Did it fail?" is not a useful question; "will it succeed if we try again?" is. A 429 backs off harder than a generic blip, a malformed payload never retries at all, and a killed worker retries with *no* backoff — the task never misbehaved, so delaying it punishes the workload for the infrastructure's problem. Backoff uses **full jitter**, because the problem is correlation: a thousand tasks failing in the same second would otherwise retry in the same second, rebuilding the herd against the still-recovering dependency.

**Immutable agent versions, pinned at submit.** A task resolves `request_type → agent → released version` once, at submission, and stores the version id. That makes execution reproducible (a retry runs the same code as attempt 1) — but it is also a *performance* decision: because versions cannot change, their fields are copied onto the task row and the claim query never joins the registry. Denormalisation is only dangerous when the source can drift. Immutability is enforced by a database trigger; only `status` may change.

**Static grants are versioned; denial is always live.** A tool grant attaches to an immutable agent *version*, so a version is a complete capability bundle and widening an agent's reach needs a reviewable new version rather than an `INSERT`. Because that would be far too slow to *revoke* during an incident, revocation doesn't go through grants at all — `tools.status = DISABLED` denies every use immediately, whatever any grant says. Same shape as certificate revocation: the certificate is immutable, but validity is checked at use.

**Authorization costs nothing at runtime.** The policy is snapshotted inside the *claim* transaction, so a tool call is a dictionary lookup and a set membership test — no query, no cache, no staleness window. An agent therefore can't gain a capability halfway through its own execution, and revocation latency is bounded by attempt duration, which is an explainable SLA rather than a cache TTL.

**Capability matching is a filter, and the cost is measured.** A task's `required_capabilities` must be a subset of its worker's (`<@` array containment). Because requirements are copied from an immutable version at submit, the predicate needs no join against the registry. But it is a *filter* on rows the ready index returns in priority order, not part of the index key — so a specialist worker facing a deep queue of work it cannot run scans that whole queue. `tests/integration/test_capability_scheduling.py` measures it: **~47 ms to claim from a 2,000-row queue**. `tasks.capability_key` already exists, unused, so the fix is a keyed partial index rather than another rewrite of a hot table.

**429 means "you"; 503 means "us."** Tenant backlog is checked first and rejects with 429, regardless of how the rest of the system is doing — a tenant over its own quota must never be told the system is struggling. Global overload is checked only once that clears, and sheds with 503, which is what keeps it a trustworthy signal rather than a catch-all excuse. The per-tenant count is exact (an indexed query); the global count is a cached, deliberately approximate gauge, because counting the whole table on every submit would make the overload check part of the overload.

**Derived state, not stored state.** There is no `RETRYING` state — a task waiting out backoff is `QUEUED` with a future `available_at`, and the claim predicate excludes it for free. There is no `CANCELLING` state — cancellation is a *request flag*, because you cannot yank work out of a remote process.

## 5. Guarantees

| Property | Value |
|---|---|
| Delivery | **At-least-once.** A task may be handed to workers more than once. |
| Committed outcome | **At-most-once.** At most one attempt ever writes a terminal state. |
| Side effects | **At-least-once.** Only as strong as the downstream API's idempotency. |
| Duplicate behaviour | Two attempts can run concurrently during a partition. The stale one is rejected and counted. |
| Failure behaviour | No task is lost. Crash recovery bounded by `lease_ttl + reaper_period`; graceful shutdown recovers in ~0s. |

Anyone claiming exactly-once without a transactional sink is wrong or hiding a two-phase commit.

## 6. The stale-worker race

The failure this design exists for — reproduced deterministically in [`tests/chaos/test_stale_worker_race.py`](tests/chaos/test_stale_worker_race.py), with no sleeps:

```
t=0    Worker A claims task 10        attempt=1, lease → t+30
t=8    A's host freezes.  A is NOT dead — still executing in memory.
t=30   lease expires (passively; expiry does nothing on its own)
t=31   reaper:  state → QUEUED, lease cleared.  attempt STAYS 1.
t=32   Worker B claims                attempt=2
       ⚠  A and B are both executing task 10. Expected and unavoidable.
t=90   A unfreezes and commits:
         UPDATE ... WHERE attempt=1 AND lease_worker_id='A' → 0 ROWS
       → STALE_WRITE_REJECTED event + acp_stale_writes_rejected_total
t=95   B commits with attempt=2 → SUCCEEDED
```

Exactly one attempt is recorded `SUCCEEDED`; the stored result is the live owner's. The fence guarantees at-most-once **committed outcome** — it does not stop A's external side effects, which is why the guarantee table above says what it says.

## 7. Demos

```bash
docker compose up -d --build --scale worker=5
python scripts/demo_chaos.py        # kills a worker mid-flight, proves recovery
python scripts/demo_governance.py   # proves tool authorization is enforced at runtime
```

`demo_chaos.py` submits 300 tasks (half with real multi-second duration, so there is genuine in-flight work), waits until ≥10 are running, then `docker kill`s one worker container outright. It reports `lease_expirations_total` / `task_recoveries_total` — read from **Prometheus**, since those counters live in the reaper's and workers' own processes, never the API's — and asserts every task still reaches a terminal state. A real run: 13 leases expired, 13 recovered, 0 lost.

`demo_governance.py` registers two agents with different tool grants and submits one request through each: `research-agent` (granted `web-search`) succeeds; `support-agent` reaching for `billing-db` (never granted) is refused at runtime with `PERMISSION_DENIED`, and the refusal is verified in **both** the task's event timeline and the audit log.

## 8. Local setup

Requires Docker and Python 3.12. On Windows use `.\make.ps1`; the `Makefile` mirrors it.

```bash
.\make.ps1 setup     # venv + dependencies
.\make.ps1 up        # PostgreSQL on :5434
.\make.ps1 migrate   # alembic upgrade head
.\make.ps1 test      # full suite
```

Full stack with a worker fleet:

```bash
docker compose up -d --build --scale worker=3
```

| Service | URL |
|---|---|
| Control API + OpenAPI docs | http://localhost:8001/docs |
| Prometheus | http://localhost:9091 |
| Grafana (anonymous admin) | http://localhost:3002 |

Ports are non-default (5434/8001/9091/3002) to avoid colliding with other local stacks; override with `ACP_PG_PORT`, `ACP_API_PORT`, `ACP_PROM_PORT`, `ACP_GRAFANA_PORT`.

## 9. Benchmarks

Full methodology and real numbers in [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Headlines from a 5-worker fleet on one machine:

| | |
|---|---|
| SIGKILL recovery latency | **30.19s** (design bound: `lease_ttl_s + reaper_period_s` = 31s) |
| SIGTERM (graceful) recovery latency | **0.03s** — ~943× faster than a crash |
| Throughput bottleneck identified | Queue wait (p50 15s) vastly exceeds execution time (p50 5ms) — the ceiling is `claim_batch_size × poll_interval`, not execution capacity |

The recovery-latency harness caught a bug in *itself*: its first version measured until a terminal state, so fast SIGTERM recovery (task reclaimed between two 50ms polls) was masked by the second attempt's full re-execution — there is no checkpointing, so a retry reruns the whole task. Fixed by detecting the moment `lease_worker_id` changes, which is the actual event being measured. The wrong number never shipped.

## 10. Tests

```bash
.\make.ps1 test-unit    # pure domain — no database, ~0.05s
.\make.ps1 test-db      # integration + concurrency, real PostgreSQL
.\make.ps1 test-race    # concurrency suite ×20, to prove it isn't flaky
```

The suite is layered by what it proves:

- **`tests/unit/`** — the state machine's full transition table (all 25 ordered pairs), an import-boundary check keeping `domain/` free of I/O, and a **metric cardinality guard** that fails if any metric gains a `task_id`/`worker_id`-shaped label.
- **`tests/integration/`** — real PostgreSQL, full request and worker flows. Includes an **EXPLAIN-based plan regression test** asserting the claim query uses `idx_tasks_ready` with **no Sort node** — a correctness test cannot catch this, because the query returns identical rows either way.
- **`tests/concurrency/`** — genuine contention (50 concurrent claims of one task, etc.) on real connections, not a mocked model of locking.
- **`tests/chaos/`** — worker death, stale writes, reaper races.

## 11. Failure model

| Failure | Detection | Behaviour |
|---|---|---|
| Worker `SIGKILL` | Lease expiry | Tasks requeued in ≤ `lease_ttl + reaper_period`; attempt recorded `LOST` |
| Worker `SIGTERM` | Self-reported drain | Work handed back with `available_at = now()`; recovery ~0s; attempt `ABANDONED` |
| Worker declared DEAD | Heartbeat rejected | Worker **self-fences**: stops claiming, exits, restarts with a fresh id |
| Reaper crash | Advisory absence | Recovery *latency* degrades; correctness does not. Nothing corrupts. |
| API crash | Client error | Retry with the same idempotency key → deduped by a partial unique index |
| PostgreSQL down | Everything fails | **Total outage — the deliberate V1 SPOF.** Workers abort rather than work on tasks they cannot prove they own. |
| Poison-pill task | `attempt` counts every handoff | Exhausts `max_attempts` → `FAILED`, instead of killing workers forever |
| Clock skew | — | **Cannot occur.** Only PostgreSQL's clock is ever compared. |

Two gaps are documented rather than hidden: a **hung worker** that keeps renewing is never detected (needs `max_execution_time`), and PostgreSQL is a single point of failure.

## 12. Observability

Metrics are namespaced `acp_*` and split by ownership: each process exports its own counters, the API exports the DB-derived gauges. Scrapes read memory only — monitoring must not be able to cause the incident it observes.

Notable metrics: `acp_stale_writes_rejected_total` (the fencing evidence), `acp_recovery_latency_seconds` (checks reality against the designed bound), `acp_leases_expired_pending` (the best single alert: non-zero means the reaper is down or behind), `acp_queue_wait_seconds`, `acp_claim_duration_seconds`.

**Cardinality is treated as a design constraint.** `task_id`, `worker_id`, and `idempotency_key` are never labels. `worker_id` is the subtle one: ids are generation-unique per process start, so a fleet redeploying daily would mint new time series daily, forever — low cardinality at any instant, unbounded over time. A unit test enforces this.

## 13. Tradeoffs

- **PostgreSQL as the queue, no Redis.** `SELECT ... FOR UPDATE SKIP LOCKED` over partial indexes is a production-grade queue. A second store means a second source of truth and a new failure mode, for no measured benefit at this scale. Redis earns its place when a benchmark shows claim contention, high-rate rate limiting, or polling latency as the bottleneck — `LISTEN/NOTIFY` should be tried first.
- **Pull-based claiming, no central dispatcher.** A central scheduler is a singleton bottleneck needing leader election and an extra `ASSIGNED` state with its own timeout. Scheduling *policy* still lives in its own pure module, so the mechanism can be swapped without touching it.
- **Lease inlined on the task row.** Lease grant and state transition must be atomic anyway, so a separate table buys only a join on the hottest query.
- **Tenant limits are enforced at claim time, not submit time**, so a busy tenant degrades in latency rather than getting errors.
- **No Kafka** — wrong data model. This needs per-item random-access mutation (extend this lease, cancel that task), which is the opposite of a log.
- **No Kubernetes** — it schedules containers; this schedules tasks, which is the interesting half.

## 14. Roadmap

| Phase | Status |
|---|---|
| 0 — skeleton, migrations, CI | ✅ |
| 1 — state machine, CAS, idempotent submit, Control API | ✅ |
| 2 — worker registry, `SKIP LOCKED` claim, leases | ✅ |
| 3 — reaper, recovery, fencing, chaos suite | ✅ |
| 4 — metrics, structured logs, Prometheus/Grafana | ✅ |
| 5 — failure classification, retry policy, backoff | ✅ |
| 6 — agent registry, immutable agent versions, routing | ✅ |
| 7 — tool registry, versioned grants, runtime authorization, audit log | ✅ |
| 8 — capability-aware placement | ✅ |
| 9 — admission control, backpressure, 429 vs 503 | ✅ |
| 10 — OpenTelemetry tracing, Grafana dashboards, the two demos | ✅ |
| 11 — benchmarks with methodology | ✅ |
| 12 — checkpointing (optional) | ⬜ |
