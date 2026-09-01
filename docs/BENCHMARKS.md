# Benchmarks

Every number below was produced by running the harness in `load/` against the real `docker compose` stack on this machine, on this date, at this fleet size — not estimated, not claimed from the design alone. Re-run `load/bench_throughput.py` and `load/bench_recovery.py` yourself; they print their own machine/config context alongside every result, because a number without its conditions is a guess with decimal points.

## Methodology

- **Stack:** `docker compose up -d --build --scale worker=5` (1 API, 1 reaper, 5 workers, Postgres, Prometheus, Grafana).
- **Machine:** Windows AMD64, Intel64 Family 6 Model 154, Python 3.12.0. Single physical machine — API, workers and Postgres all share its CPU and disk, so these numbers include that contention. They are not a claim about dedicated-hardware throughput.
- **Task simulation:** the `demo.agent` (near-instant) and `demo.slow` (configurable real duration) adapters — no network calls, no LLM, so the numbers measure the control plane, not a third-party API's latency.
- **Every latency number is read from Prometheus histograms**, not the benchmark script's own stopwatch — the worker records `acp_queue_wait_seconds` / `acp_execution_duration_seconds` / `acp_claim_duration_seconds` at the moment each event actually happens, which is immune to the client's own scheduling jitter.

## Throughput

```
python load/bench_throughput.py --tasks 2000 --concurrency 20
```

| | |
|---|---|
| Workers | 5 (capacity 5 each → 25 concurrent execution slots) |
| Tasks | 2,000, `demo.agent` (near-instant) |
| Submit rate | 153.7 tasks/s |
| End-to-end rate | 64.0 tasks/s (submit → full drain, 31.3s) |
| Queue wait p50 / p95 / p99 | 15.08s / 28.51s / 29.70s |
| Execution p50 / p95 / p99 | 0.005s / 0.0095s / 0.0099s |
| Claim query p50 / p99 | 0.0079s / 0.0906s |

**The bottleneck is identified, not hidden.** Execution takes ~5ms; queue wait is over 4,000× longer. That gap is not contention on the database or the workers being slow — it is `claim_batch_size` (5, default) × `poll_interval_ms` (250ms, default) capping the *aggregate* claim rate at roughly `workers × batch_size / poll_interval` ≈ 5 × 5 / 0.25s = **100 claims/sec**, regardless of how fast execution itself is. 2,000 tasks at 100 claims/sec is ~20s of pure claim-rate ceiling, which is the dominant term in the observed 31.3s drain time. Raising `claim_batch_size` or lowering `poll_interval_ms` would move this ceiling; neither has been tuned, because there was no measurement to tune against until this run produced one. That is the intended use of this harness: find the real constraint before changing a default to "fix" it.

## Recovery latency: crash vs. graceful shutdown

```
python load/bench_recovery.py --iterations 1 --task-duration-s 20
```

| Signal | Recovery latency | Design bound |
|---|---|---|
| `docker kill` (SIGKILL) | **30.19s** | `lease_ttl_s (30) + reaper_period_s (1)` = 31s |
| `docker stop` (SIGTERM, graceful) | **0.03s** | ~0s (hand-back, no lease wait) |

**The measured SIGKILL latency lands almost exactly on the designed bound** — the strongest single number in this document, because it is a design claim from the architecture (`docs/ARCHITECTURE.md` / `README.md`) verified against a real, unmodified process kill rather than a mock.

**Graceful shutdown recovers ~943× faster than a crash.** That gap is not incidental — it is the entire reason the worker drains and hands back in-flight work with `available_at = now()` instead of just exiting: a `SIGKILL` gives the system no warning, so recovery is bounded by how long it takes the reaper to *notice* (the lease has to actually expire); a `SIGTERM` lets the worker tell the system itself, immediately.

### A bug this benchmark caught in itself

The first version of this harness measured recovery by waiting for the task to reach `QUEUED` or a terminal state. Because SIGTERM recovery is often faster than the script's own 50ms poll interval, the task passed through `QUEUED` and was re-claimed by another worker *between two polls* — so the script never observed `QUEUED` and kept waiting, past the second attempt's **entire 20-second re-execution** (there is no checkpointing, by design — see `docs/ARCHITECTURE.md` — so a re-run repeats the full task from scratch). That produced a bogus "25.55s" SIGTERM number, conflating recovery latency with second-attempt execution time. The fix: detect recovery the moment the task's `lease_worker_id` changes to anyone other than the original owner, which is the actual event being measured. The corrected run above (0.03s) is what shipped; the wrong number never did.

## Recovery visibility in the chaos demo

`scripts/demo_chaos.py` runs the crash path at load — 300 tasks (half with genuine multi-second duration), one worker killed mid-batch:

```
tasks submitted                  300
  -> FAILED                      8
  -> SUCCEEDED                   292
lease_expirations_total          13   (tasks the killed worker was holding)
task_recoveries_total{requeued}  13   (reclaimed and re-run by a live worker)
stale_writes_rejected_total      0    (correct: a killed process cannot write)
wall-clock kill -> drain         36.4s
```

Every one of the 13 tasks the killed worker held was recovered; zero were lost. `stale_writes_rejected_total = 0` is the *correct* outcome for a `SIGKILL` — a genuinely dead process cannot come back and attempt a stale write. The fencing token's rejection of a stale writer is a different scenario (a worker that is merely *paused*, not dead) and is proven separately and deterministically — no sleeps, no live processes — in `tests/chaos/test_stale_worker_race.py` via `SIGSTOP`/`SIGCONT`.

## What is not benchmarked here, and why

- **Concurrent claim throughput under contention** (many workers racing a shallow queue) is proven *correct* by `tests/concurrency/test_transition_cas.py`, but not swept across worker counts here. That sweep is the natural next benchmark once `claim_batch_size`/`poll_interval_ms` tuning (above) is being evaluated. The specialist-worker-in-a-deep-queue case (`tests/integration/test_capability_scheduling.py`) used to be measured here with a wall-clock number, but that number turned out to be dominated by this environment's connection-establishment overhead, not query cost — it is now proven instead with `EXPLAIN (ANALYZE, BUFFERS)` asserting the keyed index is used and the buffer count stays small and queue-depth-independent, which is the honest way to show the fix without a misleading stopwatch figure.
- **Sustained throughput over minutes/hours** (autovacuum behavior, index bloat on the hot `tasks` table) — this harness runs one batch and exits; a long-soak variant is future work, not claimed here.
- **Multi-tenant fairness under load** is proven correct by `tests/integration/test_admission_control.py` (per-tenant isolation, 429 vs 503) but not measured for throughput impact under contention in this document.

Numbers are added to this file only when a harness in `load/` produces them. Nothing here is projected, estimated, or extrapolated from the design.
