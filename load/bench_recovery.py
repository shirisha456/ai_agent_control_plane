"""Recovery latency benchmark: SIGKILL vs graceful shutdown, measured separately.

The two paths have deliberately different designed bounds:

  SIGKILL     bounded by lease_ttl_s + reaper_period_s (the reaper has to
              notice the lease expired -- it cannot know the worker died any
              other way).
  SIGTERM     bounded by ~0: a draining worker hands its unfinished task back
              with available_at = now() before it exits, so another worker
              can claim it immediately.

Printing both from ONE run is the point: it is the gap between them that is
the interesting engineering result, not either number alone.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import uuid

import httpx

API = "http://localhost:8001"
PROMETHEUS = "http://localhost:9091"


def _prom_scalar(client: httpx.Client, query: str, default: float = 0.0) -> float:
    resp = client.get(f"{PROMETHEUS}/api/v1/query", params={"query": query}, timeout=10.0)
    resp.raise_for_status()
    result = resp.json()["data"]["result"]
    return float(result[0]["value"][1]) if result else default


def _workers() -> list[str]:
    out = subprocess.run(
        ["docker", "compose", "ps", "-q", "worker"], capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line.strip()]


def _submit_slow(client: httpx.Client, tenant_id: str, duration_s: list[float]) -> str:
    resp = client.post(
        f"{API}/v1/tasks",
        json={
            "tenant_id": tenant_id,
            "task_type": "demo.slow",
            "payload": {"duration_s": duration_s},
        },
    )
    resp.raise_for_status()
    return resp.json()["id"]


def _wait_for_reclaim(
    client: httpx.Client, task_id: str, original_worker: str, timeout_s: float
) -> float:
    """Time until ownership actually changes hands -- not until the task finishes.

    Recovery is often faster than this script's own poll interval: a task can
    pass through QUEUED and be re-claimed by another worker between two 50ms
    polls, so it is observed as RUNNING again before ever being seen QUEUED.
    Waiting for a terminal state (SUCCEEDED) instead would measure the
    SECOND attempt's full execution time on top of recovery -- there is no
    checkpointing (out of scope by design), so a re-run repeats the whole
    task duration from scratch. That is a real property of this system, but
    it is not what "recovery latency" means; conflating the two is exactly
    the bug this function exists to avoid.
    """
    started = time.monotonic()
    deadline = started + timeout_s
    while time.monotonic() < deadline:
        task = client.get(f"{API}/v1/tasks/{task_id}").json()
        state, lease_worker = task["state"], task.get("lease_worker_id")
        if state == "QUEUED" or (state == "RUNNING" and lease_worker != original_worker):
            return time.monotonic() - started
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            # Recovery happened faster than any poll caught the intermediate
            # state; the elapsed time up to this observation is still a
            # valid (if slightly inflated) upper bound.
            return time.monotonic() - started
        time.sleep(0.05)
    raise TimeoutError(
        f"task {task_id} was never reclaimed from {original_worker} within {timeout_s}s"
    )


def measure_one(
    client: httpx.Client, tenant_id: str, signal_kind: str, task_duration_s: float
) -> float:
    """Submit one long task, signal its worker mid-flight, time the recovery."""
    workers = _workers()
    task_id = _submit_slow(client, tenant_id, [task_duration_s, task_duration_s])

    # Wait until it is actually running, and note which worker via the task
    # events so we signal the right container in a multi-worker fleet.
    deadline = time.monotonic() + 15
    lease_worker = None
    while time.monotonic() < deadline:
        resp = client.get(f"{API}/v1/tasks/{task_id}")
        resp.raise_for_status()
        task = resp.json()
        if task["state"] == "RUNNING":
            lease_worker = task["lease_worker_id"]
            break
        time.sleep(0.05)
    if lease_worker is None:
        raise TimeoutError("task never started running")

    # lease_worker_id embeds hostname-pid-suffix; find the container whose
    # hostname prefixes it (container hostname == short container id).
    target = None
    for c in workers:
        short = c[:12]
        if lease_worker.startswith(short):
            target = c
            break
    target = target or workers[0]

    if signal_kind == "kill":
        subprocess.run(["docker", "kill", target], check=True, capture_output=True)
    else:
        subprocess.run(["docker", "stop", "-t", "10", target], check=True, capture_output=True)

    recovery_s = _wait_for_reclaim(client, task_id, lease_worker, timeout_s=90)

    # Bring the fleet back to size for the next iteration.
    subprocess.run(
        ["docker", "compose", "up", "-d", "--scale", f"worker={len(workers)}"],
        check=True,
        capture_output=True,
    )
    time.sleep(3)

    return recovery_s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--task-duration-s",
        type=float,
        default=20.0,
        help="must exceed lease_ttl_s so the task is still RUNNING when killed",
    )
    args = parser.parse_args()

    with httpx.Client(timeout=15.0) as client:
        tenant_resp = client.post(
            f"{API}/v1/tenants",
            json={"name": f"bench-recovery-{uuid.uuid4().hex[:8]}", "max_concurrent_tasks": 100},
        )
        tenant_resp.raise_for_status()
        tenant = tenant_resp.json()

        print(f"=== Recovery latency: SIGKILL vs SIGTERM ({args.iterations} iterations each) ===\n")

        kill_results = []
        for i in range(args.iterations):
            print(f"[kill {i + 1}/{args.iterations}] submit, wait for RUNNING, docker kill...")
            r = measure_one(client, tenant["id"], "kill", args.task_duration_s)
            print(f"  recovered in {r:.2f}s")
            kill_results.append(r)

        term_results = []
        for i in range(args.iterations):
            print(
                f"[stop  {i + 1}/{args.iterations}] submitting, waiting for RUNNING, "
                f"then docker stop (SIGTERM, graceful)..."
            )
            r = measure_one(client, tenant["id"], "stop", args.task_duration_s)
            print(f"  recovered in {r:.2f}s")
            term_results.append(r)

        print("\n=== RESULT ===")
        print(
            f"SIGKILL  recovery latency   min={min(kill_results):.2f}s  "
            f"max={max(kill_results):.2f}s  mean={sum(kill_results) / len(kill_results):.2f}s"
        )
        print(
            f"SIGTERM  recovery latency   min={min(term_results):.2f}s  "
            f"max={max(term_results):.2f}s  mean={sum(term_results) / len(term_results):.2f}s"
        )
        kill_mean = sum(kill_results) / len(kill_results)
        term_mean = sum(term_results) / len(term_results)
        print(
            f"\nGraceful shutdown recovers {kill_mean / max(term_mean, 0.01):.0f}x faster "
            "than a crash -- this gap IS the point of the drain-and-handback path."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
