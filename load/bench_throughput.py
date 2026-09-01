"""Sustained-throughput benchmark, against the real docker-compose stack.

Submits a fixed batch as fast as the API accepts it, waits for full drain,
and reports throughput plus the latency percentiles that actually matter:
queue wait (how long a task sits before a worker claims it) and execution
duration, both read from PROMETHEUS -- not the script's own stopwatch --
because those histograms are what the worker itself recorded at the moment
each event happened, immune to the benchmark client's own scheduling jitter.

Every number this script prints states its machine/config context in the
same breath, per the project's own rule: a number without its conditions is
not a benchmark, it is a guess with decimal points.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass

import httpx

API = "http://localhost:8001"
PROMETHEUS = "http://localhost:9091"


def _prom_query(client: httpx.Client, query: str) -> list[dict]:
    resp = client.get(f"{PROMETHEUS}/api/v1/query", params={"query": query}, timeout=10.0)
    resp.raise_for_status()
    return resp.json()["data"]["result"]


def _prom_scalar(client: httpx.Client, query: str, default: float = 0.0) -> float:
    result = _prom_query(client, query)
    return float(result[0]["value"][1]) if result else default


def _worker_count() -> int:
    out = subprocess.run(
        ["docker", "compose", "ps", "-q", "worker"], capture_output=True, text=True, check=True
    ).stdout
    return len([line for line in out.splitlines() if line.strip()])


def _wait_for_api(client: httpx.Client, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if client.get(f"{API}/healthz", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError("API never became healthy")


@dataclass
class BenchResult:
    task_count: int
    worker_count: int
    submit_duration_s: float
    submit_rate_tasks_per_s: float
    drain_duration_s: float
    end_to_end_rate_tasks_per_s: float
    queue_wait_p50_s: float
    queue_wait_p95_s: float
    queue_wait_p99_s: float
    execution_p50_s: float
    execution_p95_s: float
    execution_p99_s: float
    claim_p50_s: float
    claim_p99_s: float
    machine: str
    python_version: str


def run(task_count: int, concurrency: int) -> BenchResult:
    with httpx.Client(timeout=15.0) as client:
        _wait_for_api(client)
        worker_count = _worker_count()
        if worker_count == 0:
            raise RuntimeError(
                "no worker containers found; run docker compose up --scale worker=N first"
            )

        # max_queued_tasks raised to the batch size: this benchmark measures
        # the claim/execution pipeline, not admission control (Phase 9's
        # 429-on-backlog is a SEPARATE, deliberately tested behaviour --
        # see tests/integration/test_admission_control.py -- and correctly
        # fired here on the default 1000-task tenant limit the first time
        # this script ran with a 2000-task batch).
        tenant_resp = client.post(
            f"{API}/v1/tenants",
            json={
                "name": f"bench-{uuid.uuid4().hex[:8]}",
                "max_concurrent_tasks": 10_000,
                "max_queued_tasks": max(task_count * 2, 1000),
            },
        )
        tenant_resp.raise_for_status()
        tenant = tenant_resp.json()

        print(f"Submitting {task_count} tasks (concurrency={concurrency})...")
        submit_started = time.monotonic()

        def _submit_one(client: httpx.Client, i: int) -> str:
            resp = client.post(
                f"{API}/v1/tasks",
                json={"tenant_id": tenant["id"], "task_type": "demo.agent", "payload": {"i": i}},
            )
            resp.raise_for_status()
            return resp.json()["id"]

        # A simple bounded worker pool of sync httpx clients, rather than
        # asyncio, so this script's own concurrency model cannot be confused
        # with the system's -- we are measuring the CONTROL PLANE's
        # throughput, not how fast Python can fire requests.
        import concurrent.futures

        task_ids: list[str] = []
        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool,
            httpx.Client(timeout=15.0) as submit_client,
        ):
            futures = [pool.submit(_submit_one, submit_client, i) for i in range(task_count)]
            for f in concurrent.futures.as_completed(futures):
                task_ids.append(f.result())
        submit_duration = time.monotonic() - submit_started
        print(
            f"  submitted {len(task_ids)} in {submit_duration:.2f}s "
            f"({len(task_ids) / submit_duration:.0f} tasks/s)\n"
        )

        print("Waiting for full drain...")
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            running = _prom_scalar(client, "sum(acp_tasks_running)")
            queued = _prom_scalar(client, "sum(acp_queue_depth)")
            backlog = _prom_scalar(client, "acp_tasks_backlogged")
            if running == 0 and queued == 0 and backlog == 0:
                break
            time.sleep(1)
        drain_duration = time.monotonic() - submit_started

        def q(metric: str, quantile: str) -> float:
            return _prom_scalar(
                client,
                f"histogram_quantile({quantile}, sum(rate({metric}_bucket[2m])) by (le))",
            )

        result = BenchResult(
            task_count=task_count,
            worker_count=worker_count,
            submit_duration_s=round(submit_duration, 3),
            submit_rate_tasks_per_s=round(task_count / submit_duration, 1),
            drain_duration_s=round(drain_duration, 3),
            end_to_end_rate_tasks_per_s=round(task_count / drain_duration, 1),
            queue_wait_p50_s=round(q("acp_queue_wait_seconds", "0.50"), 4),
            queue_wait_p95_s=round(q("acp_queue_wait_seconds", "0.95"), 4),
            queue_wait_p99_s=round(q("acp_queue_wait_seconds", "0.99"), 4),
            execution_p50_s=round(q("acp_execution_duration_seconds", "0.50"), 4),
            execution_p95_s=round(q("acp_execution_duration_seconds", "0.95"), 4),
            execution_p99_s=round(q("acp_execution_duration_seconds", "0.99"), 4),
            claim_p50_s=round(q("acp_claim_duration_seconds", "0.50"), 5),
            claim_p99_s=round(q("acp_claim_duration_seconds", "0.99"), 5),
            machine=(
                f"{platform.system()} {platform.machine()}, {platform.processor() or 'unknown CPU'}"
            ),
            python_version=platform.python_version(),
        )
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON only")
    args = parser.parse_args()

    result = run(args.tasks, args.concurrency)

    if args.json:
        print(json.dumps(asdict(result), indent=2))
        return 0

    print("\n=== BENCHMARK RESULT ===")
    print(f"machine                    {result.machine}")
    print(f"python                     {result.python_version}")
    print(f"workers                    {result.worker_count}")
    print(f"tasks                      {result.task_count}")
    print(f"submit rate                {result.submit_rate_tasks_per_s} tasks/s")
    print(
        f"end-to-end rate            {result.end_to_end_rate_tasks_per_s} tasks/s "
        f"(submit -> all drained, {result.drain_duration_s}s)"
    )
    print(
        f"queue wait   p50/p95/p99   {result.queue_wait_p50_s}s / "
        f"{result.queue_wait_p95_s}s / {result.queue_wait_p99_s}s"
    )
    print(
        f"execution    p50/p95/p99   {result.execution_p50_s}s / "
        f"{result.execution_p95_s}s / {result.execution_p99_s}s"
    )
    print(f"claim query  p50/p99       {result.claim_p50_s}s / {result.claim_p99_s}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
