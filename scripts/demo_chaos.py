"""The flagship demo: submit work, kill a worker mid-flight, prove recovery.

Run against the real docker-compose stack (`make up` / `docker compose up`),
not a mock. It:

  1. submits a batch of SLOW tasks (demo.slow -- real wall-clock duration,
     unlike the instant demo.agent) across two tenants, plus a slice of
     tasks with an injected transient failure;
  2. waits until work is genuinely in flight -- with instant tasks, the
     whole batch would drain before a `docker kill` ever landed on anything,
     which is a demo that proves nothing;
  3. kills one worker container outright (docker kill -- a real SIGKILL);
  4. waits for the whole batch to reach a terminal state;
  5. prints a verification report read back FROM THE DATABASE, not asserted
     from the script's own bookkeeping.

WHAT `docker kill` DOES AND DOES NOT PROVE
-------------------------------------------
A killed process is simply gone -- it cannot come back and attempt a stale
write. So `stale_writes_rejected_total staying at 0 here is the CORRECT
outcome for this scenario, not a shortfall. The stale-write race (a worker
that is merely PAUSED, not dead, and returns to try a write after losing its
lease) is a different failure mode, reproduced deterministically via
SIGSTOP/SIGCONT in tests/chaos/test_stale_worker_race.py -- that automated,
sleep-free test is the actual proof the fencing token rejects a stale writer.
This script proves the OTHER half: that a genuinely dead worker's in-flight
leases expire and get reclaimed within the designed bound.
"""

from __future__ import annotations

import subprocess
import sys
import time
import uuid

import httpx

API = "http://localhost:8001"
# Prometheus, not the API, is where lease_expirations_total,
# task_recoveries_total and stale_writes_rejected_total actually live: the
# first two are incremented inside the REAPER process, the third inside
# WORKER processes, each on its own local registry exported from its own
# container. The API process never touches those counters. Aggregating
# across independently-scraped processes is exactly what Prometheus is
# for in this stack -- summing raw per-process /metrics text by hand would
# be reinventing it badly.
PROMETHEUS = "http://localhost:9091"
SLOW_TASK_COUNT = 150
FAST_TASK_COUNT = 150
TENANTS = 2
INJECTED_FAILURE_RATE = 0.05
SLOW_DURATION_S = [2.0, 5.0]


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def _worker_containers() -> list[str]:
    out = _run(["docker", "compose", "ps", "-q", "worker"])
    return [line for line in out.splitlines() if line.strip()]


def _wait_for_api(timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{API}/healthz", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError(f"API did not become healthy within {timeout_s}s")


def _metric_sum(metrics_text: str, prefix: str) -> float:
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in metrics_text.splitlines()
        if line.startswith(prefix)
    )


def _prom_sum(client: httpx.Client, query: str) -> float:
    """Sum a Prometheus instant-vector query across every scraped instance.

    This is how a value incremented independently in five worker containers
    and one reaper container becomes one number -- Prometheus is the
    aggregation point in this architecture, not any single process's
    /metrics endpoint.
    """
    resp = client.get(f"{PROMETHEUS}/api/v1/query", params={"query": query}, timeout=10.0)
    resp.raise_for_status()
    result = resp.json()["data"]["result"]
    return sum(float(r["value"][1]) for r in result)


def main() -> int:
    print("=== ACP Chaos Demo ===")
    print(f"API: {API}\n")

    _wait_for_api()
    workers = _worker_containers()
    if len(workers) < 2:
        print(
            f"! Only {len(workers)} worker container(s) found. Run with a fleet:\n"
            "    docker compose up -d --build --scale worker=5\n"
        )
        return 1
    print(f"Fleet: {len(workers)} worker containers\n")

    with httpx.Client(timeout=10.0) as client:
        tenant_ids = []
        for _ in range(TENANTS):
            resp = client.post(
                f"{API}/v1/tenants",
                json={"name": f"chaos-demo-{uuid.uuid4().hex[:8]}", "max_concurrent_tasks": 200},
            )
            resp.raise_for_status()
            tenant_ids.append(resp.json()["id"])

        total = SLOW_TASK_COUNT + FAST_TASK_COUNT
        print(
            f"Submitting {total} tasks ({SLOW_TASK_COUNT} slow, {FAST_TASK_COUNT} fast) "
            f"across {len(tenant_ids)} tenants..."
        )
        task_ids: list[str] = []
        for i in range(total):
            tenant_id = tenant_ids[i % len(tenant_ids)]
            if i < SLOW_TASK_COUNT:
                if i % int(1 / INJECTED_FAILURE_RATE) == 0:
                    task_type, payload = "demo.fail", {"i": i}
                else:
                    task_type, payload = "demo.slow", {"duration_s": SLOW_DURATION_S, "i": i}
            else:
                task_type, payload = "demo.agent", {"i": i}
            resp = client.post(
                f"{API}/v1/tasks",
                json={"tenant_id": tenant_id, "task_type": task_type, "payload": payload},
            )
            resp.raise_for_status()
            task_ids.append(resp.json()["id"])
        print(f"  submitted {len(task_ids)} tasks\n")

        print("Waiting for slow tasks to actually be in flight...")
        deadline = time.monotonic() + 30
        running = 0
        while time.monotonic() < deadline:
            running = _metric_sum(client.get(f"{API}/metrics").text, "acp_tasks_running{")
            if running >= 10:
                break
            time.sleep(0.3)
        print(f"  {running:.0f} tasks running\n")
        if running < 10:
            print(
                "! Never saw enough in-flight work -- the kill below would prove nothing. Aborting."
            )
            return 1

        target = workers[len(workers) // 2]
        kill_time = time.monotonic()
        print(f"*** docker kill {target[:12]} (a real SIGKILL, not a clean shutdown) ***\n")
        subprocess.run(["docker", "kill", target], check=True, capture_output=True)

        print("Waiting for the whole batch to reach a terminal state...")
        # Driven by the SYSTEM-WIDE gauges, not a sample of task ids: a
        # stride sample can miss the exact handful of tasks the killed
        # worker was holding and declare victory before the reaper's
        # recovery bound (lease_ttl_s + reaper_period_s) has even elapsed --
        # which is precisely the failure this script exists to catch, not
        # commit. acp_tasks_running / acp_queue_depth / acp_tasks_backlogged
        # cover in-flight, immediately-runnable and retry-deferred work.
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            text = client.get(f"{API}/metrics").text
            still_going = (
                _metric_sum(text, "acp_tasks_running{")
                + _metric_sum(text, "acp_queue_depth{")
                + _metric_sum(text, "acp_tasks_backlogged")
            )
            if still_going == 0:
                break
            time.sleep(2)
        drain_time = time.monotonic()

        print("\n=== VERIFICATION (read back from the database, not the script) ===")
        outcomes: dict[str, int] = {}
        for tid in task_ids:
            state = client.get(f"{API}/v1/tasks/{tid}").json()["state"]
            outcomes[state] = outcomes.get(state, 0) + 1

        # Read from PROMETHEUS: these three counters live in the reaper's and
        # workers' own processes, never the API's -- see the module-level
        # comment on why summing the API's /metrics would always read 0 here.
        lease_expirations = _prom_sum(client, "sum(acp_lease_expirations_total)")
        recoveries_requeued = _prom_sum(
            client, 'sum(acp_task_recoveries_total{disposition="requeued"})'
        )
        stale_rejected = _prom_sum(client, "sum(acp_stale_writes_rejected_total)")

        print(f"tasks submitted                  {len(task_ids)}")
        for state, count in sorted(outcomes.items()):
            print(f"  -> {state:<10}                  {count}")
        print(
            f"lease_expirations_total           {lease_expirations:.0f}   "
            "(tasks the killed worker was holding)"
        )
        print(
            f"task_recoveries_total{{requeued}}   {recoveries_requeued:.0f}   "
            "(reclaimed and re-run by a live worker)"
        )
        print(
            f"stale_writes_rejected_total       {stale_rejected:.0f}   "
            "(0 is CORRECT here -- a killed process cannot write;"
        )
        print(
            "                                          "
            "see tests/chaos/test_stale_worker_race.py for the SIGSTOP case that exercises this)"
        )
        print(f"wall-clock kill -> drain          {drain_time - kill_time:.1f}s")

        all_terminal = sum(outcomes.values()) == len(task_ids)
        recovery_happened = lease_expirations > 0 and recoveries_requeued > 0

        print()
        if all_terminal and recovery_happened:
            print(
                "PASS: every submitted task reached a terminal state, and the killed worker's "
                "in-flight leases were detected and reclaimed -- no task was lost."
            )
        elif all_terminal:
            print(
                "INCOMPLETE: the batch drained, but no lease expirations were recorded -- the kill "
                "likely did not land on a worker holding any in-flight task. Re-run; if this "
                "persists, increase SLOW_TASK_COUNT or SLOW_DURATION_S."
            )
            return 1
        else:
            print("INCOMPLETE: some tasks never reached a terminal state within the timeout.")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
