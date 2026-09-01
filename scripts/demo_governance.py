"""The second flagship demo: agent governance, not just task execution.

Registers two agents with different tool grants, submits one request that
succeeds through an authorized tool and one that is refused at runtime for an
unauthorized one -- proving this is a control plane that governs agents, not
merely a distributed queue that happens to run them.

Run against the real docker-compose stack. Requires the `demo.tools` adapter,
registered by default in the worker entrypoint.
"""

from __future__ import annotations

import sys
import time
import uuid

import httpx

API = "http://localhost:8001"


def _wait_for_api(client: httpx.Client, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if client.get(f"{API}/healthz", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError(f"API did not become healthy within {timeout_s}s")


def _wait_for_terminal(client: httpx.Client, task_id: str, timeout_s: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        task = client.get(f"{API}/v1/tasks/{task_id}").json()
        if task["state"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return task
        time.sleep(0.3)
    raise TimeoutError(f"task {task_id} did not reach a terminal state in {timeout_s}s")


def main() -> int:
    print("=== ACP Governance Demo ===")
    print(f"API: {API}\n")

    with httpx.Client(timeout=10.0) as client:
        _wait_for_api(client)

        tenant = client.post(
            f"{API}/v1/tenants", json={"name": f"gov-demo-{uuid.uuid4().hex[:8]}"}
        ).json()
        print(f"Tenant: {tenant['name']}\n")

        print("Registering tools...")
        web_search = client.post(
            f"{API}/v1/tools",
            json={"tenant_id": tenant["id"], "name": "web-search", "tool_type": "SIMULATED"},
        ).json()
        billing_db = client.post(
            f"{API}/v1/tools",
            json={"tenant_id": tenant["id"], "name": "billing-db", "tool_type": "SIMULATED"},
        ).json()
        customer_db = client.post(
            f"{API}/v1/tools",
            json={"tenant_id": tenant["id"], "name": "customer-db", "tool_type": "SIMULATED"},
        ).json()
        print(
            f"  web-search={web_search['id'][:8]} billing-db={billing_db['id'][:8]} "
            f"customer-db={customer_db['id'][:8]}\n"
        )

        print("Registering research-agent, granted web-search only...")
        research_agent = client.post(
            f"{API}/v1/agents",
            json={"tenant_id": tenant["id"], "name": "research-agent"},
        ).json()
        research_v1 = client.post(
            f"{API}/v1/agents/{research_agent['id']}/versions",
            json={"runtime_spec": {"task_type": "demo.tools"}, "max_attempts": 3},
        ).json()
        client.post(
            f"{API}/v1/agent-versions/{research_v1['id']}/grants",
            json={"tool_id": web_search["id"]},
        )
        client.post(
            f"{API}/v1/agents/{research_agent['id']}/activate",
            json={"version_id": research_v1["id"]},
        )
        client.put(
            f"{API}/v1/routes",
            json={
                "tenant_id": tenant["id"],
                "request_type": "RESEARCH_REPORT",
                "agent_id": research_agent["id"],
            },
        )

        print("Registering support-agent, granted customer-db only (NOT billing-db)...")
        support_agent = client.post(
            f"{API}/v1/agents",
            json={"tenant_id": tenant["id"], "name": "support-agent"},
        ).json()
        support_v1 = client.post(
            f"{API}/v1/agents/{support_agent['id']}/versions",
            json={"runtime_spec": {"task_type": "demo.tools"}, "max_attempts": 3},
        ).json()
        client.post(
            f"{API}/v1/agent-versions/{support_v1['id']}/grants",
            json={"tool_id": customer_db["id"]},
        )
        client.post(
            f"{API}/v1/agents/{support_agent['id']}/activate", json={"version_id": support_v1["id"]}
        )
        client.put(
            f"{API}/v1/routes",
            json={
                "tenant_id": tenant["id"],
                "request_type": "SUPPORT_TICKET",
                "agent_id": support_agent["id"],
            },
        )
        print()

        # --- Task A: routed to research-agent, uses an authorized tool -----
        print("Task A: RESEARCH_REPORT -> research-agent -> web-search (authorized)")
        task_a = client.post(
            f"{API}/v1/tasks",
            json={
                "tenant_id": tenant["id"],
                "request_type": "RESEARCH_REPORT",
                "payload": {"tools": ["web-search"]},
            },
        ).json()
        result_a = _wait_for_terminal(client, task_a["id"])
        print(f"  -> {result_a['state']}  (agent_version_id={task_a['agent_version_id'][:8]})")
        assert result_a["state"] == "SUCCEEDED", "expected the authorized tool call to succeed"
        print("  PASS: authorized tool call executed successfully.\n")

        # --- Task B: routed to support-agent, attempts an ungranted tool ---
        print("Task B: SUPPORT_TICKET -> support-agent -> billing-db (NOT authorized)")
        task_b = client.post(
            f"{API}/v1/tasks",
            json={
                "tenant_id": tenant["id"],
                "request_type": "SUPPORT_TICKET",
                "payload": {"tools": ["billing-db"]},
            },
        ).json()
        result_b = _wait_for_terminal(client, task_b["id"])
        print(f"  -> {result_b['state']}  error_class={result_b['error_class']}")
        assert result_b["state"] == "FAILED", "expected the unauthorized tool call to be refused"
        assert result_b["error_class"] == "PERMISSION_DENIED"
        print("  PASS: unauthorized tool call was refused at runtime, not attempted.\n")

        events_b = client.get(f"{API}/v1/tasks/{task_b['id']}/events").json()
        denied_events = [e for e in events_b if e["event_type"] == "TOOL_ACCESS_DENIED"]
        print(f"  task timeline: {[e['event_type'] for e in events_b]}")
        assert denied_events, "expected a TOOL_ACCESS_DENIED event on the task timeline"

        audit = client.get(f"{API}/v1/audit", params={"tenant_id": tenant["id"]}).json()
        denial_records = [a for a in audit if a["action"] == "TOOL_ACCESS_DENIED"]
        print(f"  audit log: {len(denial_records)} TOOL_ACCESS_DENIED record(s)")
        assert denial_records, "expected the denial to survive in the audit log"
        print(
            f"    resource={denial_records[0]['resource_type']} "
            f"outcome={denial_records[0]['outcome']} "
            f"data={denial_records[0]['data']}\n"
        )

        print("=== VERIFICATION ===")
        print("research-agent (web-search granted)  -> RESEARCH_REPORT task SUCCEEDED")
        print(
            "support-agent  (billing-db NOT granted) -> SUPPORT_TICKET task FAILED, refused"
        )
        print("Refusal recorded in BOTH the task's own event timeline AND the audit log,")
        print("which outlives the task's retention -- this is governance, not just execution.")
        print("\nPASS")

    return 0


if __name__ == "__main__":
    sys.exit(main())
