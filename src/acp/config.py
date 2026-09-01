"""Single settings object, environment-driven.

Timing values live here rather than in worker code on purpose: workers fetch
them from the control plane at registration, so the entire fleet can be
retuned (e.g. shortening lease_ttl after measuring spurious expirations)
without rebuilding or redeploying a single worker.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ACP_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://acp:acp@localhost:5434/acp"
    db_pool_size: int = 10
    db_max_overflow: int = 10
    # Every outbound wait in this system is bounded. An unbounded connect
    # turns a slow database into a hung process that reports nothing.
    db_connect_timeout_s: int = 5
    health_probe_timeout_s: float = 2.0

    log_level: str = "INFO"
    log_format: str = "json"  # "json" | "console"

    # --- lease / failure-detection timing -------------------------------
    # Invariant that must hold, or workers will lose leases they still own:
    #   lease_ttl_s > 3 * lease_renew_interval_s + p99 renewal RTT + pause budget
    # Recovery latency for a crashed worker is lease_ttl_s + reaper_period_s.
    lease_ttl_s: int = 30
    lease_renew_interval_s: int = 7
    heartbeat_interval_s: int = 5
    worker_dead_after_s: int = 15
    reaper_period_s: int = 1

    # --- scheduling ------------------------------------------------------
    poll_interval_ms: int = 250
    claim_batch_size: int = 5

    # What this worker process can do, comma-separated -- e.g.
    # "gpu,internet,large_context". Configuration rather than code,
    # because capability is a property of the MACHINE (does it have a GPU?
    # can it reach the internet?), not of the software running on it. Two
    # identical worker images on different hardware must be able to
    # advertise differently without a rebuild.
    #
    # Empty means a generalist worker, which can still run every task that
    # requires nothing.
    worker_capabilities: str = ""

    # --- admission control -----------------------------------------------
    # Global queue depth at which the API starts shedding with 503. Zero
    # disables it, which is the right default: shedding should be switched
    # on with a number someone derived from a measured drain rate, not left
    # at a guess that fires during a normal burst.
    global_queue_shed_threshold: int = 0
    # How long a stopping worker lets in-flight attempts finish before
    # handing them back. Docker's default SIGTERM->SIGKILL window is 10s,
    # so staying under it keeps graceful shutdown actually graceful.
    drain_grace_s: float = 5.0

    # --- observability ---------------------------------------------------
    # Workers and the reaper are separate processes, so each exposes its own
    # scrape endpoint. The API serves DB-derived gauges; per-process counters
    # come from the process that owns them, which is the only way a counter
    # can be correct without cross-process coordination.
    metrics_port: int = 0  # 0 disables the standalone exporter
    gauge_refresh_s: float = 5.0

    def capabilities(self) -> tuple[str, ...]:
        """Normalised the same way the domain normalises requirements.

        Both sides go through the same lowercasing and trimming, so a
        worker advertising "GPU " satisfies a task requiring "gpu". A
        mismatch there would be invisible -- the task would simply never be
        claimed, with no error anywhere.
        """
        return tuple(
            sorted({c.strip().lower() for c in self.worker_capabilities.split(",") if c.strip()})
        )

    def validate_timing(self) -> None:
        """Fail fast on a configuration that guarantees spurious lease loss."""
        if self.lease_ttl_s <= 3 * self.lease_renew_interval_s:
            raise ValueError(
                f"lease_ttl_s={self.lease_ttl_s} must exceed 3 renew intervals "
                f"({3 * self.lease_renew_interval_s}s); otherwise a single dropped "
                "renewal costs a worker its lease."
            )
        if self.worker_dead_after_s <= 2 * self.heartbeat_interval_s:
            raise ValueError(
                f"worker_dead_after_s={self.worker_dead_after_s} must exceed 2 "
                f"heartbeat intervals ({2 * self.heartbeat_interval_s}s)."
            )


@lru_cache
def settings() -> Settings:
    s = Settings()
    s.validate_timing()
    return s
