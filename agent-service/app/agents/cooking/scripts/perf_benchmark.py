"""HTTP load benchmark for the cooking-plan agent /generate endpoint (P1-02).

Runs N concurrent generate requests against a running server and reports:
  - p50 / p95 latency (ms)
  - max observed in-flight concurrency
  - error rate (%)

The same command can be executed BEFORE (main) and AFTER (this branch) the
async-isolation changes; the printed metrics are the before/after comparison.

Usage (server must be running, e.g. `uv run uvicorn cooking_plan_agent.main:app`):
    uv run python scripts/perf_benchmark.py --url http://127.0.0.1:8000 \
        --token dev-token --concurrency 8 --requests 40
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

# Allow running directly from a checkout: prepend the src/ directory so the
# script works even when the package is not installed into the interpreter.
_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import httpx  # noqa: E402 — third-party import after src bootstrap


def _quantile(values: list[float], q: float) -> float:
    """Linear-interpolation quantile of a sorted list (P95 convention)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    frac = index - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


async def _run_wave(
    url: str,
    token: str,
    payload: dict,
    concurrency: int,
    results: list[float],
    errors: list[str],
    inflight: list[int],
) -> None:
    """Fire `concurrency` requests, tracking in-flight count for the peak."""
    semaphore = asyncio.Semaphore(concurrency)
    active = 0
    lock = asyncio.Lock()

    async def one() -> None:
        nonlocal active
        async with semaphore:
            async with lock:
                active += 1
                inflight.append(active)
            start = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{url}/internal/v1/agents/cooking-plan/generate",
                        json=payload,
                        headers={
                            "X-Internal-Token": token,
                            "X-Request-ID": "perf-bench",
                        },
                    )
                    if response.status_code != 200:
                        errors.append(f"HTTP {response.status_code}")
                    else:
                        body = response.json()
                        if body.get("status") not in ("READY", "NEEDS_CONFIRMATION", "INFEASIBLE", "FAILED"):
                            errors.append(f"Unexpected status: {body.get('status')}")
            except httpx.HTTPError as exc:
                errors.append(str(exc))
            finally:
                results.append((time.perf_counter() - start) * 1000)
                async with lock:
                    active -= 1

    await asyncio.gather(*(one() for _ in range(concurrency)))


async def benchmark(
    url: str,
    token: str,
    concurrency: int,
    requests: int,
    scenario_seed: int = 42,
) -> dict[str, object]:
    """Run the benchmark and return summary metrics."""
    # Imported lazily so the script boots even when the package is absent
    # (the src bootstrap above makes these resolvable in a checkout).
    from perf_scenarios import generate_perf_scenario

    from cooking_plan_agent.config.settings import get_settings
    from cooking_plan_agent.domain.models import GeneratePlanRequest

    settings = get_settings()
    request = generate_perf_scenario(recipes=3, steps_per_recipe=6, seed=scenario_seed).request
    payload = GeneratePlanRequest.model_validate(request).model_dump(mode="json")

    all_results: list[float] = []
    all_errors: list[str] = []
    inflight: list[int] = []

    waves = max(1, requests // concurrency)
    for _ in range(waves):
        await _run_wave(url, token, payload, concurrency, all_results, all_errors, inflight)

    total = len(all_results) + len(all_errors)
    error_rate = len(all_errors) / total * 100 if total else 0.0
    latencies = all_results

    summary: dict[str, object] = {
        "requests": total,
        "errors": len(all_errors),
        "error_rate_pct": round(error_rate, 2),
        "p50_ms": round(_quantile(latencies, 0.50), 1),
        "p95_ms": round(_quantile(latencies, 0.95), 1),
        "mean_ms": round(statistics.mean(latencies), 1) if latencies else 0.0,
        "max_inflight": max(inflight) if inflight else 0,
        "config": {
            "llm_max_concurrency": settings.llm_max_concurrency,
            "llm_connection_pool_size": settings.llm_connection_pool_size,
            "llm_overall_timeout_seconds": settings.llm_overall_timeout_seconds,
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Load benchmark for /generate (P1-02).")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of the agent service.")
    parser.add_argument("--token", default="dev-token", help="X-Internal-Token credential.")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent requests per wave.")
    parser.add_argument("--requests", type=int, default=40, help="Total requests (waves = requests / concurrency).")
    parser.add_argument("--seed", type=int, default=42, help="Scenario seed for reproducibility.")
    args = parser.parse_args()

    summary = asyncio.run(
        benchmark(
            url=args.url,
            token=args.token,
            concurrency=args.concurrency,
            requests=args.requests,
            scenario_seed=args.seed,
        )
    )

    print("=== P1-02 Load Benchmark ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("\nRun the same command on main vs this branch for the before/after comparison.")


if __name__ == "__main__":
    main()
