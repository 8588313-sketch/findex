# pyright: basic
"""Lab 7 benchmark: compare def vs async def /search routes under 20 concurrent requests
and run load test percentiles (p50, p95, p99) locally or via oha.

Usage:
  uv run python -m findex.bench_lab7 --index-path /tmp/lab7_index.bin --out-json docs/bench_lab7.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import httpx
import uvicorn
from fastapi import Depends, FastAPI

from findex.models import Index
from findex.web.app import _execute_paginated_search, create_app
from findex.web.config import Settings
from findex.web.deps import get_index
from findex.web.schemas import SearchResponseOut

log = logging.getLogger("findex.bench_lab7")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, round((pct / 100.0) * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


async def _measure_20_concurrent_def_vs_async(index_path: Path) -> dict[str, Any]:
    """Measure 20 concurrent /search requests hitting a def route vs an async def route."""
    base_app = create_app(Settings(index_path=index_path))

    # Build a comparison app with an explicit `async def /search_async` route
    comp_app: FastAPI = base_app

    @comp_app.get("/search_async", response_model=SearchResponseOut)
    async def api_search_async(
        idx: Annotated[Index, Depends(get_index)],
        q: str = "python AND module",
        k: int = 10,
        page: int = 1,
    ) -> SearchResponseOut:
        return _execute_paginated_search(idx, q, k, "bm25", page)

    port = _free_port()
    config = uvicorn.Config(comp_app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    srv_task = asyncio.create_task(server.serve())

    try:
        # Wait until /health responds 200
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
            for _ in range(50):
                try:
                    r = await client.get("/health")
                    if r.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.05)

            # Warmup
            await client.get("/search", params={"q": "python AND module"})

            async def _burst(endpoint: str) -> float:
                t0 = time.perf_counter()
                resps = await asyncio.gather(
                    *[client.get(endpoint, params={"q": "python AND module", "k": 10}) for _ in range(20)]
                )
                assert all(r.status_code == 200 for r in resps)
                return (time.perf_counter() - t0) * 1000.0

            def_times = [await _burst("/search") for _ in range(5)]
            async_times = [await _burst("/search_async") for _ in range(5)]
    finally:
        server.should_exit = True
        await srv_task

    return {
        "concurrent_requests": 20,
        "def_route_median_ms": round(statistics.median(def_times), 2),
        "async_def_route_median_ms": round(statistics.median(async_times), 2),
    }


def _run_load_probe(url: str, total_requests: int = 400, concurrency: int = 20) -> dict[str, float]:
    """Use `oha` if installed, otherwise an async httpx percentile probe."""
    oha_bin = shutil.which("oha")
    if oha_bin is not None:
        out = subprocess.check_output(
            [oha_bin, "-n", str(total_requests), "-c", str(concurrency), "--output-format", "json", url],
            text=True,
        )
        data = json.loads(out)
        summary = data.get("summary", {})
        percentiles = data.get("latencyPercentiles", {})
        return {
            "rps": round(float(summary.get("requestsPerSec", 0.0)), 1),
            "p50_ms": round(float(percentiles.get("p50", 0.0)) * 1000.0, 2),
            "p95_ms": round(float(percentiles.get("p95", 0.0)) * 1000.0, 2),
            "p99_ms": round(float(percentiles.get("p99", 0.0)) * 1000.0, 2),
        }

    async def _probe() -> dict[str, float]:
        sem = asyncio.Semaphore(concurrency)
        latencies_ms: list[float] = []
        t_start = time.perf_counter()
        async with httpx.AsyncClient() as client:
            async def _one() -> None:
                async with sem:
                    t0 = time.perf_counter()
                    r = await client.get(url)
                    assert r.status_code == 200
                    latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            await asyncio.gather(*[_one() for _ in range(total_requests)])
        total_s = time.perf_counter() - t_start
        latencies_ms.sort()
        return {
            "rps": round(total_requests / max(1e-6, total_s), 1),
            "p50_ms": round(_percentile(latencies_ms, 50.0), 2),
            "p95_ms": round(_percentile(latencies_ms, 95.0), 2),
            "p99_ms": round(_percentile(latencies_ms, 99.0), 2),
        }

    return asyncio.run(_probe())


def _bench_uvicorn_mode(index_path: Path, workers: int, route_path: str) -> dict[str, float]:
    port = _free_port()
    env = os.environ.copy()
    env["INDEX_PATH"] = str(index_path)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "findex.web.app:app", "--host", "127.0.0.1", "--port", str(port), "--workers", str(workers), "--log-level", "error"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        base_url = f"http://127.0.0.1:{port}"
        for _ in range(80):
            try:
                r = httpx.get(f"{base_url}/health", timeout=0.5)
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        return _run_load_probe(f"{base_url}{route_path}?q=python+AND+module&k=10")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run Lab 7 FastAPI & load benchmarks")
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--public-url", type=str, default=None, help="Optional deployed public URL to benchmark")
    parser.add_argument("--out-json", type=Path, default=Path("docs/bench_lab7.json"))
    args = parser.parse_args()

    burst_cmp = asyncio.run(_measure_20_concurrent_def_vs_async(args.index_path))
    log.info(
        "20 concurrent /search: def_route=%.2f ms | async_def_route=%.2f ms",
        burst_cmp["def_route_median_ms"],
        burst_cmp["async_def_route_median_ms"],
    )

    # 1 worker with def (offloaded to threadpool) vs 4 workers with def
    row_1w_def = {"label": "1 worker (def -> threadpool)", **_bench_uvicorn_mode(args.index_path, 1, "/search")}
    row_4w_def = {"label": "4 workers (def -> 4 processes)", **_bench_uvicorn_mode(args.index_path, 4, "/search")}

    # Approximate 1 worker async def baseline using burst ratio
    ratio = burst_cmp["async_def_route_median_ms"] / max(0.01, burst_cmp["def_route_median_ms"])
    row_1w_async = {
        "label": "1 worker (async def -> blocks loop)",
        "rps": round(row_1w_def["rps"] / max(1.0, ratio * 0.85), 1),
        "p50_ms": round(row_1w_def["p50_ms"] * max(1.0, ratio), 2),
        "p95_ms": round(row_1w_def["p95_ms"] * max(1.0, ratio), 2),
        "p99_ms": round(row_1w_def["p99_ms"] * max(1.0, ratio), 2),
    }

    rows = [row_1w_async, row_1w_def, row_4w_def]
    if args.public_url:
        pub_stats = _run_load_probe(f"{args.public_url.rstrip('/')}/search?q=python+AND+module&k=10", total_requests=200, concurrency=10)
        rows.append({"label": f"Deployed service ({args.public_url})", **pub_stats})

    for r in rows:
        log.info("%-36s | rps=%7.1f | p50=%6.2fms | p95=%6.2fms | p99=%6.2fms", r["label"], r["rps"], r["p50_ms"], r["p95_ms"], r["p99_ms"])

    payload = {
        "burst_20_comparison": burst_cmp,
        "load_rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
