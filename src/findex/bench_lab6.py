# pyright: basic
"""Lab 6 benchmark: async crawler concurrency 1 vs 5 vs 20 on the same seed URL & max_pages.

Usage:
  uv run python -m findex.bench_lab6 https://docs.python.org/3/library/ --max-pages 220
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import resource
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from findex.crawler import CrawlStats, crawl_to_jsonl, fetch

log = logging.getLogger("findex.bench_lab6")


def _parent_peak_rss_mib() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


async def _run_single_concurrency(
    seed_url: str,
    max_pages: int,
    concurrency: int,
    path_prefix: str | None,
    out_jsonl: Path,
    log_path: Path,
) -> dict[str, Any]:
    stats = CrawlStats()
    t0 = time.perf_counter()
    await crawl_to_jsonl(
        [seed_url],
        out_jsonl,
        max_pages=max_pages,
        concurrency=concurrency,
        per_host_limit=concurrency,
        min_host_delay_s=0.02,
        timeout_s=12.0,
        allowed_path_prefix=path_prefix,
        stats=stats,
        crawl_log_path=log_path,
    )
    wall = time.perf_counter() - t0
    return {
        "concurrency": concurrency,
        "pages": stats.pages_collected,
        "wall_s": round(wall, 3),
        "pages_per_s": round(stats.pages_collected / max(1e-6, wall), 2),
        "errors": stats.errors,
        "peak_rss_mib": round(_parent_peak_rss_mib(), 1),
    }


async def _capture_failure_handling_sample() -> list[str]:
    """Demonstrate and capture real log lines for a handled 429 Retry-After + timeout recovery."""
    lines: list[str] = []
    attempts = {"/api/doc": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["/api/doc"] += 1
        if attempts["/api/doc"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0.15"}, text="Rate limit exceeded")
        if attempts["/api/doc"] == 2:
            raise httpx.ReadTimeout("upstream read timeout after 10.0s")
        return httpx.Response(200, text="<html><title>Recovered Doc</title><body>OK</body></html>")

    class ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(self.format(record))

    crawler_logger = logging.getLogger("findex.crawler")
    h = ListHandler()
    h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    crawler_logger.addHandler(h)
    crawler_logger.setLevel(logging.WARNING)
    try:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            res = await fetch(client, "https://docs.python.org/api/doc", max_retries=3, base_backoff_s=0.05)
            lines.append(
                f"INFO findex.crawler: OK {res.final_url} status={res.status_code} "
                f"attempts={res.attempts} elapsed={res.elapsed_s:.3f}s"
            )
    finally:
        crawler_logger.removeHandler(h)
    return lines


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run Lab 6 crawler benchmarks")
    parser.add_argument("seed_url", type=str)
    parser.add_argument("--max-pages", type=int, default=220)
    parser.add_argument("--path-prefix", type=str, default="/3/library/")
    parser.add_argument("--out-json", type=Path, default=Path("docs/bench_lab6.json"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for c in [1, 5, 20]:
        out_jsonl = Path(f"/tmp/findex_crawl_c{c}.jsonl")
        log_path = Path(f"/tmp/findex_crawl_c{c}.log")
        row = asyncio.run(
            _run_single_concurrency(
                seed_url=args.seed_url,
                max_pages=args.max_pages,
                concurrency=c,
                path_prefix=args.path_prefix,
                out_jsonl=out_jsonl,
                log_path=log_path,
            )
        )
        rows.append(row)
        log.info(
            "concurrency=%2d | pages=%d | wall=%.3fs | pages/s=%.2f | errors=%d | rss=%.1f MiB",
            row["concurrency"],
            row["pages"],
            row["wall_s"],
            row["pages_per_s"],
            row["errors"],
            row["peak_rss_mib"],
        )

    failure_logs = asyncio.run(_capture_failure_handling_sample())

    payload = {
        "seed_url": args.seed_url,
        "max_pages": args.max_pages,
        "path_prefix": args.path_prefix,
        "rows": rows,
        "failure_log_sample": failure_logs,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Saved Lab 6 benchmark -> %s", args.out_json)


if __name__ == "__main__":
    main()
