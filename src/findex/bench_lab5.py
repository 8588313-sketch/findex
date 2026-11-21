# pyright: basic
"""Lab 5 benchmark: serial vs threads vs processes across worker counts + matplotlib speedup chart.

Usage:
  uv run python -m findex.bench_lab5 data/ --out-json docs/bench_lab5.json --chart docs/speedup_lab5.png
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import resource
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from findex.builder import ExecutorKind, build_index_parallel

log = logging.getLogger("findex.bench_lab5")


def _cpu_model() -> str:
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            if out:
                return out
        except OSError as exc:
            log.debug("sysctl brand_string unavailable: %s", exc)
    return platform.processor() or platform.machine()


def _parent_peak_rss_mib() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports ru_maxrss in bytes, Linux in KiB
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def measure_cell(
    root: Path,
    executor: ExecutorKind,
    workers: int,
    repeats: int = 3,
) -> dict[str, Any]:
    wall_samples: list[float] = []
    cpu_samples: list[float] = []
    merge_samples: list[float] = []
    rss_samples: list[float] = []

    for _ in range(repeats):
        _, stats = build_index_parallel(
            root=root,
            with_positions=True,
            workers=workers,
            executor=executor,
        )
        wall_samples.append(stats.wall_time_s)
        cpu_samples.append(stats.cpu_time_s)
        merge_samples.append(stats.merge_time_s)
        rss_samples.append(_parent_peak_rss_mib())

    return {
        "executor": executor,
        "workers": workers,
        "wall_s": round(statistics.median(wall_samples), 4),
        "cpu_s": round(statistics.median(cpu_samples), 4),
        "merge_ms": round(statistics.median(merge_samples) * 1000.0, 2),
        "parent_rss_mib": round(max(rss_samples), 2),
    }


def render_speedup_chart(
    worker_counts: list[int],
    proc_speedups: list[float],
    thread_speedups: list[float],
    free_threaded_speedups: list[float] | None,
    chart_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)

    ax.plot(
        worker_counts,
        [float(w) for w in worker_counts],
        linestyle="--",
        color="#64748b",
        linewidth=1.5,
        label="Ideal linear speedup S(N) = N",
    )
    ax.plot(
        worker_counts,
        proc_speedups,
        marker="o",
        linewidth=2.3,
        color="#2563eb",
        label="processes (ProcessPoolExecutor, Python 3.11)",
    )
    ax.plot(
        worker_counts,
        thread_speedups,
        marker="s",
        linewidth=2.1,
        color="#dc2626",
        label="threads (ThreadPoolExecutor, Python 3.11 + GIL)",
    )
    if free_threaded_speedups is not None:
        ax.plot(
            worker_counts,
            free_threaded_speedups,
            marker="^",
            linewidth=2.2,
            color="#16a34a",
            label="threads (ThreadPoolExecutor, Python 3.13t free-threaded)",
        )

    ax.set_title("findex Parallel Indexing Speedup vs Worker Count", fontsize=12, fontweight="bold")
    ax.set_xlabel("Workers (N)")
    ax.set_ylabel("Speedup = T_serial / T_N")
    ax.set_xticks(worker_counts)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run Lab 5 concurrency benchmarks")
    parser.add_argument("root", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-json", type=Path, default=Path("docs/bench_lab5.json"))
    parser.add_argument("--chart", type=Path, default=Path("docs/speedup_lab5.png"))
    parser.add_argument(
        "--ft-json",
        type=Path,
        default=None,
        help="Optional JSON produced by Python 3.13t run to overlay on chart",
    )
    parser.add_argument(
        "--only-threads",
        action="store_true",
        help="Measure only serial(1) + threads (used when running under Python 3.13t)",
    )
    args = parser.parse_args()

    cpu_count = os.cpu_count() or 4
    raw_workers = [1, 2, 4, 8, cpu_count]
    worker_counts = sorted(set(raw_workers))

    log.info("Python: %s | gil_enabled=%s", sys.version.split()[0], getattr(sys, "_is_gil_enabled", lambda: True)())
    log.info("CPU: %s | logical_cores=%d", _cpu_model(), cpu_count)

    serial_row = measure_cell(args.root, "serial", 1, repeats=args.repeats)
    base_wall = serial_row["wall_s"]
    serial_row["speedup"] = 1.0
    log.info(
        "serial    | w=%2d | wall=%.4fs | cpu=%.4fs | merge=%6.2fms | rss=%.1f MiB | speedup=1.00x",
        1,
        serial_row["wall_s"],
        serial_row["cpu_s"],
        serial_row["merge_ms"],
        serial_row["parent_rss_mib"],
    )

    rows: list[dict[str, Any]] = [serial_row]
    executors: list[ExecutorKind] = ["threads"] if args.only_threads else ["threads", "processes"]

    for ex in executors:
        for w in worker_counts:
            row = measure_cell(args.root, ex, w, repeats=args.repeats)
            row["speedup"] = round(base_wall / row["wall_s"], 2) if row["wall_s"] > 0 else 0.0
            rows.append(row)
            log.info(
                "%-9s | w=%2d | wall=%.4fs | cpu=%.4fs | merge=%6.2fms | rss=%.1f MiB | speedup=%.2fx",
                ex,
                w,
                row["wall_s"],
                row["cpu_s"],
                row["merge_ms"],
                row["parent_rss_mib"],
                row["speedup"],
            )

    payload = {
        "python_version": sys.version.split()[0],
        "gil_enabled": bool(getattr(sys, "_is_gil_enabled", lambda: True)()),
        "cpu_model": _cpu_model(),
        "cpu_count": cpu_count,
        "worker_counts": worker_counts,
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not args.only_threads:
        proc_speedups = [
            next(r["speedup"] for r in rows if r["executor"] == "processes" and r["workers"] == w)
            for w in worker_counts
        ]
        thread_speedups = [
            next(r["speedup"] for r in rows if r["executor"] == "threads" and r["workers"] == w)
            for w in worker_counts
        ]
        ft_speedups: list[float] | None = None
        if args.ft_json is not None and args.ft_json.exists():
            ft_data = json.loads(args.ft_json.read_text(encoding="utf-8"))
            ft_rows = ft_data["rows"]
            ft_speedups = [
                next(r["speedup"] for r in ft_rows if r["executor"] == "threads" and r["workers"] == w)
                for w in worker_counts
            ]
        render_speedup_chart(worker_counts, proc_speedups, thread_speedups, ft_speedups, args.chart)
        log.info("Saved benchmark JSON -> %s and chart -> %s", args.out_json, args.chart)


if __name__ == "__main__":
    main()
