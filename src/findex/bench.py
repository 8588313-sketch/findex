"""Benchmarks for Lab 2 README tables.

Usage: python -m findex.bench data/
"""
import argparse
import gc
import logging
import tempfile
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from findex.builder import build_index
from findex.corpus import iter_documents
from findex.query import search_index
from findex.storage import load_binary, load_pickle, save_binary, save_pickle
from findex.tokenize import tokenize

log = logging.getLogger("findex.bench")


@dataclass
class RegularPosting:
    doc_id: int
    tf: int


@dataclass(frozen=True, slots=True)
class SlottedPosting:
    doc_id: int
    tf: int


def _collect_raw_postings(root: Path) -> dict[str, list[tuple[int, int]]]:
    raw: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for doc in iter_documents(root):
        for term, tf in Counter(tokenize(doc.text)).items():
            raw[term].append((doc.doc_id, tf))
    return dict(raw)


def bench_memory(raw: dict[str, list[tuple[int, int]]], root: Path) -> None:
    log.info("=== 1. MEMORY COMPARISON (tracemalloc) ===")

    def build_regular() -> dict[str, list[RegularPosting]]:
        return {
            term: [RegularPosting(d, tf) for d, tf in plist]
            for term, plist in raw.items()
        }

    def build_slotted() -> dict[str, list[SlottedPosting]]:
        return {
            term: [SlottedPosting(d, tf) for d, tf in plist]
            for term, plist in raw.items()
        }

    def build_arrays() -> dict[str, tuple[array[int], array[int]]]:
        return {
            term: (
                array("I", (d for d, _ in plist)),
                array("I", (tf for _, tf in plist)),
            )
            for term, plist in raw.items()
        }

    for label, fn in (
        ("Regular @dataclass", build_regular),
        ("@dataclass(slots=True)", build_slotted),
        ("Pair of array('I')", build_arrays),
    ):
        gc.collect()
        tracemalloc.start()
        obj = fn()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        log.info(
            "  %-24s | current: %8.1f KiB | peak: %8.1f KiB",
            label,
            current / 1024,
            peak / 1024,
        )
        del obj

    for with_pos in (False, True):
        gc.collect()
        tracemalloc.start()
        idx = build_index(root, with_positions=with_pos)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        log.info(
            "  Full index (pos=%-5s) | current: %8.1f KiB | peak: %8.1f KiB",
            with_pos,
            current / 1024,
            peak / 1024,
        )
        del idx


def bench_search(root: Path) -> None:
    log.info("=== 2. BOOLEAN SEARCH: merge vs set ===")
    index = build_index(root, with_positions=False)
    sorted_terms = sorted(index.items(), key=lambda kv: len(kv[1]), reverse=True)

    freq_terms = [t for t, _ in sorted_terms[:2]]
    rare_terms = [t for t, _ in sorted_terms[-2:]]

    queries = [
        ("Frequent AND", f"{freq_terms[0]} AND {freq_terms[1]}"),
        ("Frequent OR", f"{freq_terms[0]} OR {freq_terms[1]}"),
        ("Frequent NOT", f"{freq_terms[0]} NOT {freq_terms[1]}"),
        ("Rare AND", f"{rare_terms[0]} AND {rare_terms[1]}"),
        ("Rare OR", f"{rare_terms[0]} OR {rare_terms[1]}"),
    ]

    iters = 5000
    for name, q in queries:
        res_merge: list[int] = []
        t0 = time.perf_counter()
        for _ in range(iters):
            res_merge = search_index(index, q, engine="merge")
        dt_merge = (time.perf_counter() - t0) * 1e6 / iters

        res_set: list[int] = []
        t0 = time.perf_counter()
        for _ in range(iters):
            res_set = search_index(index, q, engine="set")
        dt_set = (time.perf_counter() - t0) * 1e6 / iters

        assert res_merge == res_set
        log.info(
            "  %-14s (%-22s) | hits=%2d | merge=%6.2f us | set=%6.2f us",
            name,
            q,
            len(res_merge),
            dt_merge,
            dt_set,
        )


def bench_serialization(root: Path) -> None:
    log.info("=== 3. SERIALIZATION: pickle vs binary array('I') ===")
    for with_pos in (False, True):
        index = build_index(root, with_positions=with_pos)
        with tempfile.TemporaryDirectory() as tmp:
            pkl_path = Path(tmp) / "index.pkl"
            bin_path = Path(tmp) / "index.bin"

            t0 = time.perf_counter()
            save_pickle(index, pkl_path)
            t_pkl_save = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            _ = load_pickle(pkl_path)
            t_pkl_load = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            save_binary(index, bin_path)
            t_bin_save = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            _ = load_binary(bin_path)
            t_bin_load = (time.perf_counter() - t0) * 1000

            log.info("  [with_positions=%s]", with_pos)
            log.info(
                "    pickle: size=%7.1f KiB | save=%6.2f ms | load=%6.2f ms",
                pkl_path.stat().st_size / 1024,
                t_pkl_save,
                t_pkl_load,
            )
            log.info(
                "    binary: size=%7.1f KiB | save=%6.2f ms | load=%6.2f ms",
                bin_path.stat().st_size / 1024,
                t_bin_save,
                t_bin_load,
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run Lab 2 benchmarks")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    raw = _collect_raw_postings(args.root)
    bench_memory(raw, args.root)
    bench_search(args.root)
    bench_serialization(args.root)


if __name__ == "__main__":
    main()
