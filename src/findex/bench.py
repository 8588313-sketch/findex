"""Benchmarks for Lab 2 README tables.

Usage: python -m findex.bench data/
"""
import argparse
import gc
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
    print("=== 1. MEMORY COMPARISON (tracemalloc) ===")

    def build_regular():
        return {
            term: [RegularPosting(d, tf) for d, tf in plist]
            for term, plist in raw.items()
        }

    def build_slotted():
        return {
            term: [SlottedPosting(d, tf) for d, tf in plist]
            for term, plist in raw.items()
        }

    def build_arrays():
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
        print(
            f"  {label:<24} | current: {current / 1024:8.1f} KiB | peak: {peak / 1024:8.1f} KiB"
        )
        del obj

    for with_pos in (False, True):
        gc.collect()
        tracemalloc.start()
        idx = build_index(root, with_positions=with_pos)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(
            f"  Full index (pos={with_pos!s:<5}) | current: {current / 1024:8.1f} KiB | peak: {peak / 1024:8.1f} KiB"
        )
        del idx


def bench_search(root: Path) -> None:
    print("\n=== 2. BOOLEAN SEARCH: merge vs set ===")
    index = build_index(root, with_positions=False)
    sorted_terms = sorted(index.postings.items(), key=lambda kv: len(kv[1]), reverse=True)

    freq_terms = [t for t, _ in sorted_terms[:2]]
    rare_terms = [t for t, _ in sorted_terms[-2:]]
    print(f"  2 most frequent terms: {freq_terms} (df={[len(index.postings[t]) for t in freq_terms]})")
    print(f"  2 rarest terms:        {rare_terms} (df={[len(index.postings[t]) for t in rare_terms]})")

    queries = [
        ("Frequent AND", f"{freq_terms[0]} AND {freq_terms[1]}"),
        ("Frequent OR", f"{freq_terms[0]} OR {freq_terms[1]}"),
        ("Frequent NOT", f"{freq_terms[0]} NOT {freq_terms[1]}"),
        ("Rare AND", f"{rare_terms[0]} AND {rare_terms[1]}"),
        ("Rare OR", f"{rare_terms[0]} OR {rare_terms[1]}"),
    ]

    iters = 5000
    for name, q in queries:
        t0 = time.perf_counter()
        for _ in range(iters):
            res_merge = search_index(index, q, engine="merge")
        dt_merge = (time.perf_counter() - t0) * 1e6 / iters

        t0 = time.perf_counter()
        for _ in range(iters):
            res_set = search_index(index, q, engine="set")
        dt_set = (time.perf_counter() - t0) * 1e6 / iters

        assert res_merge == res_set
        print(
            f"  {name:<14} ({q:<22}) | hits={len(res_merge):2d} | merge={dt_merge:6.2f} µs | set={dt_set:6.2f} µs"
        )


def bench_serialization(root: Path) -> None:
    print("\n=== 3. SERIALIZATION: pickle vs binary array('I') ===")
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

            print(f"  [with_positions={with_pos}]")
            print(
                f"    pickle: size={pkl_path.stat().st_size / 1024:7.1f} KiB | save={t_pkl_save:6.2f} ms | load={t_pkl_load:6.2f} ms"
            )
            print(
                f"    binary: size={bin_path.stat().st_size / 1024:7.1f} KiB | save={t_bin_save:6.2f} ms | load={t_bin_load:6.2f} ms"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lab 2 benchmarks")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    raw = _collect_raw_postings(args.root)
    bench_memory(raw, args.root)
    bench_search(args.root)
    bench_serialization(args.root)


if __name__ == "__main__":
    main()
