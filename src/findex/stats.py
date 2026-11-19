"""Corpus statistics in one streaming pass.

Usage: python -m findex.stats data/ [--limit N] [--eager]
"""
import argparse
import time
import tracemalloc
from collections import Counter
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.eager import compute_stats_eager
from findex.tokenize import tokenize


def compute_stats(root: Path, limit: int | None = None) -> tuple[int, Counter]:
    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)
    counts: Counter = Counter()
    n_docs = 0
    for doc in docs:
        n_docs += 1
        counts.update(tokenize(doc.text))
    return n_docs, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus statistics")
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eager", action="store_true",
                        help="use the greedy list-based version")
    args = parser.parse_args()

    compute = compute_stats_eager if args.eager else compute_stats

    tracemalloc.start()
    start = time.perf_counter()
    n_docs, counts = compute(args.root, args.limit)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"mode:       {'eager' if args.eager else 'lazy'}")
    print(f"documents:  {n_docs}")
    print(f"tokens:     {counts.total()}")
    print(f"vocabulary: {len(counts)}")
    print("top 50 terms:")
    for term, n in counts.most_common(50):
        print(f"  {term:<20} {n}")
    print(f"elapsed:    {elapsed:.2f} s")
    print(f"peak memory: {peak / 1024 / 1024:.1f} MiB")


if __name__ == "__main__":
    main()
