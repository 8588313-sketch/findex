"""Corpus statistics in one streaming pass.

Usage: python -m findex.stats data/ [--limit N] [--eager]
"""
import argparse
import logging
import time
import tracemalloc
from collections import Counter
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.eager import compute_stats_eager
from findex.tokenize import tokenize

log = logging.getLogger("findex.stats")


def compute_stats(root: Path, limit: int | None = None) -> tuple[int, Counter[str]]:
    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)
    counts: Counter[str] = Counter()
    n_docs = 0
    for doc in docs:
        n_docs += 1
        counts.update(tokenize(doc.text))
    return n_docs, counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Corpus statistics")
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--eager",
        action="store_true",
        help="use the greedy list-based version",
    )
    args = parser.parse_args()

    compute = compute_stats_eager if args.eager else compute_stats

    tracemalloc.start()
    start = time.perf_counter()
    n_docs, counts = compute(args.root, args.limit)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    log.info("mode:        %s", "eager" if args.eager else "lazy")
    log.info("documents:   %d", n_docs)
    log.info("tokens:      %d", counts.total())
    log.info("vocabulary:  %d", len(counts))
    log.info("top 50 terms:")
    for term, n in counts.most_common(50):
        log.info("  %-20s %d", term, n)
    log.info("elapsed:     %.2f s", elapsed)
    log.info("peak memory: %.1f MiB", peak / 1024 / 1024)


if __name__ == "__main__":
    main()
