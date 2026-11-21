"""Ranked Boolean & phrase search CLI.

Usage: python -m findex.search index.bin "query" [--scorer {bm25,tfidf}] [--top-k 5] [--repeat]
"""
import argparse
import logging
import tracemalloc
from pathlib import Path

from findex.ranking import BM25, Scorer, TfIdf, search
from findex.storage import open_index


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Ranked search over inverted index")
    parser.add_argument("index_path", type=Path, help="Path to saved index file")
    parser.add_argument("query", type=str, help="Boolean/phrase query")
    parser.add_argument(
        "--scorer",
        choices=("bm25", "tfidf"),
        default="bm25",
        help="Ranking scorer (default: bm25)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to show")
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Run the same query a second time to demonstrate @lru_cache HIT",
    )
    args = parser.parse_args()

    scorer: Scorer = BM25() if args.scorer == "bm25" else TfIdf()

    tracemalloc.start()
    with open_index(args.index_path) as index:
        results = search(index, args.query, scorer=scorer, top_k=args.top_k)
        if args.repeat:
            _ = search(index, args.query, scorer=scorer, top_k=args.top_k)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"scorer:      {scorer!r}")
    print(f"top results: {len(results)}")
    for rank, res in enumerate(results, start=1):
        print(f"  {rank}. [{res.doc_id}] score={res.score:.4f} | {res.title}")
        if res.snippet:
            print(f"     {res.snippet}")
    print(f"peak memory: {peak / 1024 / 1024:.2f} MiB")


if __name__ == "__main__":
    main()
