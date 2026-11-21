"""Search a saved inverted index.

Usage: python -m findex.search index.bin "query" [--engine {merge,set}]
"""
import argparse
import time
import tracemalloc
from pathlib import Path

from findex.query import normalize_term, search_index
from findex.storage import load_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Search inverted index")
    parser.add_argument("index_path", type=Path, help="Path to saved index file")
    parser.add_argument("query", type=str, help="Boolean search query")
    parser.add_argument(
        "--engine",
        choices=("merge", "set"),
        default="merge",
        help="Intersection/union engine (default: merge)",
    )
    args = parser.parse_args()

    tracemalloc.start()
    t0 = time.perf_counter()
    index = load_index(args.index_path)
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    doc_ids = search_index(index, args.query, engine=args.engine)
    t_search = time.perf_counter() - t1

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"engine:      {args.engine}")
    print(f"matches:     {len(doc_ids)}")
    query_terms = [
        t for raw in args.query.split()
        if raw.upper() not in {"AND", "OR", "NOT"} and not raw.startswith("-")
        if (t := normalize_term(raw)) is not None
    ]

    for doc_id in doc_ids:
        meta = index.docs[doc_id]
        tf_details = []
        for term in query_terms:
            plist = index.postings.get(term, ())
            for p in plist:
                if p.doc_id == doc_id:
                    if index.with_positions and p.positions:
                        tf_details.append(f"{term}:tf={p.tf}@pos{list(p.positions[:5])}")
                    else:
                        tf_details.append(f"{term}:tf={p.tf}")
                    break
        extra = f" ({', '.join(tf_details)})" if tf_details else ""
        print(f"  [{doc_id}] {meta.title} — {meta.path} (len={meta.length}){extra}")

    print(f"load time:   {t_load * 1000:.2f} ms")
    print(f"search time: {t_search * 1000:.3f} ms")
    print(f"peak memory: {peak / 1024 / 1024:.2f} MiB")


if __name__ == "__main__":
    main()
