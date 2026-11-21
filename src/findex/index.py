"""Build and serialize an inverted index.

Usage: python -m findex.index data/ --out index.bin [--positions] [--format {bin,pickle}]
"""
import argparse
import time
import tracemalloc
from pathlib import Path

from findex.builder import build_index
from findex.storage import save_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build inverted index")
    parser.add_argument("root", type=Path, help="Path to corpus directory")
    parser.add_argument("--out", type=Path, required=True, help="Output index file path")
    parser.add_argument(
        "--format",
        choices=("bin", "pickle"),
        default=None,
        help="Serialization format (default: inferred from --out extension or 'bin')",
    )
    parser.add_argument(
        "--positions",
        action="store_true",
        help="Store token positions inside postings",
    )
    args = parser.parse_args()

    fmt = args.format
    if fmt is None:
        fmt = "pickle" if args.out.suffix in {".pkl", ".pickle"} else "bin"

    tracemalloc.start()
    t0 = time.perf_counter()
    index = build_index(args.root, with_positions=args.positions)
    save_index(index, args.out, fmt=fmt)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_postings = sum(len(plist) for plist in index.postings.values())
    file_size = args.out.stat().st_size

    print(f"documents:   {len(index.docs)}")
    print(f"vocabulary:  {len(index.postings)}")
    print(f"postings:    {total_postings}")
    print(f"positions:   {args.positions}")
    print(f"format:      {fmt} ({file_size / 1024:.1f} KiB)")
    print(f"elapsed:     {elapsed * 1000:.2f} ms")
    print(f"peak memory: {peak / 1024 / 1024:.2f} MiB")


if __name__ == "__main__":
    main()
