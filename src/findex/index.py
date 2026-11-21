"""Build and serialize an inverted index.

Usage: python -m findex.index data/ --out index.bin [--no-positions] [--format {bin,pickle}]
"""
import argparse
import logging
import tracemalloc
from pathlib import Path

from findex.builder import build_index
from findex.storage import save_index


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
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
        "--no-positions",
        action="store_true",
        help="Disable storing token positions (enabled by default for phrase search)",
    )
    args = parser.parse_args()

    fmt = args.format
    if fmt is None:
        fmt = "pickle" if args.out.suffix in {".pkl", ".pickle"} else "bin"

    with_positions = not args.no_positions

    tracemalloc.start()
    index = build_index(args.root, with_positions=with_positions)
    save_index(index, args.out, fmt=fmt)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_postings = sum(len(plist) for plist in index.values())
    file_size = args.out.stat().st_size

    print(f"index:       {index!r}")
    print(f"postings:    {total_postings}")
    print(f"format:      {fmt} ({file_size / 1024:.1f} KiB)")
    print(f"peak memory: {peak / 1024 / 1024:.2f} MiB")


if __name__ == "__main__":
    main()
