"""Backward-compatible module entrypoint for `python -m findex.index`."""
import sys

from findex.cli import app

if __name__ == "__main__":
    sys.argv.insert(1, "index")
    app()
