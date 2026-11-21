"""Deliberately greedy version: everything is materialized as lists."""
from collections import Counter
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def compute_stats_eager(root: Path, limit: int | None = None) -> tuple[int, Counter[str]]:
    docs = list(iter_documents(root))
    if limit is not None:
        docs = docs[:limit]
    token_lists = [list(tokenize(d.text)) for d in docs]
    counts: Counter[str] = Counter()
    for tokens in token_lists:
        counts.update(tokens)
    return len(docs), counts
