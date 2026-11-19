"""Deliberately greedy version: everything is materialized as lists."""
from collections import Counter
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def compute_stats_eager(root: Path, limit: int | None = None) -> tuple[int, Counter]:
    docs = list(iter_documents(root))          # all texts in memory at once
    if limit is not None:
        docs = docs[:limit]
    token_lists = [list(tokenize(d.text)) for d in docs]  # all tokens too
    counts: Counter = Counter()
    for tokens in token_lists:
        counts.update(tokens)
    return len(docs), counts
