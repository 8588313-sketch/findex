from __future__ import annotations

import functools
import heapq
import logging
import math
import re
from pathlib import Path
from typing import Protocol

from findex.ast_query import parse_query
from findex.decorators import timed
from findex.models import Index, SearchResult

log = logging.getLogger("findex.search")


class Scorer(Protocol):
    """Duck-typed ranking scorer protocol."""

    def score(
        self,
        tf: int,
        df: int,
        doc_len: int,
        num_docs: int,
        avg_dl: float,
    ) -> float: ...


class TfIdf:
    """Sublinear TF-IDF scorer."""

    def score(
        self,
        tf: int,
        df: int,
        doc_len: int,
        num_docs: int,
        avg_dl: float,
    ) -> float:
        if tf <= 0 or df <= 0 or num_docs <= 0:
            return 0.0
        tf_weight = 1.0 + math.log(tf)
        idf = math.log((num_docs + 1.0) / (df + 1.0)) + 1.0
        return tf_weight * idf

    def __repr__(self) -> str:
        return "TfIdf()"


class BM25:
    """Okapi BM25 scorer with term saturation and length normalization."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(
        self,
        tf: int,
        df: int,
        doc_len: int,
        num_docs: int,
        avg_dl: float,
    ) -> float:
        if tf <= 0 or df <= 0 or num_docs <= 0:
            return 0.0
        idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
        norm_dl = (doc_len / avg_dl) if avg_dl > 0 else 1.0
        denom = tf + self.k1 * (1.0 - self.b + self.b * norm_dl)
        return idf * (tf * (self.k1 + 1.0)) / denom

    def __repr__(self) -> str:
        return f"BM25(k1={self.k1}, b={self.b})"


def make_snippet(
    text: str,
    terms: list[str],
    phrases: list[tuple[str, ...]] | None = None,
    window: int = 80,
) -> str:
    """Extract a +-window character snippet around the best match and highlight terms."""
    clean = " ".join(text.split())
    if not clean:
        return ""

    patterns: list[str] = []
    if phrases:
        for ph in phrases:
            patterns.append(r"\b" + r"\W+".join(re.escape(w) for w in ph) + r"\b")
    for t in dict.fromkeys(terms):
        patterns.append(r"\b" + re.escape(t) + r"\b")

    if not patterns:
        return clean[: window * 2] + ("..." if len(clean) > window * 2 else "")

    rx = re.compile("|".join(patterns), flags=re.IGNORECASE)
    matches = list(rx.finditer(clean))
    if not matches:
        return clean[: window * 2] + ("..." if len(clean) > window * 2 else "")

    # Pick window with the highest density of query matches
    best_center = matches[0].start()
    best_count = -1
    for m in matches:
        c = m.start()
        cnt = sum(1 for other in matches if abs(other.start() - c) <= window)
        if cnt > best_count:
            best_count = cnt
            best_center = c

    start = max(0, best_center - window)
    end = min(len(clean), best_center + window)
    snippet = clean[start:end]
    highlighted = rx.sub(lambda m: f"[{m.group(0)}]", snippet)

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(clean) else ""
    return f"{prefix}{highlighted}{suffix}"


# Hold strong references by index id so @lru_cache stays safe and deterministic
_INDEX_REGISTRY: dict[int, Index] = {}


@functools.lru_cache(maxsize=128)
def _cached_eval(index_key: int, query: str) -> tuple[tuple[int, ...], tuple[str, ...], tuple[tuple[str, ...], ...]]:
    index = _INDEX_REGISTRY[index_key]
    ast = parse_query(query)
    doc_ids = tuple(sorted(ast.evaluate(index)))
    pos_terms = tuple(dict.fromkeys(ast.positive_terms()))
    phrases = tuple(ast.phrases())
    return doc_ids, pos_terms, phrases


def clear_query_cache() -> None:
    _cached_eval.cache_clear()
    _INDEX_REGISTRY.clear()


def query_cache_info():
    return _cached_eval.cache_info()


@timed
def search(
    index: Index,
    query: str,
    scorer: Scorer | None = None,
    top_k: int = 5,
    include_snippets: bool = True,
) -> list[SearchResult]:
    """Execute a Boolean/phrase query and rank matching documents using a Scorer."""
    active_scorer: Scorer = scorer if scorer is not None else BM25()
    idx_key = id(index)
    _INDEX_REGISTRY[idx_key] = index

    before = _cached_eval.cache_info()
    matched_ids, pos_terms, phrases = _cached_eval(idx_key, query)
    after = _cached_eval.cache_info()
    cache_status = "HIT" if after.hits > before.hits else "MISS"
    log.info(
        "[lru_cache] %s for %r (hits=%d, misses=%d)",
        cache_status,
        query,
        after.hits,
        after.misses,
    )

    if not matched_ids:
        return []

    matched_set = set(matched_ids)
    scores: dict[int, float] = {doc_id: 0.0 for doc_id in matched_ids}
    num_docs = index.num_docs
    avg_dl = index.avg_doc_length

    for term in pos_terms:
        plist = index.get(term)
        if not plist:
            continue
        term_df = len(plist)
        for p in plist:
            if p.doc_id in matched_set:
                doc_len = index.docs[p.doc_id].length
                scores[p.doc_id] += active_scorer.score(
                    tf=p.tf,
                    df=term_df,
                    doc_len=doc_len,
                    num_docs=num_docs,
                    avg_dl=avg_dl,
                )

    top_pairs = heapq.nlargest(top_k, scores.items(), key=lambda kv: (kv[1], -kv[0]))
    results: list[SearchResult] = []
    for doc_id, sc in top_pairs:
        meta = index.docs[doc_id]
        snippet = ""
        if include_snippets:
            p_path = Path(meta.path)
            if p_path.exists():
                raw_text = p_path.read_text(encoding="utf-8", errors="ignore")
                snippet = make_snippet(raw_text, list(pos_terms), list(phrases))
        results.append(
            SearchResult(
                score=sc,
                doc_id=doc_id,
                title=meta.title,
                path=meta.path,
                snippet=snippet,
            )
        )
    return results
