from __future__ import annotations

import functools
import heapq
import logging
import math
import re
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from findex.ast_query import parse_query
from findex.decorators import timed
from findex.models import Index, SearchResult

log = logging.getLogger("findex.search")
ScorerName = Literal["bm25", "tfidf"]


class Scorer(Protocol):
    """Structural typing protocol for ranking scorers (scalar + vectorized NumPy)."""

    def score(
        self,
        tf: int,
        df: int,
        doc_len: int,
        num_docs: int,
        avg_dl: float,
    ) -> float: ...

    def score_vectorized(
        self,
        tfs: NDArray[np.int32],
        df: int,
        doc_lens: NDArray[np.int32],
        num_docs: int,
        avg_dl: float,
    ) -> NDArray[np.float64]: ...


class TfIdf:
    """Sublinear TF-IDF scorer (scalar Lab 3 baseline + vectorized NumPy)."""

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

    def score_vectorized(
        self,
        tfs: NDArray[np.int32],
        df: int,
        doc_lens: NDArray[np.int32],
        num_docs: int,
        avg_dl: float,
    ) -> NDArray[np.float64]:
        if df <= 0 or num_docs <= 0 or tfs.size == 0:
            return np.zeros(tfs.shape, dtype=np.float64)
        tf_f = tfs.astype(np.float64)
        idf = math.log((num_docs + 1.0) / (df + 1.0)) + 1.0
        out = np.zeros(tfs.shape, dtype=np.float64)
        pos = tf_f > 0.0
        out[pos] = (1.0 + np.log(tf_f[pos])) * idf
        return out

    def __repr__(self) -> str:
        return "TfIdf()"


class BM25:
    """Okapi BM25 scorer (scalar Lab 3 baseline + vectorized NumPy ufunc expression)."""

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

    def score_vectorized(
        self,
        tfs: NDArray[np.int32],
        df: int,
        doc_lens: NDArray[np.int32],
        num_docs: int,
        avg_dl: float,
    ) -> NDArray[np.float64]:
        if df <= 0 or num_docs <= 0 or tfs.size == 0:
            return np.zeros(tfs.shape, dtype=np.float64)
        tf_f = tfs.astype(np.float64)
        idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
        if avg_dl > 0.0:
            norm_dl = doc_lens.astype(np.float64) / avg_dl
        else:
            norm_dl = np.ones(doc_lens.shape, dtype=np.float64)
        denom = tf_f + self.k1 * (1.0 - self.b + self.b * norm_dl)
        scores = np.where(tf_f > 0.0, idf * (tf_f * (self.k1 + 1.0)) / denom, 0.0)
        return scores.astype(np.float64)

    def __repr__(self) -> str:
        return f"BM25(k1={self.k1}, b={self.b})"


def make_scorer(name: ScorerName = "bm25") -> Scorer:
    return BM25() if name == "bm25" else TfIdf()


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


_INDEX_REGISTRY: dict[int, Index] = {}


@functools.lru_cache(maxsize=128)
def _cached_eval(
    index_key: int,
    query: str,
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[tuple[str, ...], ...]]:
    index = _INDEX_REGISTRY[index_key]
    ast = parse_query(query)
    doc_ids = tuple(sorted(ast.evaluate(index)))
    pos_terms = tuple(dict.fromkeys(ast.positive_terms()))
    phrases = tuple(ast.phrases())
    return doc_ids, pos_terms, phrases


def clear_query_cache() -> None:
    _cached_eval.cache_clear()
    _INDEX_REGISTRY.clear()


def query_cache_info() -> functools._CacheInfo:  # pyright: ignore[reportPrivateUsage]
    return _cached_eval.cache_info()


def _top_k_from_dense_scores(
    doc_ids_arr: NDArray[np.int32],
    dense_scores: NDArray[np.float64],
    candidate_rows: NDArray[np.intp],
    top_k: int,
) -> list[tuple[int, float]]:
    """Select top-k (doc_id, score) pairs via np.argpartition with (-score, doc_id) tie-breaking."""
    n = int(candidate_rows.size)
    if n == 0 or top_k <= 0:
        return []
    k = min(top_k, n)
    cand_scores = dense_scores[candidate_rows]
    cand_doc_ids = doc_ids_arr[candidate_rows]

    # Tie-break key matches heapq.nlargest(..., key=lambda kv: (kv[1], -kv[0]))
    sort_keys = np.round(-cand_scores, 12) + cand_doc_ids.astype(np.float64) * 1e-13
    if k < n:
        part_idx = np.argpartition(sort_keys, k - 1)[:k]
        sub_scores = cand_scores[part_idx]
        sub_ids = cand_doc_ids[part_idx]
        sub_keys = sort_keys[part_idx]
    else:
        sub_scores = cand_scores
        sub_ids = cand_doc_ids
        sub_keys = sort_keys

    order = np.argsort(sub_keys)
    return [(int(sub_ids[i]), float(sub_scores[i])) for i in order]


def search_python_baseline(
    index: Index,
    query: str,
    scorer: Scorer | None = None,
    top_k: int = 5,
    include_snippets: bool = True,
) -> list[SearchResult]:
    """Original Lab 3 pure-Python dict/loop scorer kept for exact equivalence & speedup benchmarks."""
    active_scorer: Scorer = scorer if scorer is not None else BM25()
    idx_key = id(index)
    _INDEX_REGISTRY[idx_key] = index

    matched_ids, pos_terms, phrases = _cached_eval(idx_key, query)
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


@timed
def search(
    index: Index,
    query: str,
    scorer: Scorer | None = None,
    top_k: int = 5,
    include_snippets: bool = True,
) -> list[SearchResult]:
    """Execute a Boolean/phrase query and rank matching documents using vectorized NumPy arrays."""
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

    num_docs = index.num_docs
    avg_dl = index.avg_doc_length
    doc_ids_arr = index.doc_ids_array
    doc_lengths_arr = index.doc_lengths_array
    id_to_row = index.doc_id_to_row

    is_contiguous = (
        num_docs > 0
        and int(doc_ids_arr[0]) == 0
        and int(doc_ids_arr[-1]) == num_docs - 1
    )

    dense_scores = np.zeros(num_docs, dtype=np.float64)
    if is_contiguous:
        candidate_rows = np.asarray(matched_ids, dtype=np.intp)
    else:
        candidate_rows = np.array([id_to_row[d] for d in matched_ids], dtype=np.intp)

    matched_mask = np.zeros(num_docs, dtype=np.bool_)
    matched_mask[candidate_rows] = True

    for term in pos_terms:
        np_pair = index.get_numpy_postings(term)
        if np_pair is None:
            continue
        p_doc_ids, p_tfs = np_pair
        term_df = int(p_doc_ids.size)
        if is_contiguous:
            rows = p_doc_ids.astype(np.intp)
        else:
            rows = np.array([id_to_row[int(d)] for d in p_doc_ids], dtype=np.intp)
        valid = matched_mask[rows]
        if not np.any(valid):
            continue
        valid_rows = rows[valid]
        valid_tfs = p_tfs[valid]
        valid_lens = doc_lengths_arr[valid_rows]
        term_scores = active_scorer.score_vectorized(
            tfs=valid_tfs,
            df=term_df,
            doc_lens=valid_lens,
            num_docs=num_docs,
            avg_dl=avg_dl,
        )
        dense_scores[valid_rows] += term_scores

    top_pairs = _top_k_from_dense_scores(doc_ids_arr, dense_scores, candidate_rows, top_k)
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
