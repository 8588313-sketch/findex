from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from findex.decorators import timed
from findex.models import Index, SearchResult
from findex.ranking import Scorer, make_snippet, search
from findex.tokenize import tokenize

log = logging.getLogger("findex.semantic")

SearchMode = Literal["lexical", "semantic", "hybrid"]
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class TextEmbedder(Protocol):
    """Protocol for dense text embedding providers (fastembed or deterministic test doubles)."""

    def encode_texts(self, texts: list[str]) -> NDArray[np.float32]: ...


class FastEmbedProvider:
    """Lightweight ONNX-backed dense embedder using fastembed."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def encode_texts(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = list(self._model.embed(texts))
        mat = np.asarray(vectors, dtype=np.float32)
        return l2_normalize_rows(mat)


_EMBEDDER_CACHE: dict[str, FastEmbedProvider] = {}


def get_default_embedder(model_name: str = DEFAULT_EMBED_MODEL) -> FastEmbedProvider:
    provider = _EMBEDDER_CACHE.get(model_name)
    if provider is None:
        provider = FastEmbedProvider(model_name=model_name)
        _EMBEDDER_CACHE[model_name] = provider
    return provider


def l2_normalize_rows(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalize each row vector to unit L2 norm so dot product equals cosine similarity."""
    if matrix.size == 0:
        return matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-12, norms, 1.0).astype(np.float32)
    return (matrix / safe_norms).astype(np.float32)


@dataclass(frozen=True, slots=True)
class ChunkMeta:
    chunk_id: int
    doc_id: int
    chunk_idx: int
    text: str


@dataclass(slots=True)
class SemanticIndex:
    """In-memory L2-normalized chunk embedding matrix (.npy) with doc_id mapping."""

    embeddings: NDArray[np.float32]  # shape: (num_chunks, dim)
    chunks: list[ChunkMeta]
    chunk_doc_ids: NDArray[np.int32]  # shape: (num_chunks,)
    model_name: str = DEFAULT_EMBED_MODEL

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0


def chunk_text(text: str, chunk_words: int = 120, overlap_words: int = 30) -> list[str]:
    """Split text into overlapping word windows for dense passage embedding."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]
    step = max(1, chunk_words - overlap_words)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_words >= len(words):
            break
    return chunks


def default_semantic_dir(index_path: Path) -> Path:
    """Resolve companion directory for embeddings.npy next to index.bin."""
    return index_path.parent / f"{index_path.stem}_semantic"


def _read_doc_body(path_str: str, fallback_title: str) -> str:
    p = Path(path_str)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8", errors="ignore")
    return fallback_title


@timed
def build_embeddings(
    index: Index,
    out_dir: Path,
    embedder: TextEmbedder | None = None,
    model_name: str = DEFAULT_EMBED_MODEL,
    chunk_words: int = 120,
    overlap_words: int = 30,
) -> SemanticIndex:
    """Split indexed documents into chunks, compute L2-normalized embeddings, and save to .npy."""
    provider: TextEmbedder = embedder if embedder is not None else get_default_embedder(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[ChunkMeta] = []
    texts_to_encode: list[str] = []

    for doc_id in index.all_doc_ids:
        meta = index.docs[doc_id]
        raw = _read_doc_body(meta.path, meta.title)
        full_text = f"{meta.title}\n{raw}".strip()
        doc_chunks = chunk_text(full_text, chunk_words=chunk_words, overlap_words=overlap_words)
        if not doc_chunks:
            doc_chunks = [meta.title or f"doc_{doc_id}"]
        for c_idx, c_text in enumerate(doc_chunks):
            cid = len(chunks)
            chunks.append(ChunkMeta(chunk_id=cid, doc_id=doc_id, chunk_idx=c_idx, text=c_text))
            texts_to_encode.append(c_text)

    if texts_to_encode:
        raw_mat = provider.encode_texts(texts_to_encode)
        embeddings = l2_normalize_rows(np.asarray(raw_mat, dtype=np.float32))
    else:
        embeddings = np.zeros((0, 0), dtype=np.float32)

    chunk_doc_ids = np.array([c.doc_id for c in chunks], dtype=np.int32)

    npy_path = out_dir / "embeddings.npy"
    meta_path = out_dir / "chunks.json"
    np.save(npy_path, embeddings)
    payload: dict[str, Any] = {
        "model_name": model_name,
        "num_chunks": len(chunks),
        "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "chunk_idx": c.chunk_idx,
                "text": c.text,
            }
            for c in chunks
        ],
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Saved %d normalized chunk embeddings (%s) to %s", len(chunks), embeddings.shape, npy_path)
    return SemanticIndex(
        embeddings=embeddings,
        chunks=chunks,
        chunk_doc_ids=chunk_doc_ids,
        model_name=model_name,
    )


def load_semantic_index(semantic_dir: Path) -> SemanticIndex:
    """Load precomputed embeddings.npy and chunks.json from disk."""
    npy_path = semantic_dir / "embeddings.npy"
    meta_path = semantic_dir / "chunks.json"
    if not npy_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Semantic embeddings not found in {semantic_dir}")

    embeddings = np.load(npy_path).astype(np.float32)
    raw_meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    chunks = [
        ChunkMeta(
            chunk_id=int(item["chunk_id"]),
            doc_id=int(item["doc_id"]),
            chunk_idx=int(item["chunk_idx"]),
            text=str(item["text"]),
        )
        for item in raw_meta.get("chunks", [])
    ]
    chunk_doc_ids = np.array([c.doc_id for c in chunks], dtype=np.int32)
    return SemanticIndex(
        embeddings=embeddings,
        chunks=chunks,
        chunk_doc_ids=chunk_doc_ids,
        model_name=str(raw_meta.get("model_name", DEFAULT_EMBED_MODEL)),
    )


@timed
def semantic_search(
    index: Index,
    sem_index: SemanticIndex,
    query: str,
    embedder: TextEmbedder | None = None,
    top_k: int = 5,
    include_snippets: bool = True,
) -> list[SearchResult]:
    """Rank documents by maximum cosine similarity across their normalized chunk embeddings."""
    cleaned = query.strip()
    if not cleaned or sem_index.num_chunks == 0 or index.num_docs == 0:
        return []

    provider: TextEmbedder = (
        embedder if embedder is not None else get_default_embedder(sem_index.model_name)
    )
    q_mat = provider.encode_texts([cleaned])
    if q_mat.size == 0:
        return []
    q_vec = l2_normalize_rows(np.asarray(q_mat, dtype=np.float32))[0]

    chunk_scores = (sem_index.embeddings @ q_vec).astype(np.float64)

    best_doc_score: dict[int, float] = {}
    best_doc_chunk: dict[int, int] = {}
    for c_idx in range(sem_index.num_chunks):
        d_id = int(sem_index.chunk_doc_ids[c_idx])
        if d_id not in index.docs:
            continue
        sc = float(chunk_scores[c_idx])
        prev = best_doc_score.get(d_id)
        if prev is None or sc > prev:
            best_doc_score[d_id] = sc
            best_doc_chunk[d_id] = c_idx

    if not best_doc_score:
        return []

    doc_ids = np.array(list(best_doc_score.keys()), dtype=np.int32)
    scores = np.array([best_doc_score[int(d)] for d in doc_ids], dtype=np.float64)
    n = int(doc_ids.size)
    k = min(top_k, n)
    if k <= 0:
        return []

    if k < n:
        part = np.argpartition(-scores, k - 1)[:k]
        sub_ids = doc_ids[part]
        sub_scores = scores[part]
    else:
        sub_ids = doc_ids
        sub_scores = scores

    order = np.lexsort((sub_ids, -sub_scores))
    q_terms = list(tokenize(cleaned))

    results: list[SearchResult] = []
    for idx_pos in order:
        d_id = int(sub_ids[idx_pos])
        sc = float(sub_scores[idx_pos])
        meta = index.docs[d_id]
        snippet = ""
        if include_snippets:
            best_c = sem_index.chunks[best_doc_chunk[d_id]]
            snippet = make_snippet(best_c.text, q_terms)
        results.append(
            SearchResult(
                score=sc,
                doc_id=d_id,
                title=meta.title,
                path=meta.path,
                snippet=snippet,
            )
        )
    return results


@timed
def hybrid_search(
    index: Index,
    sem_index: SemanticIndex,
    query: str,
    scorer: Scorer | None = None,
    embedder: TextEmbedder | None = None,
    top_k: int = 5,
    rrf_k: int = 60,
    include_snippets: bool = True,
) -> list[SearchResult]:
    """Fuse lexical BM25/TF-IDF and dense semantic rankings via Reciprocal Rank Fusion (RRF)."""
    candidate_depth = max(top_k * 5, 50)
    lexical_hits = search(
        index,
        query,
        scorer=scorer,
        top_k=candidate_depth,
        include_snippets=include_snippets,
    )
    sem_hits = semantic_search(
        index,
        sem_index,
        query,
        embedder=embedder,
        top_k=candidate_depth,
        include_snippets=include_snippets,
    )

    rrf_scores: dict[int, float] = {}
    snippet_by_doc: dict[int, str] = {}

    for rank, hit in enumerate(lexical_hits, start=1):
        rrf_scores[hit.doc_id] = rrf_scores.get(hit.doc_id, 0.0) + 1.0 / (rrf_k + rank)
        if hit.snippet:
            snippet_by_doc[hit.doc_id] = hit.snippet

    for rank, hit in enumerate(sem_hits, start=1):
        rrf_scores[hit.doc_id] = rrf_scores.get(hit.doc_id, 0.0) + 1.0 / (rrf_k + rank)
        if hit.doc_id not in snippet_by_doc and hit.snippet:
            snippet_by_doc[hit.doc_id] = hit.snippet

    if not rrf_scores:
        return []

    doc_ids = np.array(list(rrf_scores.keys()), dtype=np.int32)
    scores = np.array([rrf_scores[int(d)] for d in doc_ids], dtype=np.float64)
    n = int(doc_ids.size)
    k = min(top_k, n)
    if k < n:
        part = np.argpartition(-scores, k - 1)[:k]
        sub_ids = doc_ids[part]
        sub_scores = scores[part]
    else:
        sub_ids = doc_ids
        sub_scores = scores

    order = np.lexsort((sub_ids, -sub_scores))
    results: list[SearchResult] = []
    for pos in order:
        d_id = int(sub_ids[pos])
        sc = float(sub_scores[pos])
        meta = index.docs[d_id]
        results.append(
            SearchResult(
                score=sc,
                doc_id=d_id,
                title=meta.title,
                path=meta.path,
                snippet=snippet_by_doc.get(d_id, ""),
            )
        )
    return results
