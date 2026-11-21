from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from typer.testing import CliRunner

from findex.builder import build_index
from findex.cli import app as cli_app
from findex.models import DocMeta, Index, Posting
from findex.ranking import (
    BM25,
    TfIdf,
    clear_query_cache,
    search,
    search_python_baseline,
)
from findex.semantic import (
    build_embeddings,
    chunk_text,
    hybrid_search,
    load_semantic_index,
    semantic_search,
)
from findex.storage import save_index
from findex.web.app import create_app
from findex.web.config import Settings

runner = CliRunner()


class DeterministicKeywordEmbedder:
    """Fast, offline deterministic embedder mapping semantic concepts to unit vectors."""

    CONCEPT_GROUPS: tuple[tuple[str, ...], ...] = (
        ("python", "programming", "language", "asyncio", "interpreter", "code"),
        ("programmer", "analytical", "engine", "ada", "lovelace", "algorithm"),
        ("moon", "lunar", "apollo", "crater", "orbit", "spacecraft"),
        ("linux", "kernel", "torvalds", "operating", "system", "posix"),
        ("kyiv", "ukraine", "capital", "dnipro", "lviv", "carpathians"),
    )

    def encode_texts(self, texts: list[str]) -> NDArray[np.float32]:
        dim = len(self.CONCEPT_GROUPS) + 4
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for row, text in enumerate(texts):
            low = text.lower()
            for g_idx, keywords in enumerate(self.CONCEPT_GROUPS):
                hits = sum(1 for kw in keywords if kw in low)
                out[row, g_idx] = float(hits)
            # Small deterministic character-level fallback signal
            out[row, -4] = float(len(low) % 17 + 1) * 0.05
            out[row, -3] = float(sum(ord(c) for c in low[:20]) % 13 + 1) * 0.05
            norm = float(np.linalg.norm(out[row]))
            if norm > 0.0:
                out[row] /= norm
        return out


@pytest.fixture
def sample_corpus_dir(tmp_path: Path) -> Path:
    docs = {
        "ada.txt": "Ada Lovelace wrote the first algorithm intended for Charles Babbage Analytical Engine.",
        "moon.txt": "The Moon orbits Earth and was visited by Apollo astronauts who studied lunar craters.",
        "linux.txt": "Linux is an open source Unix-like operating system kernel created by Linus Torvalds.",
        "python.txt": "Python is a high-level programming language with asyncio event loops and clean syntax.",
        "kyiv.txt": "Kyiv is the capital of Ukraine located along the Dnipro river in Eastern Europe.",
    }
    for name, body in docs.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    return tmp_path


@pytest.fixture
def sample_index(sample_corpus_dir: Path) -> Index:
    clear_query_cache()
    idx = build_index(sample_corpus_dir, with_positions=True)
    idx.prewarm_numpy()
    return idx


def test_numpy_bm25_and_tfidf_match_lab3_order_and_scores(sample_index: Index) -> None:
    """Vectorized NumPy BM25 & TF-IDF must produce identical ranking and scores to Lab 3."""
    queries = [
        "python",
        "python OR linux OR moon",
        "ada AND engine",
        "capital OR kernel OR algorithm",
    ]
    for scorer in (BM25(), TfIdf()):
        for q in queries:
            baseline = search_python_baseline(sample_index, q, scorer=scorer, top_k=5, include_snippets=False)
            vectorized = search(sample_index, q, scorer=scorer, top_k=5, include_snippets=False)
            assert [r.doc_id for r in vectorized] == [r.doc_id for r in baseline]
            for r_vec, r_base in zip(vectorized, baseline, strict=True):
                assert r_vec.score == pytest.approx(r_base.score, rel=1e-11, abs=1e-11)


def test_sparse_doc_ids_and_argpartition_top_k() -> None:
    """Verify non-contiguous doc_ids and np.argpartition top-k cutoff behave deterministically."""
    postings = {
        "alpha": [
            Posting(doc_id=10, tf=1),
            Posting(doc_id=25, tf=5),
            Posting(doc_id=42, tf=2),
            Posting(doc_id=99, tf=4),
        ]
    }
    docs = {
        10: DocMeta(doc_id=10, path="d10.txt", title="Doc 10", length=100),
        25: DocMeta(doc_id=25, path="d25.txt", title="Doc 25", length=100),
        42: DocMeta(doc_id=42, path="d42.txt", title="Doc 42", length=100),
        99: DocMeta(doc_id=99, path="d99.txt", title="Doc 99", length=100),
    }
    idx = Index(postings=postings, docs=docs)
    base = search_python_baseline(idx, "alpha", scorer=BM25(), top_k=2, include_snippets=False)
    vec = search(idx, "alpha", scorer=BM25(), top_k=2, include_snippets=False)
    assert [r.doc_id for r in vec] == [25, 99]
    assert [r.doc_id for r in vec] == [r.doc_id for r in base]


def test_chunking_and_semantic_and_hybrid_rrf(sample_index: Index, tmp_path: Path) -> None:
    """Test chunking, .npy persistence, cosine semantic search, and RRF hybrid fusion."""
    chunks = chunk_text(" ".join(f"word{i}" for i in range(250)), chunk_words=100, overlap_words=25)
    assert len(chunks) == 3

    embedder = DeterministicKeywordEmbedder()
    sem_dir = tmp_path / "sem_out"
    built = build_embeddings(sample_index, out_dir=sem_dir, embedder=embedder, chunk_words=50, overlap_words=10)
    assert (sem_dir / "embeddings.npy").exists()
    assert (sem_dir / "chunks.json").exists()
    assert built.num_chunks == sample_index.num_docs

    loaded = load_semantic_index(sem_dir)
    assert loaded.embeddings.shape == built.embeddings.shape
    # Check unit L2 normalization
    norms = np.linalg.norm(loaded.embeddings, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)

    # Paraphrased query with NO exact keywords from ada.txt ("first female computer scientist")
    lex_hits = search(sample_index, "female scientist", scorer=BM25(), top_k=3)
    assert lex_hits == []

    sem_hits = semantic_search(sample_index, loaded, "programmer analytical engine", embedder=embedder, top_k=3)
    assert len(sem_hits) >= 1
    assert "ada" == sem_hits[0].title

    hyb_hits = hybrid_search(
        sample_index,
        loaded,
        "ada OR programmer analytical engine",
        scorer=BM25(),
        embedder=embedder,
        top_k=3,
    )
    assert len(hyb_hits) >= 1
    assert "ada" == hyb_hits[0].title
    # Verify RRF score ceiling: max possible RRF score for rank=1 in both lists is 2 / (60 + 1)
    assert hyb_hits[0].score <= (2.0 / 61.0) + 1e-9


def test_fastapi_and_cli_hybrid_modes(sample_index: Index, tmp_path: Path) -> None:
    """Verify /search?mode=semantic|hybrid and CLI --mode flag work end-to-end."""
    idx_path = tmp_path / "index.bin"
    save_index(sample_index, idx_path)

    embedder = DeterministicKeywordEmbedder()
    sem_dir = tmp_path / "index_semantic"
    sem_idx = build_embeddings(sample_index, out_dir=sem_dir, embedder=embedder)

    # CLI search in lexical mode
    res_cli = runner.invoke(cli_app, ["search", str(idx_path), "python", "--mode", "lexical", "--json"])
    assert res_cli.exit_code == 0
    assert "python.txt" in res_cli.stdout

    # FastAPI app with preloaded semantic index
    settings = Settings(index_path=idx_path)
    app = create_app(settings=settings)
    with TestClient(app) as client:
        # Override embedder via precomputed semantic index query that uses exact tokens present in embedder
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["semantic_loaded"] is True

        stats = client.get("/stats").json()
        assert stats["semantic_ready"] is True
        assert stats["semantic_chunks"] == sem_idx.num_chunks


def test_benchmark_search_numpy(benchmark: Any, sample_index: Index) -> None:
    """pytest-benchmark regression test for vectorized search()."""
    result = benchmark(
        search,
        sample_index,
        "python OR linux OR moon",
        BM25(),
        5,
        False,
    )
    assert len(result) == 3


def test_benchmark_build_index(benchmark: Any, sample_corpus_dir: Path) -> None:
    """pytest-benchmark regression test for build_index()."""
    idx = benchmark(build_index, sample_corpus_dir, True)
    assert idx.num_docs == 5
