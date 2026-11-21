from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from findex.models import DocMeta, Index, Posting
from findex.web.app import create_app
from findex.web.config import Settings
from findex.web.deps import get_index, get_optional_index


def _make_in_memory_index() -> Index:
    docs = {
        0: DocMeta(doc_id=0, path="mem://doc0.txt", title="Asyncio Guide", length=5),
        1: DocMeta(doc_id=1, path="mem://doc1.txt", title="FastAPI Web Search", length=6),
        2: DocMeta(doc_id=2, path="mem://doc2.txt", title="Python Concurrency", length=4),
    }
    postings = {
        "python": [
            Posting(doc_id=0, tf=1, positions=(0,)),
            Posting(doc_id=1, tf=1, positions=(0,)),
            Posting(doc_id=2, tf=1, positions=(0,)),
        ],
        "asyncio": [
            Posting(doc_id=0, tf=2, positions=(1, 3)),
            Posting(doc_id=2, tf=1, positions=(2,)),
        ],
        "fastapi": [
            Posting(doc_id=1, tf=2, positions=(1, 4)),
        ],
    }
    return Index(postings=postings, docs=docs, with_positions=True)


def test_health_returns_503_when_index_not_loaded() -> None:
    app = create_app(Settings(index_path=Path("/nonexistent/index.bin")))
    app.dependency_overrides[get_optional_index] = lambda: None

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "unavailable"
    assert body["index_loaded"] is False


def test_search_pagination_stats_and_document_routes_via_dependency_overrides() -> None:
    mem_idx = _make_in_memory_index()
    app = create_app(Settings(index_path=Path("/nonexistent/index.bin")))
    app.dependency_overrides[get_index] = lambda: mem_idx
    app.dependency_overrides[get_optional_index] = lambda: mem_idx

    client = TestClient(app)

    # /health -> 200
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["num_docs"] == 3

    # /stats -> 200
    st = client.get("/stats")
    assert st.status_code == 200
    assert st.json()["vocab_size"] == 3

    # /search page 1 and page 2
    s1 = client.get("/search", params={"q": "python", "k": 2, "page": 1, "scorer": "bm25"})
    assert s1.status_code == 200
    d1 = s1.json()
    assert d1["total"] == 3
    assert d1["total_pages"] == 2
    assert len(d1["hits"]) == 2
    assert d1["has_next"] is True
    assert d1["has_prev"] is False

    s2 = client.get("/search", params={"q": "python", "k": 2, "page": 2, "scorer": "bm25"})
    assert s2.status_code == 200
    d2 = s2.json()
    assert len(d2["hits"]) == 1
    assert d2["has_next"] is False
    assert d2["has_prev"] is True

    # /docs/{doc_id} -> 200 & 404
    doc_ok = client.get("/docs/1")
    assert doc_ok.status_code == 200
    assert doc_ok.json()["title"] == "FastAPI Web Search"

    doc_404 = client.get("/docs/9999")
    assert doc_404.status_code == 404


@pytest.mark.parametrize(
    "params",
    [
        {"q": ""},
        {"q": "   "},
        {"q": "python", "k": 0},
        {"q": "python", "page": 0},
        {"q": "python", "scorer": "invalid_scorer"},
        {"q": "(unclosed_paren"},
    ],
)
def test_search_validation_errors_return_422(params: dict[str, object]) -> None:
    mem_idx = _make_in_memory_index()
    app = create_app(Settings(index_path=Path("/nonexistent/index.bin")))
    app.dependency_overrides[get_index] = lambda: mem_idx

    client = TestClient(app)
    resp = client.get("/search", params=params)  # type: ignore[arg-type]
    assert resp.status_code == 422


def test_html_ui_renders_search_results_and_highlights() -> None:
    mem_idx = _make_in_memory_index()
    app = create_app(Settings(index_path=Path("/nonexistent/index.bin")))
    app.dependency_overrides[get_optional_index] = lambda: mem_idx
    app.dependency_overrides[get_index] = lambda: mem_idx

    client = TestClient(app)
    resp = client.get("/", params={"q": "asyncio", "scorer": "bm25", "page": 1})
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Asyncio Guide" in resp.text
    assert "<mark>" in resp.text


def test_unhandled_exception_returns_500_with_request_id_and_no_traceback() -> None:
    app = create_app(Settings(index_path=Path("/nonexistent/index.bin")))

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret internal failure")

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/boom", headers={"X-Request-ID": "req-test-123"})
    assert resp.status_code == 500
    assert resp.headers["X-Request-ID"] == "req-test-123"
    body = resp.json()
    assert body == {"detail": "Internal Server Error", "request_id": "req-test-123"}
    assert "Traceback" not in resp.text
    assert "secret internal failure" not in resp.text
