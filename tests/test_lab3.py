from pathlib import Path

import pytest

from findex.ast_query import And, Not, Phrase, QuerySyntaxError, Term, parse_query
from findex.builder import build_index
from findex.models import Index
from findex.ranking import BM25, TfIdf, clear_query_cache, query_cache_info, search
from findex.storage import open_index, save_binary


def _make_corpus(tmp_path: Path) -> Path:
    (tmp_path / "en_async.txt").write_text(
        "Python uses an event loop for async and await coroutines.", encoding="utf-8"
    )
    (tmp_path / "en_java.txt").write_text(
        "Python and Java both support async programming.", encoding="utf-8"
    )
    (tmp_path / "uk_kyiv.txt").write_text(
        "Київ є столицею України. Київ розташований на Дніпрі.", encoding="utf-8"
    )
    return tmp_path


def test_index_mapping_dunders(tmp_path: Path) -> None:
    idx = build_index(_make_corpus(tmp_path), with_positions=True)
    assert isinstance(idx, Index)
    assert idx.num_docs == 3
    assert len(idx) > 0
    assert "python" in idx
    assert "nonexistent_xyz" not in idx
    assert idx.df("python") == 2
    assert idx.avg_doc_length > 0
    assert "Index(" in repr(idx)
    assert set(iter(idx)) == set(idx.keys())


def test_open_index_context_manager_cleans_up_on_exception(tmp_path: Path) -> None:
    idx = build_index(_make_corpus(tmp_path), with_positions=True)
    bin_path = tmp_path / "test.bin"
    save_binary(idx, bin_path)

    captured: Index | None = None
    with pytest.raises(RuntimeError, match="boom"), open_index(bin_path) as loaded:
        captured = loaded
        assert loaded.num_docs == 3
        raise RuntimeError("boom")

    assert captured is not None
    assert captured.num_docs == 0
    assert len(captured) == 0


def test_ast_parser_tree_equality() -> None:
    parsed = parse_query('python AND (async OR await) NOT java "event loop"')
    expected = (
        Term("python")
        & (Term("async") | Term("await"))
        & ~Term("java")
        & Phrase(("event", "loop"))
    )
    assert parsed == expected
    assert isinstance(parsed, And)
    assert isinstance(parsed.right, Phrase)
    assert isinstance(parsed.left.right, Not)

    with pytest.raises(QuerySyntaxError):
        parse_query("python AND (async OR await")


def test_phrase_search_and_ranking_cache(tmp_path: Path) -> None:
    clear_query_cache()
    idx = build_index(_make_corpus(tmp_path), with_positions=True)

    q = 'python AND (async OR await) NOT java "event loop"'
    res_bm25 = search(idx, q, scorer=BM25(), top_k=5)
    assert len(res_bm25) == 1
    assert res_bm25[0].path.endswith("en_async.txt")
    assert "[event loop]" in res_bm25[0].snippet

    # Second call should trigger @lru_cache HIT
    _ = search(idx, q, scorer=TfIdf(), top_k=5)
    info = query_cache_info()
    assert info.hits >= 1
