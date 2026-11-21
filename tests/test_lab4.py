from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from findex.ast_query import And, Not, Or, Phrase, QuerySyntaxError, Term, parse_query
from findex.builder import build_index
from findex.cli import app
from findex.models import DocMeta, Index, Posting
from findex.query import difference_sorted, intersect_sorted, search_index, union_sorted
from findex.ranking import (
    BM25,
    Scorer,
    TfIdf,
    clear_query_cache,
    make_snippet,
    query_cache_info,
    search,
)
from findex.storage import (
    load_binary,
    load_pickle,
    open_index,
    save_binary,
    save_pickle,
)
from findex.tokenize import tokenize

runner = CliRunner()


# --- 1. Parametrized tokenizer tests (including Unicode) ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hello, World!", ["hello", "world"]),
        ("Привіт, Україно!", ["привіт", "україно"]),
        ("Straße", ["strasse"]),
        ("cafe\u0301", ["caf\u00e9"]),
        ("Don't 'quoted' п’ять сімʼя", ["don't", "quoted", "п'ять", "сім'я"]),
        ("state-of-the-art Python 3.11", ["state", "of", "the", "art", "python", "3", "11"]),
        ("日本語テスト αβγ Ω", ["日本語テスト", "αβγ", "ω"]),
        ("   ...---___   ", []),
    ],
)
def test_tokenize_parametrized(raw: str, expected: list[str]) -> None:
    assert list(tokenize(raw)) == expected


# --- 2. Index building & Mapping dunder protocol ---


def test_build_index_from_fixture(sample_index: Index) -> None:
    assert sample_index.num_docs == 4
    assert len(sample_index) > 15
    assert "python" in sample_index
    assert "київ" in sample_index
    assert "missing_word_xyz" not in sample_index
    assert sample_index.df("python") == 2
    assert sample_index.avg_doc_length > 0
    assert repr(sample_index).startswith("Index(docs=4,")


# --- 3. Boolean merge vs set engine ---


@pytest.mark.parametrize(
    ("query", "expected_count"),
    [
        ("python AND async", 1),
        ("python OR київ", 3),
        ("python NOT java", 1),
        ("столиця -львів", 0),
        ("неіснуючеслово", 0),
    ],
)
def test_boolean_merge_matches_set(sample_index: Index, query: str, expected_count: int) -> None:
    res_merge = search_index(sample_index, query, engine="merge")
    res_set = search_index(sample_index, query, engine="set")
    assert res_merge == res_set
    assert len(res_merge) == expected_count


# --- 4. Query parser AST comparison ---


@pytest.mark.parametrize(
    ("query_str", "expected_ast"),
    [
        ("python", Term("python")),
        ('"event loop"', Phrase(("event", "loop"))),
        ("python AND java", And(Term("python"), Term("java"))),
        ("python java", And(Term("python"), Term("java"))),
        ("async OR await", Or(Term("async"), Term("await"))),
        ("NOT java", Not(Term("java"))),
        ("~java", Not(Term("java"))),
        (
            'python AND (async OR await) NOT java "event loop"',
            Term("python") & (Term("async") | Term("await")) & ~Term("java") & Phrase(("event", "loop")),
        ),
    ],
)
def test_parser_produces_expected_ast(query_str: str, expected_ast: object) -> None:
    assert parse_query(query_str) == expected_ast


@pytest.mark.parametrize("bad_query", ["", "   ", "(python AND java", "python )", "AND python", "python OR"])
def test_parser_rejects_invalid_queries(bad_query: str) -> None:
    with pytest.raises(QuerySyntaxError):
        parse_query(bad_query)


# --- 5. Scorer protocol & 3 ranking sanity checks ---


def test_scorers_satisfy_protocol_and_sanity_checks() -> None:
    scorers: list[Scorer] = [BM25(k1=1.5, b=0.75), TfIdf()]
    for sc in scorers:
        # 1) Rare term > frequent term
        rare = sc.score(tf=1, df=1, doc_len=100, num_docs=100, avg_dl=100.0)
        freq = sc.score(tf=1, df=90, doc_len=100, num_docs=100, avg_dl=100.0)
        assert rare > freq

        # 2) 20th repetition adds almost nothing compared to 1st->2nd
        s1 = sc.score(tf=1, df=10, doc_len=100, num_docs=100, avg_dl=100.0)
        s2 = sc.score(tf=2, df=10, doc_len=100, num_docs=100, avg_dl=100.0)
        s19 = sc.score(tf=19, df=10, doc_len=100, num_docs=100, avg_dl=100.0)
        s20 = sc.score(tf=20, df=10, doc_len=100, num_docs=100, avg_dl=100.0)
        assert (s20 - s19) < (s2 - s1) * 0.1

    # 3) Short doc with 1 occurrence beats very long doc with 1 occurrence in BM25
    bm25 = BM25()
    short_s = bm25.score(tf=1, df=2, doc_len=20, num_docs=10, avg_dl=500.0)
    long_s = bm25.score(tf=1, df=2, doc_len=5000, num_docs=10, avg_dl=500.0)
    assert short_s > long_s


def test_snippet_highlighting() -> None:
    text = "First sentence here. Python uses an event loop for async tasks. End of text."
    snip = make_snippet(text, terms=["python"], phrases=[("event", "loop")], window=30)
    assert "[Python]" in snip
    assert "[event loop]" in snip


# --- 6. Serialization & open_index context manager ---


@pytest.mark.parametrize("fmt", ["bin", "pickle"])
def test_save_and_load_formats(sample_index: Index, tmp_path: Path, fmt: str) -> None:
    out = tmp_path / f"idx.{fmt}"
    if fmt == "bin":
        save_binary(sample_index, out)
        loaded = load_binary(out)
    else:
        save_pickle(sample_index, out)
        loaded = load_pickle(out)

    assert loaded.docs == sample_index.docs
    assert loaded.postings == sample_index.postings
    assert loaded.with_positions == sample_index.with_positions


def test_open_index_cleans_up_on_error(sample_index: Index, tmp_path: Path) -> None:
    out = tmp_path / "idx.bin"
    save_binary(sample_index, out)
    with pytest.raises(ValueError), open_index(out) as loaded:
        assert loaded.num_docs == 4
        raise ValueError("simulated failure")


# --- 7. CLI tests with typer.testing.CliRunner ---


def test_cli_index_search_json_and_stats(sample_corpus: Path, tmp_path: Path) -> None:
    clear_query_cache()
    idx_file = tmp_path / "cli_idx.bin"

    res_idx = runner.invoke(app, ["-v", "index", str(sample_corpus), "--out", str(idx_file)])
    assert res_idx.exit_code == 0, res_idx.output
    assert idx_file.exists()

    # Rich table output
    res_search = runner.invoke(app, ["search", str(idx_file), 'python AND "event loop"'])
    assert res_search.exit_code == 0, res_search.output
    assert "async" in res_search.output

    # --json output must be valid parseable JSON on stdout
    res_json = runner.invoke(app, ["search", str(idx_file), "київ", "--json", "--repeat"])
    assert res_json.exit_code == 0, res_json.output
    data = json.loads(res_json.stdout)
    assert len(data) == 1
    assert data[0]["title"] == "[uk] kyiv"
    assert query_cache_info().hits >= 1

    # stats subcommand
    res_stats = runner.invoke(app, ["-vv", "stats", str(sample_corpus), "--top", "5"])
    assert res_stats.exit_code == 0, res_stats.output
    assert "Corpus Stats" in res_stats.output


def test_cli_expected_errors_exit_cleanly(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.bin"
    res_missing = runner.invoke(app, ["search", str(missing), "python"])
    assert res_missing.exit_code == 1
    assert "Traceback" not in res_missing.output
    assert "Error:" in res_missing.output

    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world", encoding="utf-8")
    idx_file = tmp_path / "ok.bin"
    assert runner.invoke(app, ["index", str(corpus), "--out", str(idx_file)]).exit_code == 0

    res_bad_q = runner.invoke(app, ["search", str(idx_file), "python AND (unclosed"])
    assert res_bad_q.exit_code == 1
    assert "Traceback" not in res_bad_q.output
    assert "Error:" in res_bad_q.output


# --- 8. Property-based tests with Hypothesis (3 properties) ---


@st.composite
def random_index_strategy(draw: st.DrawFn) -> Index:
    num_docs = draw(st.integers(min_value=1, max_value=8))
    docs = {
        d: DocMeta(doc_id=d, path=f"doc_{d}.txt", title=f"Doc {d}", length=draw(st.integers(1, 500)))
        for d in range(num_docs)
    }
    with_positions = draw(st.booleans())
    vocab = draw(st.lists(st.text(alphabet="abcdefghij", min_size=1, max_size=6), min_size=1, max_size=12, unique=True))
    postings: dict[str, list[Posting]] = {}
    for term in vocab:
        doc_subset = sorted(draw(st.lists(st.integers(0, num_docs - 1), min_size=1, max_size=num_docs, unique=True)))
        plist: list[Posting] = []
        for d in doc_subset:
            if with_positions:
                pos = tuple(sorted(draw(st.lists(st.integers(0, 200), min_size=1, max_size=5, unique=True))))
                plist.append(Posting(doc_id=d, tf=len(pos), positions=pos))
            else:
                tf = draw(st.integers(min_value=1, max_value=20))
                plist.append(Posting(doc_id=d, tf=tf))
        postings[term] = plist
    return Index(postings=postings, docs=docs, with_positions=with_positions)


@given(idx=random_index_strategy())
@settings(max_examples=40)
def test_hypothesis_save_load_roundtrip(idx: Index, tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_dir = tmp_path_factory.mktemp("hyp")
    bin_path = tmp_dir / "roundtrip.bin"
    save_binary(idx, bin_path)
    loaded = load_binary(bin_path)
    assert loaded.docs == idx.docs
    assert loaded.postings == idx.postings
    assert loaded.with_positions == idx.with_positions


@given(
    doc_texts=st.lists(
        st.lists(st.sampled_from(["alpha", "beta", "gamma", "delta"]), min_size=0, max_size=15),
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=40)
def test_hypothesis_postings_always_sorted(
    doc_texts: list[list[str]],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    corpus_dir = tmp_path_factory.mktemp("sorted_corpus")
    for i, words in enumerate(doc_texts):
        (corpus_dir / f"doc_{i:03d}.txt").write_text(" ".join(words), encoding="utf-8")

    idx = build_index(corpus_dir, with_positions=True)
    for term, plist in idx.items():
        doc_ids = [p.doc_id for p in plist]
        assert doc_ids == sorted(doc_ids), f"Postings for {term!r} are not sorted"
        assert len(doc_ids) == len(set(doc_ids)), f"Duplicate doc_ids for {term!r}"


@given(
    a=st.lists(st.integers(min_value=0, max_value=200), max_size=40).map(lambda xs: sorted(set(xs))),
    b=st.lists(st.integers(min_value=0, max_value=200), max_size=40).map(lambda xs: sorted(set(xs))),
)
@settings(max_examples=80)
def test_hypothesis_merge_equals_set_operations(a: list[int], b: list[int]) -> None:
    assert intersect_sorted(a, b) == sorted(set(a) & set(b))
    assert union_sorted(a, b) == sorted(set(a) | set(b))
    assert difference_sorted(a, b) == sorted(set(a) - set(b))


# --- 9. Slow integration benchmark test ---


@pytest.mark.slow
def test_slow_full_corpus_build_and_search(tmp_path: Path) -> None:
    data_dir = Path("data")
    if not data_dir.exists():
        pytest.skip("Full corpus data/ not present")
    t0 = time.perf_counter()
    idx = build_index(data_dir, with_positions=True)
    out = tmp_path / "full.bin"
    save_binary(idx, out)
    res = search(idx, "київ AND україна", top_k=5)
    elapsed = time.perf_counter() - t0
    assert len(res) == 5
    assert elapsed >= 0.0
