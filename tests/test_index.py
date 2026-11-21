from pathlib import Path

from findex.builder import build_index
from findex.query import search_index
from findex.storage import load_binary, load_pickle, save_binary, save_pickle


def _make_sample_corpus(tmp_path: Path) -> Path:
    (tmp_path / "uk_doc1.txt").write_text("Київ — столиця України. Київ стоїть на Дніпрі.", encoding="utf-8")
    (tmp_path / "uk_doc2.txt").write_text("Львів і Київ — великі міста України.", encoding="utf-8")
    (tmp_path / "en_doc3.txt").write_text("Python search engine with inverted index.", encoding="utf-8")
    return tmp_path


def test_build_and_boolean_search(tmp_path: Path) -> None:
    root = _make_sample_corpus(tmp_path)
    idx = build_index(root, with_positions=True)

    assert len(idx.docs) == 3
    for engine in ("merge", "set"):
        assert len(search_index(idx, "київ україни", engine=engine)) == 2
        assert len(search_index(idx, "львів OR python", engine=engine)) == 2
        assert len(search_index(idx, "київ NOT львів", engine=engine)) == 1


def test_serialization_roundtrip(tmp_path: Path) -> None:
    root = _make_sample_corpus(tmp_path)
    idx = build_index(root, with_positions=True)

    bin_file = tmp_path / "idx.bin"
    pkl_file = tmp_path / "idx.pkl"

    save_binary(idx, bin_file)
    save_pickle(idx, pkl_file)

    loaded_bin = load_binary(bin_file)
    loaded_pkl = load_pickle(pkl_file)

    assert loaded_bin.docs == idx.docs
    assert loaded_bin.postings == idx.postings
    assert loaded_pkl.docs == idx.docs
    assert loaded_pkl.postings == idx.postings
