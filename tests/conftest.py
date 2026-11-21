from pathlib import Path

import pytest

from findex.builder import build_index
from findex.models import Index


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    (tmp_path / "en_async.txt").write_text(
        "Python uses an event loop for async and await coroutines. Async IO is fast.",
        encoding="utf-8",
    )
    (tmp_path / "en_java.txt").write_text(
        "Python and Java both support concurrent programming without an event loop.",
        encoding="utf-8",
    )
    (tmp_path / "uk_kyiv.txt").write_text(
        "Київ є столицею України. Київ розташований на річці Дніпро.",
        encoding="utf-8",
    )
    (tmp_path / "uk_lviv.txt").write_text(
        "Львів — культурна столиця заходу України. Львів відомий кавою.",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def sample_index(sample_corpus: Path) -> Index:
    return build_index(sample_corpus, with_positions=True)
