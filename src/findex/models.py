from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    title: str
    length: int


@dataclass(frozen=True, slots=True, order=True)
class SearchResult:
    score: float
    doc_id: int
    title: str
    path: str
    snippet: str = ""


class Index(Mapping[str, list[Posting]]):
    """Pythonic inverted index implementing the Mapping protocol."""

    def __init__(
        self,
        postings: dict[str, list[Posting]],
        docs: dict[int, DocMeta],
        with_positions: bool = False,
    ) -> None:
        self._postings = postings
        self.docs = docs
        self.with_positions = with_positions
        self.all_doc_ids: list[int] = sorted(docs.keys())

    @property
    def postings(self) -> dict[str, list[Posting]]:
        return self._postings

    def __getitem__(self, term: str) -> list[Posting]:
        return self._postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __len__(self) -> int:
        return len(self._postings)

    def __contains__(self, term: object) -> bool:
        return term in self._postings

    def __repr__(self) -> str:
        return (
            f"Index(docs={self.num_docs}, terms={len(self)}, "
            f"avg_doc_length={self.avg_doc_length:.1f}, "
            f"with_positions={self.with_positions})"
        )

    @property
    def num_docs(self) -> int:
        return len(self.docs)

    @cached_property
    def avg_doc_length(self) -> float:
        if not self.docs:
            return 0.0
        return sum(meta.length for meta in self.docs.values()) / len(self.docs)

    def df(self, term: str) -> int:
        return len(self._postings.get(term, ()))


# Backward-compatible alias for Lab 2 code/tests
InvertedIndex = Index
