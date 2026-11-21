from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray


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
    """Pythonic inverted index implementing the Mapping protocol with NumPy int32 arrays."""

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
        self._np_postings_cache: dict[str, tuple[NDArray[np.int32], NDArray[np.int32]]] = {}
        self._doc_arrays_cache: tuple[NDArray[np.int32], dict[int, int], NDArray[np.int32]] | None = None

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

    def _ensure_doc_arrays(self) -> tuple[NDArray[np.int32], dict[int, int], NDArray[np.int32]]:
        if self._doc_arrays_cache is None or len(self._doc_arrays_cache[0]) != len(self.docs):
            sorted_ids = sorted(self.docs.keys())
            doc_ids_arr = np.array(sorted_ids, dtype=np.int32)
            doc_lengths_arr = np.array([self.docs[d].length for d in sorted_ids], dtype=np.int32)
            id_to_row = {d: idx for idx, d in enumerate(sorted_ids)}
            self._doc_arrays_cache = (doc_ids_arr, id_to_row, doc_lengths_arr)
        return self._doc_arrays_cache

    @property
    def doc_ids_array(self) -> NDArray[np.int32]:
        return self._ensure_doc_arrays()[0]

    @property
    def doc_id_to_row(self) -> dict[int, int]:
        return self._ensure_doc_arrays()[1]

    @property
    def doc_lengths_array(self) -> NDArray[np.int32]:
        """Contiguous int32 array of document token lengths aligned with doc_ids_array."""
        return self._ensure_doc_arrays()[2]

    def get_numpy_postings(self, term: str) -> tuple[NDArray[np.int32], NDArray[np.int32]] | None:
        """Return (doc_ids: int32[df], tfs: int32[df]) for a term, cached lazily."""
        cached = self._np_postings_cache.get(term)
        if cached is not None:
            return cached
        plist = self._postings.get(term)
        if not plist:
            return None
        doc_ids = np.array([p.doc_id for p in plist], dtype=np.int32)
        tfs = np.array([p.tf for p in plist], dtype=np.int32)
        pair = (doc_ids, tfs)
        self._np_postings_cache[term] = pair
        return pair

    def prewarm_numpy(self) -> None:
        """Materialize int32 arrays for all documents and terms ahead of search benchmarks."""
        _ = self._ensure_doc_arrays()
        for term in self._postings:
            _ = self.get_numpy_postings(term)


InvertedIndex = Index
