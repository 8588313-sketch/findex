from dataclasses import dataclass


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


class InvertedIndex:
    def __init__(
        self,
        postings: dict[str, list[Posting]],
        docs: dict[int, DocMeta],
        with_positions: bool = False,
    ) -> None:
        self.postings = postings
        self.docs = docs
        self.with_positions = with_positions
        self.all_doc_ids: list[int] = sorted(docs.keys())
