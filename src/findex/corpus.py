import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)


class Document(NamedTuple):
    doc_id: int
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """Yield documents one at a time, never holding the whole corpus."""
    doc_id = 0
    for path in root.rglob("*.txt"):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log.warning("skipping %s: bad encoding", path)
            continue
        yield Document(doc_id, path, text)
        doc_id += 1
