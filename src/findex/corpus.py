from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Document:
    """A single document streamed from disk or a JSONL crawl archive."""

    doc_id: int
    path: Path
    text: str
    title: str | None = None
    url: str | None = None


def iter_documents(root: Path) -> Iterator[Document]:
    """Lazily stream Document objects from a directory of .txt files or a .jsonl file."""
    if not root.exists():
        raise FileNotFoundError(f"Corpus path not found: {root}")

    if root.is_file():
        if root.suffix.lower() == ".jsonl":
            doc_id = 0
            with root.open("r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    record = json.loads(stripped)
                    text = str(record.get("text", ""))
                    url = str(record.get("url", f"{root.as_posix()}#{doc_id}"))
                    title = str(record.get("title") or url)
                    yield Document(
                        doc_id=doc_id,
                        path=Path(url),
                        text=text,
                        title=title,
                        url=url,
                    )
                    doc_id += 1
            return
        raise NotADirectoryError(f"Expected directory or .jsonl file, got: {root}")

    txt_paths = sorted(p for p in root.rglob("*.txt") if p.is_file())
    for doc_id, path in enumerate(txt_paths):
        text = path.read_text(encoding="utf-8", errors="ignore")
        yield Document(doc_id=doc_id, path=path, text=text)
