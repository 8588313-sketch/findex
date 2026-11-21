import logging
import pickle
import struct
from array import array
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from findex.decorators import timed
from findex.models import DocMeta, Index, Posting

log = logging.getLogger("findex.storage")
MAGIC = b"FIDX02"
StorageFormat = Literal["bin", "pickle"]


def save_pickle(index: Index, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Index:
    # SECURITY: Never call pickle.load() on untrusted files (RCE risk via __reduce__).
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, Index):
        raise TypeError(f"Unpickled object is {type(obj).__name__}, expected Index")
    return obj


def save_binary(index: Index, path: Path) -> None:
    """Serialize index into a compact binary format backed by array('I')."""
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(
            struct.pack(
                "<BII",
                int(index.with_positions),
                len(index.docs),
                len(index),
            )
        )

        for doc_id in index.all_doc_ids:
            meta = index.docs[doc_id]
            path_b = meta.path.encode("utf-8")
            title_b = meta.title.encode("utf-8")
            f.write(struct.pack("<III", meta.doc_id, meta.length, len(path_b)))
            f.write(path_b)
            f.write(struct.pack("<I", len(title_b)))
            f.write(title_b)

        for term, plist in index.items():
            term_b = term.encode("utf-8")
            f.write(struct.pack("<II", len(term_b), len(plist)))
            f.write(term_b)

            doc_ids = array("I", (p.doc_id for p in plist))
            tfs = array("I", (p.tf for p in plist))
            doc_ids.tofile(f)
            tfs.tofile(f)

            if index.with_positions:
                for p in plist:
                    pos_arr = array("I", p.positions)
                    f.write(struct.pack("<I", len(pos_arr)))
                    pos_arr.tofile(f)


def load_binary(path: Path) -> Index:
    """Safely deserialize an index from the custom array('I') binary format."""
    with path.open("rb") as f:
        if f.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path} is not a valid FIDX02 binary index")

        header_bytes = f.read(9)
        if len(header_bytes) != 9:
            raise ValueError(f"{path} is truncated")
        with_pos_flag, num_docs, num_terms = struct.unpack("<BII", header_bytes)
        with_positions = bool(with_pos_flag)

        docs: dict[int, DocMeta] = {}
        for _ in range(num_docs):
            doc_id, length, path_len = struct.unpack("<III", f.read(12))
            doc_path = f.read(path_len).decode("utf-8")
            (title_len,) = struct.unpack("<I", f.read(4))
            title = f.read(title_len).decode("utf-8")
            docs[doc_id] = DocMeta(
                doc_id=doc_id, path=doc_path, title=title, length=length
            )

        postings: dict[str, list[Posting]] = {}
        for _ in range(num_terms):
            term_len, df = struct.unpack("<II", f.read(8))
            term = f.read(term_len).decode("utf-8")

            doc_ids = array("I")
            tfs = array("I")
            doc_ids.fromfile(f, df)
            tfs.fromfile(f, df)

            if with_positions:
                plist: list[Posting] = []
                for d_id, tf in zip(doc_ids, tfs):
                    (pos_len,) = struct.unpack("<I", f.read(4))
                    pos_arr = array("I")
                    pos_arr.fromfile(f, pos_len)
                    plist.append(Posting(doc_id=d_id, tf=tf, positions=tuple(pos_arr)))
                postings[term] = plist
            else:
                postings[term] = [
                    Posting(doc_id=d_id, tf=tf) for d_id, tf in zip(doc_ids, tfs)
                ]

    return Index(postings=postings, docs=docs, with_positions=with_positions)


def save_index(index: Index, path: Path, fmt: StorageFormat = "bin") -> None:
    if fmt == "pickle":
        save_pickle(index, path)
    else:
        save_binary(index, path)


@timed
def load_index(path: Path) -> Index:
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    with path.open("rb") as f:
        header = f.read(len(MAGIC))
    if header == MAGIC:
        return load_binary(path)
    return load_pickle(path)


@contextmanager
def open_index(path: Path) -> Generator[Index, None, None]:
    """Context manager that loads an Index and guarantees resource cleanup on exit/error."""
    index: Index | None = None
    try:
        index = load_index(path)
        yield index
    finally:
        if index is not None:
            index.postings.clear()
            index.docs.clear()
            index.all_doc_ids.clear()
        log.debug("Released index resources for %s", path)
