from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Literal

from findex.corpus import Document
from findex.decorators import timed
from findex.models import DocMeta, Index, Posting
from findex.tokenize import tokenize

log = logging.getLogger("findex.builder")
ExecutorKind = Literal["serial", "threads", "processes"]


@dataclass(slots=True)
class PartialIndex:
    """Picklable shard of an inverted index produced by a single worker."""

    postings: dict[str, list[Posting]]
    docs: dict[int, DocMeta]
    with_positions: bool


@dataclass(slots=True)
class BuildStats:
    """Timing breakdown for parallel index construction."""

    wall_time_s: float
    cpu_time_s: float
    merge_time_s: float
    workers: int
    executor: ExecutorKind


def _title_from_path(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        lang, rest = stem.split("_", 1)
        return f"[{lang}] {rest.replace('_', ' ')}"
    return stem.replace("_", " ")


def _discover_paths(root: Path, limit: int | None = None) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Corpus directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus path is not a directory: {root}")

    paths = sorted(p for p in root.rglob("*.txt") if p.is_file())
    if limit is not None:
        return list(islice(paths, limit))
    return paths


def _chunk_with_offsets(paths: Sequence[Path], num_chunks: int) -> list[tuple[int, list[Path]]]:
    """Split file paths into contiguous chunks and pre-assign disjoint doc_id ranges."""
    n = len(paths)
    if n == 0:
        return []
    k = max(1, min(num_chunks, n))
    base, extra = divmod(n, k)
    chunks: list[tuple[int, list[Path]]] = []
    offset = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        chunk = list(paths[offset : offset + size])
        chunks.append((offset, chunk))
        offset += size
    return chunks


def build_partial(
    paths: Sequence[Path],
    start_doc_id: int = 0,
    with_positions: bool = True,
) -> PartialIndex:
    """Build a partial inverted index from a slice of file paths.

    Top-level module function so it can be pickled safely for ProcessPoolExecutor (spawn).
    """
    postings_map: dict[str, list[Posting]] = defaultdict(list)
    docs_meta: dict[int, DocMeta] = {}

    for idx_offset, path in enumerate(paths):
        doc_id = start_doc_id + idx_offset
        text = path.read_text(encoding="utf-8", errors="ignore")
        tokens = list(tokenize(text))
        docs_meta[doc_id] = DocMeta(
            doc_id=doc_id,
            path=path.as_posix(),
            title=_title_from_path(path),
            length=len(tokens),
        )

        if with_positions:
            pos_map: dict[str, list[int]] = defaultdict(list)
            for pos, term in enumerate(tokens):
                pos_map[term].append(pos)
            for term, positions in pos_map.items():
                postings_map[term].append(
                    Posting(doc_id=doc_id, tf=len(positions), positions=tuple(positions))
                )
        else:
            counts = Counter(tokens)
            for term, tf in counts.items():
                postings_map[term].append(Posting(doc_id=doc_id, tf=tf))

    return PartialIndex(
        postings=dict(postings_map),
        docs=docs_meta,
        with_positions=with_positions,
    )


def merge_partials(partials: Iterable[PartialIndex]) -> Index:
    """Merge disjoint partial indexes into a single Index with sorted posting lists."""
    partial_list = list(partials)
    if not partial_list:
        return Index(postings={}, docs={}, with_positions=True)

    with_positions = partial_list[0].with_positions
    ordered = sorted(
        partial_list,
        key=lambda part: min(part.docs.keys()) if part.docs else -1,
    )

    merged_docs: dict[int, DocMeta] = {}
    merged_postings: dict[str, list[Posting]] = defaultdict(list)

    for part in ordered:
        merged_docs.update(part.docs)
        for term, plist in part.postings.items():
            merged_postings[term].extend(plist)

    return Index(
        postings=dict(merged_postings),
        docs=merged_docs,
        with_positions=with_positions,
    )


def _build_from_jsonl(
    jsonl_path: Path,
    with_positions: bool = True,
    limit: int | None = None,
    workers: int = 1,
    executor: ExecutorKind = "serial",
    on_doc: Callable[[Document], None] | None = None,
) -> tuple[Index, BuildStats]:
    """Stream documents from a crawled .jsonl file and build an Index outside the event loop."""
    t_wall_start = time.perf_counter()
    t_cpu_start = time.process_time()

    postings_map: dict[str, list[Posting]] = defaultdict(list)
    docs_meta: dict[int, DocMeta] = {}
    doc_id = 0

    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if limit is not None and doc_id >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            text = str(record.get("text", ""))
            url = str(record.get("url", f"{jsonl_path.as_posix()}#{doc_id}"))
            title = str(record.get("title") or url)

            if on_doc is not None:
                on_doc(Document(doc_id=doc_id, path=Path(url), text=text, title=title, url=url))

            tokens = list(tokenize(text))
            docs_meta[doc_id] = DocMeta(
                doc_id=doc_id,
                path=url,
                title=title,
                length=len(tokens),
            )
            if with_positions:
                pos_map: dict[str, list[int]] = defaultdict(list)
                for pos, term in enumerate(tokens):
                    pos_map[term].append(pos)
                for term, positions in pos_map.items():
                    postings_map[term].append(
                        Posting(doc_id=doc_id, tf=len(positions), positions=tuple(positions))
                    )
            else:
                for term, tf in Counter(tokens).items():
                    postings_map[term].append(Posting(doc_id=doc_id, tf=tf))
            doc_id += 1

    t_merge_start = time.perf_counter()
    index = Index(postings=dict(postings_map), docs=docs_meta, with_positions=with_positions)
    merge_elapsed = time.perf_counter() - t_merge_start
    wall_elapsed = time.perf_counter() - t_wall_start
    cpu_elapsed = time.process_time() - t_cpu_start
    return index, BuildStats(wall_elapsed, cpu_elapsed, merge_elapsed, workers, executor)


def build_index_parallel(
    root: Path,
    with_positions: bool = True,
    limit: int | None = None,
    workers: int = 1,
    executor: ExecutorKind = "processes",
    on_doc: Callable[[Document], None] | None = None,
) -> tuple[Index, BuildStats]:
    """Build an inverted index using serial, threads, or processes and return timing metrics."""
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    if root.is_file() and root.suffix.lower() == ".jsonl":
        return _build_from_jsonl(
            jsonl_path=root,
            with_positions=with_positions,
            limit=limit,
            workers=workers,
            executor=executor,
            on_doc=on_doc,
        )

    t_wall_start = time.perf_counter()
    t_cpu_start = time.process_time()

    paths = _discover_paths(root, limit=limit)
    if not paths:
        empty = Index(postings={}, docs={}, with_positions=with_positions)
        stats = BuildStats(0.0, 0.0, 0.0, workers=workers, executor=executor)
        return empty, stats

    if executor == "serial" or (workers == 1 and on_doc is not None):
        postings_map: dict[str, list[Posting]] = defaultdict(list)
        docs_meta: dict[int, DocMeta] = {}
        for doc_id, path in enumerate(paths):
            text = path.read_text(encoding="utf-8", errors="ignore")
            doc = Document(doc_id=doc_id, path=path, text=text)
            if on_doc is not None:
                on_doc(doc)
            tokens = list(tokenize(text))
            docs_meta[doc_id] = DocMeta(
                doc_id=doc_id,
                path=path.as_posix(),
                title=_title_from_path(path),
                length=len(tokens),
            )
            if with_positions:
                pos_map: dict[str, list[int]] = defaultdict(list)
                for pos, term in enumerate(tokens):
                    pos_map[term].append(pos)
                for term, positions in pos_map.items():
                    postings_map[term].append(
                        Posting(doc_id=doc_id, tf=len(positions), positions=tuple(positions))
                    )
            else:
                for term, tf in Counter(tokens).items():
                    postings_map[term].append(Posting(doc_id=doc_id, tf=tf))

        t_merge_start = time.perf_counter()
        index = Index(postings=dict(postings_map), docs=docs_meta, with_positions=with_positions)
        merge_elapsed = time.perf_counter() - t_merge_start
    elif executor == "threads":
        chunks = _chunk_with_offsets(paths, workers)
        partials: list[PartialIndex] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(build_partial, chunk_paths, offset, with_positions): offset
                for offset, chunk_paths in chunks
            }
            for fut in as_completed(futures):
                part = fut.result()
                partials.append(part)
                if on_doc is not None:
                    for meta in part.docs.values():
                        on_doc(Document(doc_id=meta.doc_id, path=Path(meta.path), text=""))
        t_merge_start = time.perf_counter()
        index = merge_partials(partials)
        merge_elapsed = time.perf_counter() - t_merge_start
    else:
        chunks = _chunk_with_offsets(paths, workers)
        partials = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(build_partial, chunk_paths, offset, with_positions): offset
                for offset, chunk_paths in chunks
            }
            for fut in as_completed(futures):
                part = fut.result()
                partials.append(part)
                if on_doc is not None:
                    for meta in part.docs.values():
                        on_doc(Document(doc_id=meta.doc_id, path=Path(meta.path), text=""))
        t_merge_start = time.perf_counter()
        index = merge_partials(partials)
        merge_elapsed = time.perf_counter() - t_merge_start

    wall_elapsed = time.perf_counter() - t_wall_start
    cpu_elapsed = time.process_time() - t_cpu_start
    stats = BuildStats(
        wall_time_s=wall_elapsed,
        cpu_time_s=cpu_elapsed,
        merge_time_s=merge_elapsed,
        workers=workers,
        executor=executor,
    )
    return index, stats


@timed
def build_index(
    root: Path,
    with_positions: bool = True,
    limit: int | None = None,
    on_doc: Callable[[Document], None] | None = None,
    workers: int = 1,
    executor: ExecutorKind = "serial",
) -> Index:
    """Build an inverted index over the corpus (backward-compatible wrapper)."""
    index, _ = build_index_parallel(
        root=root,
        with_positions=with_positions,
        limit=limit,
        workers=workers,
        executor=executor,
        on_doc=on_doc,
    )
    return index
