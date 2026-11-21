from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.decorators import timed
from findex.models import DocMeta, Index, Posting
from findex.tokenize import tokenize


def _title_from_path(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        lang, rest = stem.split("_", 1)
        return f"[{lang}] {rest.replace('_', ' ')}"
    return stem.replace("_", " ")


@timed
def build_index(
    root: Path,
    with_positions: bool = True,
    limit: int | None = None,
) -> Index:
    """Build an inverted index in a single streaming pass over the corpus."""
    docs_iter = iter_documents(root)
    if limit is not None:
        docs_iter = islice(docs_iter, limit)

    postings_map: dict[str, list[Posting]] = defaultdict(list)
    docs_meta: dict[int, DocMeta] = {}

    for doc in docs_iter:
        tokens = list(tokenize(doc.text))
        docs_meta[doc.doc_id] = DocMeta(
            doc_id=doc.doc_id,
            path=doc.path.as_posix(),
            title=_title_from_path(doc.path),
            length=len(tokens),
        )

        if with_positions:
            pos_map: dict[str, list[int]] = defaultdict(list)
            for pos, term in enumerate(tokens):
                pos_map[term].append(pos)
            for term, positions in pos_map.items():
                postings_map[term].append(
                    Posting(doc_id=doc.doc_id, tf=len(positions), positions=tuple(positions))
                )
        else:
            counts = Counter(tokens)
            for term, tf in counts.items():
                postings_map[term].append(Posting(doc_id=doc.doc_id, tf=tf))

    return Index(
        postings=dict(postings_map),
        docs=docs_meta,
        with_positions=with_positions,
    )
