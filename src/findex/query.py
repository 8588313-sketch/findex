from typing import Literal

from findex.models import Index
from findex.tokenize import tokenize

SearchEngine = Literal["merge", "set"]


def normalize_term(raw: str) -> str | None:
    tokens = list(tokenize(raw))
    return tokens[0] if tokens else None


def intersect_sorted(a: list[int], b: list[int]) -> list[int]:
    """Two-pointer intersection of two sorted doc_id lists in O(|A| + |B|)."""
    i = j = 0
    res: list[int] = []
    len_a, len_b = len(a), len(b)
    while i < len_a and j < len_b:
        if a[i] == b[j]:
            res.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return res


def union_sorted(a: list[int], b: list[int]) -> list[int]:
    """Two-pointer union of two sorted doc_id lists in O(|A| + |B|)."""
    i = j = 0
    res: list[int] = []
    len_a, len_b = len(a), len(b)
    while i < len_a and j < len_b:
        if a[i] == b[j]:
            res.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            res.append(a[i])
            i += 1
        else:
            res.append(b[j])
            j += 1
    if i < len_a:
        res.extend(a[i:])
    if j < len_b:
        res.extend(b[j:])
    return res


def difference_sorted(universe: list[int], excluded: list[int]) -> list[int]:
    """Two-pointer difference (universe \\ excluded) on sorted lists."""
    i = j = 0
    res: list[int] = []
    len_u, len_e = len(universe), len(excluded)
    while i < len_u and j < len_e:
        if universe[i] == excluded[j]:
            i += 1
            j += 1
        elif universe[i] < excluded[j]:
            res.append(universe[i])
            i += 1
        else:
            j += 1
    if i < len_u:
        res.extend(universe[i:])
    return res


def _parse_query(query: str) -> list[tuple[list[str], list[str]]]:
    """Parse query into DNF clauses: OR of (positive_terms, negated_terms)."""
    clauses: list[tuple[list[str], list[str]]] = []
    pos_terms: list[str] = []
    neg_terms: list[str] = []
    negate_next = False

    for raw in query.strip().split():
        upper = raw.upper()
        if upper == "AND":
            continue
        if upper == "OR":
            clauses.append((pos_terms, neg_terms))
            pos_terms, neg_terms = [], []
            negate_next = False
            continue
        if upper == "NOT":
            negate_next = True
            continue
        if raw.startswith("-") and len(raw) > 1:
            term = normalize_term(raw[1:])
            if term:
                neg_terms.append(term)
            continue

        term = normalize_term(raw)
        if not term:
            continue
        if negate_next:
            neg_terms.append(term)
            negate_next = False
        else:
            pos_terms.append(term)

    clauses.append((pos_terms, neg_terms))
    return clauses


def search_index(
    index: Index,
    query: str,
    engine: SearchEngine = "merge",
) -> list[int]:
    """Execute a Boolean query using either two-pointer 'merge' or Python 'set'."""
    clauses = _parse_query(query)
    if not any(p or n for p, n in clauses):
        return []

    if engine == "set":
        universe_set = set(index.all_doc_ids)
        result_set: set[int] = set()
        for pos_terms, neg_terms in clauses:
            if not pos_terms and not neg_terms:
                continue
            if pos_terms:
                sorted_pos = sorted(pos_terms, key=lambda t: len(index.get(t, ())))
                clause_set = {p.doc_id for p in index.get(sorted_pos[0], ())}
                for term in sorted_pos[1:]:
                    clause_set &= {p.doc_id for p in index.get(term, ())}
            else:
                clause_set = set(universe_set)

            for term in neg_terms:
                clause_set -= {p.doc_id for p in index.get(term, ())}

            result_set |= clause_set
        return sorted(result_set)

    result_ids: list[int] = []
    for pos_terms, neg_terms in clauses:
        if not pos_terms and not neg_terms:
            continue
        if pos_terms:
            sorted_pos = sorted(pos_terms, key=lambda t: len(index.get(t, ())))
            clause_ids = [p.doc_id for p in index.get(sorted_pos[0], ())]
            for term in sorted_pos[1:]:
                other_ids = [p.doc_id for p in index.get(term, ())]
                clause_ids = intersect_sorted(clause_ids, other_ids)
                if not clause_ids:
                    break
        else:
            clause_ids = list(index.all_doc_ids)

        for term in neg_terms:
            if not clause_ids:
                break
            excl_ids = [p.doc_id for p in index.get(term, ())]
            clause_ids = difference_sorted(clause_ids, excl_ids)

        result_ids = union_sorted(result_ids, clause_ids)

    return result_ids
