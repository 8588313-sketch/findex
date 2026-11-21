from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from findex.models import Index
from findex.tokenize import tokenize


class QuerySyntaxError(ValueError):
    """Raised when a Boolean/phrase query cannot be parsed."""


class QueryNode(ABC):
    @abstractmethod
    def evaluate(self, index: Index) -> set[int]:
        """Return matching doc_ids for this AST node."""

    @abstractmethod
    def positive_terms(self) -> list[str]:
        """Return non-negated terms for scoring and snippet highlighting."""

    @abstractmethod
    def phrases(self) -> list[tuple[str, ...]]:
        """Return positive phrases for snippet highlighting."""

    def __and__(self, other: QueryNode) -> And:
        return And(self, other)

    def __or__(self, other: QueryNode) -> Or:
        return Or(self, other)

    def __invert__(self) -> Not:
        return Not(self)


@dataclass(frozen=True, slots=True)
class Term(QueryNode):
    term: str

    def evaluate(self, index: Index) -> set[int]:
        return {p.doc_id for p in index.get(self.term, ())}

    def positive_terms(self) -> list[str]:
        return [self.term]

    def phrases(self) -> list[tuple[str, ...]]:
        return []


@dataclass(frozen=True, slots=True)
class Phrase(QueryNode):
    terms: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        if not self.terms:
            return set()
        if not index.with_positions:
            raise ValueError(
                "Phrase search requires positional index (rebuild without --no-positions)"
            )
        if len(self.terms) == 1:
            return {p.doc_id for p in index.get(self.terms[0], ())}

        # Candidate docs must contain all terms in the phrase
        postings_per_term = [index.get(t, ()) for t in self.terms]
        if any(not pl for pl in postings_per_term):
            return set()

        doc_pos_maps: list[dict[int, tuple[int, ...]]] = [
            {p.doc_id: p.positions for p in pl} for pl in postings_per_term
        ]
        candidate_docs = set(doc_pos_maps[0].keys())
        for m in doc_pos_maps[1:]:
            candidate_docs &= m.keys()

        matched: set[int] = set()
        for doc_id in candidate_docs:
            valid_starts = set(doc_pos_maps[0][doc_id])
            for offset, pos_map in enumerate(doc_pos_maps[1:], start=1):
                next_positions = set(pos_map[doc_id])
                valid_starts = {
                    start for start in valid_starts if (start + offset) in next_positions
                }
                if not valid_starts:
                    break
            if valid_starts:
                matched.add(doc_id)
        return matched

    def positive_terms(self) -> list[str]:
        return list(self.terms)

    def phrases(self) -> list[tuple[str, ...]]:
        return [self.terms]


@dataclass(frozen=True, slots=True)
class And(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def positive_terms(self) -> list[str]:
        return self.left.positive_terms() + self.right.positive_terms()

    def phrases(self) -> list[tuple[str, ...]]:
        return self.left.phrases() + self.right.phrases()


@dataclass(frozen=True, slots=True)
class Or(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def positive_terms(self) -> list[str]:
        return self.left.positive_terms() + self.right.positive_terms()

    def phrases(self) -> list[tuple[str, ...]]:
        return self.left.phrases() + self.right.phrases()


@dataclass(frozen=True, slots=True)
class Not(QueryNode):
    operand: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return set(index.all_doc_ids) - self.operand.evaluate(index)

    def positive_terms(self) -> list[str]:
        return []

    def phrases(self) -> list[tuple[str, ...]]:
        return []


_TOKEN_SPEC = re.compile(
    r'\s*(?:"([^"]*)"|(\()|(\))|(~)|([^\s()~"]+))'
)


@dataclass(frozen=True, slots=True)
class _LexToken:
    kind: str  # LPAREN, RPAREN, AND, OR, NOT, PHRASE, WORD
    value: str


def _lex(query: str) -> list[_LexToken]:
    tokens: list[_LexToken] = []
    pos = 0
    n = len(query)
    while pos < n:
        if query[pos].isspace():
            pos += 1
            continue
        m = _TOKEN_SPEC.match(query, pos)
        if not m:
            raise QuerySyntaxError(f"Unexpected character at position {pos}: {query[pos:]!r}")
        pos = m.end()
        phrase_val, lparen, rparen, tilde, word = m.groups()
        if phrase_val is not None:
            tokens.append(_LexToken("PHRASE", phrase_val))
        elif lparen is not None:
            tokens.append(_LexToken("LPAREN", "("))
        elif rparen is not None:
            tokens.append(_LexToken("RPAREN", ")"))
        elif tilde is not None:
            tokens.append(_LexToken("NOT", "~"))
        elif word is not None:
            upper = word.upper()
            if upper == "AND" or word == "&":
                tokens.append(_LexToken("AND", word))
            elif upper == "OR" or word == "|":
                tokens.append(_LexToken("OR", word))
            elif upper == "NOT":
                tokens.append(_LexToken("NOT", word))
            elif word.startswith("-") and len(word) > 1:
                tokens.append(_LexToken("NOT", "-"))
                tokens.append(_LexToken("WORD", word[1:]))
            else:
                tokens.append(_LexToken("WORD", word))
    return tokens


class _Parser:
    """Recursive-descent parser: OrExpr -> AndExpr -> NotExpr -> Primary."""

    def __init__(self, tokens: list[_LexToken]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> _LexToken | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self) -> _LexToken:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> QueryNode:
        if not self.tokens:
            raise QuerySyntaxError("Empty query")
        node = self._parse_or()
        if self.peek() is not None:
            raise QuerySyntaxError(f"Unexpected token: {self.peek()}")
        return node

    def _parse_or(self) -> QueryNode:
        node = self._parse_and()
        while (tok := self.peek()) is not None and tok.kind == "OR":
            self.consume()
            right = self._parse_and()
            node = Or(node, right)
        return node

    def _parse_and(self) -> QueryNode:
        node = self._parse_not()
        while (tok := self.peek()) is not None:
            if tok.kind == "AND":
                self.consume()
                right = self._parse_not()
                node = And(node, right)
            elif tok.kind in {"WORD", "PHRASE", "LPAREN", "NOT"}:
                # Implicit AND between adjacent expressions: `a b`, `a NOT b`, `a (b OR c)`
                right = self._parse_not()
                node = And(node, right)
            else:
                break
        return node

    def _parse_not(self) -> QueryNode:
        if (tok := self.peek()) is not None and tok.kind == "NOT":
            self.consume()
            return Not(self._parse_not())
        return self._parse_primary()

    def _parse_primary(self) -> QueryNode:
        tok = self.peek()
        if tok is None:
            raise QuerySyntaxError("Unexpected end of query")
        if tok.kind == "LPAREN":
            self.consume()
            inner = self._parse_or()
            closing = self.peek()
            if closing is None or closing.kind != "RPAREN":
                raise QuerySyntaxError("Unclosed '(' in query")
            self.consume()
            return inner
        if tok.kind == "PHRASE":
            self.consume()
            norm_tokens = tuple(tokenize(tok.value))
            if not norm_tokens:
                raise QuerySyntaxError(f"Empty phrase after normalization: {tok.value!r}")
            if len(norm_tokens) == 1:
                return Term(norm_tokens[0])
            return Phrase(norm_tokens)
        if tok.kind == "WORD":
            self.consume()
            norm_tokens = list(tokenize(tok.value))
            if not norm_tokens:
                raise QuerySyntaxError(f"Invalid term: {tok.value!r}")
            node: QueryNode = Term(norm_tokens[0])
            for extra in norm_tokens[1:]:
                node = And(node, Term(extra))
            return node
        raise QuerySyntaxError(f"Unexpected token in primary position: {tok}")


def parse_query(query: str) -> QueryNode:
    return _Parser(_lex(query)).parse()
