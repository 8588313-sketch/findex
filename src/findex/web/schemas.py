from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ScorerLiteral = Literal["bm25", "tfidf"]


class SearchHitOut(BaseModel):
    doc_id: int
    score: float
    title: str
    path: str
    snippet: str


class SearchResponseOut(BaseModel):
    query: str
    scorer: ScorerLiteral
    page: int = Field(ge=1)
    k: int = Field(ge=1)
    total: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    has_prev: bool
    has_next: bool
    took_ms: float = Field(ge=0.0)
    hits: list[SearchHitOut]


class DocumentOut(BaseModel):
    doc_id: int
    title: str
    path: str
    length: int
    content: str


class StatsOut(BaseModel):
    num_docs: int
    vocab_size: int
    total_postings: int
    avg_doc_length: float
    with_positions: bool


class HealthOut(BaseModel):
    status: Literal["ok", "unavailable"]
    index_loaded: bool
    num_docs: int = 0
