from __future__ import annotations

import html
import logging
import math
import re
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from findex.ast_query import QuerySyntaxError
from findex.models import Index
from findex.ranking import make_scorer, search
from findex.storage import load_index
from findex.tokenize import tokenize
from findex.web.config import Settings
from findex.web.deps import get_index, get_optional_index
from findex.web.schemas import (
    DocumentOut,
    HealthOut,
    ScorerLiteral,
    SearchHitOut,
    SearchResponseOut,
    StatsOut,
)

log = logging.getLogger("findex.web")
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _highlight_snippet(snippet: str, query: str) -> str:
    """Safely HTML-escape a snippet and wrap query terms in <mark>...</mark>."""
    escaped = html.escape(snippet)
    terms = sorted(set(tokenize(query)), key=len, reverse=True)
    if not terms:
        return escaped
    pattern = re.compile(r"(" + "|".join(re.escape(t) for t in terms) + r")", re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", escaped)


def _read_doc_content(path_str: str, fallback_title: str) -> str:
    p = Path(path_str)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8", errors="ignore")
    return fallback_title


def _execute_paginated_search(
    idx: Index,
    q: str,
    k: int,
    scorer_name: ScorerLiteral,
    page: int,
) -> SearchResponseOut:
    t0 = time.perf_counter()
    scorer = make_scorer(scorer_name)
    # Retrieve enough top hits to serve requested page deterministically
    fetch_limit = max(page * k, 100)
    try:
        all_results = search(idx, q, scorer=scorer, top_k=fetch_limit)
    except QuerySyntaxError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid query syntax: {exc}",
        ) from exc

    total = len(all_results)
    total_pages = math.ceil(total / k) if total > 0 else 0
    start_idx = (page - 1) * k
    page_slice = all_results[start_idx : start_idx + k]

    hits_out = [
        SearchHitOut(
            doc_id=r.doc_id,
            score=round(r.score, 6),
            title=r.title,
            path=idx.docs[r.doc_id].path if r.doc_id in idx.docs else "",
            snippet=r.snippet,
        )
        for r in page_slice
    ]
    took_ms = (time.perf_counter() - t0) * 1000.0
    return SearchResponseOut(
        query=q,
        scorer=scorer_name,
        page=page,
        k=k,
        total=total,
        total_pages=total_pages,
        has_prev=page > 1 and total > 0,
        has_next=page < total_pages,
        took_ms=round(took_ms, 3),
        hits=hits_out,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings if settings is not None else Settings.from_env()
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
        application.state.settings = cfg
        if cfg.index_path.exists() and cfg.index_path.is_file():
            t0 = time.perf_counter()
            idx = load_index(cfg.index_path)
            application.state.index = idx
            log.info(
                "Loaded index from %s (%d docs, %d terms) in %.3fs",
                cfg.index_path,
                idx.num_docs,
                len(idx),
                time.perf_counter() - t0,
            )
        else:
            application.state.index = None
            log.warning("Index file not found at %s; /health will return 503", cfg.index_path)
        yield
        application.state.index = None

    application = FastAPI(
        title=cfg.app_title,
        version="0.7.0",
        lifespan=lifespan,
    )
    application.state.settings = cfg
    application.state.index = None

    @application.middleware("http")
    async def request_id_and_error_boundary(
        request: Request,
        call_next: Callable[[Request], Any],
    ) -> Response:
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = req_id
        try:
            response: Response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response
        except Exception:
            log.exception("Unhandled error while processing request_id=%s path=%s", req_id, request.url.path)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal Server Error", "request_id": req_id},
                headers={"X-Request-ID": req_id},
            )

    @application.get("/health", response_model=HealthOut)
    def health_check(
        response: Response,
        idx: Annotated[Index | None, Depends(get_optional_index)],
    ) -> HealthOut:
        if idx is None:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return HealthOut(status="unavailable", index_loaded=False, num_docs=0)
        return HealthOut(status="ok", index_loaded=True, num_docs=idx.num_docs)

    @application.get("/stats", response_model=StatsOut)
    def index_stats(idx: Annotated[Index, Depends(get_index)]) -> StatsOut:
        total_postings = sum(len(pl) for pl in idx.values())
        avg_len = (
            sum(meta.length for meta in idx.docs.values()) / idx.num_docs
            if idx.num_docs > 0
            else 0.0
        )
        return StatsOut(
            num_docs=idx.num_docs,
            vocab_size=len(idx),
            total_postings=total_postings,
            avg_doc_length=round(avg_len, 2),
            with_positions=idx.with_positions,
        )

    @application.get("/docs/{doc_id}", response_model=DocumentOut)
    def get_document(
        doc_id: int,
        idx: Annotated[Index, Depends(get_index)],
    ) -> DocumentOut:
        meta = idx.docs.get(doc_id)
        if meta is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {doc_id} not found",
            )
        content = _read_doc_content(meta.path, meta.title)
        return DocumentOut(
            doc_id=meta.doc_id,
            title=meta.title,
            path=meta.path,
            length=meta.length,
            content=content,
        )

    # Regular def route so FastAPI/Starlette offloads CPU-bound tokenization/scoring to threadpool
    @application.get("/search", response_model=SearchResponseOut)
    def api_search(
        idx: Annotated[Index, Depends(get_index)],
        q: Annotated[str, Query(min_length=1, max_length=300, description="Search query")],
        k: Annotated[int, Query(ge=1, le=50, description="Hits per page")] = 10,
        scorer: Annotated[ScorerLiteral, Query(description="Ranking model: bm25 or tfidf")] = "bm25",
        page: Annotated[int, Query(ge=1, le=100, description="1-based page number")] = 1,
    ) -> SearchResponseOut:
        stripped_q = q.strip()
        if not stripped_q:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Query parameter 'q' must not be blank",
            )
        return _execute_paginated_search(idx, stripped_q, k, scorer, page)

    def _enrich_for_template(resp: SearchResponseOut) -> dict[str, Any]:
        enriched_hits = [
            {
                **h.model_dump(),
                "highlighted_snippet": _highlight_snippet(h.snippet or h.title, resp.query),
            }
            for h in resp.hits
        ]
        return {
            **resp.model_dump(),
            "hits": enriched_hits,
        }

    @application.get("/", response_class=HTMLResponse)
    def home_page(
        request: Request,
        idx: Annotated[Index | None, Depends(get_optional_index)],
        q: str = "",
        scorer: ScorerLiteral = "bm25",
        k: int = 10,
        page: int = 1,
        doc_id: int | None = None,
    ) -> HTMLResponse:
        stats_data = None
        if idx is not None:
            stats_data = {
                "num_docs": idx.num_docs,
                "vocab_size": len(idx),
            }

        selected_doc = None
        if idx is not None and doc_id is not None and doc_id in idx.docs:
            meta = idx.docs[doc_id]
            selected_doc = {
                "doc_id": meta.doc_id,
                "title": meta.title,
                "path": meta.path,
                "length": meta.length,
                "content": _read_doc_content(meta.path, meta.title),
            }

        response_ctx = None
        error_msg = None
        cleaned_q = q.strip()
        if cleaned_q:
            if idx is None:
                error_msg = "Індекс ще не завантажений (503 Service Unavailable)."
            else:
                try:
                    raw_resp = _execute_paginated_search(
                        idx,
                        cleaned_q,
                        max(1, min(k, cfg.max_k)),
                        scorer,
                        max(1, page),
                    )
                    response_ctx = _enrich_for_template(raw_resp)
                except HTTPException as exc:
                    error_msg = str(exc.detail)

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "title": cfg.app_title,
                "stats": stats_data,
                "query": q,
                "scorer": scorer,
                "k": max(1, min(k, cfg.max_k)),
                "page": max(1, page),
                "selected_doc": selected_doc,
                "response": response_ctx,
                "error_message": error_msg,
            },
        )

    @application.get("/ui/results", response_class=HTMLResponse)
    def htmx_results(
        request: Request,
        idx: Annotated[Index | None, Depends(get_optional_index)],
        q: str = "",
        scorer: ScorerLiteral = "bm25",
        k: int = 10,
        page: int = 1,
    ) -> HTMLResponse:
        cleaned_q = q.strip()
        response_ctx = None
        error_msg = None
        if cleaned_q:
            if idx is None:
                error_msg = "Індекс ще не завантажений (503)."
            else:
                try:
                    raw_resp = _execute_paginated_search(
                        idx,
                        cleaned_q,
                        max(1, min(k, cfg.max_k)),
                        scorer,
                        max(1, page),
                    )
                    response_ctx = _enrich_for_template(raw_resp)
                except HTTPException as exc:
                    error_msg = str(exc.detail)
        return templates.TemplateResponse(
            request=request,
            name="_results.html",
            context={"response": response_ctx, "error_message": error_msg},
        )

    @application.get("/ui/doc/{doc_id}", response_class=HTMLResponse)
    def htmx_doc_viewer(
        doc_id: int,
        idx: Annotated[Index, Depends(get_index)],
    ) -> HTMLResponse:
        meta = idx.docs.get(doc_id)
        if meta is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
        content = html.escape(_read_doc_content(meta.path, meta.title))
        title_esc = html.escape(meta.title)
        path_esc = html.escape(meta.path)
        snippet_html = "".join(
            [
                '<div class="doc-viewer"><div class="card-header">',
                f"<strong>#{meta.doc_id} — {title_esc}</strong>",
                '<button type="button" onclick="this.parentElement.parentElement.remove()" ',
                'style="padding:0.2rem 0.6rem;font-size:0.8rem;">Закрити ✕</button></div>',
                f'<div style="font-size:0.82rem;color:var(--muted);">{path_esc} · {meta.length} токенів</div>',
                f"<pre>{content}</pre></div>",
            ]
        )
        return HTMLResponse(content=snippet_html)

    return application


app = create_app()
