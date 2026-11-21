from __future__ import annotations

from fastapi import HTTPException, Request, status

from findex.models import Index
from findex.semantic import SemanticIndex


def get_optional_index(request: Request) -> Index | None:
    """Return the loaded Index from app.state or None if unavailable."""
    return getattr(request.app.state, "index", None)


def get_optional_semantic_index(request: Request) -> SemanticIndex | None:
    """Return the loaded SemanticIndex from app.state or None if unavailable."""
    return getattr(request.app.state, "semantic_index", None)


def get_index(request: Request) -> Index:
    """Require a loaded Index; raise 503 Service Unavailable if not ready."""
    idx = get_optional_index(request)
    if idx is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search index is not loaded yet",
        )
    return idx
