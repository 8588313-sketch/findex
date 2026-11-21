from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Application settings loaded exclusively from environment variables."""

    index_path: Path
    app_title: str = "findex — Web Search"
    default_k: int = 10
    max_k: int = 50

    @classmethod
    def from_env(cls) -> Settings:
        raw_path = os.getenv("INDEX_PATH") or os.getenv("FINDEX_INDEX_PATH") or "index.bin"
        title = os.getenv("FINDEX_APP_TITLE", "findex — Ranked Search Engine")
        default_k = int(os.getenv("FINDEX_DEFAULT_K", "10"))
        max_k = int(os.getenv("FINDEX_MAX_K", "50"))
        return cls(
            index_path=Path(raw_path),
            app_title=title,
            default_k=default_k,
            max_k=max_k,
        )
