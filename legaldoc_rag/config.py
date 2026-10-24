"""集中管理配置，避免魔法数字散落在界面和检索代码中。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    root: Path
    index_dir: Path
    embedding_backend: str = "hash"
    embedding_model: str = "intfloat/multilingual-e5-small"
    top_k: int = 4
    candidate_k: int = 12
    rrf_k: int = 60
    abstain_threshold: float = 0.35
    chunk_size: int = 700
    chunk_overlap: int = 100
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"

    @classmethod
    def from_env(cls, root: Path) -> Settings:
        load_dotenv(root / ".env", override=False)
        return cls(
            root=root,
            index_dir=root / "data" / "indexes",
            embedding_backend=os.getenv("EMBEDDING_BACKEND", "hash").strip().lower(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
            top_k=int(os.getenv("TOP_K", "4")),
            candidate_k=int(os.getenv("CANDIDATE_K", "12")),
            abstain_threshold=float(os.getenv("ABSTAIN_THRESHOLD", "0.35")),
            groq_api_key=os.getenv("GROQ_API_KEY") or None,
            groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        )
