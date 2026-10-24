"""跨模块共享的数据结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class Chunk:
    """一个可被独立引用的文档证据片段。"""

    chunk_id: str
    source_name: str
    section: str
    text: str
    page: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Chunk:
        return cls(**value)


@dataclass(slots=True)
class SearchHit:
    """混合检索结果；保留各通道排名以支持排障和评测。"""

    chunk: Chunk
    score: float
    citation_id: int = 0
    bm25_rank: int | None = None
    dense_rank: int | None = None
    bm25_score: float = 0.0
    dense_score: float = 0.0
    lexical_coverage: float = 0.0


@dataclass(slots=True)
class Answer:
    text: str
    sources: list[SearchHit]
    abstained: bool


@dataclass(slots=True)
class BuildReport:
    document_count: int
    chunk_count: int
    fingerprint: str
    loaded_from_cache: bool
