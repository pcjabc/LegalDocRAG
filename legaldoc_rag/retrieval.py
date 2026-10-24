"""BM25、稠密检索与 RRF 融合实现。

这部分不依赖 LangChain，排名、分数和持久化格式都可以直接检查，方便面试解释
以及离线评测。索引仅包含用户本地文档的分块文本和数值向量。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from .embeddings import EmbeddingBackend
from .models import Chunk, SearchHit


def tokenize(text: str) -> list[str]:
    """保留英文/数字词和中文二元词，避免常见单字把无关问题误判为命中。"""

    lowered = text.lower()
    latin = re.findall(r"[a-z0-9][a-z0-9_.%-]*", lowered)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", lowered)
    chinese = [run[index : index + 2] for run in chinese_runs for index in range(len(run) - 1)]
    return latin + chinese


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


class HybridIndex:
    """小规模法律文档索引；数据量增大后可替换为 FAISS/Elasticsearch。"""

    def __init__(self, chunks: list[Chunk], vectors: list[list[float]], embedding: EmbeddingBackend) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("分块数量和向量数量不一致")
        self.chunks = chunks
        self.vectors = vectors
        self.embedding = embedding
        self.term_counts = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.lengths = [sum(counter.values()) for counter in self.term_counts]
        self.avg_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.document_frequency = Counter(token for counter in self.term_counts for token in counter)

    @classmethod
    def build(cls, chunks: list[Chunk], embedding: EmbeddingBackend) -> HybridIndex:
        vectors = embedding.embed_documents([chunk.text for chunk in chunks])
        return cls(chunks, vectors, embedding)

    def _bm25_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> list[float]:
        query_terms = set(tokenize(query))
        total = len(self.chunks)
        scores: list[float] = []
        for counter, length in zip(self.term_counts, self.lengths, strict=True):
            score = 0.0
            for term in query_terms:
                frequency = counter.get(term, 0)
                if not frequency:
                    continue
                frequency_docs = self.document_frequency[term]
                inverse_frequency = math.log(1 + (total - frequency_docs + 0.5) / (frequency_docs + 0.5))
                denominator = frequency + k1 * (1 - b + b * length / (self.avg_length or 1.0))
                score += inverse_frequency * frequency * (k1 + 1) / denominator
            scores.append(score)
        return scores

    def search(self, query: str, top_k: int = 4, candidate_k: int = 12, rrf_k: int = 60) -> list[SearchHit]:
        if not query.strip() or not self.chunks:
            return []
        bm25_scores = self._bm25_scores(query)
        query_terms = set(tokenize(query))
        query_vector = self.embedding.embed_query(query)
        dense_scores = [cosine(query_vector, vector) for vector in self.vectors]

        limit = min(candidate_k, len(self.chunks))
        bm25_order = sorted(range(len(self.chunks)), key=lambda index: bm25_scores[index], reverse=True)[:limit]
        dense_order = sorted(range(len(self.chunks)), key=lambda index: dense_scores[index], reverse=True)[:limit]
        bm25_ranks = {index: rank for rank, index in enumerate(bm25_order, 1)}
        dense_ranks = {index: rank for rank, index in enumerate(dense_order, 1)}

        fused: dict[int, float] = {}
        for index, rank in bm25_ranks.items():
            # 分数为 0 的 BM25 候选不应仅因“排在列表中”获得融合加分。
            if bm25_scores[index] > 0:
                fused[index] = fused.get(index, 0.0) + 1 / (rrf_k + rank)
        for index, rank in dense_ranks.items():
            fused[index] = fused.get(index, 0.0) + 1 / (rrf_k + rank)

        ordered = sorted(fused, key=lambda index: fused[index], reverse=True)[:top_k]
        return [
            SearchHit(
                chunk=self.chunks[index],
                score=fused[index],
                citation_id=position,
                bm25_rank=bm25_ranks.get(index) if bm25_scores[index] > 0 else None,
                dense_rank=dense_ranks.get(index),
                bm25_score=bm25_scores[index],
                dense_score=dense_scores[index],
                lexical_coverage=(
                    len(query_terms & set(self.term_counts[index])) / len(query_terms) if query_terms else 0.0
                ),
            )
            for position, index in enumerate(ordered, 1)
        ]

    def save(self, path: Path, fingerprint: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "fingerprint": fingerprint,
            "embedding_backend": self.embedding.name,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "vectors": self.vectors,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, embedding: EmbeddingBackend, fingerprint: str) -> HybridIndex | None:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 1 or payload.get("fingerprint") != fingerprint:
            return None
        if payload.get("embedding_backend") != embedding.name:
            return None
        chunks = [Chunk.from_dict(value) for value in payload["chunks"]]
        return cls(chunks, payload["vectors"], embedding)
