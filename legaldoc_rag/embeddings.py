"""可替换的稠密向量后端。

默认哈希后端不下载模型，适合演示和 CI；E5 后端用于真实语义检索。两者均把
query/passage 分开编码，避免上游项目遗漏 ``query:`` 前缀的问题。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class EmbeddingBackend(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class HashEmbedding:
    """中文字符 n-gram 哈希向量：确定、离线、无需训练，主要用于可运行兜底。"""

    name = "hash-char-ngram"

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        compact = re.sub(r"\s+", "", text.lower())
        vector = [0.0] * self.dimensions
        features = [
            compact[index : index + size] for size in (1, 2, 3) for index in range(max(0, len(compact) - size + 1))
        ]
        for feature in features:
            raw = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(raw, "little")
            vector[number % self.dimensions] += -1.0 if number & 1 else 1.0
        return _normalise(vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 哈希后端没有训练时约定的前缀；直接编码可避免固定英文前缀制造虚假相似度。
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class E5Embedding:
    """SentenceTransformers E5 后端；明确使用官方非对称前缀。"""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - 可选依赖
            raise RuntimeError("E5 模式需要安装 sentence-transformers") from exc
        self.model = SentenceTransformer(model_name)
        self.name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        values = self.model.encode([f"passage: {text}" for text in texts], normalize_embeddings=True)
        return values.tolist()

    def embed_query(self, text: str) -> list[float]:
        value = self.model.encode([f"query: {text}"], normalize_embeddings=True)[0]
        return value.tolist()


def create_embedding_backend(kind: str, model_name: str) -> EmbeddingBackend:
    if kind == "e5":
        return E5Embedding(model_name)
    if kind != "hash":
        raise ValueError("EMBEDDING_BACKEND 仅支持 hash 或 e5")
    return HashEmbedding()
