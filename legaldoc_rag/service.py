"""项目的用例层：索引文档、检索证据、拒答并生成带引用的回答。"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from .config import Settings
from .documents import chunk_document, content_fingerprint
from .embeddings import create_embedding_backend
from .models import Answer, BuildReport, SearchHit
from .retrieval import HybridIndex, tokenize


class LegalDocService:
    """供 Streamlit、评测脚本和测试共同调用的稳定入口。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedding = create_embedding_backend(settings.embedding_backend, settings.embedding_model)
        self.index: HybridIndex | None = None
        self.current_fingerprint: str | None = None

    @property
    def has_index(self) -> bool:
        return self.index is not None

    @property
    def embedding_backend_name(self) -> str:
        return self.embedding.name

    @property
    def answer_backend_name(self) -> str:
        return self.settings.groq_model if self.settings.groq_api_key else "离线证据摘录"

    def build_index(self, paths: list[Path]) -> BuildReport:
        paths = [Path(path).resolve() for path in paths]
        if not paths:
            raise ValueError("至少需要一份文档")
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"文档不存在：{', '.join(missing)}")

        signature = f"chunks={self.settings.chunk_size}/{self.settings.chunk_overlap};embedding={self.embedding.name}"
        fingerprint = content_fingerprint(paths, signature)
        cache_path = self.settings.index_dir / f"{fingerprint}.json"
        cached = HybridIndex.load(cache_path, self.embedding, fingerprint)
        loaded_from_cache = cached is not None

        if cached is None:
            chunks = [
                chunk
                for path in paths
                for chunk in chunk_document(path, self.settings.chunk_size, self.settings.chunk_overlap)
            ]
            if not chunks:
                raise ValueError("文档未生成任何有效文本片段")
            cached = HybridIndex.build(chunks, self.embedding)
            cached.save(cache_path, fingerprint)

        self.index = cached
        self.current_fingerprint = fingerprint
        return BuildReport(len(paths), len(cached.chunks), fingerprint, loaded_from_cache)

    def retrieve(self, question: str, top_k: int | None = None) -> list[SearchHit]:
        if self.index is None:
            raise RuntimeError("请先构建索引")
        return self.index.search(
            question,
            top_k=top_k or self.settings.top_k,
            candidate_k=self.settings.candidate_k,
            rrf_k=self.settings.rrf_k,
        )

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        hits = self.retrieve(question, top_k)
        if not hits or not self._has_sufficient_evidence(hits[0]):
            return Answer("未在当前文档中找到足够证据，请换一种问法或补充相关合同。", hits[:2], True)
        if self.settings.groq_api_key:
            try:
                text = self._ask_groq(question, hits)
                return Answer(text, hits, False)
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
                # 网络或供应商异常不应让本地检索不可用；界面仍返回可核验原文。
                pass
        return Answer(self._extractive_answer(question, hits), hits, False)

    def _has_sufficient_evidence(self, top: SearchHit) -> bool:
        """BM25 命中或较强稠密相似度满足其一，才允许回答。"""

        # 单个常见词（如“乙方”）不足以证明文档回答了“注册地址”这类问题。
        return (
            top.lexical_coverage >= 0.20 or top.bm25_score >= 1.50 or top.dense_score >= self.settings.abstain_threshold
        )

    @staticmethod
    def _extractive_answer(question: str, hits: list[SearchHit]) -> str:
        """无 LLM 时抽取词项重合度最高的句子，并保留引用编号。"""

        query_terms = set(tokenize(question))
        candidates: list[tuple[float, str, int]] = []
        for hit in hits:
            sentences = [part.strip() for part in re.split(r"(?<=[。！？；])", hit.chunk.text) if part.strip()]
            for sentence in sentences:
                sentence_terms = set(tokenize(sentence))
                overlap = len(query_terms & sentence_terms) / max(len(query_terms), 1)
                candidates.append((overlap, sentence, hit.citation_id))
        candidates.sort(key=lambda item: item[0], reverse=True)

        selected: list[str] = []
        seen: set[str] = set()
        for _, sentence, citation_id in candidates:
            normalised = sentence.strip()
            if normalised in seen:
                continue
            seen.add(normalised)
            selected.append(f"{normalised} [{citation_id}]")
            if len(selected) == 2:
                break
        return "根据当前合同证据：\n\n" + "\n\n".join(selected)

    def _ask_groq(self, question: str, hits: list[SearchHit]) -> str:
        evidence = "\n\n".join(
            f"[{hit.citation_id}] 文件={hit.chunk.source_name}；章节={hit.chunk.section}\n{hit.chunk.text}"
            for hit in hits
        )
        prompt = (
            "你是合同条款检索助手。只能依据给定证据回答，不得补充外部法律结论。"
            "每个事实后必须写对应引用编号，例如[1]。证据不足时只回答："
            "未在当前文档中找到足够证据。\n\n"
            f"问题：{question}\n\n证据：\n{evidence}"
        )
        payload = json.dumps(
            {
                "model": self.settings.groq_model,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.settings.groq_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - 固定 HTTPS 端点
            result = json.loads(response.read().decode("utf-8"))
        text = result["choices"][0]["message"]["content"].strip()
        if not text:
            raise ValueError("模型返回空回答")
        return text
