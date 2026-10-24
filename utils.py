"""兼容上游工具函数；新实现位于 ``legaldoc_rag`` 包。"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from legaldoc_rag.models import Chunk


def hash_filename(filename: str) -> str:
    """返回稳定的文件名摘要，仅供显示；索引缓存实际使用内容摘要。"""

    path = Path(filename)
    digest = hashlib.sha256(path.stem.encode("utf-8")).hexdigest()
    return f"{digest}{path.suffix}"


def prefix_documents_for_e5(documents: Iterable[object]) -> list[object]:
    """为兼容旧调用添加 E5 文档前缀，并避免重复添加。"""

    result = list(documents)
    for document in result:
        text = str(document.page_content).strip()
        document.page_content = text if text.startswith("passage: ") else f"passage: {text}"
    return result


def prefix_query_for_e5(query: str) -> str:
    """E5 非对称检索要求查询使用 ``query:`` 前缀。"""

    text = query.strip()
    return text if text.startswith("query: ") else f"query: {text}"


def extract_metadata(document: object) -> str:
    meta = getattr(document, "metadata", {}) or {}
    return f"[来源: {meta.get('source', '未知')} | 章节: {meta.get('section', '正文')}]"


def ensure_directory(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def format_response(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines() if line.strip())


def chunks_to_text(chunks: Iterable[Chunk]) -> str:
    """把证据片段格式化为可审阅文本，便于调试或日志记录。"""

    return "\n\n".join(f"[{item.source_name} / {item.section}] {item.text}" for item in chunks)
