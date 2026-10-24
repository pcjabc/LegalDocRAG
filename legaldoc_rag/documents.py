"""文档读取与结构化分块。"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .models import Chunk

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}
HEADING_RE = re.compile(r"^(?:#{1,6}\s+|第[一二三四五六七八九十百0-9]+[章节条]|[0-9]+(?:\.[0-9]+)*[、.\s])(.+)$")


def read_document(path: Path) -> list[tuple[int | None, str]]:
    """读取文档，返回 ``(页码, 文本)``；文本文件没有页码。"""

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的文件类型：{suffix or '无扩展名'}")
    if suffix != ".pdf":
        return [(None, path.read_text(encoding="utf-8-sig"))]

    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 由安装环境决定
        raise RuntimeError("读取 PDF 需要安装 pypdf：pip install pypdf") from exc

    reader = PdfReader(str(path))
    pages = [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, 1)]
    if not any(text.strip() for _, text in pages):
        raise ValueError(f"PDF 未提取到文字，可能是扫描件，需要先进行 OCR：{path.name}")
    return pages


def _window(text: str, size: int, overlap: int) -> list[str]:
    """对超长段落做带重叠窗口切分，并优先在标点处收尾。"""

    cleaned = re.sub(r"[ \t]+", " ", text).strip()
    if len(cleaned) <= size:
        return [cleaned] if cleaned else []
    result: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + size, len(cleaned))
        if end < len(cleaned):
            boundary = max(cleaned.rfind(mark, start + size // 2, end) for mark in "。；！？\n")
            if boundary > start:
                end = boundary + 1
        result.append(cleaned[start:end].strip())
        if end >= len(cleaned):
            break
        start = max(start + 1, end - overlap)
    return result


def _flush_buffer(
    chunks: list[Chunk],
    path: Path,
    page: int | None,
    section: str,
    buffer: list[str],
    chunk_size: int,
    overlap: int,
) -> None:
    """把当前段落缓冲区写入结果；独立函数可避免循环闭包捕获旧状态。"""

    if not buffer:
        return
    body = "\n".join(buffer)
    for piece in _window(body, chunk_size, overlap):
        seed = f"{path.name}|{page}|{section}|{piece}".encode()
        chunks.append(
            Chunk(
                chunk_id=hashlib.sha256(seed).hexdigest()[:16],
                source_name=path.name,
                section=section,
                text=piece,
                page=page,
            )
        )
    buffer.clear()


def chunk_document(path: Path, chunk_size: int = 700, overlap: int = 100) -> list[Chunk]:
    """按页、标题和自然段分块，并将来源/章节固化在每个片段中。"""

    chunks: list[Chunk] = []
    section = "正文"
    for page, raw_text in read_document(path):
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n|(?<=。)\s*\n", raw_text) if part.strip()]
        buffer: list[str] = []

        for paragraph in paragraphs:
            heading = HEADING_RE.match(paragraph)
            if heading and len(paragraph) <= 100:
                _flush_buffer(chunks, path, page, section, buffer, chunk_size, overlap)
                section = paragraph.lstrip("# ").strip()
                continue
            if buffer and sum(map(len, buffer)) + len(paragraph) > chunk_size:
                _flush_buffer(chunks, path, page, section, buffer, chunk_size, overlap)
            buffer.append(paragraph)
        _flush_buffer(chunks, path, page, section, buffer, chunk_size, overlap)
    return chunks


def content_fingerprint(paths: list[Path], settings_signature: str) -> str:
    """基于真实文件内容生成索引指纹，同名文件更新后一定会重建。"""

    digest = hashlib.sha256(settings_signature.encode("utf-8"))
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
