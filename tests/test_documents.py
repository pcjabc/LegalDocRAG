from pathlib import Path

import pytest

from legaldoc_rag.documents import chunk_document, content_fingerprint


def test_markdown_chunk_preserves_source_and_section(tmp_path: Path) -> None:
    contract = tmp_path / "合同.md"
    contract.write_text("# 示例合同\n\n## 第一条 付款\n\n甲方应支付人民币 100 元。", encoding="utf-8")
    chunks = chunk_document(contract, chunk_size=200, overlap=20)
    assert len(chunks) == 1
    assert chunks[0].source_name == "合同.md"
    assert chunks[0].section == "第一条 付款"
    assert "100 元" in chunks[0].text


def test_content_fingerprint_changes_when_content_changes(tmp_path: Path) -> None:
    contract = tmp_path / "同名合同.md"
    contract.write_text("版本一", encoding="utf-8")
    first = content_fingerprint([contract], "settings")
    contract.write_text("版本二", encoding="utf-8")
    second = content_fingerprint([contract], "settings")
    assert first != second


def test_unsupported_type_has_actionable_error(tmp_path: Path) -> None:
    unsupported = tmp_path / "合同.docx"
    unsupported.write_text("text", encoding="utf-8")
    with pytest.raises(ValueError, match="不支持的文件类型"):
        chunk_document(unsupported)
