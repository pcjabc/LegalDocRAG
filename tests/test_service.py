from pathlib import Path

from legaldoc_rag.config import Settings
from legaldoc_rag.service import LegalDocService


def make_settings(tmp_path: Path) -> Settings:
    return Settings(root=tmp_path, index_dir=tmp_path / "indexes", top_k=3)


def test_build_ask_and_cache(tmp_path: Path) -> None:
    contract = tmp_path / "保密协议.md"
    contract.write_text(
        "# 保密协议\n\n## 第一条 保密期限\n\n协议终止后，保密义务继续有效三年。",
        encoding="utf-8",
    )
    service = LegalDocService(make_settings(tmp_path))
    first = service.build_index([contract])
    answer = service.ask("协议终止后保密义务持续多久？")
    second = service.build_index([contract])
    assert first.loaded_from_cache is False
    assert second.loaded_from_cache is True
    assert answer.abstained is False
    assert "三年" in answer.text
    assert answer.sources[0].chunk.source_name == "保密协议.md"


def test_unrelated_question_is_refused(tmp_path: Path) -> None:
    contract = tmp_path / "采购合同.md"
    contract.write_text("# 采购合同\n\n## 交付\n\n设备应在上海交付。", encoding="utf-8")
    service = LegalDocService(make_settings(tmp_path))
    service.build_index([contract])
    answer = service.ask("员工每年有多少天带薪年假？")
    assert answer.abstained is True
    assert "足够证据" in answer.text
