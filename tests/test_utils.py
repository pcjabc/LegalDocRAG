from dataclasses import dataclass, field

from utils import format_response, hash_filename, prefix_documents_for_e5, prefix_query_for_e5


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict = field(default_factory=dict)


def test_e5_prefixes_are_asymmetric_and_idempotent() -> None:
    document = FakeDocument("合同正文")
    prefix_documents_for_e5([document])
    prefix_documents_for_e5([document])
    assert document.page_content == "passage: 合同正文"
    assert prefix_query_for_e5("付款期限") == "query: 付款期限"
    assert prefix_query_for_e5("query: 付款期限") == "query: 付款期限"


def test_hash_filename_keeps_extension() -> None:
    result = hash_filename("document.pdf")
    assert result.endswith(".pdf")
    assert len(result.removesuffix(".pdf")) == 64


def test_format_response_removes_blank_lines() -> None:
    assert format_response("  第一行\n\n第二行  ") == "第一行\n第二行"
