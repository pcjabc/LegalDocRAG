from legaldoc_rag.embeddings import HashEmbedding
from legaldoc_rag.models import Chunk
from legaldoc_rag.retrieval import HybridIndex, tokenize


def make_chunk(identifier: str, section: str, text: str) -> Chunk:
    return Chunk(identifier, "合同.md", section, text)


def test_tokenizer_keeps_numbers_and_chinese_terms() -> None:
    tokens = tokenize("违约金为 0.05%，上限 10%")
    assert "违约" in tokens
    assert "0.05%" in tokens
    assert "10%" in tokens


def test_hybrid_retrieval_returns_relevant_clause_first() -> None:
    chunks = [
        make_chunk("1", "交付", "乙方应在十一月三十日前完成设备交付。"),
        make_chunk("2", "保密期限", "协议终止后，保密义务继续有效三年。"),
        make_chunk("3", "付款", "甲方在验收后支付剩余款项。"),
    ]
    index = HybridIndex.build(chunks, HashEmbedding())
    hits = index.search("保密义务在协议终止后持续多久？", top_k=2)
    assert hits[0].chunk.section == "保密期限"
    assert hits[0].citation_id == 1
    assert hits[0].bm25_score > 0


def test_index_round_trip(tmp_path) -> None:
    embedding = HashEmbedding()
    index = HybridIndex.build([make_chunk("1", "付款", "合同价款为一百元。")], embedding)
    path = tmp_path / "index.json"
    index.save(path, "fingerprint")
    loaded = HybridIndex.load(path, embedding, "fingerprint")
    assert loaded is not None
    assert loaded.chunks[0].section == "付款"
    assert HybridIndex.load(path, embedding, "different") is None
