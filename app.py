"""LegalDoc RAG 的 Streamlit 入口。

界面层只负责收集输入和展示结果，检索、索引和回答逻辑都放在
``legaldoc_rag`` 包中。这样既方便测试，也避免业务逻辑被 UI 状态绑死。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from legaldoc_rag.config import Settings
from legaldoc_rag.service import LegalDocService

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "data" / "uploads"
SAMPLE_DIR = ROOT / "data" / "samples"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="LegalDoc RAG", page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner=False)
def get_service() -> LegalDocService:
    """缓存模型/索引服务，避免 Streamlit 每次交互都重新初始化。"""

    return LegalDocService(Settings.from_env(ROOT))


def save_upload(name: str, content: bytes) -> Path:
    """以内容摘要隔离同名不同内容，避免旧索引被误复用。"""

    safe_name = Path(name).name
    digest = hashlib.sha256(content).hexdigest()[:12]
    path = UPLOAD_DIR / f"{digest}-{safe_name}"
    path.write_bytes(content)
    return path


service = get_service()
st.title("⚖️ LegalDoc RAG — 合同条款检索助手")
st.caption("上传合同或加载示例文档，通过混合检索定位条款，并让每条回答都能回到原文证据。")

with st.sidebar:
    st.subheader("运行配置")
    st.write(f"向量后端：`{service.embedding_backend_name}`")
    st.write(f"回答后端：`{service.answer_backend_name}`")
    top_k = st.slider("返回证据数", 1, 8, service.settings.top_k)
    st.info("默认模式完全离线。设置 GROQ_API_KEY 后可启用大模型归纳，但引用仍来自本地检索。")

uploaded = st.file_uploader(
    "上传 PDF、TXT 或 Markdown 合同（可多选）",
    type=["pdf", "txt", "md", "markdown"],
    accept_multiple_files=True,
)
left, right = st.columns(2)
build_uploads = left.button("处理上传文档", type="primary", use_container_width=True)
build_samples = right.button("加载内置示例合同", use_container_width=True)

if build_uploads:
    if not uploaded:
        st.warning("请先选择至少一个文档。")
    else:
        paths = [save_upload(item.name, item.getvalue()) for item in uploaded]
        with st.spinner("正在解析、分块并构建索引……"):
            report = service.build_index(paths)
        st.session_state["indexed"] = True
        st.success(
            f"已处理 {report.document_count} 份文档，共生成 {report.chunk_count} 个证据片段；"
            f"索引指纹：{report.fingerprint[:12]}。"
        )

if build_samples:
    with st.spinner("正在构建示例合同索引……"):
        report = service.build_index(sorted(SAMPLE_DIR.glob("*.md")))
    st.session_state["indexed"] = True
    st.success(f"示例合同已就绪：{report.document_count} 份文档，{report.chunk_count} 个证据片段。")

st.divider()
question = st.text_input(
    "请输入合同问题",
    placeholder="例如：保密义务在合同终止后持续多久？",
    disabled=not service.has_index,
)

if question and service.has_index:
    with st.spinner("正在检索并核对证据……"):
        answer = service.ask(question, top_k=top_k)

    if answer.abstained:
        st.warning(answer.text)
    else:
        st.markdown("#### 回答")
        st.write(answer.text)

    st.markdown("#### 证据来源")
    for hit in answer.sources:
        title = f"[{hit.citation_id}] {hit.chunk.source_name} · {hit.chunk.section} · 融合分数 {hit.score:.4f}"
        with st.expander(title, expanded=True):
            st.write(hit.chunk.text)
            st.caption(
                f"BM25 排名：{hit.bm25_rank or '-'} ｜ 稠密检索排名：{hit.dense_rank or '-'} ｜ "
                f"页码：{hit.chunk.page or '不适用'}"
            )
elif not service.has_index:
    st.info("请先处理上传文档，或加载内置示例合同。")
