# LegalDoc RAG — 合同条款检索助手

一个适合学习、演示和面试讲解的轻量中文 RAG 项目：上传 PDF/TXT/Markdown 合同，系统按章节切分文档，使用 **BM25 + 稠密向量 + RRF** 检索证据，并返回文件名、章节、页码和原文片段。

本项目基于 MIT 项目 [mewerton/Rag_Juridico](https://github.com/mewerton/Rag_Juridico) 二次开发。当前版本重写了中文界面、索引缓存、检索、引用、拒答、离线运行和评测流程；原始许可证保留在 `LICENSE`。

> 这是技术演示，不提供法律意见。仓库内合同均为虚构示例，不含真实个人或企业数据。

## 先看运行效果

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
python evaluate.py
python -m streamlit run app.py
```

浏览器打开 Streamlit 给出的地址后，点击“加载内置示例合同”，即可询问：

- 保密义务在协议终止后持续多久？
- 逾期交付违约金如何计算，上限是多少？
- 一级故障多久响应，多久提供临时恢复方案？
- 员工每年有多少天带薪年假？（应拒答）

首次运行无需 API Key，也不下载模型。

## 这条 RAG 链路做了什么

1. **读取文档**：支持可提取文本的 PDF、UTF-8 TXT 与 Markdown。扫描 PDF 会明确提示先做 OCR。
2. **结构化分块**：优先识别“第几条”和 Markdown 标题；超长段落才使用带重叠窗口。
3. **可复用索引**：以文档真实内容、分块参数和向量后端共同生成 SHA-256 指纹。同名文件内容变化后会重建，不复用旧索引。
4. **混合召回**：BM25 擅长合同编号、金额、日期和专有名词；稠密向量补充表达相似性。
5. **RRF 融合**：使用排名而不是直接相加两类不可比的原始分数。
6. **证据门槛**：词项和向量证据均不足时拒答，而不是强行生成。
7. **带引用回答**：回答中的 `[1]`、`[2]` 对应界面下方的原文证据。

默认 `hash-char-ngram` 是一个确定性的离线稠密向量兜底，目的是让 CI 和首次演示稳定运行；它不是语义模型。实际中文语义检索可切换 E5：

```powershell
python -m pip install -r requirements-e5.txt
Copy-Item .env.example .env
# 修改 .env：EMBEDDING_BACKEND=e5
python -m streamlit run app.py
```

E5 后端对问题使用 `query:`、对文档使用 `passage:`。这是非对称检索的必要细节，也是对上游代码的一个关键修复。

## 可选：启用生成式回答

默认“离线证据摘录”会从命中条款中选出与问题最相关的句子，因此没有模型密钥也能完成整个流程。若需要自然语言归纳：

1. 复制 `.env.example` 为 `.env`；
2. 填写 `GROQ_API_KEY`；
3. 重启 Streamlit。

模型只能看到已检索的证据，提示词要求逐条引用。供应商请求失败时，系统自动退回离线证据摘录，检索仍可使用。不要提交 `.env`，程序也不会打印密钥。

## 评测

```powershell
python evaluate.py --top-k 3
```

固定问题集位于 `data/evaluation/questions.jsonl`，覆盖金额、日期、条款定位、跨句问题和无答案拒答。脚本输出：

- `Hit@K`：前 K 条中是否出现标注证据；
- `MRR`：正确证据出现得越靠前越好；
- 拒答准确率：无答案问题是否被拒绝；
- 平均查询耗时：仅统计查询，不包含首次模型下载和建库。

这些结果只代表仓库内的小型虚构数据集，不能外推为生产效果，也不能直接写成业务提升百分比。

## 目录说明

```text
app.py                          Streamlit 界面
legaldoc_rag/
  config.py                     环境变量与默认参数
  documents.py                  文档读取、章节识别、分块和内容指纹
  embeddings.py                 离线哈希向量与可选 E5 后端
  retrieval.py                  BM25、稠密检索、RRF 与索引持久化
  service.py                    建库、检索、拒答、摘录和 Groq 回退
data/samples/                   三份虚构合同
data/evaluation/questions.jsonl 固定评测集
evaluate.py                     命令行评测入口
tests/                          不依赖网络和模型下载的单元测试
```

## 设计边界

- 当前 JSON 索引为“小文档、可解释”设计。文档达到数万片段后，应把向量存储替换为 FAISS/Qdrant，把 BM25 替换为 Elasticsearch/OpenSearch。
- PDF 只做文本提取；扫描合同需要单独 OCR。没有把 OCR 隐式塞进依赖，是为了让基础安装保持轻量。
- 拒答阈值是工程启发式，需要结合自己的正例、近似问题和无关问题重新标定。
- 引用证明“回答使用了哪些检索片段”，不等同于自动完成法律事实核查。

## 面试时可以怎么介绍

> 我做的是一个轻量的中文合同 RAG。系统先按合同章节和自然段切分文档，并把来源、章节和页码保留在每个 Chunk 中。检索层并行使用 BM25 和稠密向量：前者补强金额、日期和条款编号，后者处理近义表达，再用 RRF 融合两路排名。为了避免旧索引和无依据回答，我分别使用内容哈希管理索引版本，并在证据不足时拒答。最后用固定问题集评测 Hit@K、MRR、拒答准确率和耗时，所有指标都来自本项目实际运行，而不是引用上游数据。

建议诚实表述为“工程组合与评测改造”，不要称为“提出了新的检索算法”。
