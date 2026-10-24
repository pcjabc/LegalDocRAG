"""旧版公开 API 的兼容层。

新代码请直接使用 :class:`legaldoc_rag.service.LegalDocService`。保留这个模块是为了
让基于上游项目编写的小脚本不会突然失效，同时明确告诉维护者迁移方向。
"""

from pathlib import Path

from legaldoc_rag.config import Settings
from legaldoc_rag.service import LegalDocService


def process_document(file_path: str) -> LegalDocService:
    """处理单份文档并返回可调用 ``ask`` 的服务实例。"""

    root = Path(__file__).resolve().parent
    service = LegalDocService(Settings.from_env(root))
    service.build_index([Path(file_path)])
    return service
