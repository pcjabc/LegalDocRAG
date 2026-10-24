"""LegalDoc RAG：轻量、可解释、可评测的合同检索包。"""

from .config import Settings
from .service import LegalDocService

__all__ = ["LegalDocService", "Settings"]
