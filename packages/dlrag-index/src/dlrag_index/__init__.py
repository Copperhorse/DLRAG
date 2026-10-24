"""Hybrid (BM25 + dense, RRF-fused) retrieval and the agent-facing tool facade."""

from .agent import AgenticIndex
from .encoders import DenseEncoder
from .hybrid import HybridSearchEngine, SearchHit
from .repo_tools import RepoBrowser

__all__ = ["AgenticIndex", "DenseEncoder", "HybridSearchEngine", "SearchHit", "RepoBrowser"]
