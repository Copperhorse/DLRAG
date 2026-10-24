"""Agent-facing facade over a persisted (or in-memory) index."""

from __future__ import annotations

import json
from pathlib import Path

from .hybrid import HybridSearchEngine
from .repo_tools import REPO_TOOL_SCHEMAS, RepoBrowser

TOOL_SCHEMAS = [
    {
        "name": "book_search",
        "description": "Hybrid (BM25 + Dense) search over the book's chunks. "
        "Returns headers, scores, repo links and code assets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "book_get_chunk",
        "description": "Fetch one full chunk by id, optionally with its linked code.",
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string"},
                "with_code": {"type": "boolean", "default": True},
            },
            "required": ["chunk_id"],
        },
    },
]


class AgenticIndex:
    """Tool surface for the coding agent. Usable in-process or from disk."""

    def __init__(
        self, engine: HybridSearchEngine | None = None, artifacts_dir: Path | None = None, repo_dir: Path | None = None
    ):
        if engine is None:
            if artifacts_dir is None:
                raise ValueError("Provide either an engine or an artifacts_dir.")
            engine = HybridSearchEngine.load(Path(artifacts_dir))
        self.engine = engine
        self.chunks = {c.chunk_id: c.to_dict() for c in engine.chunks}
        # Repo tools are only advertised when a repository is provided.
        self.repo = RepoBrowser(repo_dir) if repo_dir is not None else None
        self.tools = TOOL_SCHEMAS + (REPO_TOOL_SCHEMAS if self.repo else [])

    # ---------------------------------------------------------------- tools
    def book_search(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve the most relevant chunks (trimmed, agent-friendly)."""
        hits = self.engine.search(query, top_k=top_k)
        return [
            {
                "chunk_id": h.chunk_id,
                "section_header": h.section_header,
                "top_header": h.top_header,
                "rrf_score": round(h.rrf_score, 6),
                "github_links": h.github_links,
                "code_files": [c.file for c in h.repo_code],
                "text_preview": " ".join(h.content.split())[:400],
            }
            for h in hits
        ]

    def book_get_chunk(self, chunk_id: str, with_code: bool = True) -> dict | None:
        chk = self.chunks.get(chunk_id)
        if chk is None:
            return None
        if not with_code:
            chk = {**chk, "repo_code": []}
        return chk

    def repo_list(self, path: str = ".") -> dict:
        return self._repo().list(path)

    def repo_read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> dict:
        return self._repo().read_file(path, start_line, end_line)

    def repo_grep(self, pattern: str, path: str = ".", ignore_case: bool = True) -> dict:
        return self._repo().grep(pattern, path, ignore_case)

    def _repo(self) -> RepoBrowser:
        if self.repo is None:
            raise RuntimeError("No repo_dir configured for this AgenticIndex.")
        return self.repo
