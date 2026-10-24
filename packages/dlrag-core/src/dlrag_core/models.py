"""Data models for chunks, sections, and linked code assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CodeAsset:
    file: str
    url: str | None
    code: str
    kind: str  # e.g., 'semantic', 'exact'

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file, "url": self.url, "code": self.code, "kind": self.kind}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CodeAsset:
        return cls(**d)


@dataclass
class ChunkRecord:
    chunk_id: str
    chunk_index: int
    content: str
    section_header: str
    parent_section: str | None
    top_header: str
    content_type: str
    estimated_tokens: int
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    github_links: list[str] = field(default_factory=list)
    repo_code: list[CodeAsset] = field(default_factory=list)

    @property
    def embedding_text(self) -> str:
        """Enriches the text with header breadcrumbs for dense & BM25 indexing."""
        breadcrumb = f"{self.top_header} > {self.section_header}"
        return f"{breadcrumb}\n\n{self.content}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "section_header": self.section_header,
            "parent_section": self.parent_section,
            "top_header": self.top_header,
            "content_type": self.content_type,
            "estimated_tokens": self.estimated_tokens,
            "prev_chunk_id": self.prev_chunk_id,
            "next_chunk_id": self.next_chunk_id,
            "github_links": self.github_links,
            "repo_code": [c.to_dict() for c in self.repo_code],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChunkRecord:
        d = dict(d)
        d["repo_code"] = [CodeAsset.from_dict(c) for c in d.get("repo_code", [])]
        return cls(**d)
