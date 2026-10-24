"""Central, declarative configuration for the ingestion pipeline.

Everything book-specific lives here as data (regexes, path hints), never as
hard-coded logic, which is what makes the pipeline book-independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class PDFConfig:
    path: Path
    docling_device: str = "auto"  # "auto" | "cuda" | "cpu"
    docling_threads: int = 8
    code_enrichment: bool = True  # Docling CodeFormula model: restores code newlines/indentation (slower, ~1.5 GB VRAM)
    batch_pages: int = 50  # Docling converts this many pages at a time (progress + resume); 0 = all at once
    markdown_path: Path | None = None  # if set, skip Docling and chunk this Markdown (see markdown.md in the artifacts)


@dataclass
class RepoConfig:
    local_dir: Path | None = None
    remote_url: str | None = None  # e.g. https://github.com/org/repo/blob/main
    include_notebooks: bool = True
    include_scripts: bool = True
    semantic_fallback: bool = True  # nearest-code-file linking when nothing matches
    semantic_code_char_cap: int = 2000


@dataclass
class StructureConfig:
    # Tags inside book text that reference notebook cells, e.g.
    #   "Notebook Cell 3.1", "# Cell 3.1", "Cell_3.1", "Cell 3.1"
    cell_tag_pattern: str = r"(?i)(?:notebook\s+cell|cell)[_\s-]?(\d+(?:\.\d+)*)\b"


@dataclass
class IndexConfig:
    dense_model: str = DEFAULT_DENSE_MODEL
    batch_size: int = 16
    max_seq_len: int = 512
    rrf_k: int = 60
    candidate_pool: int = 50  # per-retriever candidate pool before RRF


@dataclass
class IngestionConfig:
    pdf: PDFConfig
    repo: RepoConfig = field(default_factory=lambda: RepoConfig())
    structure: StructureConfig = field(default_factory=StructureConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    output_dir: Path = Path("data/rag_artifacts")

    def __post_init__(self) -> None:
        self.pdf.path = Path(self.pdf.path)
        if self.pdf.markdown_path is not None:
            self.pdf.markdown_path = Path(self.pdf.markdown_path)
        self.output_dir = Path(self.output_dir)
        if self.repo.local_dir is not None:
            self.repo.local_dir = Path(self.repo.local_dir)
