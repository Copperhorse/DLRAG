"""Ingestion orchestrator: Docling PDF->Markdown -> chunk -> link -> index -> persist."""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from dlrag_core.config import IngestionConfig
from dlrag_core.models import ChunkRecord
from dlrag_index.hybrid import HybridSearchEngine
from .repo_linker import RepoLinker
from .markdown_chunker import MarkdownChunker, ChunkConfig
from .chunk_evaluator import ChunkEvaluator
from .docling_convert import convert_to_markdown

logger = logging.getLogger(__name__)


@dataclass
class IngestionArtifact:
    """Artifacts returned by the ingestion pipeline."""
    engine: HybridSearchEngine
    output_dir: Path


def _restitch_chain(raw_chunks: list[dict]) -> list[dict]:
    """Re-link prev/next chunk_id pointers among the surviving chunks.

    MarkdownChunker assigns prev_chunk_id/next_chunk_id before any quality
    filtering runs, so a chunk ChunkEvaluator drops would otherwise leave
    dangling references in its former neighbors. Order is preserved by
    ChunkEvaluator, so re-linking by list position is correct.
    """
    for i, c in enumerate(raw_chunks):
        c["prev_chunk_id"] = raw_chunks[i - 1]["chunk_id"] if i > 0 else None
        c["next_chunk_id"] = raw_chunks[i + 1]["chunk_id"] if i < len(raw_chunks) - 1 else None
    return raw_chunks


class IngestionPipeline:
    def __init__(self, cfg: IngestionConfig):
        self.cfg = cfg

    # ---------------------------------------------------------------- run
    def run(self) -> IngestionArtifact:
        out_dir = Path(self.cfg.output_dir)
        if self.cfg.pdf.markdown_path is not None:
            logger.info("[Pipeline] Skipping Docling; reading %s", self.cfg.pdf.markdown_path)
            markdown_text = self.cfg.pdf.markdown_path.read_text(encoding="utf-8")
            device_name = "none (from markdown)"
        else:
            markdown_text, device_name = convert_to_markdown(
                self.cfg.pdf.path, self.cfg.pdf.docling_device, self.cfg.pdf.docling_threads,
                self.cfg.pdf.code_enrichment, self.cfg.pdf.batch_pages, out_dir / ".docling_batches",
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "markdown.md").write_text(markdown_text, encoding="utf-8")
            shutil.rmtree(out_dir / ".docling_batches", ignore_errors=True)  # only needed to resume an interrupted run
            logger.info("[Pipeline] Saved Docling Markdown to %s (rerun with --from-markdown to skip conversion)",
                        out_dir / "markdown.md")

        logger.info("[Pipeline] Chunking Markdown with structure preservation...")
        chunker = MarkdownChunker(
            ChunkConfig(max_chunk_tokens=500, min_chunk_tokens=50, merge_short_chunks=True)
        )
        raw_chunks = chunker.process(markdown_text)

        logger.info("[Pipeline] Evaluating %d chunks for quality...", len(raw_chunks))
        evaluator = ChunkEvaluator()
        eval_stats = evaluator.evaluate_chunks(raw_chunks)
        n_before = eval_stats["total_chunks"]
        raw_chunks = [
            c for c in eval_stats["evaluated_chunks"] if c["evaluation"]["should_use"]
        ]
        n_dropped = n_before - len(raw_chunks)
        if n_dropped:
            logger.info(
                "[Pipeline] Dropped %d/%d low-quality chunks: %s",
                n_dropped, n_before, eval_stats["rejection_reasons"],
            )
        raw_chunks = _restitch_chain(raw_chunks)

        chunks: list[ChunkRecord] = []
        for c in raw_chunks:
            chunks.append(
                ChunkRecord(
                    chunk_id=c["chunk_id"],
                    chunk_index=c["chunk_index"],
                    content=c["content"],
                    section_header=c["metadata"]["section_header"],
                    parent_section=c["metadata"]["parent_section"],
                    top_header=c["metadata"]["top_header"],
                    content_type=c["metadata"]["content_type"],
                    estimated_tokens=c["estimated_tokens"],
                    prev_chunk_id=c.get("prev_chunk_id"),
                    next_chunk_id=c.get("next_chunk_id"),
                )
            )

        if self.cfg.repo.local_dir:
            logger.info(
                "[Pipeline] Linking repo assets from '%s'...", self.cfg.repo.local_dir
            )
            chunks = RepoLinker(self.cfg.repo, self.cfg.structure).link(chunks)

        logger.info(
            "[Pipeline] Initializing Hybrid Search Engine with %d chunks...", len(chunks)
        )
        engine = HybridSearchEngine(chunks, self.cfg.index)

        self._persist(chunks, engine, device_name, n_before, n_dropped, eval_stats["rejection_reasons"])
        return IngestionArtifact(engine=engine, output_dir=Path(self.cfg.output_dir))

    # ---------------------------------------------------------------- persist
    def _persist(
        self,
        chunks: list[ChunkRecord],
        engine: HybridSearchEngine,
        device_name: str,
        n_chunks_before_filter: int,
        n_chunks_dropped: int,
        rejection_reasons: dict,
    ) -> None:
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        engine.save(out_dir)  # chunks.jsonl + embeddings.npy + index.json

        n_linked = sum(1 for c in chunks if c.github_links)
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({
                "dense_model": self.cfg.index.dense_model,
                "num_chunks": len(chunks),
                "num_chunks_linked": n_linked,
                "num_chunks_before_filter": n_chunks_before_filter,
                "num_chunks_dropped": n_chunks_dropped,
                "chunk_rejection_reasons": rejection_reasons,
                "pdf_path": str(self.cfg.pdf.path),
                "repo": str(self.cfg.repo.local_dir) if self.cfg.repo.local_dir else None,
                "docling_device": device_name,
            }, f, indent=2)

        logger.info("[Pipeline] Ingestion complete. Output in %s", out_dir)
