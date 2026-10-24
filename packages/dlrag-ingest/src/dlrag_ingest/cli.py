"""CLI entry point.

Examples
--------
# Docling conversion (GPU if available, CPU fallback is automatic):
dlrag-ingest --pdf "Hands-On ML.pdf" \
    --repo ./handson-ml --remote-url https://github.com/ageron/handson-ml3/blob/main \
    --out data/rag_artifacts --query "how do I build a custom training loop"

# Force CPU:
dlrag-ingest --pdf book.pdf --device cpu
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dlrag_core.config import IngestionConfig, PDFConfig, RepoConfig
from dlrag_index.agent import AgenticIndex

from .pipeline import IngestionPipeline


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Book ingestion for agentic RAG.")
    ap.add_argument("--pdf", type=Path, help="source PDF (not needed with --from-markdown)")
    ap.add_argument("--repo", type=Path, default=None)
    ap.add_argument(
        "--remote-url", default=None, help="Base URL for deep links (e.g. .../github.com/org/repo/blob/main)"
    )
    ap.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Docling accelerator (auto = GPU if available, else CPU)",
    )
    ap.add_argument("--threads", type=int, default=8, help="Docling CPU thread count")
    ap.add_argument(
        "--code-enrichment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Docling code model: keeps code newlines/indentation (default on; slower)",
    )
    ap.add_argument(
        "--batch-pages",
        type=int,
        default=50,
        help="convert N pages at a time: progress + resume after an interruption (0 = all at once)",
    )
    ap.add_argument(
        "--from-markdown",
        type=Path,
        default=None,
        help="skip Docling and chunk this Markdown (e.g. <out>/markdown.md from an earlier run)",
    )
    ap.add_argument("--out", type=Path, default=Path("data/rag_artifacts"))
    ap.add_argument("--query", default=None, help="Optional demo query after ingestion")
    args = ap.parse_args()
    if args.pdf is None and args.from_markdown is None:
        ap.error("--pdf is required unless --from-markdown is given")

    cfg = IngestionConfig(
        pdf=PDFConfig(
            path=args.pdf or Path(""),
            docling_device=args.device,
            docling_threads=args.threads,
            code_enrichment=args.code_enrichment,
            batch_pages=args.batch_pages,
            markdown_path=args.from_markdown,
        ),
        repo=RepoConfig(local_dir=args.repo, remote_url=args.remote_url),
        output_dir=args.out,
    )
    artifact = IngestionPipeline(cfg).run()

    index = AgenticIndex(engine=artifact.engine)
    if args.query:
        print("\n=== book_search ===")
        for hit in index.book_search(args.query):
            print(f"[{hit['chunk_id']}] {hit['section_header']} ({hit['top_header']}) rrf={hit['rrf_score']}")
            for link in hit["github_links"]:
                print(f"    -> {link}")
    print(f"\nArtifacts: {args.out.resolve()}")


if __name__ == "__main__":
    main()
