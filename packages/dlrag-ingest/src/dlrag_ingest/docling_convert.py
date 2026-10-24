"""Docling converter construction: device resolution + CUDA->CPU fallback.

Extracted from the retired parsers/docling_parser.py -- that file also
contained an OutlineEntry/hierarchy tree-walk (`_outline_from_document`)
that built SectionRecord-shaped output via doc.iterate_items(); that part
is gone along with SectionRecord/ParseResult. This module keeps only the
genuinely reusable half: building a correctly-configured DocumentConverter
and retrying on CPU if CUDA conversion fails. pipeline.py uses this instead
of duplicating it inline.

Device strategy
----------------
* device='auto'  -> CUDA if torch reports a GPU, otherwise CPU.
* CUDA conversion is wrapped in a try/except: any failure re-builds the
  Docling pipeline on CPU and retries, so a broken driver never kills
  ingestion.
* device='cuda'|'cpu' pins the choice explicitly.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class ParseError(RuntimeError):
    """Raised when the PDF cannot be converted at all (e.g. Docling missing)."""


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_device(device_pref: str):
    from docling.datamodel.pipeline_options import AcceleratorDevice

    if device_pref in ("cuda", "gpu"):
        return AcceleratorDevice.CUDA
    if device_pref == "cpu":
        return AcceleratorDevice.CPU
    return AcceleratorDevice.CUDA if _torch_cuda_available() else AcceleratorDevice.CPU


def _cpu_device():
    from docling.datamodel.pipeline_options import AcceleratorDevice

    return AcceleratorDevice.CPU


def _preload_onnxruntime_cuda() -> None:
    """RapidOCR (Docling's OCR) runs on onnxruntime. With onnxruntime-gpu its CUDA provider only works once
    the CUDA/cuDNN libraries (shipped as pip `nvidia-*` packages) are loaded; otherwise every OCR call fails
    with "dlopen failed for libcudnn.so" - and Docling logs it and moves on."""
    try:
        import onnxruntime

        onnxruntime.preload_dlls()
    except Exception as exc:  # CPU-only onnxruntime, or no CUDA libs: OCR stays on the CPU
        logger.warning("[Docling] Could not preload onnxruntime CUDA libs (%s); OCR will use the CPU.", exc)


def _build_converter(device, threads: int, code_enrichment: bool = True):
    if str(device).lower().endswith("cuda"):
        _preload_onnxruntime_cuda()
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    except ImportError as exc:
        raise ParseError("Docling is required for PDF conversion. Install it with: pip install docling") from exc

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(device=device, num_threads=threads)
    # Without this the PDF's code listings come out flattened onto one line (with listing line numbers).
    pipeline_options.do_code_enrichment = code_enrichment
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_cls=StandardPdfPipeline, pipeline_options=pipeline_options)
        }
    )


def _page_count(pdf_path: Path) -> int:
    import pypdfium2 as pdfium  # installed with docling

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def convert_to_markdown(
    pdf_path: Path,
    device_pref: str,
    threads: int,
    code_enrichment: bool = True,
    batch_pages: int = 50,
    cache_dir: Path | None = None,
) -> tuple[str, str]:
    """Convert a PDF to Markdown in page batches, retrying on CPU if CUDA conversion fails.

    Batches give progress/ETA logging and, with ``cache_dir``, resumability: every finished batch is written
    to ``cache_dir`` and reused when the same conversion is started again (a manifest of the settings guards
    against reusing batches from a different configuration). Heading levels are inferred by Docling mostly from
    the PDF outline (page-matched), so they do not depend on the batch boundaries.

    Returns (markdown_text, device_name_actually_used).
    """
    device = _resolve_device(device_pref)
    n_pages = _page_count(pdf_path)
    step = batch_pages if batch_pages and batch_pages > 0 else n_pages
    manifest = {
        "pdf": Path(pdf_path).name,
        "size": Path(pdf_path).stat().st_size,
        "pages": n_pages,
        "batch_pages": step,
        "code_enrichment": code_enrichment,
    }

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            for old in cache_dir.glob("pages_*.md"):
                old.unlink()
            manifest_path.write_text(json.dumps(manifest))

    converter = None
    parts: list[str] = []
    t_start = time.time()
    converted_pages = 0
    for start in range(1, n_pages + 1, step):
        end = min(start + step - 1, n_pages)
        batch_file = cache_dir / f"pages_{start:05d}-{end:05d}.md" if cache_dir is not None else None
        if batch_file is not None and batch_file.exists():
            parts.append(batch_file.read_text(encoding="utf-8"))
            logger.info("[Docling] pages %d-%d/%d: reused from %s", start, end, n_pages, cache_dir)
            continue
        t0 = time.time()
        while True:
            try:
                if converter is None:
                    converter = _build_converter(device, threads, code_enrichment)
                markdown = converter.convert(str(pdf_path), page_range=(start, end)).document.export_to_markdown()
                break
            except Exception as exc:
                if not str(device).lower().endswith("cuda"):
                    raise
                logger.warning("[Docling] CUDA conversion failed (%s); retrying on CPU.", exc)
                device, converter = _cpu_device(), None
        if batch_file is not None:
            batch_file.write_text(markdown, encoding="utf-8")
        parts.append(markdown)
        converted_pages += end - start + 1
        rate = (time.time() - t_start) / converted_pages
        logger.info(
            "[Docling] pages %d-%d/%d done in %.0fs (avg %.1fs/page, ETA %.0f min)",
            start,
            end,
            n_pages,
            time.time() - t0,
            rate,
            rate * (n_pages - end) / 60,
        )
        gc.collect()

    device_name = str(device).split(".")[-1].lower()
    logger.info("[Docling] Converted %d pages to Markdown on %s.", n_pages, device_name.upper())
    return "\n\n".join(parts), device_name
