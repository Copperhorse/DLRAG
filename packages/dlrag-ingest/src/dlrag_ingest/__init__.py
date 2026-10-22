"""Book ingestion: Docling PDF -> Markdown -> chunks -> repo links -> hybrid index."""
from .pipeline import IngestionArtifact, IngestionPipeline

__all__ = ["IngestionArtifact", "IngestionPipeline"]
