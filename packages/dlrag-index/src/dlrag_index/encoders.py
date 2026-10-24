"""Dense encoder backed by fastembed (ONNX runtime, no torch)."""

from __future__ import annotations

import logging

from fastembed import TextEmbedding

logger = logging.getLogger(__name__)


class DenseEncoder:
    def __init__(self, model_id: str, max_seq_len: int = 512, batch_size: int = 16):
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.model = TextEmbedding(model_name=model_id, max_length=max_seq_len)
        logger.info("[Encoder] Dense '%s' loaded (fastembed).", model_id)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """L2-normalized dense embeddings -> list of dense vectors."""
        return [v.tolist() for v in self.model.embed(texts, batch_size=self.batch_size)]
