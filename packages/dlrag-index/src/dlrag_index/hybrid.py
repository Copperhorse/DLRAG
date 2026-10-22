"""Hybrid retrieval: BM25 (lexical) + Dense (neural dense) fused with RRF."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from dlrag_core.config import IndexConfig
from dlrag_core.models import ChunkRecord, CodeAsset
from dlrag_index.encoders import DenseEncoder

_STOPWORDS = frozenset("""
    a an the this that these those is are was were be been being
    of to for on with in and or as by at from into
    what which who whom how why can does do did doesn't don't didn't
    it its it's we you your our their his her they them i
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class SearchHit:
    chunk_id: str
    section_header: str
    top_header: str
    content: str
    content_type: str
    rrf_score: float
    bm25_score: float
    dense_score: float
    github_links: list[str]
    repo_code: list[CodeAsset]
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "section_header": self.section_header,
            "top_header": self.top_header,
            "content": self.content,
            "content_type": self.content_type,
            "rrf_score": self.rrf_score,
            "bm25_score": self.bm25_score,
            "dense_score": self.dense_score,
            "github_links": self.github_links,
            "repo_code": [c.to_dict() for c in self.repo_code],
            "prev_chunk_id": self.prev_chunk_id,
            "next_chunk_id": self.next_chunk_id,
        }


@dataclass
class RetrievalBundle:
    hits: list[SearchHit]
    bm25_pool: list[tuple[str, float]]      # (chunk_id, score)
    dense_pool: list[tuple[str, float]]


class HybridSearchEngine:
    def __init__(self, chunks: list[ChunkRecord], cfg: IndexConfig | None = None):
        self.cfg = cfg or IndexConfig()
        self.chunks = chunks

        # BM25 tokenized corpus using hierarchical text
        self.tokenized_corpus = [
            _tokenize(c.embedding_text) for c in self.chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self.dense = DenseEncoder(
            self.cfg.dense_model,
            max_seq_len=self.cfg.max_seq_len,
            batch_size=self.cfg.batch_size,
        )

        # Dense passage embedding with context breadcrumbs
        passages = [c.embedding_text for c in self.chunks]
        self.dense_indices = np.array(self.dense.encode(passages), dtype=np.float32)

    def search_with_pools(
        self, query: str, top_k: int = 5, pool_size: int | None = None
    ) -> RetrievalBundle:
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        bm25_ranked = np.argsort(bm25_scores)[::-1]

        query_vec = np.array(self.dense.encode([query])[0], dtype=np.float32)
        dense_scores = np.dot(self.dense_indices, query_vec)
        dense_ranked = np.argsort(dense_scores)[::-1]

        rrf: dict[int, float] = {}
        pool = pool_size or max(self.cfg.candidate_pool, top_k * 2)

        for rank, idx in enumerate(bm25_ranked[:pool]):
            if bm25_scores[idx] > 0:
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.cfg.rrf_k + rank + 1)

        for rank, idx in enumerate(dense_ranked[:pool]):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.cfg.rrf_k + rank + 1)

        top = sorted(rrf, key=rrf.get, reverse=True)[:top_k]
        hits = [
            SearchHit(
                chunk_id=self.chunks[i].chunk_id,
                section_header=self.chunks[i].section_header,
                top_header=self.chunks[i].top_header,
                content=self.chunks[i].content,
                content_type=self.chunks[i].content_type,
                rrf_score=rrf[i],
                bm25_score=float(bm25_scores[i]),
                dense_score=float(dense_scores[i]),
                github_links=self.chunks[i].github_links,
                repo_code=self.chunks[i].repo_code,
                prev_chunk_id=self.chunks[i].prev_chunk_id,
                next_chunk_id=self.chunks[i].next_chunk_id,
            )
            for i in top
        ]

        return RetrievalBundle(
            hits=hits,
            bm25_pool=[(self.chunks[i].chunk_id, float(bm25_scores[i])) for i in bm25_ranked[:pool]],
            dense_pool=[(self.chunks[i].chunk_id, float(dense_scores[i])) for i in dense_ranked[:pool]],
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        return self.search_with_pools(query, top_k=top_k).hits

    def save(self, out_dir: Path) -> None:
        """Write chunks.jsonl, embeddings.npy and index.json (no pickle).

        The BM25 index is rebuilt from the chunks on load, and the dense
        encoder is re-instantiated from ``cfg.dense_model``.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(out_dir / "embeddings.npy", self.dense_indices)
        (out_dir / "index.json").write_text(
            json.dumps({"cfg": asdict(self.cfg), "num_chunks": len(self.chunks)}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, out_dir: Path) -> HybridSearchEngine:
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
        with open(out_dir / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [ChunkRecord.from_dict(json.loads(line)) for line in f if line.strip()]
        embeddings = np.load(out_dir / "embeddings.npy")
        if not (len(chunks) == len(embeddings) == meta["num_chunks"]):
            raise ValueError(
                f"Inconsistent index in {out_dir}: {len(chunks)} chunks, "
                f"{len(embeddings)} embeddings, index.json says {meta['num_chunks']}."
            )
        engine = cls.__new__(cls)
        engine.cfg = IndexConfig(**meta["cfg"])
        engine.chunks = chunks
        engine.tokenized_corpus = [_tokenize(c.embedding_text) for c in chunks]
        engine.bm25 = BM25Okapi(engine.tokenized_corpus)
        engine.dense = DenseEncoder(
            engine.cfg.dense_model,
            max_seq_len=engine.cfg.max_seq_len,
            batch_size=engine.cfg.batch_size,
        )
        engine.dense_indices = embeddings
        return engine
