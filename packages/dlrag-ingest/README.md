# dlrag-ingest

Book-agnostic ingestion layer for a **coding agentic RAG** system. Ingest any
technical book (PDF), and expose hybrid search and chunk lookup as **agent
tools**.

## System architecture

```
                 ┌──────────────────────────────────────────────────────────┐
                 │                   IngestionPipeline                      │
                 │  Docling convert -> chunk -> link -> index -> persist    │
                 └──┬───────────────┬──────────────┬──────────────┬─────────┘
                    │               │               │              │
          ┌─────────▼──────┐  ┌─────▼──────────┐ ┌──▼─────────┐ ┌──▼────────────┐
          │ Docling        │  │ MarkdownChunker│ │ Repo Linker│ │ Index Builder │
          │ PDF -> Markdown│  │                │ │ (optional) │ │               │
          │                │  │ header-hierarchy│ 1 cell tags │ │ BM25 (lexical)│
          │ device: auto/  │  │  aware splitting│ 2 writefile │ │ + dense embed │
          │  cuda/cpu, with│  │ content-type    │ 3 semantic  │ │  (fastembed   │
          │  CUDA->CPU     │  │  detection      │  fallback   │ │   ONNX)       │
          │  retry on      │  │ (code/math/table│             │ │ fused with RRF│
          │  failure       │  │  /prose)        │             │ │ save/load     │
          │                │  │ token-budget    │             │ │               │
          │                │  │  merging        │             │ │               │
          └────────────────┘  └─────────────────┘ └────────────┘ └───────────────┘
                    │
          ┌─────────▼──────────────────────────────────┐
          │  AgenticIndex  (tool facade for the agent)  │
          │  book_search | book_get_chunk               │
          └──────────────────────────────────────────────┘
```

## Pipeline stages

1. **Conversion** — every PDF goes through Docling's `DocumentConverter`,
   configured with `PdfPipelineOptions`/`AcceleratorOptions` from
   `PDFConfig.docling_device` (`"auto"` probes `torch.cuda.is_available()`,
   or pin `"cuda"`/`"cpu"`) and `PDFConfig.docling_threads`, and exported to
   Markdown (`export_to_markdown()`). Any CUDA conversion failure is caught
   and retried once on CPU, so a broken driver doesn't kill ingestion.
2. **Chunking** — `MarkdownChunker` (`markdown_chunker.py`) walks the Markdown
   by heading level (`#`..`######`), masks and preserves code/math/table
   blocks as their own `content_type`, splits oversized prose with a
   recursive splitter (`langchain-text-splitters` if installed, otherwise a
   built-in fallback), and merges undersized chunks that share a header and
   content type back together up to `max_chunk_tokens`. Each chunk carries
   its own `section_header`, `parent_section`, and `top_header` breadcrumbs,
   plus `prev_chunk_id`/`next_chunk_id` for sequential context. `chunk_id` is
   a random UUID assigned per ingestion run — it is **not** stable across
   re-ingestion, so anything that needs to reference a chunk long-term
   (benchmarks included) should key off `section_header` text, not
   `chunk_id`.
3. **Repo linking** (when `--repo` is passed) — notebook-cell tags ->
   `%%writefile` scripts -> BM25 semantic nearest-code-file, run per chunk
   (`repo_linker.py`, `RepoLinker`). Same three-step cascade as
   before, retargeted from the retired section-level model onto
   `ChunkRecord.section_header`/`.content`. Chunks are never left unlinked
   when a repo is provided and the semantic fallback is enabled.
4. **Indexing** — `HybridSearchEngine` builds a BM25 index (regex tokenizer,
   stopword-filtered) over `chunk.embedding_text` (`top_header > 
   section_header\n\ncontent`), and a dense embedding index via
   `DenseEncoder` (`dlrag_index`; default model: `sentence-transformers/all-MiniLM-L6-v2`,
   configurable via `IndexConfig.dense_model`). Both retrievers' rankings
   are fused with Reciprocal Rank Fusion (`IndexConfig.rrf_k`,
   `IndexConfig.candidate_pool`).
5. **Persistence** — outputs land in `IngestionConfig.output_dir`
   (`data/rag_artifacts` by default):
   - `embeddings.npy` + `index.json` — dense vectors and the `IndexConfig`;
     `HybridSearchEngine.load()` rebuilds BM25 from the chunks and
     re-instantiates the dense encoder.
   - `chunks.jsonl` — every `ChunkRecord`, one per line.
   - `markdown.md` — the Docling Markdown (re-chunk it with `--from-markdown`, no PDF conversion).
   - `metadata.json` — dense model name, chunk count, chunks linked, source
     PDF path, repo path, and which device conversion actually ran on.



## Usage

Run from the repo root (`uv sync --all-packages` first):

```bash
uv run dlrag-ingest \
    --pdf "books/<book>.pdf" \
    --repo git_repo/PyTorchStepByStep \
    --remote-url https://github.com/dvgodoy/PyTorchStepByStep/blob/master \
    --query "how do I implement a step-by-step training class"
```

Outputs in `data/rag_artifacts`: `chunks.jsonl`, `embeddings.npy`, `index.json`, `metadata.json`.

```python
from dlrag_index import AgenticIndex

index = AgenticIndex(artifacts_dir="data/rag_artifacts", repo_dir="git_repo/PyTorchStepByStep")
index.tools                       # JSON schemas to register with the agent
index.book_search("custom Dataset and DataLoader")
index.book_get_chunk(chunk_id)    # fetch one chunk by id (see chunks.jsonl)
```

## Evaluating retrieval quality

```bash
uv run dlrag-eval --artifacts data/rag_artifacts --benchmark data/benchmark/benchmark.json --top-k 5
```

`benchmark.json` entries reference chunks by **section header text**
(`relevant_section_ids: ["Batch Normalization", ...]`), not by `chunk_id`,
since `chunk_id` is a random UUID re-assigned on every ingestion run. Header
matching is normalized (case, whitespace, backslash-escapes, HTML entities)
and does substring matching, so a short ground-truth header can be satisfied
by any retrieved header that contains it as a substring — worth keeping in
mind when reading recall numbers for generic single-word entries.

The evaluator reports fused RRF rank/MRR/recall as well as BM25-only and
dense-only ranks per query, plus an `attribution` field (`both` /
`bm25_only` / `dense_only` / `neither`) showing which retriever actually
found the answer.

## Requirements

`docling` (which pulls in torch for conversion), `rank_bm25`, and `fastembed` (via `dlrag-index`). `langchain-text-splitters` and `tiktoken` are optional — the chunker falls back to a native splitter and a character-count token estimate if they are not installed.
