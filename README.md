# DLRAG

Coding-oriented agentic RAG over a technical book (PDF) and its companion code repository, and an experiment:
**one-shot RAG vs an agent that calls book/repo tools** (same small model; protocol in `PROTOCOL.md`, results in `REPORT.md`). A fine-tune arm was started and is shelved (`finetune/`).

A [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/); members live in `packages/`:

| Package | Module | Role |
|---|---|---|
| `dlrag-core` | `dlrag_core` | Config dataclasses and `ChunkRecord`/`CodeAsset` models |
| `dlrag-index` | `dlrag_index` | BM25 + dense (fastembed) hybrid search (RRF), `AgenticIndex` book/repo tools |
| `dlrag-ingest` | `dlrag_ingest` | PDF → Markdown (Docling) → chunks → repo links → index (`dlrag-ingest`) |
| `dlrag-agent` | `dlrag_agent` | Experiment arms on [Agno](https://docs.agno.com): plain / one-shot RAG (+ repo map) / agent (book tools) / agent (book + repo tools), optionally with a forced final answer |
| `dlrag-eval` | `dlrag_eval` | Retrieval benchmark (`dlrag-eval`), book-derived coding tasks (`dlrag-tasks`, `dlrag-tasks-validate`) |

```bash
uv sync --all-packages
uv run dlrag-ingest --pdf books/<book>.pdf --repo git_repo/PyTorchStepByStep
uv run dlrag-eval                                   # retrieval diagnostics
uv run dlrag-tasks-validate                         # task references pass, empty stub fails

# with llama-server running the model (--jinja, port 8080):
uv run dlrag-tasks --arm plain --label base --samples 3
uv run dlrag-tasks --arm rag   --label base --samples 3
uv run dlrag-tasks --arm agent-book --label base --samples 3
uv run dlrag-tasks --arm agent --label base --samples 3
```

Data: `books/` (source PDFs), `git_repo/` (companion code), `data/rag_artifacts/` (generated index), `data/benchmark/` (retrieval queries),
`data/tasks/` (32 execution-graded coding tasks derived from the book's code), `data/results/` (run outputs, gitignored).
Shelved QLoRA code for Colab is in `finetune/` (see its README). Pipeline details: `packages/dlrag-ingest/README.md`.
