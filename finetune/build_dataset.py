"""Build the QLoRA training set from the ingested book and its companion repository.

Runs locally on CPU (no GPU, no model). Output: one JSON object per line, either
    {"kind": "doc", "text": "..."}                      raw text, plain LM loss
    {"kind": "qa",  "messages": [user, assistant]}      formatted with the chat template at train time

Same knowledge the RAG arms can reach (book chunks + repo code), nothing else. In particular
``data/tasks/**`` (task prompts, tests, reference solutions) is never read.

    uv run python finetune/build_dataset.py --dry-run
    uv run python finetune/build_dataset.py            # -> finetune/data/train.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path

REPO_SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "runs", "images", "plots", "revision"}
MAX_CHARS = 6000  # per example; ~1.5K tokens, well inside max_seq_len


_LINE_NO = re.compile(r"^(\s*)(\d{1,3})(\s+)(.*)$")


def strip_listing_numbers(content: str) -> str:
    """Remove the book's printed listing line numbers ("1 import ...", "2 x = ...") from fenced code.

    Only a counter that starts at 1 and then increases by one per numbered line is stripped, so a code line that
    merely begins with a number is left alone. Lines without a number (wrapped continuations) are kept as they are.
    """
    lines = content.split("\n")
    expected = None
    out = []
    for line in lines:
        m = _LINE_NO.match(line)
        if m and (expected is None and int(m.group(2)) == 1 or expected is not None and int(m.group(2)) == expected):
            expected = int(m.group(2)) + 1
            out.append(m.group(1) + m.group(4))
        else:
            out.append(line)
    return "\n".join(out)


def _is_flattened_code(content: str) -> bool:
    """Docling (without code enrichment) puts a whole listing on one line; such text teaches broken code."""
    inner = content.strip().strip("`").strip()
    return len(inner) > 120 and inner.count("\n") <= 1


def book_examples(chunks_path: Path, stats: dict | None = None):
    stats = stats if stats is not None else {}
    seen: set[str] = set()
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        content = c["content"].strip()
        header = c["section_header"].strip()
        if len(content) < 80:
            stats["too_short"] = stats.get("too_short", 0) + 1
            continue
        if header == "Output":  # raw cell outputs (tensors, arrays): little knowledge
            stats["cell_output"] = stats.get("cell_output", 0) + 1
            continue
        if c["content_type"] == "code" and _is_flattened_code(content):
            stats["flattened_code"] = stats.get("flattened_code", 0) + 1
            continue
        if c["content_type"] == "code":
            content = strip_listing_numbers(content)
        key = " ".join(content.split())
        if key in seen:
            stats["duplicate"] = stats.get("duplicate", 0) + 1
            continue
        seen.add(key)
        crumb_parts = []
        for part in (c.get("top_header"), c.get("parent_section"), header):
            if part and part.strip() and part.strip() not in crumb_parts:
                crumb_parts.append(part.strip())
        crumb = " > ".join(crumb_parts)
        yield {"kind": "doc", "text": f"# Book section: {crumb}\n\n{content}"[:MAX_CHARS]}
        if c["content_type"] == "code":
            q = f'Show the code from the book *Deep Learning with PyTorch Step-by-Step* for the section "{header}".'
        else:
            q = f"Explain, following the book *Deep Learning with PyTorch Step-by-Step*: {header}"
        yield {
            "kind": "qa",
            "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": content[:MAX_CHARS]}],
        }


def _notebook_text(path: Path) -> str:
    cells = json.loads(path.read_text(encoding="utf-8")).get("cells", [])
    out = []
    for c in cells:
        src = "".join(c.get("source", [])).strip()
        if src and c.get("cell_type") in ("code", "markdown"):
            out.append(f"# --- {c['cell_type']} cell ---\n{src}")
    return "\n\n".join(out)


def _repo_files(repo: Path):
    for p in sorted(repo.rglob("*")):
        if p.is_file() and p.suffix in {".py", ".ipynb"} and not (set(p.relative_to(repo).parts[:-1]) & REPO_SKIP_DIRS):
            yield p


REPO_SKIP_FILES = {"config.py", "tensorboardserverextension.py"}  # Colab/TensorBoard plumbing, not book content


def repo_examples(repo: Path):
    latest_defs: dict[str, tuple[str, str, str]] = {}  # qualified name -> (file, name, source); later versions win
    for p in _repo_files(repo):
        if p.name in REPO_SKIP_FILES:
            continue
        rel = str(p.relative_to(repo))
        text = _notebook_text(p) if p.suffix == ".ipynb" else p.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        for i in range(0, len(text), MAX_CHARS):  # long files -> several docs
            yield {"kind": "doc", "text": f"# Repository file: {rel}\n\n{text[i : i + MAX_CHARS]}"}
        if p.suffix != ".py":
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                for m in node.body:
                    if isinstance(m, ast.FunctionDef):
                        latest_defs[f"{p.parent.name}/{node.name}.{m.name}"] = (
                            rel,
                            f"{node.name}.{m.name}",
                            ast.get_source_segment(text, m) or "",
                        )
            elif isinstance(node, ast.FunctionDef):
                latest_defs[f"{p.parent.name}/{node.name}"] = (rel, node.name, ast.get_source_segment(text, node) or "")
    for rel, name, src in latest_defs.values():
        if len(src) < 60:
            continue
        q = f"Show the implementation of `{name}` from the companion repository of *Deep Learning with PyTorch Step-by-Step* (file `{rel}`)."
        yield {
            "kind": "qa",
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"```python\n{src[:MAX_CHARS]}\n```"},
            ],
        }


def _unique_prompts(rows: list[dict], stats: dict) -> list[dict]:
    """Drop QA examples whose prompt is shared with another example: e.g. 40 different answers to
    "Explain ...: Data Preparation" are conflicting targets. The text itself stays in the raw `doc` examples."""
    counts: dict[str, int] = {}
    for r in rows:
        if r["kind"] == "qa":
            q = r["messages"][0]["content"]
            counts[q] = counts.get(q, 0) + 1
    kept = [r for r in rows if r["kind"] != "qa" or counts[r["messages"][0]["content"]] == 1]
    stats["ambiguous_qa"] = stats.get("ambiguous_qa", 0) + len(rows) - len(kept)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", type=Path, default=Path("data/rag_artifacts/chunks.jsonl"))
    ap.add_argument("--repo", type=Path, default=Path("git_repo/PyTorchStepByStep"))
    ap.add_argument("--out", type=Path, default=Path("finetune/data/train.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="print statistics only")
    args = ap.parse_args()

    skipped: dict[str, int] = {}
    book = _unique_prompts(list(book_examples(args.chunks, skipped)), skipped)
    repo = _unique_prompts(list(repo_examples(args.repo)), skipped)
    rows = book + repo
    random.Random(args.seed).shuffle(rows)

    def chars(rs):
        return sum(len(r["text"]) if r["kind"] == "doc" else sum(len(m["content"]) for m in r["messages"]) for r in rs)

    print("skipped book chunks:", skipped or "none")
    for name, rs in (("book", book), ("repo", repo), ("total", rows)):
        n_doc = sum(r["kind"] == "doc" for r in rs)
        print(f"{name:<6} examples={len(rs):>6} (docs {n_doc}, qa {len(rs) - n_doc})  ~{chars(rs) // 4:,} tokens")
    if args.dry_run:
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
