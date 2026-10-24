"""Read-only, sandboxed browsing of the companion code repository."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

_SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "runs", "images", "plots", "node_modules"}
_MAP_SKIP = {"config.py", "tensorboardserverextension.py"}  # Colab/TensorBoard plumbing
_TEXT_SUFFIXES = {".py", ".ipynb", ".md", ".txt", ".yml", ".yaml", ".toml", ".cfg", ".json", ""}


class RepoBrowser:
    """list / read / grep inside ``root``; any path escaping it is rejected."""

    def __init__(self, root: Path, max_chars: int = 8000, max_results: int = 30):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"Repo directory not found: {self.root}")
        self.max_chars = max_chars
        self.max_results = max_results

    # ------------------------------------------------------------ helpers
    def _resolve(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError(f"Path escapes the repository: {rel!r}")
        return p

    def _iter_files(self, base: Path):
        for p in sorted(base.rglob("*")):
            if (
                p.is_file()
                and p.suffix.lower() in _TEXT_SUFFIXES
                and not (set(p.relative_to(self.root).parts[:-1]) & _SKIP_DIRS)
            ):
                yield p

    def _text(self, p: Path) -> str:
        raw = p.read_text(encoding="utf-8", errors="replace")
        if p.suffix != ".ipynb":
            return raw
        try:
            cells = json.loads(raw)["cells"]
        except (ValueError, KeyError):
            return raw
        return "\n".join(
            f"# --- cell {i} ({c.get('cell_type')}) ---\n{''.join(c.get('source', []))}" for i, c in enumerate(cells)
        )

    # ------------------------------------------------------------ repo map
    def repo_map(self, max_chars: int = 16000, latest_only: bool = False) -> str:
        """Compact map of the Python code: file -> classes/methods/functions with signatures.

        For one-shot prompting (no tool calls). With ``latest_only`` only the highest of numbered module versions
        (v0.py ... v4.py in one directory) is listed (curated; used by the exploratory arms); by default every file is listed,
        exactly as the repo is, with a sentence saying that the highest number is the final version.
        """
        files = [p for p in self._iter_files(self.root) if p.suffix == ".py" and p.name not in _MAP_SKIP]
        latest: dict[Path, tuple[int, Path]] = {}
        keep = []
        for p in files:
            m = re.fullmatch(r"v(\d+)\.py", p.name)
            if not m:
                keep.append(p)
            elif p.parent not in latest or int(m.group(1)) > latest[p.parent][0]:
                latest[p.parent] = (int(m.group(1)), p)
        if latest_only:
            keep += [v[1] for v in latest.values()]
        else:
            keep = files
        lines = ["Python files of the book's companion repository (import path = file path without .py, with dots)."]
        if not latest_only:
            lines.append(
                "Where a directory has several numbered versions (v0.py, v1.py, ...), the highest number is the final version used in the book."
            )
        # the library files first, so truncation can only cut the less relevant ones
        for p in sorted(keep, key=lambda q: (q.name not in ("helpers.py", "v4.py"), str(q))):
            try:
                tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            entries = []
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    entries.append(f"  class {node.name}")
                    entries += [
                        f"    def {m.name}({ast.unparse(m.args)})" for m in node.body if isinstance(m, ast.FunctionDef)
                    ]
                elif isinstance(node, ast.FunctionDef):
                    entries.append(f"  def {node.name}({ast.unparse(node.args)})")
            if entries:
                lines.append(f"{p.relative_to(self.root)}:")
                lines += entries
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[:max_chars] + "\n  ...[truncated]"

    # -------------------------------------------------------------- tools
    def list(self, path: str = ".") -> dict:
        try:
            base = self._resolve(path)
            if not base.is_dir():
                return {"error": f"Not a directory: {path}"}
            entries = [
                p.name + ("/" if p.is_dir() else "")
                for p in sorted(base.iterdir())
                if p.name not in _SKIP_DIRS and not p.name.startswith(".")
            ]
            return {"path": path, "entries": entries}
        except ValueError as e:
            return {"error": str(e)}

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> dict:
        try:
            p = self._resolve(path)
            if not p.is_file():
                return {"error": f"Not a file: {path}"}
            if p.suffix.lower() not in _TEXT_SUFFIXES:
                return {"error": f"Unsupported file type: {p.suffix}"}
            lines = self._text(p).splitlines()
        except ValueError as e:
            return {"error": str(e)}
        start = max(start_line, 1)
        chunk = "\n".join(lines[start - 1 : end_line])
        truncated = len(chunk) > self.max_chars
        return {
            "path": path,
            "start_line": start,
            "total_lines": len(lines),
            "truncated": truncated,
            "content": chunk[: self.max_chars],
        }

    def grep(self, pattern: str, path: str = ".", ignore_case: bool = True) -> dict:
        try:
            base = self._resolve(path)
            rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except (ValueError, re.error) as e:
            return {"error": str(e)}
        files = [base] if base.is_file() else self._iter_files(base)
        matches: list[dict] = []
        for p in files:
            for n, line in enumerate(self._text(p).splitlines(), 1):
                if rx.search(line):
                    matches.append({"file": str(p.relative_to(self.root)), "line": n, "text": line.strip()[:200]})
                    if len(matches) >= self.max_results:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}


REPO_TOOL_SCHEMAS = [
    {
        "name": "repo_list",
        "description": "List files and directories in the companion code repository.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "default": "."}},
        },
    },
    {
        "name": "repo_read_file",
        "description": "Read a file from the repository (notebooks are rendered as numbered cells). "
        "Output is capped; use start_line/end_line to page through long files.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "default": 1},
                "end_line": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "repo_grep",
        "description": "Regex search over repository text files; returns file, line and matching text.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "default": "."},
                "ignore_case": {"type": "boolean", "default": True},
            },
            "required": ["pattern"],
        },
    },
]
