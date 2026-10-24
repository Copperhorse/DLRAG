"""Book-derived coding tasks: loading and candidate-code extraction."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TASKS_DIR = Path("data/tasks")

OUTPUT_INSTRUCTION = (
    "Follow the conventions used in the book *Deep Learning with PyTorch Step-by-Step* "
    "(its `StepByStep` class and helper functions). "
    "Reply with ONE complete Python code block (```python ... ```) containing all needed imports "
    "and the requested code, and nothing else."
)

_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class Task:
    id: str
    tier: str
    source: str
    spec: str
    dir: Path
    needs_repo: bool = False  # tests import the companion repo's own code (it is put on PYTHONPATH)

    @property
    def prompt(self) -> str:
        return f"{self.spec}\n\n{OUTPUT_INSTRUCTION}"

    @property
    def reference(self) -> str:
        return (self.dir / "reference.py").read_text(encoding="utf-8")


def load_tasks(root: Path = DEFAULT_TASKS_DIR, ids: list[str] | None = None) -> list[Task]:
    tasks = []
    for d in sorted(Path(root).iterdir()):
        if not (d / "task.json").exists():
            continue  # skips _common and stray files
        meta = json.loads((d / "task.json").read_text(encoding="utf-8"))
        if ids and not any(meta["id"].startswith(i) for i in ids):
            continue
        tasks.append(
            Task(
                id=meta["id"],
                tier=meta["tier"],
                source=meta["source"],
                spec=(d / "prompt.md").read_text(encoding="utf-8").strip(),
                dir=d,
                needs_repo=bool(meta.get("needs_repo", False)),
            )
        )
    return tasks


def extract_code(text: str) -> str | None:
    """Last fenced code block of a model reply (None if there is no fence)."""
    blocks = _FENCE_RE.findall(text or "")
    return blocks[-1].strip() + "\n" if blocks else None


MANIFEST_NAME = "MANIFEST.sha256"


def _manifest_lines(root: Path) -> list[str]:
    lines = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.name != MANIFEST_NAME and "__pycache__" not in p.parts:
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}")
    return lines


def write_manifest(root: Path) -> str:
    """Freeze a task directory: sha256 of every file. Returns the digest of the manifest itself."""
    lines = _manifest_lines(root)
    (Path(root) / MANIFEST_NAME).write_text("\n".join(lines) + "\n")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def verify_manifest(root: Path) -> list[str]:
    """Differences between the directory and its manifest ([] = untouched; ['no manifest'] if none)."""
    m = Path(root) / MANIFEST_NAME
    if not m.exists():
        return ["no manifest"]
    want = m.read_text().splitlines()
    have = _manifest_lines(root)
    return [f"changed/added: {x}" for x in have if x not in want] + [
        f"changed/removed: {x}" for x in want if x not in have
    ]
