"""Run a candidate solution against a task's tests in a subprocess.

NOT a security sandbox: the code runs as the current user with a timeout and a
CPU-time limit only. Do not point this at untrusted models on a machine you care about.
"""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from .tasks import Task

COMMON_DIR_NAME = "_common"
REPO_DIR = Path("git_repo/PyTorchStepByStep")  # relative to the repo root, like the other default paths


@dataclass
class TestOutcome:
    passed: dict[str, bool] = field(default_factory=dict)  # test name -> passed
    error: str | None = None  # collection/timeout problems

    @property
    def pass_behavior(self) -> bool:
        b = [v for k, v in self.passed.items() if k.startswith("test_behavior")]
        return bool(b) and all(b)

    @property
    def pass_all(self) -> bool:
        return bool(self.passed) and all(self.passed.values())

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "error": self.error,
            "pass_behavior": self.pass_behavior,
            "pass_all": self.pass_all,
        }


def _limits(cpu_seconds: int):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))

    return apply


def run_tests(task: Task, code: str | None, timeout: int = 90) -> TestOutcome:
    if not code or not code.strip():
        return TestOutcome(error="no code")
    with tempfile.TemporaryDirectory(prefix=f"dlrag-{task.id}-") as tmp:
        tmp_path = Path(tmp)
        shutil.copy(task.dir.parent / COMMON_DIR_NAME / "common.py", tmp_path / "common.py")
        shutil.copy(task.dir / "tests.py", tmp_path / "tests.py")
        (tmp_path / "solution.py").write_text(code, encoding="utf-8")
        report = tmp_path / "report.xml"
        python_path = str(tmp_path) + (os.pathsep + str(REPO_DIR.resolve()) if task.needs_repo else "")
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "2",
            "MPLBACKEND": "Agg",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": python_path,
        }
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={report}", "tests.py"],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_limits(timeout),
            )
        except subprocess.TimeoutExpired:
            return TestOutcome(error="timeout")
        if not report.exists():
            return TestOutcome(error=(proc.stdout + proc.stderr)[-500:] or "no report")
        out = TestOutcome()
        for case in ET.parse(report).getroot().iter("testcase"):
            name = case.get("name", "")
            failed = case.find("failure") is not None or case.find("error") is not None
            if name.startswith("test_"):
                out.passed[name] = not failed
            elif failed:  # collection error (e.g. ImportError/SyntaxError in the candidate)
                first = case.find("error") if case.find("error") is not None else case.find("failure")
                out.error = (first.get("message") or "collection error")[:300]
        return out
