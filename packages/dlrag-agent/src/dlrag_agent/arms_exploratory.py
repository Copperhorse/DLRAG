"""EXPLORATORY arms (first two rounds, superseded by ``arms.py``; kept to reproduce ``data/results/exploratory``).

Known asymmetries vs the fair protocol: one-shot got full chunks, the agent 400-char previews; the repo map was curated to the
latest module versions; only agents got a second (forced-final) call; no tuning. Run with ``dlrag-tasks --protocol exploratory``.

Original docstring: the experiment arms, all backed by the same llama-server model via Agno.

``plain``  - no retrieval (arms A / C: base or fine-tuned weights).
``rag``    - one-shot RAG: top-k book chunks are pre-retrieved and put in the prompt.
``agent``  - Agno agent that calls book/repo tools itself (arms B / D).
"""
from __future__ import annotations

import functools
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

os.environ.setdefault("AGNO_TELEMETRY", "false")  # Agno phones home by default

from agno.agent import Agent  # noqa: E402
from agno.exceptions import StopAgentRun  # noqa: E402
from agno.models.openai.like import OpenAILike  # noqa: E402
from agno.run.base import RunStatus  # noqa: E402

from dlrag_index import AgenticIndex  # noqa: E402

ARMS = ("plain", "rag", "rag-repo", "agent-book", "agent")
AGENT_ARMS = ("agent-book", "agent")
TOOL_RESULT_CAP = 5000   # chars per tool result; keeps a multi-call run inside the context window

BASE_INSTRUCTIONS = (
    "You are an expert PyTorch engineer. Write correct, idiomatic Python. "
    "Your final answer must be a single complete Python code block."
)
AGENT_BOOK_INSTRUCTIONS = BASE_INSTRUCTIONS + (
    " Before writing code you have tools to look things up: `book_search` finds passages of the book "
    "*Deep Learning with PyTorch Step-by-Step* and `book_get_chunk` reads one passage in full. "
    "Use them to check the book's conventions (names, signatures, attributes) before answering, "
    "then give the final code block."
)
AGENT_INSTRUCTIONS = BASE_INSTRUCTIONS + (
    " Before writing code you have tools to look things up: `book_search` finds passages of the book "
    "*Deep Learning with PyTorch Step-by-Step*, `book_get_chunk` reads one passage in full, and "
    "`repo_list`, `repo_grep` and `repo_read_file` browse the book's companion code repository. "
    "Use them to check the book's conventions (names, signatures, attributes) before answering, "
    "then give the final code block."
)


@dataclass
class ModelConfig:
    base_url: str = "http://localhost:8080/v1"
    model_id: str = "sharp-minicpm5-2b"      # llama-server ignores it
    temperature: float = 1.0                 # model-card sampling defaults
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 3072
    thinking: bool = False                   # Sharp template's <think> mode; off = fast, fits 3K tokens
    timeout: float = 600.0


@dataclass
class ArmResult:
    text: str
    tool_calls: list[dict] = field(default_factory=list)   # [{name, args, error}]
    input_tokens: int = 0
    output_tokens: int = 0
    duration: float = 0.0
    error: str | None = None
    forced_final: bool = False   # the agent never produced code; a tool-free final answer was requested from its evidence


def build_model(cfg: ModelConfig) -> OpenAILike:
    return OpenAILike(
        id=cfg.model_id, base_url=cfg.base_url, api_key="none",
        temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=cfg.max_tokens,
        extra_body={"top_k": cfg.top_k, "chat_template_kwargs": {"enable_thinking": cfg.thinking}}, timeout=cfg.timeout,
    )


_CODE_FENCE = re.compile(r"```")
EVIDENCE_PER_TOOL_CHARS = 2500
EVIDENCE_TOTAL_CHARS = 14000


def _has_code(text: str) -> bool:
    return len(_CODE_FENCE.findall(text or "")) >= 2


def _evidence(tools) -> str:
    """What the tools returned during the run (the 'limit reached' notices are dropped), size-capped."""
    parts, total = [], 0
    for t in tools or []:
        result = str(t.result or "")
        if not result or result.startswith("TOOL-CALL LIMIT REACHED"):
            continue
        piece = f"[{t.tool_name} {json.dumps(t.tool_args, default=str)[:200]}]\n{result[:EVIDENCE_PER_TOOL_CHARS]}"
        if total + len(piece) > EVIDENCE_TOTAL_CHARS:
            break
        parts.append(piece)
        total += len(piece)
    return "\n\n".join(parts)


def _cap(text: str) -> str:
    return text if len(text) <= TOOL_RESULT_CAP else text[:TOOL_RESULT_CAP] + ' ...[truncated]"}'


def make_tools(index: AgenticIndex, max_calls: int | None = None, grace: int = 3,
               repo: bool = True) -> list[Callable]:
    """Plain functions (name, type hints and docstring become the tool schema).

    Agno's own ``tool_call_limit`` never invokes the tool again once reached, so a small model that keeps
    asking can loop until the context is full. The limit is therefore enforced here: past ``max_calls``
    the tools answer with a "write your answer now" notice, and after ``grace`` more attempts the run is aborted.
    """
    calls = {"n": 0}

    def counted(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            calls["n"] += 1
            if max_calls is not None and calls["n"] > max_calls:
                if calls["n"] > max_calls + grace:
                    raise StopAgentRun("tool-call loop: model ignored the tool-call limit")
                return ("TOOL-CALL LIMIT REACHED. Do not call any more tools. "
                        "Write your final answer now as one complete Python code block.")
            return fn(*a, **kw)
        return wrapper

    def book_search(query: str, top_k: int = 3) -> str:
        """Hybrid search over the book's passages. Returns section headers, previews, ids and linked code files."""
        return _cap(json.dumps(index.book_search(query, top_k=min(max(top_k, 1), 5))))

    def book_get_chunk(chunk_id: str) -> str:
        """Fetch one full book passage by the `chunk_id` returned from book_search."""
        chunk = index.book_get_chunk(chunk_id, with_code=False)
        if chunk is None:
            return json.dumps({"error": f"unknown chunk_id {chunk_id}"})
        return _cap(json.dumps({"section_header": chunk["section_header"], "content": chunk["content"]}))

    def repo_list(path: str = ".") -> str:
        """List files and directories of the book's companion code repository."""
        return _cap(json.dumps(index.repo_list(path)))

    def repo_grep(pattern: str, path: str = ".") -> str:
        """Regex search (case-insensitive) in the repository's text files; returns file, line and text."""
        return _cap(json.dumps(index.repo_grep(pattern, path)))

    def repo_read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a repository file (notebooks are shown as numbered cells). Output is capped; page with start_line/end_line."""
        return _cap(json.dumps(index.repo_read_file(path, start_line, end_line)))

    tools = (book_search, book_get_chunk) + ((repo_list, repo_grep, repo_read_file) if repo else ())
    return [counted(f) for f in tools]


class Arm:
    def __init__(self, name: str, cfg: ModelConfig, index: AgenticIndex | None = None,
                 rag_top_k: int = 3, max_tool_calls: int = 8, run_timeout: float = 240.0,
                 force_final: bool = False):
        if name not in ARMS:
            raise ValueError(f"Unknown arm {name!r}; choose from {ARMS}")
        if name != "plain" and index is None:
            raise ValueError(f"Arm {name!r} needs an index")
        self.name, self.cfg, self.index = name, cfg, index
        self.rag_top_k, self.max_tool_calls = rag_top_k, max_tool_calls
        self.force_final = force_final   # agent arms: if no code came out, ask once more without tools
        self.run_timeout = run_timeout   # wall-clock cap per run (agent arms); a run once stalled for 94 min

    def _agent(self) -> Agent:
        tools = (make_tools(self.index, max_calls=self.max_tool_calls, repo=self.name == "agent")
                 if self.name in AGENT_ARMS else None)
        instructions = {"agent": AGENT_INSTRUCTIONS, "agent-book": AGENT_BOOK_INSTRUCTIONS}.get(self.name, BASE_INSTRUCTIONS)
        return Agent(
            model=build_model(self.cfg), tools=tools, instructions=instructions,
            markdown=False, telemetry=False, retries=0,
        )

    def _rag_context(self, query: str) -> str:
        hits = self.index.engine.search(query, top_k=self.rag_top_k)
        parts = [f"### {h.section_header}\n{h.content[:1500]}" for h in hits]
        text = "Reference passages from the book (they may or may not be relevant):\n\n" + "\n\n".join(parts)
        if self.name == "rag-repo":   # one-shot counterpart of the agent's repo tools
            text += "\n\n" + self.index.repo.repo_map(latest_only=True, max_chars=9000)
        return text

    def run(self, spec: str, prompt: str) -> ArmResult:
        """``spec`` = the task description (used as the retrieval query); ``prompt`` = full user prompt."""
        text_in = f"{self._rag_context(spec)}\n\n---\n\n{prompt}" if self.name in ("rag", "rag-repo") else prompt
        t0 = time.time()
        agent, run_id = self._agent(), str(uuid.uuid4())
        # Agno only checks for cancellation between model/tool calls, so this ends a run that keeps looping.
        timer = threading.Timer(self.run_timeout, agent.cancel_run, args=(run_id,))
        timer.start()
        try:
            out = agent.run(text_in, run_id=run_id)
        except Exception as e:  # server down, context overflow, tool-call parsing failure, ...
            return ArmResult(text="", duration=time.time() - t0, error=f"{type(e).__name__}: {e}"[:500])
        finally:
            timer.cancel()
        m = out.metrics
        text = out.content if isinstance(out.content, str) else str(out.content or "")
        calls = [{"name": t.tool_name, "args": t.tool_args, "error": bool(t.tool_call_error)} for t in (out.tools or [])]
        result = ArmResult(
            text=text, tool_calls=calls,
            input_tokens=int(getattr(m, "input_tokens", 0) or 0),
            output_tokens=int(getattr(m, "output_tokens", 0) or 0),
            duration=time.time() - t0,
        )
        failure = None
        if out.status == RunStatus.cancelled:
            failure = f"timeout: cancelled after {self.run_timeout:.0f}s"
        elif out.status == RunStatus.error:  # Agno logs provider errors and returns instead of raising
            failure = f"run failed: {str(out.content)[:400]}"
            result.text = ""
        if self.force_final and self.name in AGENT_ARMS and not _has_code(result.text):
            evidence = _evidence(out.tools)
            if evidence:
                self._force_final(result, prompt, evidence)
        if failure and not _has_code(result.text):
            result.error = failure
        return result

    def _force_final(self, result: ArmResult, prompt: str, evidence: str) -> None:
        """The agent ran out of budget without answering: one tool-free call with what it found."""
        message = (f"{prompt}\n\n---\nYou already looked things up. Here is what the tools returned:\n\n{evidence}\n\n---\n"
                   "Tool use is over. Using the information above, write your final answer now: "
                   "one complete Python code block.")
        t0 = time.time()
        try:
            out = Agent(model=build_model(self.cfg), instructions=BASE_INSTRUCTIONS, markdown=False,
                        telemetry=False, retries=0).run(message)
        except Exception:
            return
        if out.status == RunStatus.error:
            return
        m = out.metrics
        result.text = out.content if isinstance(out.content, str) else str(out.content or "")
        result.forced_final = True
        result.input_tokens += int(getattr(m, "input_tokens", 0) or 0)
        result.output_tokens += int(getattr(m, "output_tokens", 0) or 0)
        result.duration += time.time() - t0


def make_arm(name: str, cfg: ModelConfig | None = None, artifacts_dir: Path | None = None,
             repo_dir: Path | None = None, **kw) -> Arm:
    cfg = cfg or ModelConfig()
    index = None
    if name != "plain":
        index = AgenticIndex(artifacts_dir=artifacts_dir, repo_dir=repo_dir if name in ("agent", "rag-repo") else None)
    return Arm(name, cfg, index, **kw)
