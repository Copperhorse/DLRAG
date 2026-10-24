"""The experiment arms of the fair protocol (see PROTOCOL.md), all on the same llama-server model via Agno.

Design rule: **the agent arms are supersets of the one-shot arms.** A one-shot arm and its agent counterpart receive the
*identical* first user message (pre-retrieved book passages C, optionally the repo map R, then the task); the agent
additionally has tools to fetch more. So any difference measures the marginal value of the tools, not a difference in
what was shown up front.

=================  ==========  ==========  ===========================================
arm                context C   repo map R  tools
=================  ==========  ==========  ===========================================
plain              -           -           -
rag                top-k       -           -
rag-repo           top-k       yes         -
agent-rag          top-k       -           book_search
agent-rag-repo     top-k       yes         book_search, repo_list, repo_grep, repo_read_file
agent-scratch      -           yes         same as agent-rag-repo (no pre-retrieval)
=================  ==========  ==========  ===========================================

Same finalization policy everywhere: if a run ends without a code block, one more request is made (agents: tool-free, with
what their tools returned; one-shot: the same prompt again with a reminder to answer with a code block).
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

ARMS = ("plain", "rag", "rag-repo", "agent-rag", "agent-rag-repo", "agent-scratch")
AGENT_ARMS = ("agent-rag", "agent-rag-repo", "agent-scratch")
CONTEXT_ARMS = ("rag", "rag-repo", "agent-rag", "agent-rag-repo")  # get the pre-retrieved passages C
REPO_MAP_ARMS = ("rag-repo", "agent-rag-repo", "agent-scratch")  # get the repo map R
REPO_TOOL_ARMS = ("agent-rag-repo", "agent-scratch")  # may browse the repo

CHUNK_CHARS = 2000  # one rendering for pre-retrieved passages AND for the book_search tool
TOOL_RESULT_CAP = 6000  # chars per tool result (keeps a multi-call run inside the context window)

BASE_INSTRUCTIONS = (
    "You are an expert PyTorch engineer. Write correct, idiomatic Python. "
    "Your final answer must be a single complete Python code block."
)


def agent_instructions(arm: str, max_calls: int) -> str:
    text = (
        BASE_INSTRUCTIONS + " You may look things up with tools before answering: `book_search(query, top_k)` returns "
        "passages of the book *Deep Learning with PyTorch Step-by-Step*"
    )
    if arm in REPO_TOOL_ARMS:
        text += ", and `repo_list`, `repo_grep` and `repo_read_file` browse the book's companion code repository"
    text += ". "
    if arm in CONTEXT_ARMS or arm in REPO_MAP_ARMS:
        text += "Some material is already given in the message; use the tools only if it is not enough. "
    text += (
        f"You can make at most {max_calls} tool calls. Check the book's conventions (names, signatures, attributes) "
        "before answering, then give the final code block."
    )
    return text


@dataclass
class ModelConfig:
    base_url: str = "http://localhost:8080/v1"
    model_id: str = "sharp-minicpm5-2b"  # llama-server ignores it
    temperature: float = 1.0  # model-card sampling defaults
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 3072
    thinking: bool = False  # Sharp template's <think> mode; off = fast, fits 3K tokens
    timeout: float = 600.0


@dataclass
class ArmResult:
    text: str
    tool_calls: list[dict] = field(default_factory=list)  # [{name, args, error}]
    input_tokens: int = 0
    output_tokens: int = 0
    duration: float = 0.0
    error: str | None = None
    forced_final: bool = False  # a second request was needed because no code block came out the first time


def build_model(cfg: ModelConfig) -> OpenAILike:
    return OpenAILike(
        id=cfg.model_id,
        base_url=cfg.base_url,
        api_key="none",
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_tokens,
        extra_body={"top_k": cfg.top_k, "chat_template_kwargs": {"enable_thinking": cfg.thinking}},
        timeout=cfg.timeout,
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


def render_passages(hits) -> str:
    """The ONE rendering of retrieved book passages, used for the pre-retrieved context and for `book_search`."""
    parts = [f"### {h.section_header}\n{h.content[:CHUNK_CHARS]}" for h in hits]
    return "\n\n".join(parts)


def _cap(text: str) -> str:
    return text if len(text) <= TOOL_RESULT_CAP else text[:TOOL_RESULT_CAP] + "\n...[truncated]"


def make_tools(index: AgenticIndex, max_calls: int | None, repo: bool, grace: int = 3) -> list[Callable]:
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
                return (
                    "TOOL-CALL LIMIT REACHED. Do not call any more tools. "
                    "Write your final answer now as one complete Python code block."
                )
            return fn(*a, **kw)

        return wrapper

    def book_search(query: str, top_k: int = 3) -> str:
        """Hybrid search over the book's passages; returns the most relevant passages in full (section header + text)."""
        hits = index.engine.search(query, top_k=min(max(top_k, 1), 5))
        return _cap(render_passages(hits))

    def repo_list(path: str = ".") -> str:
        """List files and directories of the book's companion code repository."""
        return _cap(json.dumps(index.repo_list(path)))

    def repo_grep(pattern: str, path: str = ".") -> str:
        """Regex search (case-insensitive) in the repository's text files; returns file, line and text."""
        return _cap(json.dumps(index.repo_grep(pattern, path)))

    def repo_read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a repository file (notebooks are shown as numbered cells). Output is capped; page with start_line/end_line."""
        return _cap(json.dumps(index.repo_read_file(path, start_line, end_line)))

    tools = (book_search,) + ((repo_list, repo_grep, repo_read_file) if repo else ())
    return [counted(f) for f in tools]


class Arm:
    def __init__(
        self,
        name: str,
        cfg: ModelConfig,
        index: AgenticIndex | None = None,
        context_k: int = 3,
        max_tool_calls: int = 8,
        run_timeout: float = 240.0,
    ):
        if name not in ARMS:
            raise ValueError(f"Unknown arm {name!r}; choose from {ARMS}")
        if name != "plain" and index is None:
            raise ValueError(f"Arm {name!r} needs an index")
        if name in REPO_MAP_ARMS and (index is None or index.repo is None):
            raise ValueError(f"Arm {name!r} needs a repo_dir")
        self.name, self.cfg, self.index = name, cfg, index
        self.context_k, self.max_tool_calls, self.run_timeout = context_k, max_tool_calls, run_timeout

    # ---------------------------------------------------------------- inputs
    def initial_message(self, spec: str, prompt: str) -> str:
        """First user message. Identical for an agent arm and its one-shot counterpart (see module docstring)."""
        parts = []
        if self.name in CONTEXT_ARMS:
            hits = self.index.engine.search(spec, top_k=self.context_k)
            parts.append(
                "Reference passages from the book (they may or may not be relevant):\n\n" + render_passages(hits)
            )
        if self.name in REPO_MAP_ARMS:
            parts.append(self.index.repo.repo_map())
        parts.append(prompt)
        return "\n\n---\n\n".join(parts)

    def _agent(self) -> Agent:
        if self.name in AGENT_ARMS:
            return Agent(
                model=build_model(self.cfg),
                tools=make_tools(self.index, max_calls=self.max_tool_calls, repo=self.name in REPO_TOOL_ARMS),
                instructions=agent_instructions(self.name, self.max_tool_calls),
                markdown=False,
                telemetry=False,
                retries=0,
            )
        return Agent(
            model=build_model(self.cfg), instructions=BASE_INSTRUCTIONS, markdown=False, telemetry=False, retries=0
        )

    # ------------------------------------------------------------------- run
    def run(self, spec: str, prompt: str) -> ArmResult:
        """``spec`` = the task description (the retrieval query); ``prompt`` = the full task prompt."""
        message = self.initial_message(spec, prompt)
        t0 = time.time()
        agent, run_id = self._agent(), str(uuid.uuid4())
        # Agno only checks for cancellation between model/tool calls, so this ends a run that keeps looping.
        timer = threading.Timer(self.run_timeout, agent.cancel_run, args=(run_id,))
        timer.start()
        try:
            out = agent.run(message, run_id=run_id)
        except Exception as e:  # server down, context overflow, tool-call parsing failure, ...
            return ArmResult(text="", duration=time.time() - t0, error=f"{type(e).__name__}: {e}"[:500])
        finally:
            timer.cancel()
        m = out.metrics
        result = ArmResult(
            text=out.content if isinstance(out.content, str) else str(out.content or ""),
            tool_calls=[
                {"name": t.tool_name, "args": t.tool_args, "error": bool(t.tool_call_error)} for t in (out.tools or [])
            ],
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
        if not _has_code(result.text):
            self._finalize(result, message, out)
        if failure and not _has_code(result.text):
            result.error = failure
        return result

    def _finalize(self, result: ArmResult, message: str, out) -> None:
        """Same policy for every arm: one more request when no code block came out."""
        if self.name in AGENT_ARMS:
            evidence = _evidence(out.tools)
            if not evidence:
                return
            follow_up = (
                f"{message}\n\n---\nYou already looked things up. Here is what the tools returned:\n\n{evidence}\n\n---\n"
                "Tool use is over. Using the information above, write your final answer now: one complete Python code block."
            )
        else:
            follow_up = (
                f"{message}\n\n---\nYour previous reply contained no code block. "
                "Write your final answer now: one complete Python code block."
            )
        t0 = time.time()
        try:
            second = Agent(
                model=build_model(self.cfg), instructions=BASE_INSTRUCTIONS, markdown=False, telemetry=False, retries=0
            ).run(follow_up)
        except Exception:
            return
        if second.status == RunStatus.error:
            return
        m = second.metrics
        result.text = second.content if isinstance(second.content, str) else str(second.content or "")
        result.forced_final = True
        result.input_tokens += int(getattr(m, "input_tokens", 0) or 0)
        result.output_tokens += int(getattr(m, "output_tokens", 0) or 0)
        result.duration += time.time() - t0


def make_arm(
    name: str, cfg: ModelConfig | None = None, artifacts_dir: Path | None = None, repo_dir: Path | None = None, **kw
) -> Arm:
    cfg = cfg or ModelConfig()
    index = None
    if name != "plain":
        index = AgenticIndex(artifacts_dir=artifacts_dir, repo_dir=repo_dir if name in REPO_MAP_ARMS else None)
    return Arm(name, cfg, index, **kw)
