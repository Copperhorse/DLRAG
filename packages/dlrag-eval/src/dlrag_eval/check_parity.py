"""`dlrag-check-parity`: verify the invariants of the fair protocol (no model or server needed).

* an agent arm and its one-shot counterpart get the identical first user message,
* the agents' `book_search` returns exactly the rendering used for the pre-retrieved passages,
* each arm has exactly the tools the protocol table lists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dlrag_agent.arms import ModelConfig, agent_instructions, make_arm, make_tools, render_passages

from .tasks import DEFAULT_TASKS_DIR, load_tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    ap.add_argument("--artifacts", type=Path, default=Path("data/rag_artifacts"))
    ap.add_argument("--repo", type=Path, default=Path("git_repo/PyTorchStepByStep"))
    ap.add_argument("--context-k", type=int, default=3)
    args = ap.parse_args()

    cfg = ModelConfig()
    arms = {
        n: make_arm(n, cfg, artifacts_dir=args.artifacts, repo_dir=args.repo, context_k=args.context_k)
        for n in ("plain", "rag", "rag-repo", "agent-rag", "agent-rag-repo", "agent-scratch")
    }
    problems = []

    def check(ok: bool, what: str):
        if not ok:
            problems.append(what)

    tasks = load_tasks(args.tasks_dir)
    for t in tasks:
        m = {n: a.initial_message(t.spec, t.prompt) for n, a in arms.items()}
        check(m["rag"] == m["agent-rag"], f"{t.id}: rag vs agent-rag first message differ")
        check(m["rag-repo"] == m["agent-rag-repo"], f"{t.id}: rag-repo vs agent-rag-repo first message differ")
        check(m["plain"] == t.prompt, f"{t.id}: plain message is not the bare prompt")
        check(
            t.prompt in m["agent-scratch"] and "Reference passages" not in m["agent-scratch"],
            f"{t.id}: agent-scratch must have the repo map and no pre-retrieved passages",
        )
        check(
            m["rag-repo"].count("Python files of the book's companion repository") == 1,
            f"{t.id}: repo map missing/duplicated",
        )

    index = arms["agent-rag"].index
    names = lambda repo: [f.__name__ for f in make_tools(index, max_calls=8, repo=repo)]
    check(names(False) == ["book_search"], f"agent-rag tools: {names(False)}")
    check(
        names(True) == ["book_search", "repo_list", "repo_grep", "repo_read_file"],
        f"agent-rag-repo tools: {names(True)}",
    )
    probe = tasks[0].spec
    k = min(
        args.context_k, 3
    )  # one tool call returns at most 5 passages / 6000 chars; the agent can call it again for more
    tool_out = make_tools(index, max_calls=8, repo=False)[0](probe, k)
    check(
        tool_out == render_passages(index.engine.search(probe, top_k=k)),
        "book_search output differs from the pre-retrieved rendering",
    )
    for n in ("agent-rag", "agent-rag-repo", "agent-scratch"):
        check("at most 8 tool calls" in agent_instructions(n, 8), f"{n}: budget missing from instructions")

    if problems:
        print("PARITY CHECK FAILED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print(
        f"parity OK for {len(tasks)} tasks (context_k={args.context_k}): paired first messages identical, tools as specified, "
        "book_search == pre-retrieved rendering"
    )


if __name__ == "__main__":
    main()
