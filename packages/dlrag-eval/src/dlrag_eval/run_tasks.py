"""`dlrag-tasks`: run an experiment arm over the book-derived tasks and score it with the tests."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from dlrag_agent import arms as fair
from dlrag_agent import arms_exploratory as exploratory
from dlrag_agent.arms import ModelConfig

from .sandbox import run_tests
from .tasks import DEFAULT_TASKS_DIR, MANIFEST_NAME, extract_code, load_tasks, verify_manifest


TIERS = ("recall", "transfer", "explore")


def summarize(rows: list[dict]) -> str:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["tier"]].append(r)
        groups["ALL"].append(r)
    lines = [f"{'tier':<10}{'runs':>5}{'behavior%':>11}{'all%':>7}{'tools/run':>11}{'out_tok':>9}{'sec/run':>9}{'errors':>8}{'forced%':>9}"]
    for tier in [t for t in TIERS if t in groups] + ["ALL"]:
        g = groups[tier]
        n = len(g)
        lines.append(
            f"{tier:<10}{n:>5}{100 * sum(r['pass_behavior'] for r in g) / n:>11.1f}"
            f"{100 * sum(r['pass_all'] for r in g) / n:>7.1f}"
            f"{sum(len(r['tool_calls']) for r in g) / n:>11.2f}"
            f"{sum(r['output_tokens'] for r in g) / n:>9.0f}"
            f"{sum(r['duration'] for r in g) / n:>9.1f}"
            f"{sum(bool(r['error']) for r in g):>8}"
            f"{100 * sum(bool(r.get('forced_final')) for r in g) / n:>9.1f}"
        )
    return "\n".join(lines)


def compare(paths: list[Path]) -> str:
    """One table per tier across several result files (label = the run's --label), plus a per-task matrix."""
    runs: dict[str, list[dict]] = {}
    for path in paths:
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if rows:
            runs.setdefault(rows[0]["label"], []).extend(rows)
    out = [f"{'label':<14}{'tier':<10}{'runs':>5}{'all%':>7}{'behav%':>8}{'tools/run':>11}{'out_tok':>9}{'sec/run':>9}{'errors':>8}{'forced%':>9}"]
    for tier in TIERS + ("ALL",):
        for label, rows in runs.items():
            g = [r for r in rows if tier == "ALL" or r["tier"] == tier]
            if not g:
                continue
            n = len(g)
            out.append(
                f"{label:<14}{tier:<10}{n:>5}{100 * sum(r['pass_all'] for r in g) / n:>7.1f}"
                f"{100 * sum(r['pass_behavior'] for r in g) / n:>8.1f}"
                f"{sum(len(r['tool_calls']) for r in g) / n:>11.2f}"
                f"{sum(r['output_tokens'] for r in g) / n:>9.0f}"
                f"{sum(r['duration'] for r in g) / n:>9.1f}"
                f"{sum(bool(r['error']) for r in g):>8}"
                f"{100 * sum(bool(r.get('forced_final')) for r in g) / n:>9.1f}"
            )
        out.append("")
    tasks = sorted({r["task_id"] for rows in runs.values() for r in rows})
    out.append(f"{'task (pass_all k/n)':<32}" + "".join(f"{label:>14}" for label in runs))
    for t in tasks:
        cells = []
        for rows in runs.values():
            g = [r for r in rows if r["task_id"] == t]
            cells.append(f"{sum(r['pass_all'] for r in g)}/{len(g)}" if g else "-")
        out.append(f"{t:<32}" + "".join(f"{c:>14}" for c in cells))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", choices=("fair", "exploratory"), default="fair",
                    help="fair (default): information-equalized arms, see PROTOCOL.md; exploratory: the superseded first-round arms")
    ap.add_argument("--arm", choices=sorted(set(fair.ARMS) | set(exploratory.ARMS)),
                    help="fair: plain | rag | rag-repo | agent-rag | agent-rag-repo | agent-scratch;  "
                         "exploratory: plain | rag | rag-repo | agent-book | agent")
    ap.add_argument("--label", default=None, help="name for the run (e.g. base, finetuned); default = arm")
    ap.add_argument("--samples", type=int, default=1, help="samples per task (for pass@1 averaging)")
    ap.add_argument("--tasks", nargs="*", help="task id prefixes (default: all)")
    ap.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    ap.add_argument("--base-url", default=ModelConfig.base_url)
    ap.add_argument("--request-timeout", type=float, default=ModelConfig.timeout, help="timeout in seconds of one model request (raise it when thinking is on)")
    ap.add_argument("--model-id", default=ModelConfig.model_id, help="model name sent to the server and recorded in every result row")
    ap.add_argument("--temperature", type=float, default=ModelConfig.temperature)
    ap.add_argument("--max-tokens", type=int, default=ModelConfig.max_tokens)
    ap.add_argument("--thinking", action="store_true", help="enable the model's <think> mode (slow; raise --max-tokens)")
    ap.add_argument("--artifacts", type=Path, default=Path("data/rag_artifacts"))
    ap.add_argument("--repo", type=Path, default=Path("git_repo/PyTorchStepByStep"))
    ap.add_argument("--context-k", "--rag-top-k", dest="context_k", type=int, default=3,
                    help="number of pre-retrieved book passages (one-shot arms and the agents' initial context)")
    ap.add_argument("--max-tool-calls", type=int, default=8)
    ap.add_argument("--force-final", action="store_true",
                    help="exploratory protocol only (the fair protocol always finalizes every arm the same way)")
    ap.add_argument("--run-timeout", type=float, default=240.0, help="wall-clock cap per run in seconds (a timed-out run counts as a failure)")
    ap.add_argument("--out", type=Path, default=Path("data/results"))
    ap.add_argument("--summarize", type=Path, nargs="+",
                    help="only print the summary of existing results .jsonl file(s); several files = comparison table")
    args = ap.parse_args()

    if args.summarize:
        if len(args.summarize) == 1:
            print(summarize([json.loads(l) for l in args.summarize[0].read_text().splitlines() if l.strip()]))
        else:
            print(compare(args.summarize))
        return
    if not args.arm:
        ap.error("--arm is required unless --summarize is used")

    cfg = ModelConfig(base_url=args.base_url, model_id=args.model_id, temperature=args.temperature, max_tokens=args.max_tokens, thinking=args.thinking, timeout=args.request_timeout)
    if args.protocol == "exploratory":
        if args.arm not in exploratory.ARMS:
            ap.error(f"--arm {args.arm} is not an exploratory arm {exploratory.ARMS}")
        arm = exploratory.make_arm(args.arm, cfg, artifacts_dir=args.artifacts, repo_dir=args.repo,
                                   rag_top_k=args.context_k, max_tool_calls=args.max_tool_calls,
                                   run_timeout=args.run_timeout, force_final=args.force_final)
    else:
        if args.arm not in fair.ARMS:
            ap.error(f"--arm {args.arm} is not a fair-protocol arm {fair.ARMS}")
        if args.force_final:
            ap.error("--force-final only applies to --protocol exploratory")
        arm = fair.make_arm(args.arm, cfg, artifacts_dir=args.artifacts, repo_dir=args.repo,
                            context_k=args.context_k, max_tool_calls=args.max_tool_calls, run_timeout=args.run_timeout)
    if (args.tasks_dir / MANIFEST_NAME).exists():   # a frozen task set (the test set) must be untouched
        diff = verify_manifest(args.tasks_dir)
        if diff:
            raise SystemExit("Frozen task set was modified since MANIFEST.sha256 was written:\n  " + "\n  ".join(diff))
    tasks = load_tasks(args.tasks_dir, args.tasks or None)
    label = args.label or args.arm
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{label}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"

    rows = []
    with open(path, "w", encoding="utf-8") as f:
        for task in tasks:
            for s in range(args.samples):
                res = arm.run(task.spec, task.prompt)
                code = extract_code(res.text)
                outcome = run_tests(task, code)
                row = {
                    "label": label, "arm": args.arm, "model": args.model_id, "thinking": args.thinking, "max_tokens": args.max_tokens, "task_id": task.id, "tier": task.tier, "sample": s,
                    "response": res.text, "code": code, **outcome.to_dict(),
                    "tool_calls": res.tool_calls, "input_tokens": res.input_tokens,
                    "output_tokens": res.output_tokens, "duration": res.duration, "error": res.error,
                    "forced_final": res.forced_final,
                }
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{task.id:<30} s{s} behavior={outcome.pass_behavior!s:<5} all={outcome.pass_all!s:<5} "
                      f"tools={len(res.tool_calls)} {res.duration:5.1f}s"
                      + (f"  ERR {res.error[:80]}" if res.error else ""), flush=True)
    print("\n" + summarize(rows) + f"\n\nResults: {path}")


if __name__ == "__main__":
    main()
