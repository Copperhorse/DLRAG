"""`dlrag-analyze`: the pre-registered analysis of PROTOCOL.md, generated from the result files (no hand-copied numbers).

Primary: pass_all of agent-rag-repo minus rag-repo, per task averages paired by task, 95% bootstrap CI over tasks.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .tasks import DEFAULT_TASKS_DIR, load_tasks, verify_manifest

ARMS = ["plain", "rag", "rag-repo", "agent-rag", "agent-rag-repo", "agent-scratch"]
PAIRS = [  # (a, b, role)
    ("agent-rag-repo", "rag-repo", "PRIMARY"),
    ("agent-rag", "rag", "secondary"),
    ("agent-rag-repo", "agent-rag", "secondary (repo access for agents)"),
    ("rag-repo", "rag", "secondary (repo map for one-shot)"),
    ("agent-scratch", "rag-repo", "secondary"),
    ("agent-scratch", "agent-rag-repo", "secondary (pre-retrieval for agents)"),
    ("rag", "plain", "vs plain"), ("rag-repo", "plain", "vs plain"), ("agent-rag", "plain", "vs plain"),
    ("agent-rag-repo", "plain", "vs plain"), ("agent-scratch", "plain", "vs plain"),
]


def load_arm(results: Path, arm: str) -> list[dict]:
    files = sorted(results.glob(f"test-{arm}-2*.jsonl"))
    if not files:
        raise SystemExit(f"no test results for {arm} in {results}")
    return [json.loads(l) for l in files[-1].read_text().splitlines() if l.strip()]


def per_task(rows: list[dict], key: str = "pass_all") -> dict[str, float]:
    d = defaultdict(list)
    for r in rows:
        d[r["task_id"]].append(float(bool(r[key])))
    return {t: sum(v) / len(v) for t, v in d.items()}


def bootstrap(fn, tasks: list[str], n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    xs = sorted(fn([rng.choice(tasks) for _ in tasks]) for _ in range(n))
    return xs[int(0.025 * n)], xs[int(0.975 * n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("data/results/test"))
    ap.add_argument("--tasks-dir", type=Path, default=Path("data/tasks_test"))
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--arms", nargs="+", default=None, help="restrict to these arms (default: all six)")
    ap.add_argument("--compare-with", type=Path, default=None,
                    help="results dir of another model run on the same tasks/arms: adds paired (this - other) differences per arm")
    ap.add_argument("--other-samples", type=int, default=None, help="samples per task in the compared run (default: same as --samples)")
    ap.add_argument("--this", default="this model", help="name of this model in the cross-model table")
    ap.add_argument("--other", default="other model", help="name of the other model in the cross-model table")
    args = ap.parse_args()
    global ARMS, PAIRS
    if args.arms:
        ARMS = [a for a in ARMS if a in args.arms]
        PAIRS = [(a, b, r) for a, b, r in PAIRS if a in ARMS and b in ARMS]

    diff = verify_manifest(args.tasks_dir)
    print(f"Test set manifest: {'OK (untouched since frozen)' if not diff else 'MISMATCH ' + str(diff)}\n")
    tasks_meta = {t.id: t for t in load_tasks(args.tasks_dir)}
    ids = sorted(tasks_meta)
    data = {a: load_arm(args.results, a) for a in ARMS}
    for a, rows in data.items():
        expected = len(ids) * args.samples
        assert len(rows) == expected, f"{a}: {len(rows)} rows, expected {expected}"
    PT = {a: per_task(rows) for a, rows in data.items()}
    PB = {a: per_task(rows, "pass_behavior") for a, rows in data.items()}

    print("### Main table (24 test tasks x 5 samples)\n")
    print("| arm | runs | all tests pass | 95% CI (over tasks) | behaviour pass | tokens/run | sec/run | tool calls/run | needed 2nd request | errors |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        rows = data[a]; n = len(rows)
        lo, hi = bootstrap(lambda s, a=a: sum(PT[a][t] for t in s) / len(s), ids)
        print(f"| `{a}` | {n} | {100 * sum(PT[a].values()) / len(ids):.1f}% | {100 * lo:.1f} - {100 * hi:.1f} | {100 * sum(PB[a].values()) / len(ids):.1f}% | "
              f"{sum(r['input_tokens'] + r['output_tokens'] for r in rows) / n:.0f} | {sum(r['duration'] for r in rows) / n:.1f} | "
              f"{sum(len(r['tool_calls']) for r in rows) / n:.2f} | {100 * sum(bool(r.get('forced_final')) for r in rows) / n:.0f}% | {sum(bool(r['error']) for r in rows)} |")

    print("\n### By tier (all tests pass)\n")
    print("| arm | recall (8 tasks) | transfer (8) | explore (8) |\n|---|---|---|---|")
    for a in ARMS:
        cells = []
        for tier in ("recall", "transfer", "explore"):
            ts = [t for t in ids if tasks_meta[t].tier == tier]
            cells.append(f"{100 * sum(PT[a][t] for t in ts) / len(ts):.1f}%")
        print(f"| `{a}` | " + " | ".join(cells) + " |")

    print("\n### Paired differences in pass_all (percentage points, 95% bootstrap CI over the 24 tasks)\n")
    print("| difference | role | points | 95% CI | excludes 0? |\n|---|---|---|---|---|")
    for a, b, role in PAIRS:
        f = lambda s, a=a, b=b: sum(PT[a][t] - PT[b][t] for t in s) / len(s)
        lo, hi = bootstrap(f, ids)
        print(f"| `{a}` - `{b}` | {role} | {100 * f(ids):+.1f} | {100 * lo:+.1f} to {100 * hi:+.1f} | {'yes' if lo > 0 or hi < 0 else 'no'} |")

    print("\n### Per task (all tests pass, k of 5 samples)\n")
    print("| task | tier | " + " | ".join(f"`{a}`" for a in ARMS) + " |\n|---|---|" + "---|" * len(ARMS))
    for t in ids:
        print(f"| {t} | {tasks_meta[t].tier} | " + " | ".join(str(round(PT[a][t] * args.samples)) for a in ARMS) + " |")
    unsolved = [t for t in ids if all(PT[a][t] == 0 for a in ARMS)]
    print(f"\nUnsolved by every arm: {', '.join(unsolved) or 'none'}")

    print("\n### Failure anatomy (share of runs)\n")
    print("| arm | no code block | import/collection failure | behaviour ok, a convention wrong | all pass |\n|---|---|---|---|---|")
    for a in ARMS:
        rows = data[a]; n = len(rows)
        nocode = sum(not r["code"] for r in rows); imp = sum(1 for r in rows if r["code"] and not r["passed"])
        conv = sum(1 for r in rows if r["pass_behavior"] and not r["pass_all"])
        print(f"| `{a}` | {100 * nocode / n:.1f}% | {100 * imp / n:.1f}% | {100 * conv / n:.1f}% | {100 * sum(bool(r['pass_all']) for r in rows) / n:.1f}% |")

    thinking = sum(1 for rows in data.values() for r in rows if "<think" in (r["response"] or "") or "</think" in (r["response"] or ""))
    print(f"\nThinking tags found in replies: {thinking} of {sum(len(v) for v in data.values())} runs (must be 0: thinking is returned separately, never in the reply text)")

    agent_arms = [a for a in ("agent-rag", "agent-rag-repo", "agent-scratch") if a in data]
    if agent_arms:
        print("\n### Tool use (agent arms)\n")
    for a in agent_arms:
        rows = data[a]; calls = [len(r["tool_calls"]) for r in rows]
        names = defaultdict(int)
        for r in rows:
            for t in r["tool_calls"]:
                names[t["name"]] += 1
        print(f"- `{a}`: {sum(calls) / len(calls):.2f} calls/run, {100 * sum(c == 0 for c in calls) / len(calls):.0f}% of runs used no tool, calls by tool {dict(names)}")


    if args.compare_with is not None:
        other = {a: load_arm(args.compare_with, a) for a in ARMS}
        for a, rows in other.items():
            assert len(rows) == len(ids) * (args.other_samples or args.samples), f"{a} (other): {len(rows)} rows"
        OT = {a: per_task(rows) for a, rows in other.items()}
        OB = {a: per_task(rows, "pass_behavior") for a, rows in other.items()}
        print(f"\n### Comparison: {args.this} minus {args.other}, same tasks and arms (percentage points, paired by task, 95% bootstrap CI over the 24 tasks)\n")
        print(f"| arm | {args.this} all pass | {args.other} all pass | difference (all pass) | 95% CI | excludes 0? | {args.this} behaviour | {args.other} behaviour | difference (behaviour) | 95% CI |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for a in ARMS:
            f = lambda s, a=a: sum(PT[a][t] - OT[a][t] for t in s) / len(s)
            g = lambda s, a=a: sum(PB[a][t] - OB[a][t] for t in s) / len(s)
            lo, hi = bootstrap(f, ids); lo2, hi2 = bootstrap(g, ids)
            print(f"| `{a}` | {100 * sum(PT[a].values()) / len(ids):.1f}% | {100 * sum(OT[a].values()) / len(ids):.1f}% | {100 * f(ids):+.1f} | {100 * lo:+.1f} to {100 * hi:+.1f} | {'yes' if lo > 0 or hi < 0 else 'no'} | "
                  f"{100 * sum(PB[a].values()) / len(ids):.1f}% | {100 * sum(OB[a].values()) / len(ids):.1f}% | {100 * g(ids):+.1f} | {100 * lo2:+.1f} to {100 * hi2:+.1f} |")
        print(f"\nTokens and seconds per run, {args.this} vs {args.other}:\n")
        print("| arm | tokens (this) | tokens (other) | sec/run (this) | sec/run (other) |\n|---|---|---|---|---|")
        for a in ARMS:
            t1 = sum(r["input_tokens"] + r["output_tokens"] for r in data[a]) / len(data[a]); t2 = sum(r["input_tokens"] + r["output_tokens"] for r in other[a]) / len(other[a])
            d1 = sum(r["duration"] for r in data[a]) / len(data[a]); d2 = sum(r["duration"] for r in other[a]) / len(other[a])
            print(f"| `{a}` | {t1:.0f} | {t2:.0f} | {d1:.1f} | {d2:.1f} |")


if __name__ == "__main__":
    main()
