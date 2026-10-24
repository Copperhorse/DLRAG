"""`dlrag-tune`: pick a setting from dev runs with the rule written in PROTOCOL.md.

Rule: highest dev `pass_all` (mean over runs); an exact tie goes to the smaller setting (the number after `--param` in the label,
e.g. `dev-rag-repo-k6` -> 6). No human discretion: the selected label is printed with the full table.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load(results_dir: Path, label: str) -> list[dict]:
    rows = []
    for f in sorted(Path(results_dir).glob(f"{label}-2*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return rows


def setting(label: str) -> int:
    return int(re.search(r"(\d+)$", label).group(1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("labels", nargs="+", help="dev run labels, each ending in the setting value (e.g. dev-rag-repo-k3)")
    ap.add_argument("--results", type=Path, default=Path("data/results/dev"))
    args = ap.parse_args()

    table = []
    for label in args.labels:
        rows = load(args.results, label)
        if not rows:
            raise SystemExit(f"no results for {label} in {args.results}")
        n = len(rows)
        table.append(
            {
                "label": label,
                "setting": setting(label),
                "runs": n,
                "pass_all": sum(bool(r["pass_all"]) for r in rows) / n,
                "pass_behavior": sum(bool(r["pass_behavior"]) for r in rows) / n,
                "tokens": sum(r["input_tokens"] + r["output_tokens"] for r in rows) / n,
                "sec": sum(r["duration"] for r in rows) / n,
                "tools": sum(len(r["tool_calls"]) for r in rows) / n,
            }
        )
    print(f"{'label':<26}{'setting':>8}{'runs':>6}{'all%':>8}{'behav%':>8}{'tokens/run':>12}{'sec/run':>9}{'tools':>7}")
    for t in table:
        print(
            f"{t['label']:<26}{t['setting']:>8}{t['runs']:>6}{100 * t['pass_all']:>8.1f}{100 * t['pass_behavior']:>8.1f}{t['tokens']:>12.0f}{t['sec']:>9.1f}{t['tools']:>7.1f}"
        )
    best = max(table, key=lambda t: (round(t["pass_all"], 9), -t["setting"]))
    print(f"\nSELECTED: {best['label']} (setting {best['setting']}, dev pass_all {100 * best['pass_all']:.1f}%)")


if __name__ == "__main__":
    main()
