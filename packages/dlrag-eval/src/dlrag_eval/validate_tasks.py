"""`dlrag-tasks validate`: references must pass every test, an empty stub must fail."""
from __future__ import annotations

import argparse
from pathlib import Path

from .sandbox import run_tests
from .tasks import DEFAULT_TASKS_DIR, load_tasks, verify_manifest, write_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    ap.add_argument("ids", nargs="*", help="task id prefixes (default: all)")
    ap.add_argument("--write-manifest", action="store_true", help="freeze the directory: write MANIFEST.sha256 (after validating)")
    ap.add_argument("--verify-manifest", action="store_true", help="check the directory against its MANIFEST.sha256")
    args = ap.parse_args()
    if args.verify_manifest:
        diff = verify_manifest(args.tasks_dir)
        print("manifest OK: task directory untouched since it was frozen" if not diff else "MANIFEST MISMATCH:\n  " + "\n  ".join(diff))
        raise SystemExit(1 if diff else 0)

    bad = 0
    for t in load_tasks(args.tasks_dir, args.ids or None):
        ref = run_tests(t, t.reference)
        stub = run_tests(t, "# empty\n")
        n_beh = sum(k.startswith("test_behavior") for k in ref.passed)
        n_conv = len(ref.passed) - n_beh
        ok = ref.pass_all and not stub.pass_all and n_beh >= 1 and n_conv >= 1
        bad += not ok
        failing = [k for k, v in ref.passed.items() if not v]
        print(f"{'OK  ' if ok else 'FAIL'} {t.id:<28} tests={len(ref.passed)} (behavior {n_beh}/convention {n_conv})"
              + (f" ref-failing={failing}" if failing else "")
              + (f" ref-error={ref.error}" if ref.error else "")
              + (" stub-passes!" if stub.pass_all else ""))
    if args.write_manifest and not bad:
        print("manifest written, sha256 of the manifest:", write_manifest(args.tasks_dir))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
