"""Print chunks whose section header contains any of the given terms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("terms", nargs="+")
    ap.add_argument("--artifacts", type=Path, default=Path("data/rag_artifacts"))
    args = ap.parse_args()

    terms = [t.lower() for t in args.terms]
    with open(args.artifacts / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            header = chunk.get("section_header", "")
            if any(t in header.lower() for t in terms):
                print(f"Header: {header}")
                print(f"ID: {chunk['chunk_id']}")
                print(f"Content preview: {chunk['content'][:100]}...\n")


if __name__ == "__main__":
    main()
