"""Extract the chat template embedded in a GGUF so training uses exactly what llama-server uses.

    uv run --with gguf python finetune/extract_chat_template.py ~/models/Sharp-MiniCPM5-2B-Q8_K_M.gguf
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gguf import GGUFReader


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("chat_template.jinja"))
    args = ap.parse_args()

    field = GGUFReader(str(args.gguf)).fields.get("tokenizer.chat_template")
    if field is None:
        raise SystemExit("No tokenizer.chat_template in this GGUF.")
    template = bytes(field.parts[field.data[0]]).decode("utf-8")
    args.out.write_text(template, encoding="utf-8")
    print(f"Wrote {len(template)} chars to {args.out}")


if __name__ == "__main__":
    main()
