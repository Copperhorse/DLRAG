> **Shelved.** The fine-tune arm is on hold; the experiment now compares RAG vs agentic RAG. Known blocker: the 577 QA pairs in
> `data/train.jsonl` are template prompts ("Show the implementation of `X` from the companion repository ...") and the 110
> repo *method* answers have broken indentation (method cut out of its class). The raw-doc part of the data is fine.

# QLoRA fine-tuning (run on Colab, not on the dev machine)

Goal: arms **C** (fine-tuned, no retrieval) and **D** (fine-tuned + tools) of the experiment, built from the *same* base the
GGUF arms use: `openbmb/MiniCPM5-2B` (the Sharp-MiniCPM5-2B GGUF is a quantization of it).

```bash
# 1. locally (CPU only): template + data
uv run --with gguf python finetune/extract_chat_template.py ~/models/Sharp-MiniCPM5-2B-Q8_K_M.gguf   # done once; file is committed
uv run python finetune/build_dataset.py --dry-run        # statistics
uv run python finetune/build_dataset.py                  # -> finetune/data/train.jsonl

# 2. Colab: open finetune/colab.ipynb, upload train_qlora.py + chat_template.jinja + data/train.jsonl, run all
#    -> out/gguf/*.gguf (Q8_0) + out/adapter

# 3. back here: copy the GGUF to ~/models, serve it like the base model, then
uv run dlrag-tasks --arm plain --label finetuned          # arm C
uv run dlrag-tasks --arm agent --label finetuned-agent    # arm D
```

## Where the data comes from

Only the book and its repo, nothing from HuggingFace (on purpose: the tuned model should know exactly what the RAG arms can look up):
- the ingested book chunks (`data/rag_artifacts/chunks.jsonl`): raw text docs and "show the code for / explain ..." pairs;
- the companion repo `git_repo/PyTorchStepByStep`: every `.py` file and notebook (code + markdown cells) as docs, plus one
  "show the implementation of `Class.method`" pair per function/method (latest version of each).

`build_dataset.py` skips: chunks under 80 chars, raw cell outputs (header `Output`), duplicate chunks, QA pairs whose prompt is shared with other chunks (e.g. 40 different answers to "Explain: Data Preparation"; their text stays in the raw docs), any code chunk that
Docling flattened to one line, and repo plumbing (`config.py`, `tensorboardserverextension.py`). It strips the printed listing
line numbers (`1 import ...`) from book code. `data/tasks/**` is never read (checked: no task prompt or test text is in the data;
~79% of the `recall` reference-solution lines are, since the repo code is part of the data by design).

What the training does: 4-bit NF4 base + LoRA (r=16, all attention/MLP projections) + TRL `SFTTrainer`,
~1M tokens (book chunks and repo files as raw text, plus mechanical "show the code for ..." pairs rendered with the
Sharp chat template, thinking off), 2 epochs, lr 1e-4. The merged model is exported to GGUF `q8_0`.
`train_qlora.py --no-unsloth` is the fallback if Unsloth does not support the architecture (it prints the llama.cpp
`convert_hf_to_gguf.py` command to produce the GGUF).

## Things that make the comparison imperfect (keep them in mind when reading results)

- **Quantization differs.** Arms A/B use the Sharp GGUF (coding-weighted imatrix, per-tensor policy); the fine-tuned export is a
  plain Q8_0 of the merged model. For a clean baseline, also convert the untouched `openbmb/MiniCPM5-2B` to plain Q8_0
  (`convert_hf_to_gguf.py <dir> --outtype q8_0`) and run arm A on that too.
- **Training is on 4-bit weights, inference on Q8**: QLoRA's adapters learn against NF4-quantized base weights.
- **System prompt**: training QA examples use the template's default system prompt; the Agno arms send their own instructions.
- **Loss is computed on all tokens** (no response-only masking).
- Training data = the same knowledge RAG can reach (book chunks + repo code), which includes the reference solutions of the
  `recall` tasks by design. `data/tasks/**` is never read. The noisy chunk `top_header` is not used.
- Hyper-parameters are untuned defaults; do not tune them on the eval tasks (that would leak the test set).
