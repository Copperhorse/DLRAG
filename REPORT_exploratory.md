> **Superseded.** Exploratory rounds (2026-10-01/02) with known asymmetries between the arms; see `REPORT.md` and `PROTOCOL.md` for the fair re-run. Kept for the record.

# RAG vs agentic RAG on book-derived coding tasks

Status: second full run, 2026-10-02. One model (Sharp-MiniCPM5-2B Q8), 32 tasks, 5 samples per task per arm.
The first run (2026-10-01, four arms, 24 tasks) is described under "How the study changed".

## Question

Does a small coding model do better on tasks that depend on one book's conventions when it
(a) has no retrieval, (b) gets the top book passages pasted in once (one-shot RAG), or (c) can search the book and the
book's code repository itself through tool calls (agentic RAG)? The original plan also had a QLoRA fine-tuned arm; it is
shelved (see the end).

## Setup

- **Model:** `Sharp-MiniCPM5-2B-Q8_K_M.gguf` (a quantization of `openbmb/MiniCPM5-2B`), served by llama.cpp `llama-server`
  with `--jinja`, 32K context. Thinking mode off; the model card's sampling (temperature 1.0, top-p 0.95, top-k 20); 3,072 max new tokens.
- **Corpus:** the PDF of *Deep Learning with PyTorch Step-by-Step* converted with Docling (code enrichment on, 1,187 pages),
  3,058 chunks, hybrid BM25 + dense (fastembed MiniLM) retrieval fused with RRF. Companion repo `PyTorchStepByStep` for the repo tools.
- **Tasks:** `data/tasks`, 32 execution-graded Python tasks. The prompt states behaviour only; hidden tests check
  `behavior` (works on synthetic data) and `convention` (the book's exact names, checkpoint keys, attribute names, call order).
  `pass_behavior` = all behaviour tests pass; `pass_all` = every test passes. All reference solutions pass and an empty stub fails.
  - 18 `recall`: re-derive something the book/repo contains. 6 `transfer`: variations on it. (Together: "the 24 original tasks".)
  - 8 `explore` (added after the first run): use the repo's own library (`StepByStep`, `lr_range_test`, `attach_hooks`, `make_balanced_sampler`,
    `index_splitter`, `loader_apply`/`correct`, clipping + scheduler, `make_normalizer`) to do something. The prompt never names the classes or
    methods, so the model has to find them. The tests import the repo code (it is put on `PYTHONPATH`).
- **Arms** (`packages/dlrag-agent`, built on Agno):

  | arm | what the model gets |
  |---|---|
  | `plain` | nothing |
  | `rag` | top-3 book chunks, retrieved with the task description, pasted into the prompt |
  | `rag-repo` | the same, plus a compact map of the repo's Python code (file paths, classes, method signatures; only the latest `vN.py` of each module) |
  | `agent-book` | Agno agent with `book_search`, `book_get_chunk` |
  | `agent` | the same plus `repo_list`, `repo_grep`, `repo_read_file` on the companion repo |
  | `agent-book-ff`, `agent-ff` | as above, with a **forced final answer** (below) |

  Agent runs are limited to 8 tool calls: further calls get a "limit reached, answer now" reply, and after 3 more attempts the run is aborted.
  Each agent run also has a 240 s wall-clock cap (never triggered in the final runs). **Forced final answer:** if an agent run ends without any
  code block, one more tool-free request is made containing the task and everything the tools returned, asking for the final code block.
- **Scoring:** the last fenced Python block of the reply is run against the task's tests in a subprocess. A reply without a code block counts as a failure.

## Results

### The 24 original tasks (120 runs per arm)

| arm | runs | all tests pass | behaviour pass | tool calls/run | out tokens/run | sec/run | forced answer |
|---|---|---|---|---|---|---|---|
| `plain` | 120 | 5.0% | 32.5% | 0.0 | 205 | 15.3 | 0% |
| `rag` | 120 | 49.2% | 56.7% | 0.0 | 214 | 4.3 | 0% |
| `rag-repo` | 120 | 50.8% | 59.2% | 0.0 | 195 | 4.5 | 0% |
| `agent-book` | 120 | 28.3% | 35.8% | 9.4 | 497 | 12.1 | 0% |
| `agent-book-ff` | 120 | 56.7% | 60.8% | 9.0 | 635 | 15.4 | 38% |
| `agent` | 120 | 39.2% | 40.8% | 9.6 | 688 | 17.6 | 0% |
| `agent-ff` | 120 | 54.2% | 62.5% | 9.3 | 838 | 21.9 | 38% |

By tier (`pass_all`, includes the explore tier):

| arm | recall (90 runs) | transfer (30 runs) | explore (40 runs) |
|---|---|---|---|
| `plain` | 6.7% | 0.0% | 0.0% |
| `rag` | 56.7% | 26.7% | 0.0% |
| `rag-repo` | 63.3% | 13.3% | 17.5% |
| `agent-book` | 32.2% | 16.7% | 7.5% |
| `agent-book-ff` | 63.3% | 36.7% | 10.0% |
| `agent` | 44.4% | 23.3% | 2.5% |
| `agent-ff` | 63.3% | 26.7% | 15.0% |

**Uncertainty.** The 5 samples of one task are not independent, so the effective sample size is closer to 24 than 120.
95% bootstrap intervals resample the tasks:

| arm | all tests pass | 95% CI (resampling the 24 tasks) |
|---|---|---|
| `plain` | 5.0% | 0.8 - 10.8 |
| `rag` | 49.2% | 34.2 - 65.0 |
| `rag-repo` | 50.8% | 33.3 - 68.3 |
| `agent-book` | 28.3% | 20.0 - 37.5 |
| `agent-book-ff` | 56.7% | 45.0 - 68.3 |
| `agent` | 39.2% | 26.7 - 51.7 |
| `agent-ff` | 54.2% | 40.0 - 68.3 |

| difference (paired by task) | points | 95% CI |
|---|---|---|
| `rag` - `plain` | +44.2 | +30.0 to +58.3 |
| `rag-repo` - `rag` | +1.7 | -9.2 to +14.2 |
| `agent-book-ff` - `rag` | +7.5 | -4.2 to +19.2 |
| `agent-ff` - `rag` | +5.0 | -10.0 to +20.8 |
| `agent-ff` - `rag-repo` | +3.3 | -10.0 to +18.3 |
| `agent-ff` - `agent-book-ff` | -2.5 | -12.5 to +7.5 |
| `agent-book-ff` - `agent-book` | +28.3 | +16.7 to +39.2 |
| `agent-ff` - `agent` | +15.0 | +3.3 to +25.8 |

Only "retrieval beats no retrieval" and "the forced final answer rescues the agents" are clearly separated from zero. All other gaps between the
good arms (`rag`, `rag-repo`, `agent-book-ff`, `agent-ff`) are within the noise.

### The 8 explore tasks (40 runs per arm)

| arm | runs | all tests pass | behaviour pass | tool calls/run | out tokens/run | sec/run | forced answer |
|---|---|---|---|---|---|---|---|
| `plain` | 40 | 0.0% | 0.0% | 0.0 | 384 | 7.7 | 0% |
| `rag` | 40 | 0.0% | 0.0% | 0.0 | 378 | 7.7 | 0% |
| `rag-repo` | 40 | 17.5% | 17.5% | 0.0 | 337 | 7.6 | 0% |
| `agent-book` | 40 | 7.5% | 7.5% | 11.6 | 450 | 11.0 | 0% |
| `agent-book-ff` | 40 | 10.0% | 10.0% | 11.4 | 697 | 17.9 | 78% |
| `agent` | 40 | 2.5% | 7.5% | 11.6 | 421 | 11.8 | 0% |
| `agent-ff` | 40 | 15.0% | 15.0% | 11.8 | 886 | 23.8 | 85% |

Per task (`pass_all`, k of 5 samples):

| task | `plain` | `rag` | `rag-repo` | `agent-book` | `agent-book-ff` | `agent` | `agent-ff` |
|---|---|---|---|---|---|---|---|
| x01_train_with_library | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| x02_suggest_lr | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| x03_layer_output_shapes | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| x04_balanced_loader | 0 | 0 | 5 | 0 | 2 | 0 | 2 |
| x05_split_loaders | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| x06_accuracy_per_class | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| x07_train_with_tricks | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| x08_image_normalizer | 0 | 0 | 1 | 2 | 0 | 0 | 1 |

### The 24 original tasks, per task (`pass_all`, k of 5)

| task | `plain` | `rag` | `rag-repo` | `agent-book` | `agent-book-ff` | `agent` | `agent-ff` |
|---|---|---|---|---|---|---|---|
| r01_train_step_fn | 0 | 5 | 5 | 1 | 4 | 5 | 2 |
| r02_val_step_fn | 2 | 5 | 5 | 1 | 4 | 4 | 5 |
| r03_mini_batch | 2 | 5 | 5 | 2 | 4 | 2 | 5 |
| r04_index_splitter | 0 | 2 | 3 | 1 | 3 | 2 | 3 |
| r05_balanced_sampler | 0 | 1 | 5 | 2 | 4 | 4 | 5 |
| r06_train_loop | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| r07_checkpoint | 0 | 5 | 5 | 4 | 4 | 2 | 4 |
| r08_predict | 0 | 5 | 5 | 2 | 4 | 0 | 2 |
| r09_loader_apply | 0 | 5 | 4 | 0 | 3 | 1 | 3 |
| r10_clipping | 1 | 5 | 5 | 2 | 5 | 5 | 4 |
| r11_set_lr_scheduler | 0 | 2 | 4 | 2 | 4 | 3 | 4 |
| r12_epoch_schedulers | 0 | 2 | 0 | 1 | 2 | 0 | 0 |
| r13_batch_schedulers | 0 | 1 | 2 | 1 | 1 | 1 | 3 |
| r14_make_lr_fn | 1 | 2 | 0 | 1 | 3 | 2 | 5 |
| r15_correct | 0 | 0 | 0 | 0 | 3 | 1 | 1 |
| r16_make_data_split | 0 | 0 | 0 | 4 | 2 | 3 | 4 |
| r17_count_parameters | 0 | 2 | 5 | 3 | 5 | 3 | 5 |
| r18_build_loaders | 0 | 4 | 4 | 2 | 2 | 2 | 2 |
| t01_grad_accumulation_step | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| t02_clipped_train_step | 0 | 4 | 3 | 1 | 4 | 3 | 4 |
| t03_checkpoint_with_scheduler | 0 | 2 | 0 | 2 | 3 | 0 | 1 |
| t04_predict_proba | 0 | 2 | 1 | 1 | 3 | 2 | 3 |
| t05_train_early_stopping | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| t06_evaluate | 0 | 0 | 0 | 1 | 1 | 2 | 0 |

Tasks that no arm solves: `r06_train_loop`, `t01_grad_accumulation_step`, `t05_train_early_stopping`, `x02_suggest_lr`, `x03_layer_output_shapes`.

## What the numbers say

**1. Retrieval matters a lot.** Without it the model writes working but unconventional code (`plain`: 32.5% behaviour pass, 5.0% full pass).
Pasting in 3 book chunks lifts the full pass to 49.2%.

**2. The first run's "agents lose" was a finishing problem.** Without a forced answer, 45-48% of agent runs ended with no code block at all: the 2B model keeps calling tools
until the run is aborted. With the forced final answer both agents rise by 15-28 points (`agent-book` 28.3% -> 56.7%, `agent` 39.2% -> 54.2%).
Runs that needed the forced answer (46 of 120) passed 59% (`agent-book-ff`) and 54% (`agent-ff`), the same as runs that did not
(55% and 54%), so the forced step is not a lower-quality shortcut.

**3. On the original tasks, agentic and one-shot RAG are level.** `agent-book-ff` 56.7%, `agent-ff` 54.2%, `rag-repo` 50.8%, `rag` 49.2%; every pairwise gap is inside the confidence
intervals. The agents cost 3-5x more time per run (15-22 s against 4-5 s) and about 3-4x the output tokens. The intuition "an agent can fetch text whenever it needs it, so it should win"
is **not confirmed and not refuted** here: these tasks mostly need one lookup, and the task description is already a good search query for one-shot RAG.

**4. Repo access helps only when the task needs the repo.** On the original tasks it adds nothing once the finishing problem is fixed
(`agent-ff` vs `agent-book-ff`: -2.5 points). On the explore tasks the arms without repo information score 0-10%, and the two repo-aware arms that reliably produce an answer score 15-17.5%
(`rag-repo` 17.5%, `agent-ff` 15.0%); `agent` without the forced answer scores 2.5% because most of its runs never answered. All explore numbers are small (40 runs per arm, 6-7 passes), and `rag-repo` vs `agent-ff` is not a meaningful difference.

**5. Why the explore tasks stay hard.** Failures by cause (explore tier, 40 runs each; import failure = the code did not even import the right modules):

| arm | import / collection failures | imported `stepbystep.v0`-`v3` | imported `stepbystep.v4` |
|---|---|---|---|
| `rag` | 29 | 0 | 0 |
| `rag-repo` | 8 | 0 | 23 |
| `agent-book-ff` | 23 | 6 | 2 |
| `agent-ff` | 15 | 11 | 3 |

Without repo information the model invents module names (`step_by_step`, `stepbystep.model_training.v1`, ...). The repo keeps five versions of the same class (`stepbystep/v0.py` ... `v4.py`);
`agent-ff` often reads and imports an old version that lacks the needed methods, while `rag-repo` imported the latest `v4` in 23 of 40 runs.

## Threats to validity

- Small: 24 + 8 tasks, 5 correlated samples each; one model; sampling at temperature 1.0.
- **The `rag-repo` map was curated by me** (latest version of each module only), while the agent saw the raw repo with all five versions. That favours `rag-repo` on the explore tasks and is the main confound of that comparison.
- The one-shot RAG query is the clean task description, while the agent writes its own queries (favours one-shot RAG).
- **The forced final answer is a harness design choice** that one-shot arms do not need. It is defensible (a budgeted agent must answer) but it changes the agent results a lot, so other budget designs (cap, notice, grace) could move them too. None were tuned.
- `explore` tasks were written after seeing the first results, by the same author as the harness; their prompts, tests and the `rag-repo` map all encode my idea of what the library API is.
- The book and its repo are probably in the model's pretraining data; `plain` still scored 5%, so recall of the exact conventions is weak.
- Two `explore` tasks (`x02`, `x03`) and three original ones are unsolved by every arm; I did not check whether their tests are too strict (`r06` was checked: the generated code is wrong).

## How the study changed

1. **First run (2026-10-01):** four arms (`plain`, `rag`, `agent-book`, `agent`), 24 tasks. Result: `rag` 49.2% > `agent` 39.2% > `agent-book` 28.3% > `plain` 5.0%. About 45% of agent runs produced no code.
2. A first `agent-book` attempt stalled for 94 minutes on one run (a tool-calling loop). It was killed, its partial results moved to `data/results/incomplete/`, a per-run wall-clock cap added, and both agent arms re-run.
3. **Forced final answer** added (`--force-final`) and both agent arms re-run on the 24 tasks (labels `*-ff`). `plain`, `rag` and the first-run agent files were kept.
4. **Explore tasks** added (they needed `matplotlib` and `tensorboard` as dependencies because the repo's library imports them, and a `needs_repo` flag in `task.json` to put the repo on `PYTHONPATH`).
   First results showed that without repo information the explore tasks are unsolvable by construction, so the `rag-repo` arm was added for a fair one-shot counterpart.
5. Docling initially ran OCR on the CPU (only the CPU onnxruntime was installed); switching to `onnxruntime-gpu` with `preload_dlls()` fixed it. Chunk headers polluted by notebook comment lines
   and flattened code listings were fixed before these runs (see `CLAUDE.md`).
6. **Fine-tune arm shelved:** the QLoRA dataset's 577 QA pairs are template prompts and the 110 method answers have broken indentation. The code is kept in `finetune/` and was never run.

## Reproduce

```bash
uv sync --all-packages
~/Git/llama.cpp-spark/build/bin/llama-server -m ~/models/Sharp-MiniCPM5-2B-Q8_K_M.gguf --jinja -c 32768 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 --port 8080
for arm in plain rag rag-repo agent-book agent; do uv run dlrag-tasks --arm $arm --label $arm --samples 5 --out data/results; done
for arm in agent-book agent; do uv run dlrag-tasks --arm $arm --force-final --label $arm-ff --samples 5 --out data/results; done
uv run dlrag-tasks --summarize $(ls data/results/*.jsonl | tr '\n' ' ')     # comparison table + per-task matrix, all labels merged
```

Result files: `data/results/<label>-<timestamp>.jsonl` (one row per run: response, extracted code, per-test results, tool calls, tokens, time, `forced_final`).
The first runs of the arms on the original 24 tasks and the later runs on the explore tasks are separate files with the same label; the summary merges them by label.
(Result files written before the forced-final feature have no `forced_final` field; it counts as false.)

## Suggested next steps

1. Give the agent the same information as the one-shot arm, or the reverse, so the explore comparison is not confounded by my curated repo map
   (for example, tell the agent that the highest-numbered `vN.py` is the latest, or give `rag-repo` the unfiltered map).
2. Check why `x02`, `x03`, `t01`, `t05` are unsolved (tests too strict, or just too hard for a 2B model).
3. Try a larger coding model from `~/models` (e.g. OxCoder-9B) to see whether the agent-vs-one-shot ranking depends on model size.
4. Tasks that really need several lookups (find where something is defined, then what it depends on) would be a better test of agency than the current mostly single-lookup tasks.
