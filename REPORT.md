# Small model + RAG vs small model + agent: a controlled comparison on book-derived coding tasks

Status: final report of the pre-registered protocol (`PROTOCOL.md`), 2026-10-02/03. Main study: one model (Sharp-MiniCPM5-2B Q8), settings tuned on 32 dev tasks, evaluated once on 24 frozen test tasks, 5 samples per task per arm. A second model (Sharp-Spark-X2.5-4B, thinking off) was added on 2026-10-06 with the same frozen tasks and settings, without the agent arms (see "Second model").
The earlier exploratory rounds, which had asymmetries between the arms, are in `REPORT_exploratory.md` and are superseded.

## Summary

**Motivation.** Can a small model (under ~4B parameters), helped by retrieval from a coding book and its code repository, write good enough code that a bigger model is not needed? The project built a retrieval system over *Deep Learning with PyTorch Step-by-Step* and its repo,
a set of execution-graded coding tasks derived from them, and a harness to compare ways of giving the retrieved knowledge to the model.

**What was tested.** Holding the model fixed (a 2B model), which way of using retrieval works best: nothing, one-shot RAG (the retrieved passages are put in the prompt once), or an agent that can also call search/read tools. The comparison was made fair on purpose
(same first message for paired arms, settings tuned on separate tasks, a frozen test set, a pre-registered analysis); see `PROTOCOL.md`.

**Findings** (24 frozen test tasks, 5 samples each):

1. **Retrieval helps a lot.** With nothing, 3.3% of runs pass every test (20.8% pass the behaviour tests); with one-shot RAG plus a repo map, 33.3% (50.8%). Every retrieval arm beats no retrieval by 15-30 points, and all those intervals exclude zero.
2. **Tools did not add value.** One-shot RAG vs an agent that starts from the same first message and can also call tools: -1.7 points (95% CI -10.8 to +6.7), no detectable difference, at about 3x the tokens (16.1K vs 5.5K per run) and 2x the time.
   The agents rarely used their tools when the material was already in the message (84% and 79% of their runs made no tool call).
3. **An agent that has to find everything itself is worse.** Without the pre-retrieved passages the agent scored 18.3%, 13-15 points below the arms that had them (intervals exclude zero); a 2B model is a weak searcher.
4. **The repo map fixes imports.** Import failures fall from 34.2% to 9.2% when the model is shown the repository's module paths and signatures.
5. **A 4B model did not do measurably better than the 2B** on the same frozen tasks and arms (e.g. one-shot RAG with repo map: 41.7% vs 33.3%, difference +8.3 points, 95% CI -5.8 to +24.2; no arm shows a detectable difference). Doubling the size bought much less than retrieval did (+25 to +37 points). Without thinking, the 4B also could not use tools at all (see "Second model").

**In practice, for a small model:** put roughly ten retrieved passages (the dev optimum) and, if the code must use a codebase, a signature map of that codebase in the first message. That was as good as the agent at about a third of the tokens and half the time.
If an agent is used anyway: seed it with the same context, cap its tool calls, always request a final answer if it ends without code, and put a wall-clock limit on runs (one run stalled for 94 minutes before limits were added).

**What this study does not answer.** The original goal, whether a small model + harness can replace a *much bigger* model, was **not tested**: the largest model run was 4B, and no 7-9B or larger model was run as a baseline. So nothing here says how close 2-4B + RAG comes to a 9B model alone, nor whether bigger models use tools better. The only size evidence is 2B vs 4B (no detectable difference). The work stopped here by decision; if it is continued, the missing piece is exactly that baseline (see "Suggested next steps").

## What was done (details in `PROTOCOL.md`)

- **Information equality:** an agent arm is a superset of its one-shot counterpart. Both get the *identical* first message (the same top-10 book passages, and in the repo variants the same map of the repo's Python code with every `vN.py` version and the note that the highest number is final);
  the agent additionally has tools (`book_search` returning passages in the same rendering and size cap; `repo_list`, `repo_grep`, `repo_read_file`). `dlrag-check-parity` verifies this. Every arm has the same finalization rule (one more request if no code block came out).
- **Tuning on dev, once on test:** the 32 earlier tasks were the dev set (settings chosen from them only); the 24 test tasks in `data/tasks_test` were written and validated against reference solutions before any model ran on them, frozen with a sha256 manifest (verified unchanged before this analysis), and run once.
- **Arms:** `plain` (nothing), `rag`, `rag-repo` (one-shot), `agent-rag`, `agent-rag-repo` (agents with the same first message), `agent-scratch` (diagnostic: repo map but no pre-retrieved passages).
- **Tasks:** 24 execution-graded Python tasks (8 `recall`, 8 `transfer`, 8 `explore`) from code the dev tasks do not use (the attention/encoder/positional-encoding classes of the sequence chapters, data generators, remaining `StepByStep` methods). The prompt states behaviour only; hidden `behavior` tests check that the code works and `convention` tests check the book's exact names and call order. `pass_all` = every test passes.
- **Model/sampling:** Sharp-MiniCPM5-2B Q8 on llama.cpp (`--jinja`), thinking off, temperature 1.0, top-p 0.95, top-k 20, 3,072 new tokens.

## Tuning on the dev tasks (32 tasks x 3 samples)

One-shot context size (`rag-repo`):

```
label                      setting  runs    all%  behav%  tokens/run  sec/run  tools
dev-rag-repo-k3                  3    96    38.5    45.8        4205      7.3    0.0
dev-rag-repo-k6                  6    96    44.8    54.2        4794      6.8    0.0
dev-rag-repo-k10                10    96    54.2    60.4        5534      7.4    0.0
dev-rag-repo-k15                15    96    47.9    60.4        6397      7.8    0.0
dev-rag-repo-k20                20    96    51.0    61.5        7234      9.8    0.0

SELECTED: dev-rag-repo-k10 (setting 10, dev pass_all 54.2%)
```

Agent tool budget (`agent-rag-repo`, 10 passages):

```
label                      setting  runs    all%  behav%  tokens/run  sec/run  tools
dev-agent-b4                     4    96    52.1    55.2        9180     10.7    0.3
dev-agent-b8                     8    96    51.0    58.3       14681     15.4    0.7
dev-agent-b12                   12    96    51.0    56.2       16080     13.7    0.9

SELECTED: dev-agent-b4 (setting 4, dev pass_all 52.1%)
```

The context-size grid was extended from {3, 6, 10} to {3, 6, 10, 15, 20} because the best value was at the edge (documented as "Deviation 1" in `PROTOCOL.md`, decided from dev data only); the optimum stayed at k = 10.
The tool budgets could not be told apart on dev (52.1 / 51.0 / 51.0%, one run in 96 apart), so the pre-registered rule picked the cheapest, 4.

## Test results (24 frozen tasks x 5 samples; generated by `dlrag-analyze`)

### Main table (24 test tasks x 5 samples)

| arm | runs | all tests pass | 95% CI (over tasks) | behaviour pass | tokens/run | sec/run | tool calls/run | needed 2nd request | errors |
|---|---|---|---|---|---|---|---|---|---|
| `plain` | 120 | 3.3% | 0.0 - 8.3 | 20.8% | 680 | 6.0 | 0.00 | 3% | 0 |
| `rag` | 120 | 30.0% | 15.8 - 45.0 | 43.3% | 2668 | 6.9 | 0.00 | 0% | 0 |
| `rag-repo` | 120 | 33.3% | 18.3 - 49.2 | 50.8% | 5518 | 5.9 | 0.00 | 0% | 0 |
| `agent-rag` | 120 | 28.3% | 15.8 - 41.7 | 38.3% | 6396 | 11.1 | 0.98 | 5% | 0 |
| `agent-rag-repo` | 120 | 31.7% | 17.5 - 47.5 | 40.8% | 16123 | 12.1 | 1.18 | 7% | 0 |
| `agent-scratch` | 120 | 18.3% | 7.5 - 30.8 | 28.3% | 22475 | 21.4 | 3.24 | 28% | 0 |

### By tier (all tests pass)

| arm | recall (8 tasks) | transfer (8) | explore (8) |
|---|---|---|---|
| `plain` | 5.0% | 5.0% | 0.0% |
| `rag` | 45.0% | 37.5% | 7.5% |
| `rag-repo` | 40.0% | 42.5% | 17.5% |
| `agent-rag` | 40.0% | 32.5% | 12.5% |
| `agent-rag-repo` | 47.5% | 20.0% | 27.5% |
| `agent-scratch` | 17.5% | 15.0% | 22.5% |

### Paired differences in pass_all (percentage points, 95% bootstrap CI over the 24 tasks)

| difference | role | points | 95% CI | excludes 0? |
|---|---|---|---|---|
| `agent-rag-repo` - `rag-repo` | PRIMARY | -1.7 | -10.8 to +6.7 | no |
| `agent-rag` - `rag` | secondary | -1.7 | -10.0 to +6.7 | no |
| `agent-rag-repo` - `agent-rag` | secondary (repo access for agents) | +3.3 | -7.5 to +14.2 | no |
| `rag-repo` - `rag` | secondary (repo map for one-shot) | +3.3 | -4.2 to +12.5 | no |
| `agent-scratch` - `rag-repo` | secondary | -15.0 | -26.7 to -5.0 | yes |
| `agent-scratch` - `agent-rag-repo` | secondary (pre-retrieval for agents) | -13.3 | -25.8 to -2.5 | yes |
| `rag` - `plain` | vs plain | +26.7 | +12.5 to +41.7 | yes |
| `rag-repo` - `plain` | vs plain | +30.0 | +15.8 to +45.8 | yes |
| `agent-rag` - `plain` | vs plain | +25.0 | +12.5 to +38.3 | yes |
| `agent-rag-repo` - `plain` | vs plain | +28.3 | +13.3 to +44.2 | yes |
| `agent-scratch` - `plain` | vs plain | +15.0 | +4.2 to +27.5 | yes |

### Per task (all tests pass, k of 5 samples)

| task | tier | `plain` | `rag` | `rag-repo` | `agent-rag` | `agent-rag-repo` | `agent-scratch` |
|---|---|---|---|---|---|---|---|
| tr01_subsequent_mask | recall | 0 | 4 | 3 | 1 | 5 | 1 |
| tr02_positional_encoding | recall | 0 | 0 | 0 | 0 | 0 | 0 |
| tr03_encoder | recall | 0 | 1 | 0 | 3 | 1 | 0 |
| tr04_attention | recall | 0 | 0 | 0 | 1 | 1 | 0 |
| tr05_set_seed | recall | 0 | 5 | 5 | 3 | 5 | 2 |
| tr06_to_device | recall | 0 | 3 | 3 | 3 | 2 | 0 |
| tr07_preprocessed_dataset | recall | 2 | 5 | 5 | 5 | 5 | 4 |
| tr08_generate_sequences | recall | 0 | 0 | 0 | 0 | 0 | 0 |
| tt01_padding_mask | transfer | 0 | 0 | 0 | 0 | 0 | 0 |
| tt02_positional_encoding_noscale | transfer | 0 | 1 | 0 | 1 | 0 | 0 |
| tt03_bidirectional_encoder | transfer | 0 | 2 | 2 | 2 | 0 | 0 |
| tt04_predict_classes | transfer | 0 | 1 | 4 | 1 | 2 | 3 |
| tt05_three_way_split | transfer | 0 | 5 | 4 | 3 | 1 | 3 |
| tt06_decoder_mean_hidden | transfer | 0 | 1 | 1 | 1 | 0 | 0 |
| tt07_train_with_checkpoints | transfer | 2 | 0 | 1 | 0 | 0 | 0 |
| tt08_mini_batch_with_metric | transfer | 0 | 5 | 5 | 5 | 5 | 0 |
| tx01_train_with_tensorboard | explore | 0 | 0 | 0 | 0 | 1 | 0 |
| tx02_plot_training_curves | explore | 0 | 2 | 4 | 4 | 5 | 5 |
| tx03_weight_history | explore | 0 | 1 | 0 | 0 | 1 | 0 |
| tx04_gradient_history | explore | 0 | 0 | 0 | 0 | 0 | 0 |
| tx05_ball_loader | explore | 0 | 0 | 0 | 0 | 1 | 1 |
| tx06_square_sequence_loader | explore | 0 | 0 | 3 | 1 | 3 | 3 |
| tx07_one_cycle_training | explore | 0 | 0 | 0 | 0 | 0 | 0 |
| tx08_resume_training | explore | 0 | 0 | 0 | 0 | 0 | 0 |

Unsolved by every arm: tr02_positional_encoding, tr08_generate_sequences, tt01_padding_mask, tx04_gradient_history, tx07_one_cycle_training, tx08_resume_training

### Failure anatomy (share of runs)

| arm | no code block | import/collection failure | behaviour ok, a convention wrong | all pass |
|---|---|---|---|---|
| `plain` | 0.0% | 30.0% | 17.5% | 3.3% |
| `rag` | 0.0% | 34.2% | 13.3% | 30.0% |
| `rag-repo` | 0.0% | 9.2% | 17.5% | 33.3% |
| `agent-rag` | 0.0% | 29.2% | 10.0% | 28.3% |
| `agent-rag-repo` | 0.0% | 19.2% | 9.2% | 31.7% |
| `agent-scratch` | 0.0% | 10.8% | 10.0% | 18.3% |

### Tool use (agent arms)

- `agent-rag`: 0.98 calls/run, 84% of runs used no tool, calls by tool {'book_search': 118}
- `agent-rag-repo`: 1.18 calls/run, 79% of runs used no tool, calls by tool {'repo_read_file': 124, 'repo_grep': 14, 'repo_list': 3}
- `agent-scratch`: 3.24 calls/run, 52% of runs used no tool, calls by tool {'book_search': 21, 'repo_read_file': 295, 'repo_grep': 53, 'repo_list': 20}


## Reading the results

1. **Primary result: no detectable difference** between the best one-shot arm and the best agent arm (-1.7 points, CI -10.8 to +6.7). With 24 tasks the data rule out an agent advantage much larger than ~7 points for this model, but cannot exclude a few points either way.
   The same holds without the repo (`agent-rag` - `rag` = -1.7, CI -10.0 to +6.7).
2. **The agents were mostly one-shot arms with overhead.** Given the passages (and map) up front, 79-84% of runs made no tool call. Where they did, `agent-rag-repo` read repo files (124 `repo_read_file` calls) and `agent-rag` searched the book again (118 `book_search`). The extra cost (tokens 5.5K -> 16.1K) bought no measurable gain.
3. **Starting from scratch is worse.** `agent-scratch` has to find everything with tools (3.2 calls per run, 28% of runs needed the finalization request) and passes only 18.3%, significantly below the same agent with pre-retrieved passages. Pre-retrieval carries real value; a 2B model is a weak searcher (its budget of 4 was not tuned separately, so this arm is a lower bound).
4. **Retrieval and the repo map matter, in the expected places.** Import/collection failures fall from 34.2% (`rag`) to 9.2% (`rag-repo`): the map gives the correct module paths. All retrieval arms beat `plain` (3.3%) by 15-30 points.
5. **Tiers are too small to read.** Each tier has 8 tasks. `agent-rag-repo` is ahead of `rag-repo` on `explore` (27.5% vs 17.5%) and behind on `transfer` (20.0% vs 42.5%); opposite directions, neither significant, so no tier pattern should be inferred.
6. **Absolute levels are lower on test than on dev** (`rag-repo`: 54.2% dev vs 33.3% test). Expected: settings were selected on dev (optimism), and the test tasks (sequence-model code, data generators) are harder or different; six of the 24 are unsolved by every arm. Only comparisons within the test set are meaningful.

## Second model: Sharp-Spark-X2.5-4B (added 2026-10-06; thinking off, no agent arms)

**Setup** (pre-registered in `PROTOCOL.md`, Addendum 1 and Deviation 2). The same 24 frozen test tasks, the same first messages, finalization rule and settings as the 2B (10 passages, no re-tuning for the 4B), 5 samples per task, thinking **off** and the same 3,072-token reply cap as the 2B so that the comparison isolates model size
(the user first asked for an 8K reasoning budget, then asked for a fair comparison with the non-reasoning 2B; matching won). Served with `scripts/serve_spark.sh` (`llama serve`; `llama-server` does not know this architecture) using the settings of `~/run_goose.sh`, with the model's recommended sampling (temperature 0.6, repetition penalty 1.1).
Arms run: `plain`, `rag`, `rag-repo`. **The three agent arms were not run:** with thinking off this model never emits a tool call; it invents the tool's output as text (observed in direct requests, also with `tool_choice: required`, and in 4 `agent-scratch` smoke runs with 0 tool calls). With thinking on, the same request produced a correct tool call.
This was seen in a handful of checks, not a systematic study. No result contains thinking tags (0 of 360 runs).

### Main table (24 test tasks x 5 samples)

| arm | runs | all tests pass | 95% CI (over tasks) | behaviour pass | tokens/run | sec/run | tool calls/run | needed 2nd request | errors |
|---|---|---|---|---|---|---|---|---|---|
| `plain` | 120 | 5.0% | 0.0 - 12.5 | 27.5% | 593 | 6.6 | 0.00 | 0% | 0 |
| `rag` | 120 | 30.8% | 16.7 - 45.8 | 40.8% | 2777 | 7.7 | 0.00 | 0% | 0 |
| `rag-repo` | 120 | 41.7% | 25.8 - 58.3 | 51.7% | 5964 | 7.9 | 0.00 | 0% | 0 |

### By tier (all tests pass)

| arm | recall (8 tasks) | transfer (8) | explore (8) |
|---|---|---|---|
| `plain` | 12.5% | 2.5% | 0.0% |
| `rag` | 47.5% | 40.0% | 5.0% |
| `rag-repo` | 75.0% | 32.5% | 17.5% |

### Failure anatomy (share of runs)

| arm | no code block | import/collection failure | behaviour ok, a convention wrong | all pass |
|---|---|---|---|---|
| `plain` | 0.0% | 43.3% | 22.5% | 5.0% |
| `rag` | 0.0% | 27.5% | 10.0% | 30.8% |
| `rag-repo` | 0.0% | 3.3% | 10.0% | 41.7% |

### Cross-model: Spark 4B minus MiniCPM 2B, same tasks and arms (percentage points, paired by task, 95% bootstrap CI over the 24 tasks)

| arm | Spark 4B all pass | MiniCPM 2B all pass | difference (all pass) | 95% CI | excludes 0? | Spark 4B behaviour | MiniCPM 2B behaviour | difference (behaviour) | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| `plain` | 5.0% | 3.3% | +1.7 | -5.0 to +10.0 | no | 27.5% | 20.8% | +6.7 | -9.2 to +21.7 |
| `rag` | 30.8% | 30.0% | +0.8 | -13.3 to +15.0 | no | 40.8% | 43.3% | -2.5 | -20.0 to +14.2 |
| `rag-repo` | 41.7% | 33.3% | +8.3 | -5.8 to +24.2 | no | 51.7% | 50.8% | +0.8 | -12.5 to +15.0 |

Tokens and seconds per run, Spark 4B vs MiniCPM 2B:

| arm | tokens (this) | tokens (other) | sec/run (this) | sec/run (other) |
|---|---|---|---|---|
| `plain` | 593 | 680 | 6.6 | 6.0 |
| `rag` | 2777 | 2668 | 7.7 | 6.9 |
| `rag-repo` | 5964 | 5518 | 7.9 | 5.9 |

**Reading the 4B results**
1. **Primary pre-registered comparison: no detectable difference between the 4B and the 2B** on `rag-repo` (+8.3 points, CI -5.8 to +24.2), nor on `rag` (+0.8) or `plain` (+1.7); the behaviour-test differences are likewise inside their intervals. The data cannot rule out a difference of up to roughly 15-25 points in either direction.
2. **Retrieval helped the 4B as much as the 2B:** `rag-repo` minus `plain` is +36.7 points (CI +21.7 to +53.3) for the 4B against +30.0 (CI +15.8 to +45.8) for the 2B. The repo map again removed most import failures (43.3% -> 27.5% -> 3.3% across plain / rag / rag-repo; 2B: 30.0% -> 34.2% -> 9.2%).
3. **Neither model knows this book's conventions on its own:** `plain` passes 5.0% (4B) and 3.3% (2B) of runs fully.
4. **Similar cost:** `rag-repo` took 7.9 s and 6.0K tokens per run for the 4B against 5.9 s and 5.5K for the 2B; `plain` and `rag` were within a second of each other.
5. **Tool use needs thinking in this model.** That is why there is no agent comparison for the 4B. Testing agents with thinking on would be a different reasoning setting from the 2B and is left open.

Caveats: one more model only; the 4B differs from the 2B in architecture and training as well as size; settings were tuned for the 2B; each model used its own recommended sampling; with 24 tasks the intervals are wide.

## Limits

- **One model.** A 2B model uses tools poorly; with a larger model the ranking could change. This study cannot say whether agentic RAG helps in general, only that it did not help this model on these tasks at 3x the cost.
- **24 test tasks.** The intervals are wide (about +-8 points on a difference). 5 samples of one task are correlated, so they were not treated as independent (intervals resample tasks).
- **Task type.** Most tasks need one lookup, the regime where one-shot retrieval is strongest; tasks that need several dependent lookups would test agency better and are not included.
- **The corpus is the book and its repo, which are probably in the model's pretraining data**, and the tasks were derived from the same repo by the same author as the harness.
- **Settings:** three tool budgets and five context sizes were swept (on 96 runs each); prompts and sampling were not tuned; `agent-scratch` was not tuned separately. The dev set had been run before under the exploratory protocol (used only to choose settings).
- Temperature 1.0 makes single runs noisy.

## Reproduce

```bash
uv sync --all-packages
~/Git/llama.cpp-spark/build/bin/llama-server -m ~/models/Sharp-MiniCPM5-2B-Q8_K_M.gguf --jinja -c 65536 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 --port 8080
uv run dlrag-tasks-validate --tasks-dir data/tasks_test --verify-manifest        # frozen test set untouched
uv run dlrag-check-parity --tasks-dir data/tasks_test --context-k 10             # paired arms see identical first messages
for ARM in rag-repo agent-rag-repo rag agent-rag plain agent-scratch; do
  uv run dlrag-tasks --arm $ARM --context-k 10 --max-tool-calls 4 --samples 5 --tasks-dir data/tasks_test --label test-$ARM --out data/results/test
done
uv run dlrag-analyze                                                             # regenerates the tables above
```

Dev tuning: `dlrag-tasks --arm rag-repo --context-k K --samples 3 --label dev-rag-repo-kK --out data/results/dev` and `... --arm agent-rag-repo --context-k 10 --max-tool-calls B --label dev-agent-bB`, then `dlrag-tune <labels>`.

## Suggested next steps

1. **The missing baseline for the original goal:** run larger models from `~/models` (for example OxCoder-9B Q4, or Ornith-9B) with no retrieval, and with the same one-shot `rag-repo` arm, on the same frozen test tasks (the 2B and the 4B are already done; other small models: Nanbeige-3B, gemma E2B/E4B).
   Pre-register the question before running: "small model + RAG within 10 points of the big model alone". Use the behaviour tests as the primary metric, because the convention tests check this particular book's exact names, which a big model alone cannot know either; a model with retrieval would win those by construction and the comparison would not show whether the code is "good enough".
2. Tasks that really need several dependent lookups (find where something is defined, then what it depends on), and more test tasks to narrow the intervals.
3. Tune `agent-scratch` separately (its budget and instructions), since it is the arm that depends most on tool-use skill.
