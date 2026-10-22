# Protocol: one-shot RAG vs agentic RAG (fair re-run)

Written before any run on the test tasks. It supersedes the exploratory rounds in `data/results/exploratory/` (see `REPORT.md`,
"exploratory" sections), which had asymmetries between the arms: different views of the book, a hand-curated repo map for only one arm,
an extra model call for the agents only, untuned settings, and tasks written after seeing results.

## Question

With the same small model, are the answers better when the model can fetch more information through tool calls
(agentic RAG) than when it receives the retrieved information once (one-shot RAG)? At what cost?

## Fixed for every arm

- Model: Sharp-MiniCPM5-2B Q8 on llama-server (`--jinja`, 32K ctx), thinking off, model-card sampling (temperature 1.0, top-p 0.95, top-k 20), 3,072 new tokens max.
- Corpus: `data/rag_artifacts` (3,058 book chunks, hybrid BM25 + dense with RRF) and the book's companion repo `git_repo/PyTorchStepByStep`.
- Scoring: execution of hidden tests; primary metric `pass_all` (every test passes), secondary `pass_behavior`. A reply without a code block is a failure.
- Same finalization policy: if a run ends without a code block, exactly one more request is made (agents: tool-free, given what their tools
  returned; one-shot arms: the same message plus a reminder to answer with a code block).

## Information equality (the central rule)

An agent arm is a **superset** of its one-shot counterpart: both receive the *identical* first user message (pre-retrieved passages C for the task text,
optionally the repo map R, then the task); the agent additionally has tools. The agents' `book_search` returns passages in exactly the same rendering and size cap as C.
R is the real repo (every file, all `vN.py` versions) plus one sentence saying the highest version number is the final one. `dlrag-check-parity` verifies these invariants.

| arm | C (top-k passages) | R (repo map) | tools |
|---|---|---|---|
| `plain` | - | - | - |
| `rag` | yes | - | - |
| `rag-repo` | yes | yes | - |
| `agent-rag` | yes | - | `book_search` |
| `agent-rag-repo` | yes | yes | `book_search`, `repo_list`, `repo_grep`, `repo_read_file` |
| `agent-scratch` (diagnostic) | - | yes | same as `agent-rag-repo` |

Primary one-shot arm: `rag-repo`. Primary agent arm: `agent-rag-repo`. (The repo-free pair `rag` / `agent-rag` is a secondary comparison.)

## Tasks

- **Dev** = the 32 tasks in `data/tasks` (18 recall, 6 transfer, 8 explore). Used only to choose settings.
- **Test** = 24 new tasks in `data/tasks_test` (8 recall, 8 transfer, 8 explore) on code not used in dev. They were written and validated with their reference solutions
  (reference passes, empty stub fails, at least one behaviour and one convention test each) **before any model run on them**, and frozen with a sha256 manifest
  (`data/tasks_test/MANIFEST.sha256`). Test results never change a setting, a prompt, a task or an analysis choice. Editing a test task after the first test run voids the test.

## Compute

Cost is reported for every run (tool calls, tokens, seconds) and settings are swept, not hidden.

## Tuning on dev (sequential, rule fixed here)

1. **One-shot context size:** `rag-repo` with `--context-k` in {3, 6, 10}, 32 dev tasks x 3 samples. Choose the k with the highest dev `pass_all`; ties -> the smaller k. Call it k*.
2. **Agent tool budget:** `agent-rag-repo` with context k* and `--max-tool-calls` in {4, 8, 12}, same dev runs. Choose the highest dev `pass_all`; ties -> the smaller budget. Call it b*.
3. The chosen k* and b* (and the dev table) are appended to this file, with the date, before the test run. All test arms use k*; agent arms use b*.

## Test run (once)

`plain`, `rag`, `rag-repo`, `agent-rag`, `agent-rag-repo`, `agent-scratch` on the 24 test tasks, 5 samples each, at the frozen settings.

## Analysis (fixed in advance)

- Primary: `pass_all` of `agent-rag-repo` minus `rag-repo`, averaged per task and paired by task, with a 95% bootstrap confidence interval over the 24 tasks.
  The conclusion is stated from that interval (agent better / worse only if the interval excludes 0; otherwise "no detectable difference").
- Secondary, reported with the same intervals: `agent-rag` vs `rag`, each arm vs `plain`, per tier (recall / transfer / explore), `agent-scratch` vs the others.
- Cost: tokens and seconds per run for every arm, and `pass_all` against tokens for the dev sweep.
- Everything in the report is generated from the result files by script.

## Known limits (stated up front)

One model (2B, weak at tool use; a larger model could rank the arms differently); one book (probably in the model's pretraining data); 24 test tasks give wide intervals;
temperature 1.0 makes single runs noisy; the tasks are my own and are derived from the repo that is also the retrieval corpus; tasks on seq2seq code (test) may be harder than the dev tasks, which could push all arms toward 0.

## Freeze record

- 2026-10-02: `data/tasks_test` (24 tasks: 8 recall `tr*`, 8 transfer `tt*`, 8 explore `tx*`) validated (every reference solution passes, every empty stub fails,
  at least one behaviour and one convention test per task; a generic solution fails the convention tests of the spot-checked tasks) and frozen.
  Manifest digest (sha256 of `MANIFEST.sha256`): `2f9cf995654c84bf775ce418ac57384827b6b7355d7ebe699d20577bbcbf1d29`.
  **No model had been run on any of these tasks at that point.** `dlrag-tasks` refuses to run a directory whose files differ from its manifest.

## Deviation 1 (2026-10-02, decided from dev results only; no test task had been run)

Stage 1 on dev (`rag-repo`, 32 dev tasks x 3 samples): k=3 -> 38.5%, k=6 -> 44.8%, k=10 -> 54.2% `pass_all` (tokens/run 4.2K / 4.8K / 5.5K).
The best setting is the largest value of the grid and the curve was still rising, so capping the one-shot arm there would handicap it
(and, because the agents start from the same context, would cap them too). The grid is therefore extended by k in {15, 20} (20 passages + repo map is about 14K tokens,
inside the 32K context). Rule unchanged: highest dev `pass_all` over all five values, exact ties to the smaller k. The same boundary rule is applied to
stage 2 (the agent budget): if the best budget is the largest in the grid, the grid is extended by one step (16).
`llama-server` context is raised to 65,536 for the remaining runs so that long agent transcripts cannot overflow it (it only matters if a run would otherwise exceed 32K tokens).

## Stage 1 result (2026-10-02, dev, `rag-repo`, 32 tasks x 3 samples, `dlrag-tune`)

| k | dev pass_all | dev pass_behavior | tokens/run | sec/run |
|---|---|---|---|---|
| 3 | 38.5% | 45.8% | 4205 | 7.3 |
| 6 | 44.8% | 54.2% | 4794 | 6.8 |
| 10 | **54.2%** | 60.4% | 5534 | 7.4 |
| 15 | 47.9% | 60.4% | 6397 | 7.8 |
| 20 | 51.0% | 61.5% | 7234 | 9.8 |

**Selected k\* = 10** (the optimum is inside the grid, so no further extension). Stage 2 uses `agent-rag-repo` with `--context-k 10` and tool budgets {4, 8, 12} (plus 16 only if 12 wins), same dev tasks and samples.

## Stage 2 result and frozen settings (2026-10-02, dev, `agent-rag-repo`, context k=10, 32 tasks x 3 samples, `dlrag-tune`)

| tool budget | dev pass_all | dev pass_behavior | tokens/run | sec/run | tool calls/run |
|---|---|---|---|---|---|
| 4 | **52.1%** | 55.2% | 9180 | 10.7 | 0.3 |
| 8 | 51.0% | 58.3% | 14681 | 15.4 | 0.7 |
| 12 | 51.0% | 56.2% | 16080 | 13.7 | 0.9 |

The budgets differ by about one run in 96, i.e. the dev data cannot tell them apart; the pre-registered rule selects budget 4 (highest, and cheapest). Budget 12 did not win, so the edge rule did not trigger a budget-16 run.
For reference, the one-shot arm at its selected k=10 scored 54.2% on the same dev tasks.

**FROZEN SETTINGS FOR THE TEST RUN: context k* = 10 passages, tool budget b* = 4** (`agent-scratch`, which has no pre-retrieved passages and was not tuned separately, also uses b* = 4).
Test run order: `rag-repo`, `agent-rag-repo` (primary comparison), then `rag`, `agent-rag`, `plain`, `agent-scratch`; 24 test tasks x 5 samples, labels `test-<arm>`, results in `data/results/test`.
A technical failure (server crash) is handled by re-running that whole arm unchanged; nothing is changed after seeing test results.

## Addendum 1: second model, Sharp-Spark-X2.5-4B (2026-10-06, written before any run with this model)

Question: with everything else held fixed, how does a larger small model (4B) compare with the 2B on the frozen test tasks, with and without retrieval?

- **Model:** `~/models/Sharp-Spark-X2.5-4B-Q6_K_XL.gguf`, served with `scripts/serve_spark.sh` (the `llama serve` binary: the `llama-server` of `~/Git/llama.cpp-spark` does not know the `spark2_5` architecture), using the flags of `~/run_goose.sh`.
- **Reasoning matched to the 2B: thinking OFF** (`enable_thinking=false`, no reasoning budget), same 3,072-token reply cap. `--reasoning-budget 8192` stays on the server only as a safety cap and must never be reached; every result is checked for thinking tags / non-empty `reasoning_content`.
  (The user first asked for an 8K reasoning budget, then asked for a fair comparison with the 2B, which had no reasoning; matching won. An 8K-reasoning study for both models is a possible follow-up, not part of this run.)
- **Same everything else:** the frozen test set (manifest verified before the run), the six arms, 5 samples per task, the same first messages and finalization rule, context k = 10 and tool budget 4. **These settings were tuned for the 2B and are not re-tuned for the 4B.**
- **Sampling:** each model uses its own recommended sampling (2B: temperature 1.0; 4B: temperature 0.6, repetition penalty 1.1 over 256 tokens, as in `run_goose.sh`; top-p 0.95 and top-k 20 for both).
- **Analysis fixed in advance:** (1) primary: 4B minus 2B on `rag-repo` `pass_all`, paired by task, 95% bootstrap CI over the 24 tasks; secondary: the same for every other arm and for `pass_behavior`; (2) the original primary comparison, `agent-rag-repo` minus `rag-repo`, repeated for the 4B; (3) tokens and seconds per run. No directional hypothesis; "no detectable difference" is reported as such.
- **Known limits:** the models differ in more than size (architecture, training data, template); settings tuned on the 2B; one more model only.

### Deviation 2 (2026-10-06, before any test run with the 4B): agent arms cannot run with thinking off

Smoke tests against `scripts/serve_spark.sh` (`llama serve`, build 11046): with `enable_thinking=false` the model never emits a tool call; it invents the tool's answer in plain text (e.g. a made-up directory listing),
even with `tool_choice: "required"` (observed in direct requests and in 4 `agent-scratch` runs: 0 tool calls). With thinking on, the same request produces a correct `repo_list` call. The chat template reports tool support, so this is model behaviour, not a harness error.
Consequence: with thinking off the three agent arms would be one-shot arms with extra overhead, not agents. They are therefore **not run (and not reported as agent results)** in the thinking-off run, and the pre-registered item (2) "agent-rag-repo minus rag-repo for the 4B" is not available there.
The thinking-off run covers `rag-repo`, `rag` and `plain` (24 tasks x 5 samples, everything else as in Addendum 1). Whether and how to test the agent arms for this model (thinking on, which changes the reasoning setting relative to the 2B) is left to a separate decision and, if done, is a separate, clearly labelled condition.

### Addendum 1 outcome (2026-10-06)

Run as specified for `rag-repo`, `rag`, `plain` (24 tasks x 5 samples, thinking off, manifest verified before and after; no thinking tags in any reply). Primary: Spark 4B minus MiniCPM 2B on `rag-repo` `pass_all` = +8.3 points, 95% CI -5.8 to +24.2 (no detectable difference); all other arm comparisons also inside their intervals. Item (2) (agent minus one-shot for the 4B) was not run (Deviation 2).
Full tables: `data/results/spark4b/analysis.md` and `REPORT.md`, section "Second model". Nothing was changed after seeing the 4B results.

## Addendum 2: reasoning study, 4K thinking budget, 2B and 4B (2026-10-06, written before any run in this condition)

Why: every earlier run used thinking off. The 2B agents called tools without thinking; the Sharp-Spark 4B cannot call tools without thinking (Deviation 2), so it has no agent results. The user asked for both models with a 4K reasoning budget in the agent arms. The one-shot arm gets the same budget so that agency is not confounded with reasoning.

- **Condition:** thinking ON (`enable_thinking=true` per request), server `--reasoning-budget 4096`, per-request `max_tokens` 8192 (4,096 thinking + room for the answer or a tool call), request timeout 900 s, run timeout 1800 s. Finalization requests also think. Everything else as in the frozen protocol: the 24 frozen test tasks (manifest verified), same first messages, context k = 10, tool budget 4 (tuned for the 2B without thinking, not re-tuned), each model's own sampling (2B temperature 1.0; 4B temperature 0.6 with repetition penalty 1.1 over 256 tokens), **3 samples per task** (the thinking-off runs have 5).
- **Arms:** `rag-repo` (matched one-shot) and `agent-rag-repo`, for the 2B and for the 4B. Out of scope: `agent-rag`, `agent-scratch`, `rag`, `plain`.
- **Analysis fixed in advance (per model):** (1) primary: `pass_all` of `agent-rag-repo` minus `rag-repo`, both with 4K thinking, paired by task, 95% bootstrap CI over the 24 tasks; (2) tool use: calls per run and share of runs with at least one call; (3) reasoning effect: `rag-repo` with thinking vs the existing thinking-off results on the same tasks, and for the 2B also `agent-rag-repo` with vs without thinking (the 4B has no thinking-off agent results); (4) 4B minus 2B for both arms with thinking; (5) output tokens and seconds per run. No directional hypothesis; "no detectable difference" is reported as such.
- **Checks:** no thinking tags in reply text; per-request tokens within budget + answer (sampled in calibration); runs that hit a limit are reported.
- **Known limits:** 3 samples (noisier); settings tuned for the 2B without thinking; the budget is a cap, not a requirement; one budget only; 24 tasks give wide intervals.
- **Procedure:** a short calibration (3 tasks x 2 arms x 1 sample per model, results kept apart) checks the points above and measures speed; the full run follows if the projected time is under about 12 h, otherwise the user is asked.

### Addendum 2 status (2026-10-06 ~20:38)

The reasoning study was **stopped by the user** before any arm finished (2B `rag-repo` reached 66 of 72 runs; nothing else ran). No result was analysed or reported. The addendum above remains the plan if the study is resumed; see `data/results/STATUS_think4k.md`.
