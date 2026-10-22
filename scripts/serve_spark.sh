#!/usr/bin/env bash
# Serve Sharp-Spark-X2.5-4B for the experiments (non-interactive derivative of ~/run_goose.sh, without Goose).
# Needs the `llama` binary (~/.local/bin/llama): the llama-server of ~/Git/llama.cpp-spark does not know the spark2_5 architecture.
# Thinking is switched off per request by the harness (enable_thinking=false); --reasoning-budget stays as a safety cap only.
#   scripts/serve_spark.sh            # foreground; logs to /tmp/llama-serve-spark.log
set -euo pipefail
MODEL="${MODEL:-$HOME/models/Sharp-Spark-X2.5-4B-Q6_K_XL.gguf}"
PORT="${PORT:-8080}"
LOG="${LOG:-/tmp/llama-serve-spark.log}"
exec llama serve -m "$MODEL" --host 127.0.0.1 --port "$PORT" -ngl 99 \
  -c "${LLAMA_CTX:-147456}" -np 1 -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
  --temp 0.6 --top-p 0.95 --top-k 20 --repeat-penalty 1.1 --repeat-last-n 256 \
  -lv "${LLAMA_LOG_VERBOSITY:-1}" --reasoning-budget "${REASONING_BUDGET:-8192}" --jinja >"$LOG" 2>&1
