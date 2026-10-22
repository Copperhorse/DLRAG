#!/usr/bin/env bash
# Reasoning study (PROTOCOL.md, Addendum 2): 4K thinking budget, arms rag-repo and agent-rag-repo, 2B then 4B.
#   scripts/run_think4k.sh calibrate [models]   # 3 tasks x 2 arms x 1 sample per model -> data/results/think4k_calib_<model>
#   scripts/run_think4k.sh full [models]        # 24 frozen tasks x 3 samples          -> data/results/think4k_<model>
# models: "2b 4b" (default). Starts and stops its own llama server per model (by process name, never `pkill -f`).
# Logs live inside the project (data/results/...), so the run survives closing Claude Code.
set -u
cd "$(dirname "$0")/.."
MODE="${1:-full}"; MODELS="${2:-2b 4b}"
BUDGET=4096; MAX_TOKENS=8192
DRIVER_LOG="data/results/think4k_driver_${MODE}.log"
mkdir -p data/results
log() { echo "$*" | tee -a "$DRIVER_LOG"; }

uv run dlrag-tasks-validate --tasks-dir data/tasks_test --verify-manifest 2>&1 | tail -1 | tee -a "$DRIVER_LOG"
if pgrep -x llama >/dev/null || pgrep -x llama-server >/dev/null; then log "a llama server is already running; stop it first"; exit 1; fi

for M in $MODELS; do
  if [ "$M" = "2b" ]; then MID=sharp-minicpm5-2b; TEMP=1.0; else MID=sharp-spark-x2.5-4b; TEMP=0.6; fi
  if [ "$MODE" = "calibrate" ]; then OUT="data/results/think4k_calib_$M"; EXTRA=(--tasks tr05 tx02 tt08); SAMPLES=1; else OUT="data/results/think4k_$M"; EXTRA=(); SAMPLES=3; fi
  mkdir -p "$OUT/logs"
  log "[$M] starting server $(date +%T)"
  if [ "$M" = "2b" ]; then
    setsid nohup "$HOME/Git/llama.cpp-spark/build/bin/llama-server" -m "$HOME/models/Sharp-MiniCPM5-2B-Q8_K_M.gguf" --jinja -c 65536 -ngl 99 -fa on \
      -ctk q8_0 -ctv q8_0 --port 8080 --alias "$MID" --reasoning-budget $BUDGET >"$OUT/logs/server.log" 2>&1 &
  else
    REASONING_BUDGET=$BUDGET LOG="$OUT/logs/server.log" setsid nohup scripts/serve_spark.sh >/dev/null 2>&1 &
  fi
  for i in $(seq 1 150); do curl -sf localhost:8080/health 2>/dev/null | grep -q ok && break; sleep 2; done
  if ! curl -sf localhost:8080/health 2>/dev/null | grep -q ok; then log "[$M] server did not come up"; continue; fi
  for ARM in rag-repo agent-rag-repo; do
    log "[$M] START $ARM $(date +%T)"
    uv run dlrag-tasks --arm "$ARM" --context-k 10 --max-tool-calls 4 --samples $SAMPLES --tasks-dir data/tasks_test "${EXTRA[@]}" \
      --thinking --max-tokens $MAX_TOKENS --request-timeout 900 --run-timeout 1800 --model-id "$MID" --temperature $TEMP \
      --label "test-$ARM" --out "$OUT" >"$OUT/logs/$ARM.log" 2>&1
    log "[$M] DONE $ARM exit=$? $(date +%T)"
  done
  P=$(pgrep -x llama-server || pgrep -x llama); [ -n "$P" ] && kill $P; sleep 5
  log "[$M] server stopped $(date +%T)"
done
log "THINK4K $MODE FINISHED $(date +%T)"
