#!/usr/bin/env bash
# S4: MTP acceptance + decode-speed bench, beellama 8-boosts+0001 native (GPU1)
# Compares: draft-mtp (fixed n-max 3) vs draft-mtp-adaptive (floor 1, max 3)
set -uo pipefail

BIN=/tmp/be-src-native0001/bin
OUT=/home/baramofme/IdeaProjects/llama-cpp-turboquant/bench-results-s4-0001
M27=/opt/llm/models/Qwen3.8-27B-MTP-Q4_K_M.gguf
M2B=/opt/llm/models/Qwen3.5-2B-MTP-Q4_K_M.gguf
mkdir -p "$OUT"
export HIP_VISIBLE_DEVICES=1

run_s4() { # tag [extra-spec-flags...]
  local tag=$1; shift
  "$BIN/llama-server" -m "$M27" -c 32768 -b 4096 -ub 1024 -ngl 99 -fa on \
    --jinja --metrics --host 127.0.0.1 --port 8094 "$@" \
    > "$OUT/${tag}.log" 2>&1 &
  local PID=$!
  for i in $(seq 1 120); do
    curl -s http://127.0.0.1:8094/health >/dev/null 2>&1 && break
    sleep 2
  done
  # load KV: warm a decent context before measuring acceptance
  curl -s http://127.0.0.1:8094/v1/completions \
    -d '{"model":"x","max_tokens":512,"prompt":"Write a detailed technical essay about KV cache quantization for efficient LLM inference, covering accuracy, memory footprint, and throughput trade-offs. "}' \
    > "$OUT/${tag}.completion.json"
  curl -s http://127.0.0.1:8094/metrics > "$OUT/${tag}.metrics"
  kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
  echo "$tag:" | tee -a "$OUT/summary.txt"
  grep -E "draft acceptance|draft p_accept|speculative" "$OUT/${tag}.log" | tail -3 | tee -a "$OUT/summary.txt"
  grep -E "KV self size|KV cross size|timings" "$OUT/${tag}.log" | tail -4 | tee -a "$OUT/summary.txt"
}

run_s4 mtp_fixed          --spec-type draft-mtp --spec-draft-n-max 3
run_s4 mtp_adaptive       --spec-type draft-mtp-adaptive --spec-draft-n-max 3 --spec-draft-n-min-adaptive 1
run_s4 mtp_adaptive_mid   --spec-type draft-mtp-adaptive --spec-draft-n-max 3 --spec-draft-n-min-adaptive 2
echo "=== S4 DONE ==="