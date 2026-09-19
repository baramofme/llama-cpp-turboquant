#!/usr/bin/env bash
# turboquant(llm-main image) bench: S1/S2/S3 + turbo variants on GPU1, fresh docker run
set -uo pipefail
IMG=baramofme/llama-cpp-rocm:gfx1100-rocm7.2-tbqplus-rebuild
OUT=/home/baramofme/IdeaProjects/llama-cpp-turboquant/bench-results-kv-vs-tq
mkdir -p "$OUT"
LOG=$OUT/runner.log
M=/models/Qwen3.8-27B-MTP-Q4_K_M.gguf
echo "=== tbq docker bench $(date) ===" >> "$LOG"

bench() {
  local tag=$1 ctk=$2 ctv=$3
  timeout 600 docker run --rm --device /dev/kfd --device /dev/dri/renderD130 --device /dev/dri/card2 --group-add video \
    -v /opt/llm/models:/models \
    -e HIP_VISIBLE_DEVICES=0 \
    --entrypoint /app/llama-bench \
    "$IMG" -m "$M" -ngl 99 -fa on -ctk "$ctk" -ctv "$ctv" -b 4096 -ub 1024 -p 512 -n 512 -r 3 -o jsonl \
    > "$OUT/${tag}.jsonl" 2> "$OUT/${tag}.err"
  local pp tg
  pp=$(grep -oE '"n_prompt": 512[^}]*"avg_ts": [0-9.]+' "$OUT/${tag}.jsonl" 2>/dev/null | grep -oE '[0-9.]+$' | head -1)
  tg=$(grep -oE '"n_gen": 512[^}]*"avg_ts": [0-9.]+' "$OUT/${tag}.jsonl" 2>/dev/null | grep -oE '[0-9.]+$' | head -1)
  echo "tbq ${tag}: pp=${pp} tg=${tg}" | tee -a "$LOG"
}

bench tq_f16_f16 f16 f16
bench tq_q8_q5 q8_0 q5_0
bench tq_q8_turbo3 q8_0 turbo3
bench tq_q8_turbo4 q8_0 turbo4
bench tq_q8_turbo2 q8_0 turbo2
echo "=== tbq done ===" >> "$LOG"
