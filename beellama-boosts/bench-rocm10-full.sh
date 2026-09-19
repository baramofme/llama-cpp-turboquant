#!/usr/bin/env bash
# ROCm10 beellama (8 boosts + 0001) S1-S3 on GPU1, fresh rocm10 docker
set -uo pipefail
IMG=rocm/dev-ubuntu-26.04:10.0.0-full
OUT=/home/baramofme/IdeaProjects/llama-cpp-turboquant/bench-results-kv-vs-tq
mkdir -p "$OUT"
LOG=$OUT/runner.log
M=/models/Qwen3.8-27B-MTP-Q4_K_M.gguf
echo "=== rocm10 bench $(date) ===" >> "$LOG"

run() {
  local tag=$1 ctk=$2 ctv=$3; shift 3
  timeout 600 docker run --rm --device /dev/kfd --device /dev/dri/renderD130 --device /dev/dri/card2 --group-add video \
    -v /opt/llm/models:/models -v /home/baramofme/IdeaProjects/llama-cpp-turboquant/beellama-boosts/src:/src/src \
    -e HIP_VISIBLE_DEVICES=0 -e LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib \
    --entrypoint /src/src/build-rocm10/bin/llama-bench \
    "$IMG" -m "$M" -ngl 99 -fa on -ctk "$ctk" -ctv "$ctv" -b 4096 -ub 1024 -p 512 -n 512 -r 3 -o jsonl "$@" \
    > "$OUT/${tag}.jsonl" 2> "$OUT/${tag}.err"
  local pp tg
  pp=$(grep -oE '"n_prompt": 512,"[^}]*"avg_ts": [0-9.]+' "$OUT/${tag}.jsonl" 2>/dev/null | grep -oE '[0-9.]+$' | head -1)
  tg=$(grep -oE '"n_gen": 512,"[^}]*"avg_ts": [0-9.]+' "$OUT/${tag}.jsonl" 2>/dev/null | grep -oE '[0-9.]+$' | head -1)
  echo "r10 ${tag}: pp=${pp} tg=${tg}" | tee -a "$LOG"
}

run r10_s1 f16 f16
run r10_s2 q8_0 q5_0
run r10_s3 kvarn5 kvarn4 --kv-tail-tokens 1024
run r10_s3q kvarn7 kvarn5 --kv-tail-tokens 1024
echo "=== rocm10 done ===" >> "$LOG"
