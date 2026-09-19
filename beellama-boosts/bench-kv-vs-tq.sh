#!/usr/bin/env bash
# Goal #2/#3: kvarn vs turboquant cross-fork benchmark + KV memory measurement
# beellama native (GPU1) vs turboquant docker llm-main
# Phase A: pp/tg (llama-bench, no -c in this fork)
# Phase B: KV memory (llama-server -c 32768 + valid prefill, rocm-smi delta)
set -uo pipefail

M27=/opt/llm/models/Qwen3.8-27B-MTP-Q4_K_M.gguf
M27_CT=/models/Qwen3.8-27B-MTP-Q4_K_M.gguf
CTX=32768
OUT=/home/baramofme/IdeaProjects/llama-cpp-turboquant/bench-results-kv-vs-tq
mkdir -p "$OUT"
LOG=$OUT/runner.log
BIN=/tmp/be-src-native0001/bin
TBQ_BIN=/app/llama-bench
echo "=== bench start $(date) ===" > "$LOG"

vram1() { rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[1\].*Total Used/{print $NF}'; }

summ_json() {
  local f=$1
  python3 -c "
import json
ts={}
for line in open('$f'):
    d=json.loads(line)
    if d.get('n_prompt',0)>0: ts['pp']=d['avg_ts']
    if d.get('n_gen',0)>0: ts['tg']=d['avg_ts']
print(f\"pp={ts.get('pp',0):.1f} tg={ts.get('tg',0):.2f}\")
" 2>/dev/null
}

bench_tbq() {
  local tag=$1 ctk=$2 ctv=$3
  docker exec llm-main bash -lc "export HIP_VISIBLE_DEVICES=0; ${TBQ_BIN} -m ${M27_CT} -ngl 99 -fa on -ctk '$ctk' -ctv '$ctv' -b 4096 -ub 1024 -p 512 -n 512 -r 3 -o jsonl" > "$OUT/${tag}.jsonl" 2> "$OUT/${tag}.err"
  echo "tbq ${tag}: $(summ_json "$OUT/${tag}.jsonl")" | tee -a "$LOG"
}

bench_be() {
  local tag=$1 ctk=$2 ctv=$3; shift 3
  HIP_VISIBLE_DEVICES=1 "$BIN/llama-bench" -m "$M27" -ngl 99 -fa on -ctk "$ctk" -ctv "$ctv" -b 4096 -ub 1024 -p 512 -n 512 -r 3 -o jsonl "$@" > "$OUT/${tag}.jsonl" 2> "$OUT/${tag}.err"
  echo "be  ${tag}: $(summ_json "$OUT/${tag}.jsonl")" | tee -a "$LOG"
}

kvmem_be() {
  local tag=$1 ctk=$2 ctv=$3; shift 3
  local vb va
  vb=$(vram1)
  HIP_VISIBLE_DEVICES=1 "$BIN/llama-server" -m "$M27" -c $CTX -b 4096 -ub 1024 -ngl 99 -fa on \
    -ctk "$ctk" -ctv "$ctv" --host 127.0.0.1 --port 8095 "$@" > "$OUT/mem_${tag}.log" 2>&1 &
  local PID=$!
  for i in $(seq 1 150); do curl -s http://127.0.0.1:8095/health >/dev/null 2>&1 && break; sleep 2; done
  echo "KV memory benchmark prompt. $(printf 'The quick brown fox jumps over the lazy dog. %.0s' $(seq 1 2000))" > /tmp/kvprompt.txt
  curl -s http://127.0.0.1:8095/v1/completions -d '{"model":"x","max_tokens":4,"prompt":"The quick brown fox jumps over the lazy dog. '"$(printf 'The quick brown fox jumps over the lazy dog. %.0s' $(seq 1 2000))"'"}' > "$OUT/mem_${tag}.resp" 2>/dev/null
  sleep 1
  va=$(vram1)
  kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
  local kvs
  kvs=$(grep -oE "KV self size:[^,]*" "$OUT/mem_${tag}.log" | head -1)
  echo "mem ${tag}: vram_delta=$(( (va-vb)/1024/1024 ))MiB | $kvs" | tee -a "$LOG"
}

bench_be be_f16_f16 f16 f16
bench_be be_q8_q5 q8_0 q5_0
bench_be be_kvarn5_4 kvarn5 kvarn4 --kv-tail-tokens 1024
for kv in kvarn2 kvarn3 kvarn4 kvarn6 kvarn8; do
  bench_be "be_kvn_${kv}" "$kv" "$kv" --kv-tail-tokens 1024
done

bench_tbq tq_f16_f16 f16 f16
bench_tbq tq_q8_q5 q8_0 q5_0
bench_tbq tq_q8_turbo3 q8_0 turbo3
bench_tbq tq_q8_turbo4 q8_0 turbo4
bench_tbq tq_q8_turbo2 q8_0 turbo2

kvmem_be f16 f16 f16
kvmem_be q8_q5 q8_0 q5_0
kvmem_be kvarn5_4 kvarn5 kvarn4 --kv-tail-tokens 1024
kvmem_be kvarn4_2 kvarn4 kvarn2 --kv-tail-tokens 1024

echo "=== bench done $(date) ===" >> "$LOG"
