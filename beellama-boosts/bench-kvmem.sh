#!/usr/bin/env bash
# KV memory measurement: llama-server VRAM delta @ 32768 ctx, one config per server, serial
set -uo pipefail
OUT=/home/baramofme/IdeaProjects/llama-cpp-turboquant/bench-results-kv-vs-tq
mkdir -p "$OUT"
LOG=$OUT/runner.log
BIN=/tmp/be-src-native0001/bin
M27=/opt/llm/models/Qwen3.8-27B-MTP-Q4_K_M.gguf

vram1() { rocm-smi --showmeminfo vram 2>/dev/null | awk '/GPU\[1\].*Total Used/{print $NF}'; }

# baseline: model loaded, minimal KV (4096 ctx) -> gives model footprint
baseline() {
  local vb va
  vb=$(vram1)
  HIP_VISIBLE_DEVICES=1 "$BIN/llama-server" -m "$M27" -c 4096 -b 4096 -ub 1024 -ngl 99 -fa on -ctk f16 -ctv f16 \
    --host 127.0.0.1 --port 8098 > /dev/null 2>&1 &
  local PID=$!
  for i in $(seq 1 120); do curl -s http://127.0.0.1:8098/health >/dev/null 2>&1 && break; sleep 2; done
  sleep 2; va=$(vram1)
  kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
  echo "$(( (va-vb)/1024/1024 ))"
}

kvmem() { # tag ctk ctv [extra...]
  local tag=$1 ctk=$2 ctv=$3; shift 3
  local vb va
  vb=$(vram1)
  HIP_VISIBLE_DEVICES=1 "$BIN/llama-server" -m "$M27" -c 32768 -b 4096 -ub 1024 -ngl 99 -fa on \
    -ctk "$ctk" -ctv "$ctv" --host 127.0.0.1 --port 8099 "$@" > "$OUT/mem_${tag}.log" 2>&1 &
  local PID=$!
  for i in $(seq 1 150); do curl -s http://127.0.0.1:8099/health >/dev/null 2>&1 && break; sleep 2; done
  # prefill ~8K words of real text to force KV allocation
  local PAYLOAD
  PAYLOAD=$(python3 -c "import json; p=('KV cache quantization must preserve attention fidelity across heads and layers. '*1000)[:60000]; print(json.dumps({'model':'x','max_tokens':8,'prompt':p}))")
  curl -s http://127.0.0.1:8099/v1/completions -d "$PAYLOAD" > "$OUT/mem_${tag}.resp" 2>/dev/null
  sleep 2; va=$(vram1)
  kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
  echo "mem ${tag} (ctx32768): total_vram=$(( (va-vb)/1024/1024 ))MiB" | tee -a "$LOG"
}

BASE=$(baseline)
echo "baseline model+4096ctx f16 KV: ${BASE} MiB" | tee -a "$LOG"

kvmem f16 f16 f16
kvmem q8_q5 q8_0 q5_0
kvmem kvarn5_4 kvarn5 kvarn4 --kv-tail-tokens 1024
kvmem kvarn4_2 kvarn4 kvarn2 --kv-tail-tokens 1024
echo "=== kvmem done ===" >> "$LOG"
