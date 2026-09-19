#!/bin/bash
# ROCm 10 빌드 S1/S2/S3 벤치 (GPU1 = renderD130/card2 = PCI 06:00.0)
set -u
export LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
export HIP_VISIBLE_DEVICES=0
B=/src/src/build-rocm10/bin
M27=/models/Qwen3.8-27B-MTP-Q4_K_M.gguf
OUT=/src/rocm10
mkdir -p "$OUT"

run() {
  local name="$1" ctk="$2" ctv="$3"
  echo "=== $name (ctk=$ctk ctv=$ctv) ==="
  local tail=""
  case "$ctk" in kvarn*) tail="--kv-tail-tokens 1024" ;; esac
  "$B/llama-bench" -m "$M27" -ngl 99 -fa on -ctk "$ctk" -ctv "$ctv" $tail \
    -b 4096 -ub 1024 -p 512 -n 512 -r 3 -o jsonl > "$OUT/${name}.jsonl" 2> "$OUT/${name}.err"
  echo "rc=$?"
  tail -5 "$OUT/${name}.jsonl" 2>/dev/null | python3 -c "
import json,sys
for line in sys.stdin:
    d=json.loads(line)
    print(f\"  {d['type_k']}/{d['type_v']} pp={d['n_prompt']} gen={d['n_gen']} ts={d['avg_ts']:.2f}\")
" 2>/dev/null
}

run s1 f16 f16
run s2 q8_0 q5_0
run s3 kvarn5 kvarn4
