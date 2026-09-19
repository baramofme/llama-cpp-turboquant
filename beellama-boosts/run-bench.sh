#!/bin/bash
# ROCm 10 dev 이미지 안에서 실행되는 벤치 스크립트
set -u
export LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
export HIP_VISIBLE_DEVICES=0
B=/src/src/build-rocm10/bin
M=/models/Qwen3.8-27B-MTP-Q4_K_M.gguf

run() {
  local tag=$1; shift
  "$B/llama-bench" -m "$M" -ngl 99 -fa on "$@" -b 4096 -ub 1024 -p 512 -n 512 -r 3 -o jsonl > "/src/$tag.jsonl" 2> "/src/$tag.err"
  python3 -c "
import json
pp=tg=0
for line in open('/src/$tag.jsonl'):
    line=line.strip()
    if not line: continue
    d=json.loads(line)
    if d.get('n_prompt',0)>0: pp=d['avg_ts']
    if d.get('n_gen',0)>0: tg=d['avg_ts']
print('$tag: pp=%.2f tg=%.2f' % (pp,tg))
" 2>/dev/null || echo "$tag: FAIL"
}

run r10_s1 -ctk f16 -ctv f16
run r10_s2 -ctk q8_0 -ctv q5_0
run r10_s3 -ctk kvarn5 -ctv kvarn4 --kv-tail-tokens 1024
