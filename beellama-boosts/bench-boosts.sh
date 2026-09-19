#!/usr/bin/env bash
# beellama v0.4.5 + rdna-boosts clean blocks (0002/0005/0006/0007/0012) native bench
# GPU1(HIP_VISIBLE_DEVICES=1) 단독 사용
set -uo pipefail

BIN=/tmp/beellama-src/build-boosts/bin
OUT=bench-results-boosts
M27=/opt/llm/models/Qwen3.8-27B-MTP-Q4_K_M.gguf
M2B=/opt/llm/models/Qwen3.5-2B-MTP-Q4_K_M.gguf
mkdir -p "$OUT"
export HIP_VISIBLE_DEVICES=1

bench() {
  local tag=$1; shift
  echo "=== $tag ==="
  "$@" -o jsonl > "$OUT/${tag}.jsonl" 2> "$OUT/${tag}.log"
  echo "rc=$? -> $OUT/${tag}.jsonl"
}

bench boost_s1 "$BIN/llama-bench" -m "$M27" -ngl 99 -fa on -ctk f16 -ctv f16 -b 4096 -ub 1024 -p 512 -n 512 -r 3
bench boost_s2 "$BIN/llama-bench" -m "$M27" -ngl 99 -fa on -ctk q8_0 -ctv q5_0 -b 4096 -ub 1024 -p 512 -n 512 -r 3
bench boost_s3 "$BIN/llama-bench" -m "$M27" -ngl 99 -fa on -ctk kvarn5 -ctv kvarn4 --kv-tail-tokens 1024 -b 4096 -ub 1024 -p 512 -n 512 -r 3

# S7: batched
echo "=== boost_s7 ==="
"$BIN/llama-batched-bench" -m "$M27" -c 32768 -b 4096 -ub 1024 -ngl 99 -fa on -ctk f16 -ctv f16 \
  -p 512 -n 512 -np 4 > "$OUT/boost_s7.out" 2> "$OUT/boost_s7.log"
echo "rc=$?"

# S4: llama-server draft-mtp
echo "=== boost_s4 (llama-server draft-mtp) ==="
"$BIN/llama-server" -m "$M27" -c 32768 -b 4096 -ub 1024 -ngl 99 -fa on --jinja --metrics \
  --spec-type draft-mtp --spec-draft-n-max 3 --host 127.0.0.1 --port 8091 \
  > "$OUT/boost_s4.log" 2>&1 &
S4PID=$!
for i in $(seq 1 120); do
  curl -s http://127.0.0.1:8091/health > /dev/null 2>&1 && break
  sleep 2
done
sleep 2
curl -s http://127.0.0.1:8091/v1/completions -d '{"model":"x","max_tokens":256,"prompt":"Explain speculative decoding in one paragraph."}' > "$OUT/boost_s4.completion.json"
curl -s http://127.0.0.1:8091/metrics > "$OUT/boost_s4.metrics" 2>/dev/null
kill $S4PID 2>/dev/null; wait $S4PID 2>/dev/null
grep 'draft acceptance' "$OUT/boost_s4.log" | tail -2
echo done
