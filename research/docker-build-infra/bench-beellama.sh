#!/usr/bin/env bash
# bench-beellama.sh - beellama v0.4.4 vs tbqplus 빌드 성능 벤치
#
# 사용법:
#   ./bench-beellama.sh smoke        # 2B 모델 파이프라인 검증
#   ./bench-beellama.sh bench        # 27B S1~S3 llama-bench (양쪽 빌드)
#   ./bench-beellama.sh server       # S4~S6 llama-server (양쪽 빌드, 수용률 측정)
#   ./bench-beellama.sh batched      # S7 llama-batched-bench (양쪽 빌드)
#   ./bench-beellama.sh all          # 순서대로 전부 실행
#
# 원시 출력: bench-results/
# GPU1(renderD129/card2) 단독 사용, llm-main(GPU0) 간섭 없음

set -uo pipefail

# 이미지/출력 디렉터리/빌드 목록 환경변수로 오버라이드 가능
# v0.4.5 단독: BEELLAMA_IMG=ghcr.io/anbeeld/beellama.cpp:server-rocm-preview-v0.4.5 OUT=bench-results-v045 BUILDS=beellama ./bench-beellama.sh ...
BEELLAMA_IMG=${BEELLAMA_IMG:-ghcr.io/anbeeld/beellama.cpp:server-rocm-v0.4.4}
MAIN_IMG=${MAIN_IMG:-baramofme/llama-cpp-rocm:gfx1100-rocm7.2-tbqplus-rebuild}
BUILDS=${BUILDS:-beellama main}

M27=/models/Qwen3.8-27B-MTP-Q4_K_M.gguf
M2B=/models/Qwen3.5-2B-MTP-Q4_K_M.gguf
OUT=${OUT:-bench-results}
PORT=8090
PROMPT='Explain in detail how speculative decoding works in large language model inference. Cover the draft model, the verification step, the acceptance rate, and how it trades compute for latency. Be thorough and technical.'

DEV=(--device /dev/kfd --device /dev/dri/renderD130 --device /dev/dri/card2 --group-add video)
VOL=(-v /opt/llm/models:/models -v /mnt/nvmedata/models:/models2)

mkdir -p "$OUT"

log() { echo "[$(date '+%F %T')] $*"; }

# 벤치 대상 GPU(renderD130) VRAM 사용량 (B)
vram1() { cat /sys/class/drm/renderD130/device/mem_info_vram_used 2>/dev/null || echo 0; }

# llama-bench 실행 (jsonl 출력 + 로깅)
run_bench() {
  local img="$1" tag="$2" entry="$3"; shift 3
  local base="${img%%:*}"; base="${base##*/}"
  local f="$OUT/${tag}_${base}"
  local vb va rc
  vb=$(vram1)
  log "bench 시작: $tag ($base) VRAM before: $vb B"
  docker run --rm "${DEV[@]}" "${VOL[@]}" \
    -e HIP_VISIBLE_DEVICES=0 \
    --entrypoint "$entry" \
    "$img" "$@" -o jsonl > "${f}.jsonl" 2> "${f}.log"
  rc=$?
  va=$(vram1)
  log "bench 종료: $tag rc=$rc VRAM after: $va B"
  if [ $rc -ne 0 ]; then
    log "ERROR: ${f}.log:"
    tail -5 "${f}.log"
  fi
  echo "$base $f" > /tmp/.bench_last
}

# llama-batched-bench 실행 (표 형식 stdout)
run_batched() {
  local img="$1" tag="$2"; shift 2
  local base="${img%%:*}"; base="${base##*/}"
  local f="$OUT/${tag}_${base}"
  local vb va rc
  vb=$(vram1)
  log "batched 시작: $tag ($base) VRAM before: $vb B"
  docker run --rm "${DEV[@]}" "${VOL[@]}" \
    -e HIP_VISIBLE_DEVICES=0 \
    --entrypoint /app/llama-batched-bench \
    "$img" "$@" > "${f}.out" 2> "${f}.log"
  rc=$?
  va=$(vram1)
  log "batched 종료: $tag rc=$rc VRAM after: $va B"
  [ $rc -ne 0 ] && { log "ERROR: ${f}.log:"; tail -5 "${f}.log"; }
  return $rc
}

# llama-server 시나리오 (S4~S6): ready 대기 -> completion -> metrics/로그 스냅샷 -> 종료
run_server() {
  local img="$1" tag="$2"; shift 2
  local base="${img%%:*}"; base="${base##*/}"
  local f="$OUT/${tag}_${base}"
  local cname="bench-${tag}-${base}"
  local vb va
  vb=$(vram1)
  docker rm -f "$cname" >/dev/null 2>&1
  log "server 시작: $tag ($base) VRAM before: $vb B"
  docker run -d --name "$cname" "${DEV[@]}" "${VOL[@]}" \
    -e HIP_VISIBLE_DEVICES=0 \
    -p ${PORT}:$PORT \
    --entrypoint /app/llama-server \
    "$img" -m "$M27" \
    -c 32768 -b 4096 -ub 1024 -ngl 99 -fa on --jinja --metrics \
    --host 0.0.0.0 --port $PORT "$@"
  # ready 대기 (최대 5분)
  local ok=0
  for i in $(seq 1 150); do
    if curl -s --max-time 3 "http://localhost:${PORT}/health" 2>/dev/null | grep -q ok; then ok=1; break; fi
    if ! docker ps --filter "name=^${cname}$" --format '{{.Names}}' | grep -q "$cname"; then break; fi
    sleep 2
  done
  if [ $ok -ne 1 ]; then
    log "ERROR: $tag 서버 ready 실패. 로그:"
    docker logs "$cname" 2>&1 | tail -30 | tee "${f}.log"
    docker rm -f "$cname" >/dev/null 2>&1
    return 1
  fi
  # metrics baseline 스냅샷
  curl -s "http://localhost:${PORT}/metrics" > "${f}.metrics.before"
  # completion 요청 (tg 256토큰)
  curl -s "http://localhost:${PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"bench\",\"max_tokens\":256,\"prompt\":\"${PROMPT}\"}" > "${f}.completion.json"
  # metrics 종료 스냅샷 + 로그
  curl -s "http://localhost:${PORT}/metrics" > "${f}.metrics.after"
  docker logs "$cname" > "${f}.log" 2>&1
  va=$(vram1)
  docker stop "$cname" >/dev/null 2>&1
  docker rm -f "$cname" >/dev/null 2>&1
  log "server 종료: $tag VRAM after: $va B"
  # 로딩 시간 / spec 수용률 로그 추출
  grep -E 'load time|spec:' "${f}.log" | head -10 > "${f}.summary" 2>/dev/null || true
}

# jsonl에서 pp/tg 평균 추출 (양 스키마 호환: .test/.result.mean 또는 n_prompt/n_gen/avg_ts)
bench_pp() { jq -r 'if .test? then (select(.test|test("^pp")) | .result.mean) else (select(.n_prompt>0) | .avg_ts) end' "$1" 2>/dev/null | head -1; }
bench_tg() { jq -r 'if .test? then (select(.test|test("^tg")) | .result.mean) else (select(.n_gen>0) | .avg_ts) end' "$1" 2>/dev/null | head -1; }

# results.csv 누적
CSV="$OUT/results.csv"
[ -f "$CSV" ] || echo 'build,scenario,pp_tps,tg_tps,accept_rate,vram_after_B,notes' > "$CSV"
csv_row() { echo "$*" >> "$CSV"; }

# ---------------------------------------------------------------- scenario 정의

smoke() {
  log "=== 스모크: 2B 모델 S1 (beellama) ==="
  run_bench "$BEELLAMA_IMG" smoke_s1 /app/llama-bench \
    -m "$M2B" -ngl 99 -fa on -ctk f16 -ctv f16 -b 4096 -ub 1024 -p 512 -n 512 -r 1
  local f=$(awk '{print $2}' /tmp/.bench_last)
  local pp=$(bench_pp "${f}.jsonl"); local tg=$(bench_tg "${f}.jsonl")
  local va=$(vram1)
  csv_row "beellama,smoke-S1,${pp:-NA},${tg:-NA},NA,${va},2B Q4_K_M 파이프라인 검증"
}

bench27() {
  local K
  for K in $BUILDS; do
    local img=$BEELLAMA_IMG; [ $K = main ] && img=$MAIN_IMG
    local tag=${K}_s1
    log "=== S1 ($K): f16 baseline ==="
    run_bench "$img" "$tag" /app/llama-bench \
      -m "$M27" -ngl 99 -fa on -ctk f16 -ctv f16 -b 4096 -ub 1024 -p 512 -n 512 -r 3
    csv_row "$K,S1-f16,$(bench_pp $(awk '{print $2}' /tmp/.bench_last).jsonl),$(bench_tg $(awk '{print $2}' /tmp/.bench_last).jsonl),NA,NA,f16 KV baseline"

    log "=== S2 ($K): q8_0/q5_0 (TheTom 비대칭) ==="
    run_bench "$img" "${K}_s2" /app/llama-bench \
      -m "$M27" -ngl 99 -fa on -ctk q8_0 -ctv q5_0 -b 4096 -ub 1024 -p 512 -n 512 -r 3
    csv_row "$K,S2-q8_0_q5_0,$(bench_pp $(awk '{print $2}' /tmp/.bench_last).jsonl),$(bench_tg $(awk '{print $2}' /tmp/.bench_last).jsonl),NA,NA,TheTom 비대칭 KV"

    if [ $K = beellama ]; then
      log "=== S3 (beellama): kvarn5/kvarn4 + tail 1024 ==="
      run_bench "$img" beellama_s3 /app/llama-bench \
        -m "$M27" -ngl 99 -fa on -ctk kvarn5 -ctv kvarn4 --kv-tail-tokens 1024 -b 4096 -ub 1024 -p 512 -n 512 -r 3
      csv_row "beellama,S3-kvarn5_4,$(bench_pp $(awk '{print $2}' /tmp/.bench_last).jsonl),$(bench_tg $(awk '{print $2}' /tmp/.bench_last).jsonl),NA,NA,KVarN 캐시 (tail 1024)"
    else
      log "=== S3 (llm-main): q8_0/turbo3 (현행 운영 구성) ==="
      run_bench "$img" main_s3 /app/llama-bench \
        -m "$M27" -ngl 99 -fa on -ctk q8_0 -ctv turbo3 -b 4096 -ub 1024 -p 512 -n 512 -r 3
      csv_row "llm-main,S3-q8_0_turbo3,$(bench_pp $(awk '{print $2}' /tmp/.bench_last).jsonl),$(bench_tg $(awk '{print $2}' /tmp/.bench_last).jsonl),NA,NA,현행 운영 구성(turbo3)"
    fi
  done
}

server27() {
  local K
  for K in $BUILDS; do
    local img=$BEELLAMA_IMG; [ $K = main ] && img=$MAIN_IMG
    log "=== S4 ($K): draft-mtp n-max 3 ==="
    run_server "$img" "${K}_s4" --spec-type draft-mtp --spec-draft-n-max 3
    log "=== S5 ($K): draft-dflash + profit 컨트롤러 ==="
    run_server "$img" "${K}_s5" --spec-type draft-dflash --spec-dm-controller profit
    if [ $K = beellama ]; then
      log "=== S6 (beellama): draft-mtp + kvarn5/kvarn4 ==="
      run_server "$img" beellama_s6 --spec-type draft-mtp --spec-draft-n-max 3 -ctk kvarn5 -ctv kvarn4 --kv-tail-tokens 1024
    else
      log "=== S6 (llm-main): draft-mtp + q8_0/turbo3 ==="
      run_server "$img" main_s6 --spec-type draft-mtp --spec-draft-n-max 3 -ctk q8_0 -ctv turbo3
    fi
  done
}

batched() {
  local K
  for K in $BUILDS; do
    local img=$BEELLAMA_IMG; [ $K = main ] && img=$MAIN_IMG
    log "=== S7 ($K): llama-batched-bench -npl 4 (4 병렬 시퀀스) ==="
    run_batched "$img" "${K}_s7" \
      -m "$M27" -c 32768 -b 4096 -ub 1024 -ngl 99 -fa on -ctk f16 -ctv f16 \
      -npl 4 -npp 512 -ntg 512
    local base="${img%%:*}"; base="${base##*/}"
    local f="$OUT/${K}_s7_${base}"
    local pp=$(awk '$6==4 && $12!~/^-/ {print $12; exit}' "$f.out" 2>/dev/null)
    local tg=$(awk '$6==4 && $16!~/^-/ {print $16; exit}' "$f.out" 2>/dev/null)
    csv_row "$K,S7-batched_npl4,${pp:-NA},${tg:-NA},NA,NA,4병렬 시퀀스 동시 디코딩"
  done
}

case "${1:-all}" in
  smoke)    smoke ;;
  bench)    bench27 ;;
  server)   server27 ;;
  batched)  batched ;;
  all)      smoke; bench27; server27; batched ;;
  *)        echo "usage: $0 {smoke|bench|server|batched|all}"; exit 2 ;;
esac
log "완료. 원시 출력: $OUT/"
