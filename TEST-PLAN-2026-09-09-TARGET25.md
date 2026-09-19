# 테스트 계획: Qwen3.8-Flash-Next decode 25+ t/s 달성 (2026-09-09)

> 목표: 듀얼 7900 XTX (48GB)에서 Flash-Next decode **25 t/s 이상** 실서버 검증
> 현재 알려진 최고: MTP 적용 23.6 t/s @ 20k ctx (이전 세션 문서화, expert cache 이식 전)
> 이번 세션 이식: MoE expert cache (+26% llama-bench), async CPU, MTP head
>
> **주의: 본 문서는 계획이며, vLLM 운영 서버가 GPU를 점유 중이므로 당장 실행하지 않음.
> GPU 독점 조건 확보 후 아래 순서대로 실행.**

---

## 1. 목표 수치의 논리적 근거

```
이전 세션 최고 (MTP @20k ctx):        23.6 t/s
이번 expert cache 이식 (+26%):        23.6 x 1.26 = 29.7 t/s (곱연산 가정)
슬롯 최적화 추가 여지:                 16 -> 40~56 슬롯
-------------------------------------------------------------------------
목표 25 t/s = 23.6의 +6% = "MTP + expert cache(일부)" 만으로도 도달 가능 범위
```

**핵심 가정**: expert cache의 +26% (llama-bench, GPU 점유 상태) 가 실서버 MTP 구성에서도
유지되거나 일부(절반 이상) 발휘되면 25 t/s 달성 가능.

---

## 2. 테스트 환경 (고정 조건)

| 항목 | 값 |
|---|---|
| GPU | 7900 XTX x2 (gfx1100, ROCm 7.2.0), **독점 필요** (vLLM 종료/일시 중지) |
| 모델 | `/mnt/nvmedata/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/` (82GB, 3파트) |
| MTP | `/mnt/nvmedata/models/Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64/MTP/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` |
| 프로필 | `/home/baramofme/IdeaProjects/llama-cpp-turboquant/qwen38-merged.csv` (코드+채팅 병합, 15,153줄) |
| 빌드 | `build/bin/llama-server` (이번 세션 이식 완료본) |
| CPU | i5-14600K, -t 6 (물리 코어 수) |

**사전 확인 (실측 전)**
```
rocm-smi --showpids    # vLLM(1090729/1090730) 등 GPU 프로세스 0개인지
rocm-smi --showmeminfo vram   # GPU 여유 >= 20GB/카드
```

---

## 3. 테스트 시나리오 (순서대로, 각 5분 이내)

### Phase 0: 기준선 재확립 (GPU 독점 후)
| # | 구성 | 측정 항목 | 기대치 |
|---|---|---|---|
| 0a | `-ngl 99 -ncmoe 99 -fa on -ctk q8_0 -ctv q8_0 -c 16384`, 캐시 없음 | tg64 | ~5~7 t/s (GPU-free) |
| 0b | 동일 + `--moe-cache-profile ... --moe-cache-slots 16` | tg64 | 0a 대비 상승 확인 |
| 0c | 동일 + MTP draft (`-md ... --spec-type draft-mtp`) | tg64 | 0b 대비 상승 |

**통과 기준**: 0a < 0b < 0c 순서로 단조 증가 + expert cache 로그
(`init_moe_expert_cache: expert cache: 48 layers x N slots`) 확인

### Phase 1: expert cache 슬롯 스윕 (MTP + 캐시)
| # | 슬롯 | VRAM | 기대 |
|---|---|---|---|
| 1a | 16 | 1.4GB | 기준 |
| 1b | 24 | 2.2GB | 상승 |
| 1c | 32 | 2.9GB | 상승 |
| 1d | 40 | 3.6GB | 상승 |
| 1e | 48 | 4.3GB | peak 후보 |
| 1f | 56 | 5.1GB | VRAM 여유 시 |

**통과 기준**: tg64 기준 최고 슬롯 식별. 40슬롯에서 `pack allocation failed`가
뜨면 VRAM 부족 → 32~40 사이에서 재탐색.
(주의: 런타임 CUDA pool 성장이 load-time 체크보다 크므로, 슬롯은 "최대치 - 여유"로 선정)

### Phase 2: 컨텍스트 길이 영향
| # | ctx | 기대 (이전 세션) |
|---|---|---|
| 2a | 20k | **23.6 t/s (MTP 기준)** |
| 2b | 50k | 17.5 t/s |
| 2c | 65k | 낮음 (KV 증가) |

**판단**: 25 t/s 목표는 **20k 이하 컨텍스트**에서 유효. 실사용 컨텍스트에 맞춰 설정.

### Phase 3: async CPU overlap (기본 on) 확인
- `--sched-async-cpu` (기본 true) vs `--no-sched-async-cpu` 비교
- 기대: MTP + 캐시 조합에서 +4~5% (codacus 기준, HIP 미검증)

### Phase 4: KV 캐시 압축 (VRAM 여유 확보)
| # | 설정 | 용도 |
|---|---|---|
| 4a | `-ctk q8_0 -ctv q8_0` | 기본 (이전 세션 검증) |
| 4b | `-ctk q4_0 -ctv q4_0` | VRAM 절약 → 슬롯 증가로 전환 |

**판단**: KV를 줄여 확보한 VRAM을 expert cache 슬롯으로 전환하는 것이
KV 정밀도 손실보다 이득 (codacus README 권장).

---

## 4. 성공/실패 판정 기준

| 판정 | 기준 |
|---|---|
| ✅ 목표 달성 | tg64 >= 25 t/s (20k ctx, MTP + expert cache) |
| ✅ 부분 성공 | 20~25 t/s (이전 23.6 대비 개선) |
| ❌ 회귀 | < 23.6 t/s (이전 세션 최고보다 낮음) → 슬롯/설정 원복 |

**측정 프로토콜** (모든 시나리오 공통)
1. 동일 프롬프트 (고정 텍스트 20토큰 + 생성 256토큰)
2. 3회 반복, 중앙값 채택
3. GPU 점유 재확인 (다른 프로세스 없음)
4. 로그에서 draft acceptance 기록 (MTP 품질 지표)

---

## 5. 위험 요소와 대응

| 위험 | 대응 |
|---|---|
| vLLM GPU 점유로 측정 불가 | Phase 0 전에 vLLM 일시 중지 (운영 영향 확인 후) |
| expert cache pack alloc 실패 (VRAM) | 슬롯 축소 + KV 압축(-ctk q4_0) |
| MTP draft OOM (이전 세션 1GB 부족) | 슬롯 4~8 감소로 VRAM 확보 |
| HIP 불안정 (host-pin/prefetch) | 이미 자동 off 확인됨. 재발 시 `GGML_CUDA_REGISTER_HOST` 미설정 유지 |
| draft acceptance 낮음 (<30%) | 프로필/슬롯 재조정, `--spec-draft-n-max` 튜닝 |

---

## 6. 실행 명령 템플릿

```bash
# GPU 독점 확인
rocm-smi --showpids | grep -c "PID"  # 1 (헤더) 이어야 함

# Phase 0b: expert cache만
./build/bin/llama-server \
  -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  --moe-cache-profile /home/baramofme/IdeaProjects/llama-cpp-turboquant/qwen38-merged.csv \
  --moe-cache-slots 40 \
  -ngl 99 --n-cpu-moe 99 -t 6 -fa on -ctk q8_0 -ctv q8_0 \
  -c 20480 -np 1 -b 2048 -ub 512 --jinja --host 0.0.0.0 --port 12000 &

# Phase 0c: MTP 추가
#   + -md /mnt/nvmedata/models/.../mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
#     --spec-type draft-mtp --spec-draft-n-max 1

# 측정 (20토큰 프롬프트, 256 생성)
curl -s http://127.0.0.1:12000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen4exp","messages":[{"role":"user","content":"<고정 20토큰 프롬프트>"}],"max_tokens":256,"stream":false}' \
  | python3 -c "import sys,json; t=json.load(sys.stdin)['timings']; print(t['predicted_per_second'])"

# 로그에서 확인
grep -E "expert cache|draft acceptance|print_timing" /tmp/server.log
```

---

## 7. 결과 기록 형식

각 시나리오 결과를 `bench-results/2026-09-09-target25/*.json`에 저장하고
아래 표에 채움:

| 시나리오 | 슬롯 | ctx | MTP | async | tg (t/s) | draft accept | 비고 |
|---|---|---|---|---|---|---|---|
| 0a baseline | - | 16k | - | on | | | |
| 0b +cache16 | 16 | 16k | - | on | | | |
| 0c +MTP | 16 | 16k | on | on | | | |
| 1e +cache40 | 40 | 16k | on | on | | | |
| 2a ctx20k | 40 | 20k | on | on | | | 목표 판정 |
| 3a async off | 40 | 20k | on | off | | | |
| 4b KV q4 | 48 | 20k | on | on | | | |

---

## 8. 최종 목표 대비 로드맵

```
현재(이식 후, 미검증): 23.6 t/s (이전 세션 MTP)
  + expert cache: 23.6 x 1.26 = 29.7 (가정)  -> 목표 25 초과 가능
  + 슬롯 최적화: 추가 +5~10%
  - 컨텍스트 제약: 20k 이하에서만 유효
  - HIP 불안정 요소: host-pin/prefetch 제외 (자동 off)

결론: 25 t/s는 "MTP + expert cache(슬롯 40 전후)" 조합으로 도달 가능성이 높음.
핵심 검증 포인트는 expert cache 이득(+26%)이 실서버 MTP 구성에서 유지되는가.
```