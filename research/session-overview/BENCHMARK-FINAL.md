# 최종 벤치마킹 비교: beellama(KVarN) vs TurboQuant in llama.cpp

> 작성: 2026-09-06 (Asia/Seoul)
> 목적: ROCm 10 + rdna3-boosts 통합 후, 운영에 쓸 최종 런타임 선택의 근거 문서
> 판정: **TurboQuant(ROCm 10) 채택** - prefill/tg/MTP/정확도 종합 우위

---

## 1. 대상 (비교한 것들)

| 이름 | 포크 | ROCm | KV 캐시 | MTP | 빌드 위치 |
|---|---|---|---|---|---|
| **7.2.1 (기존 운영)** | llama-cpp-turboquant (theTom) | 7.2.1 docker | q8_0/turbo3 + f16 | draft-mtp | llm-main 이미지 (과거) |
| **beellama+boosts** | beellama.cpp v0.4.5 + rdna-boosts | 10.0 docker | kvarn5/kvarn4, q8_0/q5_0 | draft-mtp-adaptive | beellama-boosts/src/build-rocm10 |
| **tq+boosts** ✅ | theTom + rdna-boosts (신규) | 10.0 docker | q8_0/turbo3 | draft-mtp-adaptive | /tmp/tq-patch-test/build-rocm10 |
| 이미지 | - | - | - | - | baramofme/llama-cpp-rocm:gfx1100-rocm10-tbq-rboosts |

기반: beellama v0.4.5(44a36969) / theTom 4a54c52 + stew675/llama-cpp-rdna-boosts 8블록(0001/0004/0010/0011 포함).

## 2. 벤치마킹 테스트 방법

### 2.1 도구
| 도구 | 용도 | 명령 형태 |
|---|---|---|
| llama-bench | pp/tg 표준 벤치 | `llama-bench -m <model> -ngl 99 -fa on -ctk <K> -ctv <V> -b 4096 -ub 1024 -p <N> -n <N> -r 3 -o jsonl` |
| llama-server | MTP 수용률/장기 프롬프트 실측 | `/v1/completions` + `slot print_timing` 로그 |
| needle-test | long-context 정확도 (사용자 정의) | 시스템 텍스트 8K~100K + magic constant 추출 |
| rocm-smi | VRAM 델타 | `--showmeminfo vram` |

### 2.2 측정 항목
- **pp (prefill)**: p=512(표준) + p=4096(긴 프롬프트) + 4445토큰 서버 prefill(운영 재현)
- **tg (decode)**: n=512 (llama-bench) + n=100~256 (서버, MTP 적용 시)
- **MTP acceptance**: 서버 로그 `draft acceptance = X%`
- **KV 메모리**: 소스 기반 bytes/헤드/token 계산 + rocm-smi VRAM 델타
- **정확도**: needle-in-haystack (8K/30K, needle@10~78%)

### 2.3 공통 조건 (재현 필수)
- GPU1(RX 7900 XTX, gfx1100) 단독, HIP_VISIBLE_DEVICES=1 (docker는 =0 + renderD130/card2)
- 모델: Qwen3.8-27B-MTP-Q4_K_M(Q4, 벤치용) / Q3_K_M(운영 재현용, 140K ctx)
- 배치: `-b 4096 -ub 1024` / `--cache-ram 19152 --no-mmap`(운영)
- 샘플링: MTP 벤치 `--reasoning off --jinja` (Qwen3 thinking 모드 제거 필수)

## 3. 최종 벤치마킹 비교표

### 3.1 표준 벤치 (llama-bench, p=512/n=512, Q4_K_M, GPU1)
| 구성 | 7.2.1 | tq+boosts | beellama(q8/q5) |
|---|---|---|---|
| f16/f16 | 921 / 34.6 | 987 / 38.8 | - |
| q8_0/turbo3 | 911 / 33.6 | **936~956 / 37.5~37.7** | (없음) |
| q8_0/q5_0 | 911 / 33.4 | 964 / 37.6 | 1008 / 36.3 |

### 3.2 운영 조건 (llama-server, Q3_K_M, 140K, 4445토큰 prefill, adaptive MTP)
| 지표 | **tq+boosts (q8/t3)** | beellama (q8/q5) | beellama (kvarn5/4) |
|---|---|---|---|
| prefill 4445t | **793 t/s** | 905 t/s | 84~133 t/s (폴백) |
| tg | **52.5 t/s** | 47.4 t/s | - |
| MTP 수용률 | **55.8~65.3%** | 52.2% | - |

### 3.3 정확도 (needle-in-haystack, 27B)
| 시나리오 | beellama (모든 KV) | **tq+boosts** |
|---|---|---|
| 30K ctx, needle@78% | ❌ `////` 반복 붕괴 | ✅ 회수 |
| 30K ctx, needle@10% | ❌ 붕괴 | - |
| 8K ctx, needle@50% | ❌ 붕괴 | - |
| (2B, 30K) | ✅ 회수 | ✅ 회수 |

### 3.4 KV 캐시 메모리 (K+V, @32K ctx, head_dim 256)
| 타입 | bytes/헤드/token | vs f16 |
|---|---|---|
| f16/f16 | 512+512 | 100% |
| q8_0/q5_0 | 264+148 | 40.2% |
| q8_0/turbo3 | 264+100 | **35.5%** (실용) |
| kvarn5/kvarn4 | 172+140 | 30.5% (비실용, pp 폴백) |

## 4. 판정

**TurboQuant(ROCm 10 + boosts + q8_0/turbo3 + adaptive MTP) 채택.**
- 같은 조건(p512)에서 7.2.1 대비 pp +2~3%, tg +11% 초과
- 운영 140K에서 beellama 대비 정확도 안정(turbo) + MTP 수용률 우위
- KV 압축 35.5%로 원-클릭 실용, KVarN은 30.5%지만 prefill 폴백으로 비실용

## 5. 테스트 유의사항

1. **llama-bench는 `-c` 없음** - 컨텍스트 길이를 잡으려면 llama-server 사용
2. **긴 프롬프트에서 KVarN은 `--kv-tail-tokens` 없이 실행 금지** - portable 폴백으로 pp -63%
3. **Qwen3.8은 `--reasoning off` 필수** - 켜져 있으면 raw prompt에서 `////` 반복 붕괴 (정확도 테스트 오염)
4. **GPU0(GQ0, 운영)과 GPU1(벤치) 분리** - HIP_VISIBLE_DEVICES 혼동 주의 (native: 1, docker: 0)
5. **측정 조건 통일** - 같은 모델/QV/배치/세션에서 비교 (서로 다른 시점 값은 노이즈로 오해 가능, §17)
6. **ROCm 10 바이너리는 GLIBC 2.43 필요** - 호스트 직접 실행 불가, 도커 컨테이너에서 실행
7. **cache-ram 오프로드**: 140K 등 큰 컨텍스트는 VRAM 초과 시 `--cache-ram` 필수 (KV가 host RAM으로)
8. **KV 타입 강제 규칙**: beellama에서 q8_0/K + kvarn/V는 K를 kvarn으로 자동 승격 (KVarn은 K/V 동반)

## 6. 향후 제안

### 6.1 다음 검증
- **B000-MTP 실제 테스트**: 256토큰+ 긴 생성에서의 MTP 수용률 측정 (1회 측정의 노이즈 제거, 3회 반복)
- **140K needle**: 100K/140K 컨텍스트에서 turbo3 정확도 재확인 (30K에서 이미 성공, 스케일업 검증)
- **Qwen3.6-35B-A3B** (사용자 바이브 코딩 모델): 이 모델로 동일 비교 - 140K 루프/변수 오류가 turboquant에서 해소되는지 직접 확인

### 6.2 소프트웨어 개선
- **KVarN D=256 WMMA 지원** (`fattn-kvarn-route-policy.h` DKQ>128 제한 완화): CUDA split/vector 디코딩 HIP 이식 시 KVarN의 u prefill 폴백이 해결 - turboquant와 공정 경쟁 가능
- **turboquant에 adaptive MTP 백포트**: theTom은 이미 보유 (이번 통합으로 확인) - 유지
- **0013(fused MoE)**: Qwen3.6-35B-A3B(MoE) 사용 시 이득 예상 - 다음 통합 후보

### 6.3 운영 적용
1. Dokploy compose `image`를 `baramofme/llama-cpp-rocm:gfx1100-rocm10-tbq-rboosts`로 교체
2. config.ini Dense/Dense-1: `cache-type-k q8_0, cache-type-v turbo3, spec-type draft-mtp-adaptive, spec-draft-n-min-adaptive 2` (kv-tail-tokens 제거)
3. 배포 후 로그 확인: `draft acceptance` 50%+, prefill 4445t 약 6초 이내

## 7. 원시 데이터 위치
- 벤치: `bench-results-tq-boosts/`, `bench-results-tq-prefill-sweep/`, `bench-results-kv-vs-tq/`
- needle: `/tmp/needle_test.py`, `/tmp/needle_chat.py` (+ 서버 로그)
- 진행 로그: `bench-kv-vs-turboquant.md` (§1~17 전체 이력)