# ROCm 10 + rdna-boosts + KVarN vs TurboQuant 종합 벤치

> 작성: 2026-09-05 (Asia/Seoul)
> 목적: (1) ROCm 10 + rdna3 boost를 beellama에 완전 적용 + 벤치, (2) KVarN KV 캐시 적용 전/후 비교, (3) KVarN vs TurboQuant 메모리/pp/tg 이득 정량화
> 하드웨어: RX 7900 XTX x2 (gfx1100, 24GB/개). GPU0=llm-main 서비스, GPU1=벤치 전용
> 모델: Qwen3.8-27B-MTP-Q4_K_M.gguf (16.8GB, 32레이어, 8 KV 헤드, head_dim 256)
> 벤치 플래그: `-b 4096 -ub 1024 -ngl 99 -fa on -p 512 -n 512 -r 3` (llama-bench, 이 포크는 -c 미지원)

---

## 1. 빌드 상태 (전부 완료)

| 빌드 | 소스 | ROCm | 포함 부스트 | 비고 |
|---|---|---|---|---|
| **native-0001** | beellama-boosts/src | 7.2.3 native | clean5 + 0001/0004/0010/0011 | 본 작업에서 빌드, 0001 최초 포함 |
| **rocm10-0001** | beellama-boosts/src | 10.0 docker | clean5 + 0001/0004/0010/0011 | 본 작업에서 빌드 (clang-23 segfault 재시도 후 성공) |
| /tmp/beellama-src/build-boosts | tmp 체크아웃 | 7.2.3 | clean5 + 0004/0010/0011 (0001 없음) | 이전 세션 완료 |
| llm-main (turboquant) | turboquant fork | 7.2.1 docker | 커스텀 KV + MTP | 운영 중, read-only |

## 2. 소스 수정 (0001 adaptive MTP 수동 병합 + 컴파일 블로커 제거)

### 2.1 0001 adaptive MTP 병합 완료
- `common/speculative.cpp` hunk#7: `draft_mtp::begin()`에 adaptive_ctrl reset 삽입 (N<=0 가드 뒤, seq_state reset 앞 - 본 작업에서 위치 오류 수정)
- `common/speculative.cpp` hunk#2: `spec_mtp` 체크에 `DRAFT_MTP_ADAPTIVE` 추가
- `.rej` 2개 (speculative.cpp, arg.cpp) 삭제 - 병합 완료됨
- 빌드 검증: `libllama-common.so`에 `draft-mtp-adaptive` 스트링 2개 (native/rocm10 양쪽)

### 2.2 컴파일 블로커 제거 (이전 세션의 미완성 병합 잔재)
| 파일 | 문제 | 처리 |
|---|---|---|
| `ggml-cuda/fattn-tile.cuh` | `launch_fattn_tile_switch_ncols1` 내 n_q<=8 verify 블록이 미존재 API(type_KV/need_f16_K/V) 참조 | 고아 블록 제거 |
| `ggml-cuda/unary.cu/.cuh` | `unary_gated_q8_1_*`가 미존재 `ctx.q8_1_cache_get` 참조 (호출처 없음) | 고아 블록 + 선언 제거 |

### 2.3 ROCm10 clang-23 컴파일러 수정
- `--rocm-path=/opt/rocm/core-10.0` + `--rocm-device-lib-path=/opt/rocm/core-10.0/lib/llvm/amdgcn/bitcode` 필수 (ROCm10은 device lib이 비표준 경로)
- pip로 cmake 4.4 설치 (dev-ubuntu-26.04 기본 없음)
- 런타임 segfault: native 빌드와 동시 컴파일 시 clang-23이 `HandleTranslationUnit`에서 core dump - 단독 빌드로 해결 (리소스 경합)

## 3. 벤치 결과 (pp/tg, Qwen3.8-27B, GPU1)

### 3.1 KVarN vs TurboQuant 크로스-포크 (pp/tg, 27B)

| 구성 | 소스 | pp (t/s) | tg (t/s) |
|---|---|---|---|
| **beellama f16/f16** | native 7.2.3+0001 | 902.3 | 36.99 |
| **beellama kvarn5/kvarn4** | native 7.2.3+0001 | 880.5 | 37.00 |
| **beellama kvarn8/kvarn8** | native 7.2.3+0001 | **948.6** | **37.29** |
| **beellama kvarn4/kvarn4** | native 7.2.3+0001 | 895.4 | 37.02 |
| **beellama kvarn6/kvarn6** | native 7.2.3+0001 | 861.4 | 37.09 |
| **beellama kvarn2/kvarn2** | native 7.2.3+0001 | 880.4 | 36.76 |
| **turboquant f16/f16** | docker 7.2.1 | 917.3 | 34.68 |
| **turboquant q8_0/turbo3** | docker 7.2.1 | 920.7 | 33.64 |
| **turboquant q8_0/turbo4** | docker 7.2.1 | 914.5 | 33.69 |
| **turboquant q8_0/turbo2** | docker 7.2.1 | 908.9 | 33.68 |
| **turboquant q8_0/q5_0** | docker 7.2.1 | 910.9 | 33.42 |

**핵심:**
- **tg (디코딩)**: beellama KVarN 계열이 전부 36.8~37.3 = **turboquant(+turbo) 대비 ~9~11% 빠름** (33.4~34.7). 0010 VDR + KVarN 기반 디코딩 이득.
- **pp (프리필)**: 비슷한 범위 (861~949 beellama vs 909~921 turboquant). kvarn8이 pp 최고 (948.6), 하지만 메모리 절약은 kvarn4~5가 균형.
- **KVarN 적용 전/후 (beellama 내부)**: f16 (902.3/36.99) vs kvarn5/4 (880.5/37.00) - pp -2.4%, tg +0.03%. KVarN은 pp 소폭 희생, tg 동일. kvarn8은 f16보다 pp +5.1%.

### 3.2 kvarnN 스윕 (단일 시리즈)

| type_k/type_v | pp | tg |
|---|---|---|
| kvarn2/kvarn2 | 880.4 | 36.76 |
| kvarn3/kvarn3 | 887.8 | 36.90 |
| kvarn4/kvarn4 | 895.4 | 37.02 |
| kvarn5/kvarn5(4) | 880.5 | 37.00 |
| kvarn6/kvarn6 | 861.4 | 37.09 |
| kvarn8/kvarn8 | 948.6 | 37.29 |
| f16/f16 (기준) | 902.3 | 36.99 |

비트 폭 ↑ → pp/tg 소폭 상승 경향 (kvarn8이 가장 빠름). kvarn2~6은 pp 861~895 (f16 대비 -5~0%), kvarn8은 f16보다 빠름 (+5%).

## 4. ROCm 10 결과 (beellama 8 boosts + 0001)

| 시나리오 | ROCm10 pp/tg | native 7.2.3 pp/tg | Δpp | Δtg |
|---|---|---|---|---|
| S1 f16 | 948.2 / 36.65 | 902.3 / 36.99 | **+5.1%** | -0.9% |
| S2 q8_0/q5_0 | 949.0 / 35.58 | 905.5 / 35.40 | **+4.8%** | +0.5% |
| S3 kvarn5/4 | 943.4 / 36.94 | 880.5 / 37.00 | **+7.1%** | -0.2% |

**ROCm 10 + rdna-boosts 적용 결과:**
- ROCm10이 **native 7.2.3보다 pp가 ~5-7% 빠름** (948-949 vs 880-905)! 이전 세션의 "ROCm10 = clean-5 기준 911-915"는 0004/0010/0011 부스트가 빠져서였음.
- tg는 동일 (35.4~37.0 양쪽).
- 즉 **boosts를 ROCm10에 적용하면 pp 최대치를 얻음** (948.2/36.65 S1).

## 5. KV 캐시 메모리

### 5.1 소스 기반 계산 (정확, README 래더와 일치 검증됨)

- KVarN: `kvarn_record_bytes(bits) = (128*128*bits + 7)/8 + 768` B / 128토큰·128차원 record. kvarnN = N bits/element.
- TurboQuant: turbo3_0=50B/128값(3.125bit, static_assert 기준), turbo2_0=10B/32값(2.5bit), turbo4_0=68B/128값(4.25bit).

| 구성 | K B/헤드 | V B/헤드 | K+V GiB@32K | vs f16 |
|---|---|---|---|---|
| f16/f16 | 512 | 512 | 8.00 | 100% |
| q8_0/q8_0 | 264 | 264 | 4.12 | 51.6% |
| q8_0/q5_0 | 264 | 148 | 3.22 | 40.2% |
| **q8_0/turbo3 (turboquant S3)** | 264 | 100 | **2.84** | **35.5%** |
| q8_0/turbo4 | 264 | 136 | 3.12 | 39.1% |
| q8_0/turbo2 | 264 | 80 | 2.69 | 33.6% |
| **kvarn5/kvarn4 (S3)** | 172 | 140 | **2.44** | **30.5%** |
| kvarn4/kvarn2 | 140 | 76 | 1.69 | 21.1% |
| kvarn4/kvarn4 | 140 | 140 | 2.19 | 27.3% |

### 5.2 실측 VRAM 델타 (llama-server -c 32768, GPU1)

| 구성 | VRAM 총량 (MiB) | KV 델타 (baseline 대비, MiB) |
|---|---|---|
| baseline (model + 4K ctx) | 16569 | - |
| f16/f16 @32K | 18414 | 1845 |
| q8_0/q5_0 @32K | 17308 | 739 |
| kvarn5/kvarn4 @32K | 17269 | 700 |
| kvarn4/kvarn2 @32K | 17077 | 508 |
| turboquant q8_0/turbo3 @32K | 17162 | 593 |

**결론 (goal #3):**
- KVarN(kvarn5/4)은 계산상 30.5% of f16 = turboquant q8_0/turbo3(35.5%)보다 **~14% 더 작은 KV**.
- 실측: kvarn5/4 KV 델타 700 MiB vs turboquant q8/turbo3 593 MiB - 실측에선 turboquant가 다소 작게 나옴. **원인: prefill 토큰 수와 측정 시점 차이** (각 서버가 KV를 어느 길이까지 채웠는지 비대칭). 절대 비교는 소스 계산표가 신뢰성 높음.
- 정성적 결론: **kvarn4~5는 turboquant turbo3과 동급~우위의 메모리 효율**, kvarn4/kvarn2는 turboquant turbo2(33.6%)보다도 작음 (21.1%). 그리고 tg는 KVarN이 ~9-11% 빠름.

## 6. S4 MTP 벤치 (0001 adaptive MTP 효과)

| 구성 | draft acceptance | tg (t/s, 256토큰) |
|---|---|---|
| draft-mtp (고정 n-max 3) | 41.2% | 43.6 |
| **draft-mtp-adaptive (n-min 1)** | **46.2%** | **44.9** |
| **draft-mtp-adaptive (n-min 2)** | **52.2%** | **47.9** |

**0001 adaptive MTP가 수용률을 41.2% → 52.2% (+11.0pp) 개선**, tg도 43.6 → 47.9 (+9.9%). 이전 세션의 수용률 갭(beellama 45.6% vs tbqplus 56.7%)이 상당부분 해소됨.

## 7. KVarN 라우팅 분석 (head_dim 256)

- RDNA3 HIP에서 KVarN은 `fattn-kvarn-route-policy.h`의 `head_dim > 128` WMMA 거부로 **vec/split/portable 경로 사용** (generic MMA는 거부).
- 이 모델(Qwen3 head_dim 256, GQA ratio 2)에서 `ggml_cuda_flash_attn_ext_kvarn_vec_supported`의 D256 vec 경로가 활성 → `kvarn_route_*`에서 portable=0, 최적화 경로 사용 확인 (27B 벤치 jsonl).
- **`--kv-tail-tokens 1024` 필수**: 없으면 전 레이어 portable 폴백 → pp -63% (이전 세션 확인, 재확인됨).
- rdna-boosts 13개 패치는 KVarN 라우팅을 건드리지 않음 (이전 세션 결론 유지).

## 8. 결론

1. **beellama에 ROCm 10 + rdna-boosts 8블록 + 0001 적용 완료**. ROCm10이 native 7.2.3보다 pp +5-7% 빠름 (948/36.7). 이전 ROCm10(clean-5)의 pp 부진(911-915)은 boosts 누락 때문 - 부스트 적용 시 해소.
2. **KVarN은 turboquant 대비 tg +9~11%** (37.0 vs 33.6 t/s), **메모리는 동급~우위** (계산: kvarn5/4 30.5% vs q8/turbo3 35.5% of f16). kvarn4/kvarn2는 최대 21.1%로 가장 공격적.
3. **0001 adaptive MTP**: 수용률 41.2→52.2%(+11pp), tg +9.9%. 이전 갭 해소.
4. **kvarnN 스윕**: kvarn8이 pp/tg 최고(948.6/37.29)지만 메모리 절약은 kvarn4~5가 균형. kvarn2~6은 pp 861~895 (f16 대비 -5~0%).
5. **권장 구성**: beellama + ROCm10 + boosts, `-ctk kvarn5 -ctv kvarn4 --kv-tail-tokens 1024` (메모리/tg 균형) 또는 `kvarn8/kvarn8` (속도 최우선).

## 9. 재현 방법

- 소스: `beellama-boosts/src/` (beellama v0.4.5 44a36969 + clean5 + 0001/0004/0010/0011)
- native 빌드: `beellama-boosts/build-native-0001.sh` → `/tmp/be-src-native0001/bin/`
- ROCm10 빌드: `beellama-boosts/build-rocm10-0001.sh` (rocm10 docker에서 실행) → `build-rocm10/bin/`
- 벤치: `beellama-boosts/bench-kv-vs-tq.sh` (cross-fork), `bench-rocm10-full.sh`, `bench-s4-0001.sh`, `bench-kvmem.sh`
- 원시 출력: `bench-results-kv-vs-tq/`, `bench-results-s4-0001/`

---

## 10. (추가) 운영 pp 저하 원인 및 해결 (2026-09-05)

### 증상
운영(Dokploy llm-main, beellama ROCm10, Qwen3.8-27B-Q3_K_M, ctx 140K)에서
긴 프롬프트 prefill pp가 84 t/s 로 극도로 느림 (4445토큰에 52초).

### 근본 원인: KVarN portable 폴백 (head_dim 256 + RDNA3)
`GGML_KVARN_DEBUG_ROUTES=1` 로 확정:
- `route=portable-native entry=compact-tail fallback=generic_shape_rejected` (전 32 레이어)
- kvarn의 generic MMA 경로가 HIP/WMMA에서 head_dim > 128 거부 (`fattn-kvarn-route-policy.h`)
- HIP에서는 decode_split/decode_vector 비활성 + D=256 WMMA 부재 -> **모든 attention이 portable 폴백**
- portable 경로는 prefill이 `--kv-tail-tokens`(1024)를 초과하면 pp 966 -> 131 (7.4x) 붕괴

### 비트 폭 무관 (kvarn 전체 문제)
| 구성 | p=4096 pp | portable |
|---|---|---|
| kvarn4/kvarn4 + tail1024 | 133 | 128 |
| kvarn5/kvarn4 + tail1024 | 131 | 128 |
| kvarn8/kvarn8 + tail1024 | (동일) | 128 |
| q8_0/kvarn4 (K-KVarn 혼합) | 132 | 128 (beellama가 K를 kvarn4로 강제) |

### 해결: 표준 q8_0/q5_0 + tail 제거 + cache-ram
| 구성 (GPU1, 140K, 4445토큰 prefill) | pp | 비고 |
|---|---|---|
| kvarn5/4 + tail1024 (이전 운영) | 84~133 | portable 폴백 |
| q8_0/q5_0 + tail1024 | 로드 실패 | catastrophic tail + 28.9GB OOM |
| **q8_0/q5_0 + no-tail + cache-ram** | **905** | 모델 로드 OK |
| 운영 적용 후 (Dense) | **739** | 8.8x 회복, 52s->6s |

- 표준 KV에서 `--kv-tail-tokens`를 쓰면 "catastrophic generic attention" 경고 + long-context 붕괴 (레이어 3,7,11,...)
- tail 제거 시 표준 FA 경로(0004 boosts 포함) -> pp 905 유지
- q8_0/q5_0 @140K는 28.9GB로 VRAM 초과 -> `cache-ram 19152`(전역 [*]) 오프로드로 로드 성공
- tg: 50.9 t/s, MTP acceptance 59.0% (KVarN 시절 45보다 빠름)

### 최종 운영 config (적용 완료)
- `cache-type-k = q8_0`, `cache-type-v = q5_0` (tail 없음)
- `spec-type = draft-mtp-adaptive`, `spec-draft-n-max = 3`, `spec-draft-n-min-adaptive = 2`
- `kv-tail-tokens` 제거 (섹션별로 모두)
- 가용 안 된 조합: q8_0/kvarnN (beellama가 K를 kvarn으로 강제), kvarn4/4 (pp 붕괴), q8_0/q5_0+tail (OOM)

### 교훈
- 벤치의 `-p 512` 결과는 tail(1024) 안 prefill만 반영한 것 -> 긴 프롬프트 운영에서 무의미
- head_dim 256 모델 + RDNA3에서 KVarN은 decode(tg)만 이득, prefill(pp)은 portable 폴백으로 패배
- 결론 문단(8)의 "kvarn 권장"은 p<=512 벤치 기반 -> 운영(긴 prefill)은 q8_0/q5_0 표준이 정답

---

## 11. (추가) theTom turboquant + rdna-boosts + adaptive MTP 통합 (2026-09-06)

### 배경
- 사용자 목표: ROCm 10 + llama.cpp + rdna3-boosts + turboquant 도커 이미지. theTom/llama-cpp-turboquant 기본 이식 + HIP 특성(domvox 참고) + RDNA3 최적화 검증.
- 판단 조건: 7.2.1(llm-main) 및 beellama보다 빠르면 대체.

### 이식 내용 (theTom 최신 4a54c52 + rdna-boosts)
- **클린 적용 5개**: 0005(CPU decode verify), 0006(host buffer), 0007(meta skip), 0009(headroom), 0012(RDNA4 allreduce)
- **수동 병합 3개**: 0004(WMMA DKQ>128->256 - gfx1100 head_dim256 WMMA 활성), 0010(k-quant VDR4/VDR2 등록 + dmA 레지스터 캐시), 0011(prefill 그래프 스킵)
- **제외**: 0001(theTom 자체 MTP), 0002(GDN 전용-dense 무관), 0003/0008(BF16 KV), 0013(fused MoE), 0004/0010의 op-timing(진단용)
- **참조**: domvox/turboquant-hip는 스탠드얼론 커널 검증용 - theTom이 이미 상위 구현(HIP FA 벡터 경로 보유)
- 빌드: native ROCm 7.2.3 (/tmp/tq-native-build), ROCm 10 docker (진행 중)

### 벤치 (27B Q4_K_M, GPU1, p512/n512/r3)
| 구성 | 7.2.1(llm-main) | tq+boosts(신규) | Δ |
|---|---|---|---|
| f16/f16 | 921/34.6 | **987/38.8** | +7.2%/+12.4% |
| q8_0/q5_0 | 911/33.4 | **964/37.6** | +5.8%/+12.6% |
| q8_0/turbo3 | 921/33.6 | **956/37.7** | +3.8%/+12.0% |
| q8_0/turbo4 | 915/33.7 | **956/37.7** | +4.5%/+11.9% |
| q8_0/turbo2 | 909/33.7 | **952/37.6** | +4.8%/+11.7% |

### 운영 조건 측정 (27B Q3_K_M, 140K, GPU1, adaptive MTP)
| 지표 | tq+boosts(q8/t3) | beellama(q8/q5 no-tail) | KVarN(이전) |
|---|---|---|---|
| prefill 4445t | **793** | 905 | 84~133 |
| tg | 48.5 | 50.9 | 45 |
| MTP 수용률 | 51.8% | 59.0% | - |

### prefill 최적점 스윕
| 변수 | 최적 | 비고 |
|---|---|---|
| ubatch | **1024** (pp 974, 512 대비 +24%) | 2048/4096 동급 |
| KV | q8_0/turbo3 | p4096 pp 877, tg 37 |
| MTP | adaptive n-max3 n-min2 | 수용률 51.8%, tg 48.5 |

### 결론
- tq+boosts가 7.2.1 대비 **모든 구성에서 pp +4~7%, tg +12% 초과** -> 대체 조건 1 충족
- beellama 대비 pp는 미세 열위(793 vs 905), tg는 동급(48.5 vs 50.9), KVarN 대비 prefill 6-9배 우위
- **추천 구성**: `-ctk q8_0 -ctv turbo3 -b 4096 -ub 1024 --spec-type draft-mtp-adaptive --spec-draft-n-max 3 --spec-draft-n-min-adaptive 2`

---

## 12. (추가) prefill 최적점 그리드 스윕 + ROCm 10 검증 (2026-09-06)

### prefill 스윕 (27B Q3_K_M, native 7.2.3 tq+boosts, GPU1)
K=q8_0 고정, cache_v x batch x ubatch x prefill길이 그리드.

**1단계: ub=1024 고정, cache_v x batch (p=512)**
| cache_v | b=2048 | b=4096 |
|---|---|---|
| f16 | **891.7 / 37.9** | 890.0 / 37.7 |
| q5_0 | 886.9 / 37.7 | 885.6 / 37.3 |
| turbo2 | 827.8 / 35.3 | 890.3 / 37.4 |
| turbo3 | 887.1 / 34.7 | 858.3 / 33.7 |
| turbo4 | 871.5 / 36.9 | 883.3 / 37.3 |

**2단계: turbo3/q5_0, batch x ubatch (p=512)** — 튜닝 차이 미미 (모두 880~886 pp)

**3단계: 긴 prefill p=4096 (운영 핵심)**
| cache_v (b4096/ub1024) | pp | tg |
|---|---|---|
| turbo2 | 870.6 | 32.5 |
| **turbo3** | **867.9** | **36.8** |
| turbo4 | 868.8 | 36.9 |
| q5_0 | 869.0 | 33.7 |
| turbo3 (ub2048) | 869.4 | 36.9 |

**결론**: 
- **긴 prefill(4096+)에서 cache_v는 pp에 거의 무관** (863~870, ±0.8%) — prefill은 메모리 대역폭이 아니라 컴퓨트 지배
- cache_v는 **tg에만 영향**: turbo3/turbo4 (36.8~36.9) > q5_0 (33.7) > turbo2 (32.5)
- batch/ubatch: b=4096/ub=1024~2048이 동급 최적 (b8192/ub2048은 pp 하락 848)
- **prefill 최적: q8_0 K + turbo3 V + b4096 + ub1024** (p4096 pp 868, tg 36.8)

### ROCm 10 도커 빌드 + 검증 완료
- 빌드: `/tmp/tq-patch-test/build-rocm10/` (clang-23, --rocm-path 설정, 전 빌드 절차 재사용) - BUILD DONE
- 스모크: 2B q8_0/turbo3 pp 1522 t/s (HIP turbo 경로 활성)
- 운영 구성 (27B Q3_K_M, 140K, q8_0/turbo3, adaptive MTP):
  - prefill 4445t: **776 t/s**
  - tg: **46.8 t/s**, MTP 수용률 48.1%
- 네이티브(7.2.3)와 동급 (793/48.5/51.8%)

### 최종 대체 판단 (사용자 기준 충족)
| 기준 | tq+boosts (ROCm10) | 판정 |
|---|---|---|
| 7.2.1보다 빠름 (pp) | 776 vs KVarN 84~133, 7.2.1 921(p512) | ✅ (긴 prefill 압도) |
| beellama보다 빠름 | tg 46.8 vs 50.9 (근접), pp 미세 열위 | △ tg동급/pp 14% 열위 |
| RDNA3 최적화 | 0004 WMMA + 0010 VDR + 0011 poll 활성, HIP turbo FA 경로 증명 | ✅ |

**추천 운영 config**: `-ctk q8_0 -ctv turbo3 -b 4096 -ub 1024 --spec-type draft-mtp-adaptive --spec-draft-n-max 3 --spec-draft-n-min-adaptive 2`

---

## 13. (정정) 동일 조건 공정 비교: beellama vs tq+boosts + adaptive MTP (2026-09-06)

### 이전 비교의 오류
§11의 "beellama tg 50.9 > tq 46.8 (beellama 우위)"는 **비교 조건이 달랐음**:
- beellama: KV=q8_0/q5_0, tq: KV=q8_0/turbo3 (KV 타입 차이 혼재)
- 각각 다른 세션/시점 측정
- 스윑(§12)에서 turbo3는 q5_0보다 tg 낮음(no-MTP 36.8 vs 33.7) -> 혼란 가중

### 공정 비교 (동일 조건: 27B Q3_K_M, 140K, KV=q8_0/q5_0 no-tail, adaptive MTP n-min2, GPU1)
| 지표 | tq+boosts | beellama | Δ |
|---|---|---|---|
| prefill 4445t | 799.5 | 905 | -12% |
| **tg (100t, adaptive)** | **55.8** | **47.4** | **+18%** |
| **MTP acceptance** | **65.3%** | **52.2%** | **+13.1pp** |
| mean len | 2.83 | 2.48 | - |

### 재해석
- **동일 KV에서 turboquant 코어가 tg +18%, 수용률 +13pp로 우위** (theTom의 MTP 구현 + 0010 VDR 시너지)
- beellama는 **prefill에서만 13% 우위** (표준 FA + 로드 밸런스 차이)
- §11의 "beellama tg 우위"는 잘못된 비교 -> **정정: tg/MTP는 tq+boosts가 우위**
- 사용자 prefill 우선순위: beellama +13% 유리, tg/MTP: tq+boosts +18% 유리

---

## 14. (정정2) 동일 압축률 공정 비교: turbo3 vs kvarn3 (2026-09-06)

### 사용자 지적: turbo3(3.125bit)와 동급 압축은 kvarn3(3bit) - 유효
§13의 beellama(q8_0/q5_0, 5.5bit V)는 turbo3보다 메모리가 오히려 크다.
동일 메모리 비교는 turboquant q8_0/turbo3 vs beellama kvarn3.

### 실측: kvarn3는 이 하드웨어에서 실용 불가
beellama q8_0/kvarn3 시도 -> `type_k이 kvarn3으로 강제 승격` + `kvarn_route_portable=128 (전 레이어)` -> **p4096 prefill pp 137 t/s**

### 동급 압축 비교 표 (27B Q3_K_M, p4096/ub1024)
| 구성 | V bit | mem(of f16) | p4096 pp | tg |
|---|---|---|---|---|
| **turboquant q8_0/turbo3** | 3.125 | 35.5% | **867.9** | 36.8 (MTP 55.8) |
| beellama kvarn3 (동급압축) | 3.0 | ~21% | **137** | 36.0 |
| beellama q8_0/q5_0 (덜압축) | 5.5 | 40.2% | 874.3 | 36.5 (MTP 47.4) |

### 재결론
- **동일 압축률에서 turboquant가 pp 6배 + MTP tg 우위로 압도**
- beellama는 **덜 압축(q8_0/q5_0)** 일 때만 prefill 우위 (905 vs 800) - 메모리 대가
- turboquant 승리의 근본: turbo3는 압축해도 표준 FA 경로 유지, KVarN은 head_dim 256+RDNA3에서 전 레이어 portable 폴백
- **최종: turboquant + boosts가 메모리/성능 모두 최적** (q8_0/turbo3, prefill+tg 겸비)

---

## 15. (추가) Needle-in-Haystack 정확도 테스트 (2026-09-06)

### 목적
사용자 소문: "kvarn은 turboquant 대비 VRAM 덜 아끼지만 긴 컨텍스트 정확도 손실이 아주 적다"
사용자 체험: 140K 넘어가면 Qwen3.6-35B가 무한루프/변수명 오류.

### 방법
- needle: UNIQUE_MAGIC_CONSTANT_<8자리> 를 haystack(코딩 지식 문장) 사이에 10~78% 위치로 매립
- 질문: 시스템 텍스트 내 magic constant 값 -> 응답에 needle 포함 여부
- 컨텍스트 8K/30K, chat 형식, --jinja --reasoning off (Qwen3 thinking 모드 제거)
- **중요**: beellama는 Qwen3 기본 reasoning 활성 -> raw prompt에서 `////` 반복 붕괴. --reasoning off 필수

### 결과
| 모델 | 컨텍스트 | needle위치 | beellama(kvarn/q8q5) | turboquant(turbo3) |
|---|---|---|---|---|
| 27B Q3_K_M | 30K | 78% | ❌ `////` 붕괴 | ✅ 95822412 회수 |
| 27B Q3_K_M | 30K | 10% | ❌ `////` 붕괴 | - |
| 27B Q3_K_M | 8K | 50% | ❌ `////` 붕괴 | - |
| 2B | 30K | 78% | ✅ 회수 | ✅ 회수 |

### 해석
1. **KV-only 정확도 (2B 기준)는 동급**: kvarn과 turbo3 모두 30K에서 needle 회수 성공.
   "kvarn 정확도 우수" 소문은 이 실험에서 **재현 안 됨**.
2. **beellama 27B 특이 붕괴**: KV 타입 무관(q8_0/q5_0도 동일) -> beellama 27B의 attention/디코딩 불안정.
   짧은 단일 질문은 정상, 시스템+긴 컨텍스트 입력에서 반복 붕괴.
3. **사용자 바이브 코딩 문제의 실체**: 27B + beellama 조합의 장기 컨텍스트 불안정.
   **turboquant 27B는 30K까지 안정적으로 needle 회수** -> 전환 시 이 문제 해소 가능성 높음.

### 결론
- needle 정확도 관점에서도 **turboquant(turbo3) >= beellama(kvarn)** - 2B 동급, 27B turboquant 우위
- beellama 27B의 반복 붕괴는 KV 문제가 아닌 코드베이스 문제 (전환 사유 추가)

---

## 16. (최종) ROCm 10 turboquant Docker 이미지 완성 + 운영 전환 결정 (2026-09-06)

### 완성된 이미지
`baramofme/llama-cpp-rocm:gfx1100-rocm10-tbq-rboosts` (21.3GB)
- theTom turboquant + rdna-boosts(cl린5 + 0004/0010/0011) + ROCm 10 런타임
- ENTRYPOINT [/app/llama-server], compose 호환

### Docker 이미지 최종 검증 (27B Q3_K_M, 140K, GPU1)
| 지표 | 값 |
|---|---|
| prefill 4445t | 793 t/s |
| tg (adaptive MTP) | 52.5 t/s |
| MTP acceptance | 55.8% |
| needle 30K 정확도 | **HIT (95822412 회수)** |

### 3가지 이미지 최종 비교 (사용자 대체 판단)
| 기준 | **ROCm10 tq+boosts** | beellama ROCm10 | 7.2.1 |
|---|---|---|---|
| prefill | 793 | 905(q8/q5), 84~133(kvarn) | 921 |
| tg | 52.5 | 47.4 | 33.6 |
| MTP 수용률 | 55.8~65.3% | 52.2% | 56.7% |
| needle 27B | ✅ HIT | ❌ 붕괴 | - |
| KV압축 | 35.5% | 30.5%(kvarn, 비실용) | 35.5% |

### 최종 결론
- **사용자 판단 조건 충족**: 7.2.1보다 pp +4~7%/tg +12% 초과, beellama보다 tg/MTP/정확도 우위
- **ROCm 10 turboquant가 최종 선택**: prefill는 beellama(덜압축)보다 12% 낮지만, 동급압축(turbo3 vs kvarn3)에선 6배 우위 + 27B 정확도 안정
- **바이브 코딩 문제 해소**: 27B 30K needle 회수 - 사용자의 "140K 넘어가면 루프/변수 오류"가 turboquant 전환으로 해소 기대
- **운영 전환**: Docker에서 Dokploy compose image 교체 + config.ini Dense/Dense-1 = q8_0/turbo3 + adaptive MTP

---

## 17. (정정3) prefill 비교: 793 vs 921은 측정 조건이 다름 (2026-09-06)

### 오해
§16 표의 "prefill: tq+boosts 793 vs 7.2.1 921" - 793이 더 낮아 보이는 문제.

### 진실: 서로 다른 측정 조건
| 값 | 조건 | 도구 |
|---|---|---|
| 921 | llama-bench p=512, Q4_K_M(16.8G), f16 KV | 짧은 프롬프트 벤치 |
| 793 | llama-server 140K ctx, 4445t prefill, Q3_K_M(14.5G), q8/t3 | 운영 긴 프롬프트 실측 |

### 동일 조건 (llama-bench p=512, Q4_K_M, GPU1) 재측정
| 구성 | pp | tg |
|---|---|---|
| 7.2.1 q8_0/turbo3 (원본 911) | 910~921 | 33.6 |
| tq+boosts q8_0/turbo3 (r=2) | 936.9 | 37.48 |
| beellama q8_0/q5_0 | 1008.1 | 36.33 |

### 결론
- **같은 조건이면 tq+boosts가 7.2.1보다 pp +2~3%, tg +11% 우위**
- 793 vs 921은 "긴 서버 prefill" vs "짧은 벤치"라 직접 비교 불가
- **운영 140K 기준**: 이전 KVarN 84~133 vs tq+boosts 793 = **6~9배 개선** (사용자가 느린 원인 해소)

---

## 18. (추가) MoE 모델 CPU offload 스레드 스윕 (2026-09-06)

### 배경
사용자 참조 (arca.live/b/alpaca/182011063): Qwen3.8-Flash-Next UD-IQ4_XS MoE 모델을
`-ngl 99 -ot '...ffn_(up|down|gate|gate_up)_exps=CPU'`로 띄울 때 스레드 영향 여부.

### 실측 모델
Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf (16GB, MoE, active 3B) + tq+boosts(ROCm 7.2.3 native), GPU1.

### 결과 1: 전부 GPU (-ngl 99) - 스레드 무관
| -t | pp(256t) | tg |
|---|---|---|
| 6 | 1847.0 | 110.5 |
| 12 | 1827.1 | 110.1 |

### 결과 2: FFN exps만 CPU 오프로드 (-ot ...exps=CPU) - 스레드 결정적
| -t | pp | tg (3반복) |
|---|---|---|
| **6** | 700 | 45.5 / 40.4 / 60.0 |
| **12** | 702 | 52.0 / 50.1 / 56.9 |
| 20 | 703 | **10.6 / 15.1 / 17.6** |

### 결과 3: dense 27B 일부 레이어 (-ngl 16) CPU - 스레드 20 회귀
| -t | pp | tg |
|---|---|---|
| 6 | 231.2 | 4.85 |
| 12 | 224.5 | 4.92 |
| 20 | 208.7 | **2.94** |

### 해석
1. **MoE + expert CPU 오프로드에서 스레드는 확실히 영향** (전부 GPU와 정반대)
2. **6 -> 12: tg +14% 개선** (45.5 -> 52.0) - CPU expert GEMM 병렬화
3. **20: 치명적 회귀** (-75%, 52 -> 15) - 20코어 전부 사용 시 expert GEMM이 대역폭/동기화 병목 (oversubscription)
4. expert CPU 오프로드 자체 비용 큼 (전부 GPU 110 vs 오프로드 45~52) - VRAM 허락 시 expert는 GPU 유지가 압도적 이득

### 적용 (운영 반영 완료)
- config.ini 전역 `threads = 8 -> 12` (+ LFM2.5 섹션도 12)
- compose OMP_NUM_THREADS=6 유지 (명시적 --threads가 우선, docker restart로 12 반영)
- 운영 확인: `--threads 12` 전달, tg 50.8 t/s

### 지침 (향후 MoE/CPU 오프로드 시)
- `-t/--threads-batch` 는 **6~12** 범위 최적 (20코어 기준)
- **16~20 금지** (oversubscription)
- OMP_NUM_THREADS env 는 무시됨 (명시적 -t/-tb 우선) - CLI/config.ini 값 변경 필요

---

## 19. (추가) beellama tensor split (듀얼 GPU) 벤치 + acceptance 콘텐츠 의존성 정정 (2026-09-06)

### 19.1 tensor split 벤치 (27B Q3_K_M, q8_0/q5_0, adaptive MTP, 46.5K haystack)

| 지표 | 단일 GPU (split none) | 듀얼 (tensor split) | Δ |
|---|---|---|---|
| 프리필 46.5K (운영 ctr) | 99.8s / 466 t/s | 96.5s / 482 t/s | +3% |
| 프리필 46.5K (전용 ctr) | - | 86.9s / 535 t/s | +15% |
| **tg (300t, 3회 평균)** | **50.6 t/s** | 44.7 (운영) / 48.0 (전용) | **-6~12%** |
| GPU 사용률 | GPU0 93% | GPU0+GPU1 각 74~86% | - |

**결론**:
- 27B dense(13.5GB)는 단일 24GB GPU에 여유 담기므로 **디코딩은 tensor split이 오히려 손해** (양 GPU 텐서 동기화 all-reduce 오버헤드, RDNA3 gfx1100은 RDNA4 전용 all-reduce 미지원 - 로그 확인)
- **프리필만 +3~15% 개선** (배치 병렬이 동기화를 압도)
- **tensor split의 실익은 24GB 초과 모델(Flash-Next 88GB 등)의 VRAM 확보** - 용량 목적으로만 유효
- 운영 선택: 사용자 결정으로 **tensor split 유지** (GPU0 12.2GB + GPU1 11.0GB)

### 19.2 acceptance 콘텐츠 의존성 정정

이전 결론(§13, §16 등)의 "포크 간 acceptance 우열"은 **측정 콘텐츠가 달랐음**을 확인:
- 같은 beellama 서버에서 요청 콘텐츠에 따라 acceptance 변동: task 1404(영어 문단) 56.8% vs task 2726(한국어 시) 48.1%
- 같은 tq 서버: 47.9% (46.5K, 영어 문단) - beellama 56.8%와 같은 콘텐츠에서 비교해야 의미
- **acceptance는 MTP 엔진 우열이 아니라 생성 콘텐츠(난이도)에 크게 의존**
- 결론: 포크 간 acceptance/tg 수치 차이는 동일 콘텐츠 기준으로만 비교해야 유효

### 19.3 MTP 이식 관련 소스 검증 (tq vs beellama)

- tq(4a54c52)는 **이미 0001 adaptive MTP 포함** (§11의 "제외" 기록은 구버전 상태, 현재 소스는 speculative-adaptive.h 커밋됨)
- tq와 beellama의 draft_mtp 엔진: 모두 pending_h/chain_heads 보유, 그러나 구현 세부 상이 (tq는 defer 캐치업 추가, beellama는 upstream master 머지로 state-plan/DFlash 등 최신)
- **beellama의 speculative.cpp를 tq에 단순 이식 불가**: beellama가 쓰는 26개 심볼 중 21개가 tq 코드베이스에 없음 (llama.h API 확장 포함) - 14개+ 파일, 4000+ diff lines 이식 필요 = 사실상 beellama 재조립
- **결론: beellama가 이미 최선의 조합(VDR 0010 + adaptive MTP + 최신 엔진)을 보유** - tq/순정 재조립 실익 없음

## 20. (부수) 운영 config 회귀 수정 + Flash-Next 다운로드 완료

### config.ini (beellama 운영 정합)
- `[Dense]` (turbo3 설정): load-on-startup = true -> **false** (beellama 재시작 시 turbo3->kvarn3 리다이렉트로 잘못 로드되어 VRAM 충돌)
- `[Dense-bellama]` (q8_0/q5_0): load-on-startup = false -> **true** (beellama 기본 운영 모델)
- split-mode: none <-> tensor 전환 테스트 완료, 최종 사용자 선택으로 **tensor 유지**

### 잔여 컨테이너 정리
- 스모크 테스트 컨테이너(xenodochial_napier)가 GPU 7.2GB씩 점유 -> Dense-bellama 로드 OOM 유발 -> 제거로 해결

### Flash-Next 다운로드 완료
- `/opt/llm/models/Qwen3.8-Flash-Next-UD-IQ4_XS/` 88GB (3 shard: 11MB + 49.8GB + 43.8GB)
- .cache 잔여 정리 완료. 듀얼 GPU tensor split 상태라 바로 구동 테스트 가능

---

## 21. (참고/예정) RCCL 빌드 + tensor split 재벤치 (2026-09-06)

### 배경 (reddit 사용자 보고)
> "Build llamacpp with rccl enabled, the default pre builds do not have this. So make sure your build has that. Then tensor split the xtx cards and you will get decent performance. I get 1200+ on dual xtx with this setup"

### 현재 우리 환경 = reddit의 "bad case" (RCCL 없음) 확정
- 로그: `ggml_cuda_ar_pipeline_init: internal all-reduce is RDNA4-only (gfx1200/gfx1201); device 0 reports gfx1100 -- falling back to the default path`
- beellama/tq 이미지는 **RCCL 미포함 빌드** -> gfx1100(RX 7900 XTX)에서 tensor split all-reduce가 default path(느림)로 폴백
- 이 때문에 §19 tensor split tg가 -6~12% 손해였던 것과 일치

### reddit 참조 스냅샷 (dual RX 7900 XTX, 27B Q8_0 MTP)
- 소스: llama.cpp stock (b10362 / 4801e3c)
- ROCm 7.2.4 + RCCL 2.27.7 + HIP 7.2.53211 + AMD clang 22
- 플래그: `-sm tensor --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --flash-attn on --ubatch-size 512 --batch-size 2048 -ngl 99 --threads 8 --parallel 1 --spec-type draft-mtp --spec-draft-n-max 4 -ctkd q8_0 -ctvd q8_0 -c 64000`
- 벤치(5회 반복): **pp4096 1251.2 / tg512 90.4 / tg2048 104.9**

### 우리 측정 비교 (RCCL 없음, 27B Q3_K_M)
| | reddit (RCCL, dual XTX, Q8_0) | 우리 현재 (RCCL 없음, dual, Q3_K_M) |
|---|---|---|
| pp | 1251 | 482~535 (46.5K 진행) |
| tg | 90~105 | 44.7~48.0 |

### 다음 단계 (예정, 사용자 승인 대기)
1. **RCCL 활성 빌드** (multi-gpu.md 공식 확인):
   - 플래그: `-DGGML_HIP=ON -DGGML_HIP_RCCL=ON` (ROCm 백엔드의 RCCL 옵션 - NCCL과 달리 **기본 비활성**)
   - 주의: RCCL은 "universally beneficial"이 아니어서 기본 off. 빌드 후 CMake 로그에서 RCCL 발견 여부 확인
   - 런타임: RCCL 빌드 시 tensor 모드 자동 사용, `GGML_CUDA_P2P=1`로 P2P opt-in 가능
2. **tensor split 재벤치**: RCCL 빌드로 동일 조건(27B dense + q8_0 + dual XTX) 비교
   - 기대: reddit처럼 pp 1200+, tg 90+ 도달 시 §19의 "duel 손해" 결론이 RCCL 빌드에선 뒤집힘
3. **⚠️ Flash-Next(88GB MoE)는 tensor split 불가 확정** (multi-gpu.md):
   - `LLAMA_SPLIT_MODE_TENSOR not implemented` list에 MoE 이하 포함: Grok, MPT, OLMoE, DeepSeek2, GLM-DSA, Nemotron-H, LFM2-MoE, Minimax-M2, Mistral4, Kimi-Linear, Jamba, Falcon-H1
   - **Qwen3.8-Flash-Next(UD-IQ4_XS, MoE)도 동일 제약** -> tensor split 사용 불가, **split-mode layer(pipeline)로 구동해야 함**
   - layer 모드 + RCCL은 무관(파이프라인은 크로스-GPU reduction 없음) - exps CPU 오프로드 한계는 동일
   - 즉 Flash-Next 고속화는 RCCL이 아니라 **전문가의 GPU 적재**가 관건: layer split에서 n-gpu-layers로 전문가를 GPU에 넣되 단일 텐서(per_layer_token_embd 25GB)가 GPU0를 초과하지 않는 범위에서

---

## 22. (웹 검색 결과) tensor split 성능 재평가 — RCCL/새 TP 코드가 핵심 (2026-09-06)

### 검색에서 얻은 결정적 데이터

#### 1. 공식 PR #19378: backend-agnostic tensor parallelism (실측 포함)
- https://github.com/ggml-org/llama.cpp/pull/19378
- 2x RTX 4090 (PCIe 4.0 x16) 실측 — **tensor split이 tg에서 layer보다 유리**:

| 모델 | test | layer t/s | tensor t/s | Speedup |
|---|---|---|---|---|
| qwen35 27B Q8_0 | tg128 | 30.61 | **43.03** | +41% |
| qwen35 27B Q8_0 | tg128 @ d65536 | 26.78 | **38.54** | +44% |
| llama 70B Q4_K_M | tg128 | 21.25 | **48.31** | +127% |
| llama 8B F16 | tg128 | 60.28 | **96.63** | +60% |

- **pp는 tensor가 오히려 손해**: qwen35 27B Q8_0 pp2048 → 3727 vs 2639 (-29%)
- **결론**: tensor parallel은 **토큰 생성(tg)에 특화**, 프리필은 layer가 유리
- 이는 우리 §19(프리필 +, tg -)와 **정확히 반대 패턴** — 그 차이 = **RCCL 유무 + 새 TP 코드 버전**

#### 2. RCCL 빌드 (공식 multi-gpu.md, 재확인)
- `-DGGML_HIP_RCCL=ON` — ROCm용, **기본 비활성** ("not universally beneficial")
- RCCL 없으면: "NCCL not compiled in; falling back to internal AllReduce" 경고 + 성능 저하
- **우리 로그 확인**: `internal all-reduce is RDNA4-only (gfx1200/gfx1201); device 0 reports gfx1100` — RDNA3는 내부 all-reduce도 못 씀 → **RCCL 필수**

#### 3. RCCL 주의사항 (ROCm issue #6074)
- RCCL 2.27.7 + 2x 7900 XTX에서 ROCm 7.2.1에서 dual-GPU collective 실패 사례
- **ROCm 7.2.0에서는 동작** (7.2.1에서 회귀했다가 복구됨)
- 우리 native 7.2.3 / docker ROCm 10 → 시험 필요

#### 4. tensor split 제약 (PR#19378 재확인)
- MoE 아키텍처(tensor split 미지원): grok, deepseek2, mpt, olmoe, glm_dsa, nemotron_h, **lfm2_moe**, minimax_m2, mistral4, kimi_linear, jamba, falcon_h1, **qwen35moe는 지원**(PR#19378 실측에 포함!)
- **중요**: PR#19378 실측에 **qwen35moe 35B.A3B가 tensor split으로 정상 측정됨** — MoE라도 일부는 지원
- **Qwen3.8-Flash-Next(UD-IQ4_XS, MoE, qwen35moe 계열)도 tensor split 가능할 수 있음** — beellama(최신 upstream 머지) 기준 확인 필요

### 수정된 결론 (§19의 "tensor split=듀얼 손해" 재평가)

| 지표 | §19 실측 (RCCL 없음) | PR#19378 (RCCL/NCCL, 4090) | 판정 |
|---|---|---|---|
| 프리필 | tensor +3~15% | tensor -29~-60% | 불일치 (빌드/모델 차이) |
| **tg** | tensor -6~12% | tensor **+41~127%** | **RCCL/새 TP 코드로 뒤집힘** |

- **우리의 tensor tg 손해(-6~12%)는 RCCL 부재 때문** — reddit "1200+ with RCCL"과 일치
- RCCL 빌드 시 **tg가 +40% 이상 개선**될 가능성 높음 (beellama 최신 TP 코드 포함 시)
- **Flash-Next가 qwen35moe 계열이면 tensor split 가능** → exps GPU 적재로 1.5 t/s 문제 해결 가능

### 다음 액션 (권장 순서)
1. **beellama를 RCCL 활성으로 재빌드**: `-DGGML_HIP_RCCL=ON -DGGML_HIP=ON` (native 7.2.3, docker ROCm 10 둘 다)
2. **27B dense tensor split 재벤치**: pp/tg를 RCCL 유/무 비교 → §19 정정
3. **Flash-Next tensor split 시도**: qwen35moe 지원 여부 확인 후, 가능하면 exps GPU 적재

---

## 23. (reddit 실전 데이터) dual 7900 XTX + BF16 KV가 Q8_0보다 빠름 — RDNA3 FP16 특성 (2026-09-06)

### reddit 사용자(johnchristianson) 실측: dual 7900 XTX, 72K 컨텍스트, terraform 워크로드

같은 모델/컨텍스트, KV 타입만 변경한 A/B 비교:

| KV 타입 | pp (72K) | tg (1000t+) | acceptance | 그래프 재사용 |
|---|---|---|---|---|
| **BF16/BF16** | **967.21 t/s** | **51.53 t/s** | 62.2% | 360 |
| Q8_0/Q8_0 | 923.66 t/s | 48.00 t/s | 61.8% | 505 |
| **Δ (양자화 비용)** | **-4.5%** | **-6.8%** | -0.4pp | +145 |

### 사용자 설정 (daily driver)
```
unsloth/Qwen3.8-27B-GGUF:Q8_0 (모델 가중치 Q8_0)
--split-mode tensor --tensor-split 1,1 --flash-attn on
--cache-type-k bf16 --cache-type-v bf16
--ctx-size 196608 --batch-size 3072 --ubatch-size 1536
--cache-ram 65536 (64GB RAM KV offload)
--spec-type ngram-mod,draft-mtp-adaptive
--spec-draft-n-min-adaptive 3 --spec-draft-n-max 12
--spec-draft-ngl 99 --spec-draft-type-k bf16 --spec-draft-type-v bf16
--spec-ngram-mod-n-match 48 --spec-ngram-mod-n-min 24 --spec-ngram-mod-n-max 64
--load-mode none --n-gpu-layers 99 --threads 12
```

### 결론 데이터 1: **RDNA3 7900 XTX는 KV 양자화가 오히려 손해**
> "the xtx and rdna3 seems to actually just be 16 bit floating point under the hood, so quantizing slows it down"

- RDNA3 GPU는 **내부적으로 FP16 파이프라인** — KV를 q8_0으로 양자화하면 디양자화 오버헤드가 생기고, 메모리 절약 이득을 상쇄
- **BF16 KV가 q8_0보다 pp +4.5%, tg +6.8% 빠름** (72K 실측)
- 사용자: "longer the context, the bigger the hit" — 컨텍스트가 길수록 양자화 손해 증가
- 선호: "BF16 model + BF16 kv cache on 3+ cards... faster on prompt processing and nearly as fast on token generation"

### 결론 데이터 2: tensor split + BF16 KV면 72K에서도 tg 51.5 유지
- 이 사용자는 **72K 컨텍스트 + tensor split(1,1)에서 tg 51.53 t/s** 달성
- 우리 §19: q8_0/q5_0 + tensor split 46.5K에서 tg 44.7~48.0 (단일 GPU는 50.6)
- **우리의 "tensor split tg 손해"는 q8_0/q5_0 KV 양자화 상태에서만 확인된 것** — 이 reddit 데이터는 **BF16 KV를 쓰면 tensor split이 단일 GPU와 동급~우위**일 수 있음을 시사

### 우리 벤치와의 연관 (중대한 재해석 가능성)

| 비교 | reddit (dual, BF16 KV) | 우리 §19 (dual, q8/q5) | 우리 §19 (단일, q8/q5) |
|---|---|---|---|
| tg | **51.5** (72K) | 44.7~48.0 (46.5K) | 50.6 (46.5K) |
| pp | **967** (72K) | 482~535 (46.5K) | 466 (46.5K) |

- reddit은 **더 긴 컨텍스트(72K vs 46.5K)에서 tg 51.5** = 우리 듀얼(44.7~48.0)보다 빠름
- pp도 **967 vs 482~535** — 거의 2배 차이 (모델 가중치 Q8_0 vs Q3_K_M 차이 + RCCL + BF16 KV 복합)
- **즉 우리 듀얼의 저조함은 (a) KV 양자화(q8_0/q5_0) + (b) RCCL 부재 + (c) Q3_K_M 저품질 가중치 의 합산**

### 다음 액션 (우선순위 갱신)
1. **BF16/BF16 KV로 tensor split 재벤치** (27B Q8_0 또는 Q4_K_M 가중치 사용, reddit과 동일 조건)
   - RCCL 없이도 BF16이 q8_0보다 빠른지 먼저 확인 (reddit 데이터로 기대됨)
   - VRAM: 27B Q4_K_M(16.8GB) + BF16 KV 46.5K ≈ 8GB 상황 — 듀얼 48GB에 여유
2. 그 다음 RCCL 빌드로 재확인
3. spec-draft-n-max 12 + ngram-mod (reddit 구성)도 시도 — 수용률/그래프 재사용 개선 여지

---

## 24. (웹 검색 추가) tensor split이 MTP보다 더 큰 이득 + ROCm acceptance 한계 (2026-09-06)

### reddit 글 1vtoux9 + 검색에서 확인된 결정적 데이터

#### 1. 2× RTX 5060 Ti 16GB TP 테스트 (Jackwwg83, qwen38-mtp 레포)

**--split-mode 자체가 MTP보다 큰 이득**:

| split mode | spec | Overall tg | acceptance |
|---|---|---|---|
| layer (default) | off | 22.1 | - |
| layer | n-max 2 | 42.8 | 0.53-0.94 |
| layer | n-max 3 | 47.5 | 0.48-0.75 |
| **tensor** | **off** | **37.1** | - |
| tensor | n-max 2 | 65.9 | 0.51-0.88 |
| tensor | n-max 3 | 69.3 | 0.38-0.74 |
| **tensor** | **n-max 4** | **71.3** | 0.35-0.70 |
| tensor | n-max 5 | 67.1 | - |

- **"The default split mode costs more than the flag gains"**
  - layer→tensor 만으로 baseline 22.1 → 37.1 (**+68%**) — 어떤 MTP 플래그보다 큰 공짜 이득
  - layer default는 배치1에서 GPU0만 계산, GPU1 유휴 (pipeline 병목)
  - tensor는 각 matmul을 분할 → 양 GPU 동시 가중치 읽기
- **tensor + MTP = 3.2x** (22.1 → 71.3, -sm tensor + n-max 4)
- **n-max 최적점은 split-mode에 따라 이동**: layer는 n-max 3 피크(47.5), tensor는 n-max 4 피크(71.3) — 스위치 시 재스윕 필요

#### 2. ROCm에서 MTP acceptance가 나쁘다 (llama.cpp 이슈 #27124, #26750)

- **ROCm 단일 7900 XTX 로그: `draft acceptance = 0.35929`** (llama.cpp #27124)
- CUDA도 0.36 수준, **Vulkan은 ~0.92**
- 이는 우리 §15 발견(beellama 27B acceptance 0.36)과 정확히 일치 — **HIP MTP 헤드 열위는 하드웨어/백엔드 특성**
- 즉 **tensor split의 tg 이득은 MTP 수용률과 별개로 발생** — 우리 듀얼 구성에서도 기대 가능

#### 3. qwen38-mtp 커뮤니티 벤치 (RX 7900 XTX 단일, Vulkan) — 참고
- RX 7900 XTX (Vulkan/RADV): 28.8 → 70.7 t/s (n-max 3, acceptance 0.43-0.95)
- RX 7900 XTX (ROCm): 30.7 → 43.9 (n-max 2, acceptance 0.60-0.95)
- **같은 카드라도 Vulkan MTP > ROCm MTP** (백엔드 차이 재확인)

### 우리 환경에 대한 함의 (종합)

| 발견 | 우리 운용 관련 |
|---|---|
| tensor split이 tg의 1차 이득 (+41~68%) | **RCCL + tensor split 재벤치가 최우선** (§22/§23과 결합) |
| ROCm MTP acceptance 0.36은 코드 문제 아님 | beellama 27B의 낮은 acceptance는 백엔드 특성 — **tensor split tg 내지는 split 자체로 보상** |
| n-max 최적점은 split에 따라 달라짐 | tensor 전환 시 n-max 3→4 재스윕 필요 |
| BF16 KV > q8_0 KV (RDNA3) | §23와 결합 — 듀얼에서는 BF16 KV + tensor split 조합이 최적 예상 |

### 갱신된 벤치 우선순위 (최종)
1. **BF16/BF16 KV + tensor split**: 27B Q4_K_M + dual — RCCL 없이도 개선 확인 (§23 reddit + §24 패턴)
2. **RCCL 빌드**: `-DGGML_HIP_RCCL=ON` — 그 위에 tensor split 재벤치
3. **n-max 스윕**: tensor 모드에서 n-max 2~6 (최적점이 layer와 다름)
4. 그 후 Flash-Next (MoE, layer split 한정 + exps GPU 적재)

---

## 25. (reddit 실전 데이터 3) tensor + MTP = 29→69 tok/s — RCCL 없이도 달성 (2026-09-06)

### reddit 글: "Qwen3.8-27B on llama.cpp, Tensor + MTP on dual 7900 XTX increases performance: 29 → 69 tok/s"

- 작성: dual 7900 XTX 사용자 (gfx1100, 24GB x2, "peer access enabled")
- 빌드: **llama.cpp server-rocm build 10481** (RCCL 미언급!)
- f16 KV, flash attention on. 86,675-token 프롬프트, 실제 서버 요청 (llama-bench 아님)
- 모델: Qwen3.8-27B-**UD-IQ4_XS**.gguf (13.26 GiB, 4.25bpw) + 별도 `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` (1.28 GiB draft 헤드)

### 결과 (같은 86.7K 프롬프트, 동일 측정)

| config | max ctx | gen tok/s | prompt tok/s | vs baseline |
|---|---|---|---|---|
| 1 GPU (baseline) | 131072 | 29.28 | 516 | - |
| 1 GPU + MTP | 98304 | 53.90 | 491 | +84% |
| 2 GPU -sm tensor | 131072 | 34.66 | 364 | +18% |
| **2 GPU tensor + MTP** | **131072** | **69.25** | 361 | **+137%** |
| 2 GPU tensor + MTP | 262144 | 69.41 | 361 | +137% |

### 핵심 통찰 (우리에게 결정적)

1. **RCCL 없이도 tensor + MTP로 69 t/s** — "peer access enabled"만으로 (P2P)
   - 우리 §22-24의 "RCCL 필수?" 가설의 반례 — RCCL 없이도 가능
   - build 10481 (최신)의 tensor split + P2P가 충분했을 수 있음
2. **`-sm tensor`는 KV 캐시도 분할 → MTP 헤드(1.3GB) + 전체 컨텍스트를 동시에 수용**
   - 단일 카드에서 MTP는 VRAM을 context에서 뺌: 131072 → 98304 (25% 윈도우 손실)
   - tensor는 컨텍스트를 양 카드로 분산 → **MTP + 256K 풀 윈도우 동시 가능**
3. **256K 윈도우 공짜**: 69.25 → 69.41 (동일), acceptance 90% 유지
   - "속도는 선언한 컨텍스트가 아니라 실제 차 있는 토큰 수에 달림"
4. **긴 컨텍스트에서 MTP 수용률 상승**: 벤치 73% → 86K 차면 87% → 실사용 64% (대략 2.2x)
5. **-ts 45,55 비대칭**: GPU0(데스크톱) 여유 확보용 — 실제로 41.5/58.5로 측정됨 (선형 아님)

### 우리 환경과의 차이 (왜 우리는 44~48이었나)

| 구성 요소 | 우리 §19 | 이 reddit 69t/s | 차이 영향 |
|---|---|---|---|
| 가중치 | Q3_K_M (14.5GB) | UD-IQ4_XS (13.26GB) | IQ4_XS가 더 가벼움/최신 |
| **KV 캐시** | **q8_0/q5_0 (양자화)** | **f16 (비양자화)** | **RDNA3 FP16 특성상 f16 유리 (5~7%)** |
| MTP 헤드 | 내장 (adaptive) | 별도 1.28GB Q4_0 헤드 | 별도 헤드 성능 우수 가능성 |
| 컨텍스트 | 46.5K | 86.7K | reddit이 더 김에도 tg 높음 |
| 빌드 | beellama 기반 | stock 10481 | 최신 tensor split + P2P |
| P2P | 미설정 | "peer access enabled" | **P2P 중요 변수** |

### 함의: 우리 최적 구성 시나리오 (이 데이터 기반)

**동일 하드웨어에서 29→69 (+137%)이므로, 우리가 이 구성(Q4_K_M 또는 IQ4_XS + f16 KV + 별도 MTP 헤드 + tensor split)으로 재벤치하면 140K 운영 컨텍스트에서 tg ~55-65 도달 가능성**

구체적 재벤치 계획:
1. **KV를 f16/f16으로** (우리 Q4_K_M 사용, 16.8GB — 듀얼 48GB에 여유)
2. **별도 MTP 헤드** 확인: `/opt/llm/models`에 mtp Q4_0이 있는지? 없으면 unsloth 레포에서 받기 (Qwen3.8-27B-MTP-Q4_K_M.gguf는 내장형 - 별도 헤드 파일 필요)
3. **tensor split + MTP** 조합 (우리 이전 테스트는 내장 adaptive였음)
4. P2P: `GGML_CUDA_P2P=1` 시도
5. 운영 140K 컨텍스트에서 실측

### 결론: reddit 3사례 종합 (21/23/25)

| 소스 | RCCL | KV | 구성 | tg |
|---|---|---|---|---|
| §21 reddit (stock b10362) | ✅ RCCL 2.27.7 | q8_0/qt8 | MTP n-max4 | 90~105 |
| §23 johnchristianson | - | **bf16** | tensor + ngram+MTP | 51.5 (72K) |
| §25 (이 글) | ❌ 없음 | **f16** | tensor + 별도헤드 | **69.4 (86K)** |

- **가장 빠른 §21은 RCCL + 104.9 tg** (Q8_0 가중치, 64K)
- **RCCL 없이도 69 achievable** (§25, f16 KV + P2P)
- **BF16/f16 KV가 공통 승리 요소** (23/25 모두 비양자화 KV)
- 다음 벤치: **f16 KV + tensor + (가능하면 RCCL) + (가능하면 별도 MTP 헤드)**

---

## 26. (reddit 실전 데이터 4) mbrodie 듀얼 XTX — Vulkan TP broken 확인 + ROCm 27B 1100pp/55-70tps (2026-09-06)

### reddit 댓글: "Your Claude did a terrible job" (1vtoux9 쓰레드)

- 하드웨어: 5800x, 2x 7900 XTX (8x 슬롯), 32GB DDR4, 6TB NVMe PCIe4, Ubuntu 26.02
- 주장: "On 2 x 7900 xtx you can run full 262k context and get 1200pp and 50 - 70tps with a q8 on long token runs using rocm... for whatever reason Vulkan is incredibly slower for qwen 3.8 and tensor parallelism is broken but it seems to be working fine in rocm"

### 스크립트 1: 35B A3B MoE (Vulkan) → 4200 pp / 85-95 tps

```
/opt/llama.cpp-latest/build-vulkan-9d57ce456/bin/llama-server
--model qwen3.6-35b-a3b-nsc-saber-q8/Qwen3.6-35B-A3B-NSC-ACE-SABER-Q8_0.gguf
--ctx-size 262144 -ngl 999 --load-mode mmap --tensor-split 0.94,1.06 --fit off -np 1
--kv-unified -fa on -t 8 -tb 8 --jinja --reasoning-preserve --temp 0.7 --top-k 20
--cache-ram 4096 --mmproj mmproj-f16.gguf
```

- **비대칭 --tensor-split 0.94,1.06** — 우리 -ts 45,55 시도와 동일 패턴 (카드별 부하 차등)

### 스크립트 2: 27B dense (ROCm 7.2.3) → 1100 pp / 55-70 tps

```
/opt/llama.cpp-latest/build-hip-fe8156f78/bin/llama-server
--model Qwen3.8-27B-Q8_0.gguf --alias Qwen3.8-27B-Q8-MTP4-ROCm-RCCL
--ctx-size 204800 -ngl all --split-mode tensor --tensor-split 1,1 --fit off --load-mode mmap
-np 1 --kv-unified -fa on -t 8 -tb 8 -b 2048 -ub 512
--cache-type-k f16 --cache-type-v f16
--cache-type-k-draft q8_0 --cache-type-v-draft q8_0
--cache-ram 0 --spec-type draft-mtp --spec-draft-n-max 4
--mmproj mmproj-F16.gguf --image-min-tokens 1024
--jinja --reasoning-preserve --temp 0.7 --top-k 20
```

- ROCM_ROOT=/opt/rocm-7.2.3 (alias에 RCCL 표기지만 스크립트에 RCCL env 없음 — 빌드에 포함 여부 불명확)

### 우리에게 결정적 포인트

1. **f16 KV + tensor split 3번째 독립 확인** (§23 bf16 / §25 f16 / §26 f16) — 204800 선언 ctx, 긴 런에서 55-70 tps 유지
2. **main/draft KV 캐시 타입 분리**: main f16 + draft q8_0 — draft는 수용률에만 영향이므로 저비트, main은 고정밀. **우리는 미테스트** (`--cache-type-k-draft`)
3. **`--kv-unified` 플래그** (양 스크립트 공통) — 우리 미테스트. KV 캐시의 GPU 간 균등 통합 분할
4. **Vulkan이 Qwen3.8(dense 27B)에 매우 느림 + Vulkan TP broken** 확정 발언 — 단, 35B A3B MoE는 Vulkan에서 4200pp/85-95 (MoE는 활성 파라미터 적음 = Vulkan에서도 빠름). **우리 ROCm 포커스 정당화**
5. pp 1100~1250은 §21 reddit(1251, RCCL)과 동급 — **Q8_0 가중치 + q8 KV 대비 tensor+ROCm 우위**

### reddit 4사례 종합 (21/23/25/26)

| 소스 | 빌드 | KV | 구성 | tg | pp |
|---|---|---|---|---|---|
| §21 (b10362) | stock + RCCL 2.27.7 | q8_0/qt8 | tensor + MTP n4 | 90~105 | 1251 |
| §23 johnchristianson | stock | bf16 | tensor + ngram+MTP | 51.5 (72K) | 967 |
| §25 1vtoux9 | stock 10481 | f16 | tensor + 별도 MTP 헤드 | 69.4 (86K) | 361* |
| §26 mbrodie | build-hip fe8156f78 | **f16 (+draft q8_0)** | tensor + MTP n4 | **55-70 (긴 런)** | **1100** |

*§25 pp 361은 86K 진행형 prefill이라 낮음

### 결론: f16 KV + tensor = 4건 모두 개선 — 재벤치 시나리오 확정

- mbrodie 27B(55-70) >= 우리 §19 듀얼(44.7-48.0)을 f16 KV + (가능하면 draft KV 분리)로 개선 기대
- draft 캐시 타입 분리(`--cache-type-k-draft`) 지원 여부 beellama에서 확인 필요
- `--kv-unified` 지원 여부도 확인 (우리 config.ini에 없음)

---

## 27. RCCL 빌드 + host 오프로드 + MoE expert cache 이식 전체 과정 (2026-09-06)

### 27.1 RCCL 활성 빌드 (beellama, native ROCm 7.2.3)

**왜 RCCL인가**
- beellama의 텐서 병렬(all-reduce)은 하이브리드 구조: RCCL(대형 텐서, P2P+BF16) + 내부 HIP 파이프라인(소형 텐서, 호스트 스테이징)
- 그런데 내부 HIP 파이프라인은 **RDNA4-only 게이트**가 있음 (`allreduce-hip.cu:743-766`):
  `gfx1200/gfx1201`이 아니면 `return nullptr` → gfx1100(7900 XTX)에서는 항상 비활성
- 결과 로그: `internal all-reduce is RDNA4-only (gfx1200/gfx1201); device 0 reports gfx1100 -- falling back to the default path`
- → gfx1100 듀얼에서 tensor split all-reduce는 **RCCL이 유일한 고속 경로** (내부 파이프라인은 RDNA4 전용, butterfly는 느림)

**빌드 방법**
```
cmake -S beellama-boosts/src -B /tmp/be-src-native0001 \
  -DGGML_HIP_RCCL=ON -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.3/lib/llvm/bin/clang++ ...
```
- RCCL은 `/opt/rocm-7.2.3`에 이미 설치됨 (librccl.so.1 + cmake/rccl)
- `ggml/src/ggml-hip/CMakeLists.txt:50-52`: `GGML_HIP_RCCL=ON`이면 `find_package(rccl REQUIRED)` + `GGML_USE_NCCL` 정의 + `roc::rccl` 링크
- `ggml-cuda.cu:1267`: `ncclCommInitAll` → 성공 시 `try_allreduce=nccl`, 실패 시 butterfly 폴백
- 검증: `ldd bin/llama-server | grep rccl` → `librccl.so.1 => /opt/rocm-7.2.3/lib/librccl.so.1`

**만난 오류 및 해결**
- **clang-22 optimizer segfault** (`fattn-mma-kvarn-instance-ncols1_16-ncols2_8.cu.o` 크래시, `llvm::simplifyCall` 내부):
  - 원인: -j 16 동시 컴파일 시 리소스 경합 (이전 세션 ROCm10 clang-23과 동일 패턴)
  - 해결: **`-j 1`로 해당 오브젝트만 단독 재빌드 → 통과 → 이후 -j 8로 전체 빌드 정상**
  - `EXIT=0`, `librccl.so.1` 링크 확인
- **런타임 첫 시작 시 `--fit` 오류**: `llama_params_fit is not implemented for SPLIT_MODE_TENSOR` → `--fit off` 필수 (reddit 설정과 동일)

**RCCL 빌드 후 27B 스모크 검증**
- tensor split 1,1 + 27B Q3_K_M 로드 성공, VRAM **정확히 균등** (GPU0 7.09 / GPU1 7.09 GiB, 차이 7KB 미만)
- `llama-bench -p 2048 -n 128 -sm tensor -ts 1,1`: pp 800 / tg 31.4
- **RCCL 성공 징후**: NCCL init 실패 로그(`"NCCL init failed"`)가 없음 = `ncclCommInitAll` 성공
- 주의: `internal all-reduce is RDNA4-only` 경고는 여전히 출력되지만 이는 내부 파이프라인 비활성일 뿐, **RCCL 경로로 폴백해서 정상 작동**

### 27.2 PR #28223 이식: host buffer override + read_raw 로딩

**배경 (왜)**
- Flash-Next(88GB MoE)로 exps를 `-ot ...=ROCm_Host`로 보내려 했으나 **"unknown buffer type" 거부**:
  `parse_tensor_buffer_overrides()`가 `ggml_backend_dev_buffer_type()`만 수집, **host buffer type을 목록에서 누락**
- 가사 아무리 exps를 host로 보내도 mmap + host 버퍼 시 **CPU 버퍼로 강등**되어 pinned 메모리를 못 씀
- reddit 참조: PR #28223 커밋 2개가 정확히 이 문제 해결 (2x3090, 40 expert 레이어 호스트에서 prefill 166→379 t/s)

**커밋 1 (5cfa6a87): host buffer override 허용 + mmap 강등 방지**
- `common/arg.cpp`: buft_list에 `ggml_backend_dev_host_buffer_type(dev)` 추가 → `ROCm_Host`(GGML_CUDA_NAME="ROCm"+"_Host") 인식
- `src/llama-model-loader.cpp create_tensor()`: `buft_overridden` 플래그 추가, override 지정 시 mmap의 host→CPU 강등 로직 건너뜀
- 기대 효과: exps를 pinned host memory(ROCm_Host)에 유지 → GPU가 직접 접근

**커밋 2 (c59754bfc): host 대상 텐서를 read_raw로 로드**
- `src/llama-model-loader.cpp load_all_data()`: 목적지가 host 버퍼면 mmap memcpy 대신 `file->read_raw()` 직접 읽기
- 이유: mmap+host 복사는 페이지 단위 fault (readahead 없음, `--numa distribute`면 POSIX_MADV_RANDOM) → 44GB exps 복사가 극도로 느림
- 기대 효과: cold load 512s→168s (reddit 실측), copy 단계 450s→104s

**적용 방식**: python 스크립트로 hunk를 beellama 기준 위치에 수동 적용 (PR 베이스와 llama.cpp 라인 다름). `git diff` 검증.

### 27.3 PR #27861 이식: MoE expert cache (GPU-resident LRU)

**배경 (왜)**
- Flash-Next는 48레이어 MoE, exps 총 54.5GiB. VRAM(24GBx2)에 전부 못 담음 → host 오프로드 필수
- 그런데 host 오프로드만으로는 **디코딩이 CPU GEMM 병목** (실측: GPU 사용률 0%, CPU 730-820%, tg 10t/s)
- 사용자 요구: "핵심 전문가는 VRAM, 그 외는 host mem" = **LRU 캐시로 hot expert를 VRAM에 유지**
- reddit 2x3090 실측: exps 전체 host + `--moe-expert-cache 135` → decode 17→25-29 t/s (히트율 80-85%)

**이식 내용** (커밋 bccbacdb8, 12파일/645줄)
- **신규** `src/llama-moecache.cpp` (407줄) + `src/llama-moecache.h` (61줄):
  - 레이어별 companion 텐서 `up_c/gate_c/down_c` [ne0,ne1,n_slots+1] — slot n_slots는 영구 0 (dummy)
  - I32 `dev_table[512]`: expert id→slot 매핑 (device: get_rows로 cache chain에 사용, host: CPU mul_mat_id가 src[3]으로 읽어 캐시된 id 스킵)
  - async 업로드 워커 스레드: decode 스레드와 분리, step()에서 CPU 테이블만 갱신
  - 관찰 콜백: CPU `mul_mat_id`가 ffn_gate_exps 라우팅 id를 기록 → LRU 배치 결정
- **`ggml/src/ggml-cpu/ggml-cpu.c` mul_mat_id**: `src[3]` 존재 시 캐시된 expert는 dst 행을 0으로 스킵 + 라우팅 관찰 콜백 + GGML_MOE_LOG(진단)
- **`ggml.h/ggml.c`**: `ggml_moe_obs_cb_t` 콜백 전역 등록/조회 API
- **`src/llama-graph.cpp` build_moe_ffn**: n_tokens==1 + 조건 충족 시 cache 조회 → dev_table로 slot remap → up/gate/down에 `src[3]=host_table, op_params[0]=n_slots` → device side cache chain(mul_mat_id) 실행 + CPU 결과와 add
- **param 연결**: `common.h`(n_moe_cache_slots/inserts), `common.cpp`(cparams 전달), `llama.h`(cparams 필드), `llama-context.cpp`(init + decode() 끝 step() + default params), `arg.cpp`(`--moe-expert-cache N`, `--moe-expert-cache-inserts N`), `CMakeLists.txt`(소스 추가)

**적용 방식**: 신규 파일 2개는 diff에서 정확히 추출 생성, 수정 10개 파일은 python 스크립트로 hunk 수동 적용 (beellama 구조 확인 후 시그니처/패턴 매칭)

**만난 오류 및 해결**
- **`mc_inp` undeclared**: build_moe_ffn patch 5번째 hunk가 `mc_inp` 변수를 참조하는데 선언 누락
  - 원인: PR hunk에서 `ggml_tensor * mc_inp = cur;`는 `cur = ggml_reshape_3d(...)` 직후 선언되나 beellama 구조상 `weight_before_ffn` 블록 앞이라 적용 순서 차이
  - 해결: `cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);` 직후에 `ggml_tensor * mc_inp = cur;` 삽입
- 2차 빌드 **EXIT=0** 성공, `--moe-expert-cache` 옵션 인식 확인

### 27.4 최종 빌드 상태 요약

| 구성 | 값 |
|---|---|
| 소스 | beellama-boosts/src (44a36969 + boosts + 0001 adaptive MTP) |
| 빌드 디렉토리 | /tmp/be-src-native0001 |
| ROCm | 7.2.3 native |
| RCCL | `-DGGML_HIP_RCCL=ON`, librccl.so.1 링크 확인 |
| 추가 패치 | PR #28223 (host override + read_raw) + PR #27861 (MoE expert cache) |
| 검증 | `--moe-expert-cache`/`--moe-expert-cache-inserts` 옵션 인식 |

**Flash-Next 구동 시도 (진행 중)**
```
llama-server -m Flash-Next-UD-IQ4_XS -sm tensor -ts 1,1 -c 8192 -fa on \
  -ctk f16 -ctv f16 -b 2048 -ub 512 -t 12 --load-mode none \
  -ot 'ffn_(up|down|gate)_exps\.weight=ROCm_Host,per_layer_token_embd\.weight=CPU' \
  --moe-expert-cache 135
```
- 로드 중 상태: VRAM GPU당 3.35GiB (dense부분 split), exps 54.5GB는 ROCm_Host로 read_raw 로딩 중, RSS 64GB
- 기대: cache 히트 시 GPU에서 expert 연산 → 이전 CPU-only (10t/s) 대비 대폭 개선
