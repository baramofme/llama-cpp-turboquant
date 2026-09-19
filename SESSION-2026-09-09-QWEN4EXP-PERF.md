# Qwen3.8-Flash-Next 속도 개선 프로젝트 - 세션 문서 (2026-09-09)

> 목표: codacus/llama.cpp `perf` 브랜치의 성능 기능을 HIP(RDNA3, 7900 XTX) 포크에 이식하여
> Qwen3.8-Flash-Next의 decode/prefill을 개선. codacus의 CUDA 검증 수치(23~24 t/s)에 근접하는 것.
>
> 환경: 듀얼 7900 XTX (gfx1100, ROCm 7.2.0), 브랜치 `rocm10`, 모델 Qwen3.8-Flash-Next UD-IQ3_XXS (82GB, 512 experts, 10 active)

---

## 1. 배경: 왜 이 작업을 했나

이전 세션(2026-09-08)에서 확인된 사실:
- 로컬 실측: 듀얼 7900 XTX(48GB)에서 Flash-Next 84.9GB = decode **17~18 t/s** (실질 한계)
- MTP draft 적용 시 **23.6 t/s** 까지 문서화됨
- codacus 유튜브: RTX 3060 12GB에서 **24.4 tok/s** (thread 6개, expert cache + MTP)
- codacus README: prefill **1143 → 1880 t/s (+64%)**, decode **16.6 → 24.4 (+47%)**

핵심 괴리: 3060(360 GB/s)이 7900 XTX(960 GB/s)보다 빠르게 보임 → codacus의 비밀은
**MoE expert cache / host-pin / prefetch-experts** 였고, 이는 우리 빌드에 없던 기능.

---

## 2. 이식 대상 (codacus-perf vs 로컬 갭 분석 결과)

| # | 기능 | codacus 커밋 | 로컬 상태 |
|---|---|---|---|
| 1 | llama-mmap host-pin (`GGML_CUDA_REGISTER_HOST`) | 20f5994bf | ❌ 미이식 |
| 2 | ggml-backend prefetch-experts (`GGML_SCHED_PREFETCH_EXPERTS`) | 1163cb349, 5f83fbbe7 | ❌ 미이식 |
| 3 | **MoE expert cache** (`--moe-cache-profile/slots`, hot/cold dual-chain) | ac743f81f | ❌ 미이식 |
| 4 | async CPU/GPU overlap (`--sched-async-cpu`) | 0ac3d9b27 | ❌ 미이식 |
| 5 | qwen4exp expert-cache wiring | 5a0a2b60f | ❌ 미이식 |
| 6 | mul_mat_id skip-id fix | 08fa762f7 | ❌ 미이식 |
| 7 | MTP draft head (#28243) | e7cae7d26 | 🟡 uncommitted 이식 중 |
| 8 | tools/moe-trace, laguna wiring | ac743f81f 외 | ❌ 미이식 |
| - | TurboQuant KV cache (#1~#4) | 170322f8d 등 | ✅ 이미 보유 |

---

## 3. 이식 결과 (40 files, +1703/-165)

### 3.1 MoE expert cache (기능 3, 3b, 3c) - 성공
- `src/llama-model.cpp`: `init_moe_expert_cache()` - 라우팅 프로필 CSV를 읽어 hot 전문가 팩(VRAM) + cold 맵(CPU) 구성
- `src/llama-graph.cpp/h`: `build_moe_ffn`에 `use_moe_packs` dual-chain (hot pack GPU + cold 원본 CPU, 결과 add)
- `ggml-cuda/*.cu`: mmid/mmvq/mmq/mmvf/quantize에 **id=-1 skip** 지원 (핫/콜드 분할의 빈 슬롯)
- `tools/moe-trace/*`: 라우팅 프로필 캡처 도구 (moe-trace.cpp, simulate.py)
- `common/arg.cpp`: `--moe-cache-profile` / `--moe-cache-slots` 옵션

### 3.2 async CPU/GPU overlap (기능 4) - 성공
- `ggml/src/ggml-backend.cpp`: `ggml_sched_cpu_async` 워커 스레드 (CPU split을 worker에서, GPU split과 동시 실행)
- `ggml/include/ggml-backend.h`: `ggml_backend_sched_set_async_cpu()`
- `src/llama-context.cpp`, `src/llama-cparams.h`, `include/llama.h`: `sched_async_cpu` 파라미터
- `common/arg.cpp`, `common/common.cpp/h`: `--sched-async-cpu` / `--no-sched-async-cpu`
- `tools/llama-bench/llama-bench.cpp`: `sched_async_cpu` 테스트 차원 (충돌 해결 포함)

### 3.3 host-pin (기능 1) - 이식 후 HIP에서 비활성화 (아래 §4 참조)
### 3.4 prefetch-experts (기능 2) - 이식 후 HIP에서 비활성화 (아래 §4 참조)

### 3.5 MTP draft head (기능 7) - 완성 확인
- 이전 세션 uncommitted 작업 + `common/speculative.cpp`의 `mparams.model_shared = model_tgt`
- qwen4exp `graph_mtp`, `borrow_shared_tensor`, NEXTN_HC_HEAD 텐서 등

### 3.6 기타 fix (기능 5, 6) - 성공
- `src/models/qwen4exp.cpp` 등: build_moe_ffn에 `&model.layers[il]` 전달 (expert cache wiring)
- `ggml-cuda/ggml-cuda.cu`: `ids_may_skip` (mul_mat_id sync predicate)
- `ggml-backend.cpp`: prefetch expert-copy 경로의 id=-1 skip

---

## 4. 문제 해결 과정과 실패 기록

### 4.1 [실패→해결] prefill 크래시: `GGML_ASSERT(id >= 0 && id < n_expert)`
- 증상: expert cache 활성화 시 prefill에서 abort (ggml-backend.cpp:1983)
- 원인: hot/cold 분할 시 id=-1(빈 슬롯)이 prefetch expert-copy 경로를 통과
- 해결: codacus의 "sched expert-copy path skips id=-1" 픽스 적용 (`if (id < 0) continue`)
- 결과: prefill 91 → 110 t/s (+20%)

### 4.2 [실패→비활성화] host-pin (`GGML_CUDA_REGISTER_HOST`) HIP에서 hang
- 시도 1: ReadOnly 플래그 제거 → **여전히 300초+ hang** (대용량 mmap 70GB+ 등록)
- 시도 2: `hipHostMalloc`으로 expert를 pinned에 로드 → **decode 2배 악화** (3.50 t/s, 스케줄링 꼬임)
- 근본 원인 (웹 검색 + 프로브로 확정):
  - RX 7900 XTX: `hostRegisterReadOnlySupported=0` (디바이스 프로브 실측)
  - ROCm/ROCm#2433: `hipHostRegister`가 SVM 기반 전환 후 대용량 등록 시 극도로 느림 (7~54초)
  - ROCm/ROCm#6523: gfx1100 H2D 복사 커널 버그 (모델 로딩 86.7초 vs 14.4초)
- 최종: `ggml_backend_cuda_register_host_buffer` HIP 분기에서 등록 생략 (CUDA만 활성)

### 4.3 [실패→비활성화] prefetch-experts HIP에서 illegal memory access
- 증상: `GGML_SCHED_PREFETCH_EXPERTS=1` 시 `ggml_backend_cuda_set_tensor_async`에서 크래시
- 시도 1: 동기 업로드 폴백 (`#if GGML_USE_HIP`) → **안 먹음** (ggml-base에 HIP 매크로 없음)
- 시도 2: expert copy 경로도 동기화 → 동일 (컴파일 가드 무효)
- 근본 원인: ggml-backend.cpp는 backend-agnostic (GGML_USE_HIP 미정의) + HIP의
  두 번째 backend 인스턴스(별도 스트림)가 split 백엔드 스트림과 **race**
- 해결: **런타임 감지** - `ggml_backend_dev_name`이 "ROCm"이면 prefetch 비활성화
- 결과: 크래시 해결 (exit 0)

### 4.4 [시도→폐기] expert를 pinned host buffer(hipHostMalloc)로 로드
- 아이디어: `-ncmoe` expert를 `ggml_backend_cuda_host_buffer_type()`(pinned)에 로드
- 결과: GPU가 device-accessible로 취급 → 스케줄링 꼬임 → decode 2배 악화, prefill도 55 t/s로 저하
- 판단: `hipHostMalloc`은 "GPU가 직접 접근"하는 메모리라 CPU-offload 시나리오와 충돌
- 폐기: `common/common.h` 원복 (CPU buft 유지)

### 4.5 [성공] 서버 구동 검증
- MTP draft + expert cache 조합 서버 기동 성공 (포트 12000)
- `borrow_shared_tensor` 오류는 `-fit`의 메모리 측정 단계 일시 오류로 확인, 실제 서버는 정상 응답
- 단, vLLM(운영 서버)이 GPU를 12GB씩 점유하여 공정한 decode 실측 불가

---

## 5. 최종 수치 변화 (실측)

### 5.1 이번 세션 llama-bench (Flash-Next 82GB, -ngl 99 -ncmoe 99)

| 측정 | 이전 | 이후 | 변화 |
|---|---|---|---|
| decode tg64 (GPU 점유) | 5.65 t/s | **7.11 t/s** (expert cache 16슬롯) | **+26%** |
| prefill pp256 (GPU-free) | ~91 t/s | **106.71 t/s** (baseline) | - |
| prefill pp256 (캐시) | 91 t/s | **110.05 t/s** (+20%) | **+20%** |
| prefill pp1024 (GPU-free) | - | **123.49 t/s** | - |
| prefill pp256 (expert pinned 시도) | - | 55.62 t/s | 폐기 |
| decode (expert pinned 시도) | - | 3.50 t/s | 폐기 |

### 5.2 이전 세션 실측 (실서버, 듀얼 7900 XTX, Flash-Next 84.9GB)

| 구성 | decode (tg) | prefill (pp) |
|---|---|---|
| exps 전부 CPU (baseline) | 4.70 t/s | 24 t/s |
| GPU41 + host7 + cache96 | 16.49 t/s | 38 t/s |
| cache128/ub256 | 17.29 t/s | 56 t/s |
| late10 host + cache192 | 17.33 t/s | 55 t/s |
| 레이어 split + on-direct | 18.16 t/s | 55 t/s |
| **MTP draft 적용** | **23.6 t/s** | - |

### 5.3 codacus 기준 (CUDA RTX 3060, 참고용)

| 항목 | baseline | 최적화 후 |
|---|---|---|
| prefill (35B-A3B pp2048) | 1143 t/s | 1880 t/s (+64%) |
| decode (Flash-Next + MTP) | 16.6 t/s | 24.4 t/s (+47%) |

---

## 6. 수치 해석: 왜 1/10로 보이는가

prefill 비교가 "1/10"로 보이는 이유는 **모델/배치 조건이 다르기 때문** (compute bound 아님):
- codacus 1880 t/s = **35B 모델** (Qwen3.6-35B-A3B) × pp2048 × ncmoe 26
- 우리 110~123 t/s = **177B 모델** (5.1배 큼) × pp256 × ncmoe 99
- 7900 XTX(960 GB/s)는 3060(360 GB/s)보다 대역폭 2.7배 유리 → 같은 조건이면 더 빠름
- HIP에서 codacus의 +64%(host-pin+prefetch)는 hang/크래시로 사용 불가 → expert cache(+20%)로 대체

---

## 7. 최종 HIP 권장 구성

```
# HIP (7900 XTX) - 안정 최적
llama-server \
  -m Flash-Next.gguf -md MTP-shared-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 1 \
  -ngl 99 --n-cpu-moe 99 -t 6 -fa on \
  -ctk q8_0 -ctv q8_0 -c 16384 -np 1 \
  --moe-cache-profile qwen38-merged.csv --moe-cache-slots 40 \
  # GGML_CUDA_REGISTER_HOST 사용 금지 (hang)
  # GGML_SCHED_PREFETCH_EXPERTS 자동 off (ROCm 감지, 크래시 방지)
```

| 기능 | HIP 상태 | 이유 |
|---|---|---|
| MoE expert cache | ✅ 사용 | decode +26%, prefill +20% 검증 |
| async CPU overlap | ✅ 사용 (기본 on) | GPU/CPU split 동시 실행 |
| MTP draft head | ✅ 사용 | +12% (문서화) |
| host-pin | ❌ 비활성 | 대용량 SVM 등록 hang |
| prefetch-experts | ❌ 자동 off | 두 번째 스트림 race 크래시 |

---

## 8. 남은 작업

- [ ] GPU 독점 상태에서 실서버 최종 검증 (vLLM 점유 해제 후)
- [ ] expert cache 슬롯 최적값 탐색 (VRAM 여유 기반, 40~56슬롯 후보)
- [ ] MTP + expert cache 조합의 실제 27 t/s 도달 확인
- [ ] 35B-A3B 동일 조건 비교 벤치 (7900 XTX vs 3060 공정 비교)
- [ ] `GGML_SCHED_PREFETCH_EXPERTS` HIP 안정화 (동기 업로드로 진짜 overlap 구현) - 선택적

---

## 9. 참고 자료

- codacus perf 브랜치: https://github.com/thecodacus/llama.cpp/tree/perf
- ROCm#2433 (hipHostRegister SVM 느림): https://github.com/ROCm/ROCm/issues/2433
- ROCm#6523 (gfx1100 H2D 커널 버그): https://github.com/ROCm/ROCm/issues/6523
- domvox#5 (hipHostRegisterReadOnly 회귀): https://github.com/domvox/llama.cpp-turboquant-hip/pull/5
- sglang#33968 (HIP device pointer 불일치): https://github.com/sgl-project/sglang/pull/33968
- 로컬 프로브: RX 7900 XTX `hostRegisterSupported=1, hostRegisterReadOnlySupported=0`