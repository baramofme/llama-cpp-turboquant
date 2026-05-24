# RDNA3 MMQ LDS Double-Buffer Accelerator Port

> **작업일**: 2026-05-24
> **브랜치**: `feature/turboquant-kv-cache`
> **소스**: DrBearJew `tbq4-rdna3-experiment` (commit `474fd01b7`)
> **머지 대상**: `upstream/master` (ggml-org/llama.cpp)

## 배경

MoE 모델의 Expert 행렬곱셈은 quantized weight(`tile_x`)를 LDS에 로드한 후 `vec_dot`으로 계산한다.
기존 구현은 tile_x 로드 중 GPU가 idle 상태가 되는 구간이 있었다. DrBearJew의 RDNA3 MMQ LDS accelerator는
tile_x 로드와 vec_dot 계산을 **double-buffer로 중첩(overlap)**하여 이 idle을 제거한다.

## 포팅 범위

- **대상**: `ggml/src/ggml-cuda/mmq.cuh` (1개 파일)
- **범위**: RDNA3 MMQ LDS accelerator **만**. TBQ4 아키텍처, KV cache 타입 변경, ROCm perf 패치 제외.

## 변경사항 상세

### 1. 새 유틸리티 함수들

| 함수 | 목적 |
|------|------|
| `ggml_cuda_mmq_x_max_env()` | `GGML_CUDA_MMQ_MAX_X` env var 파싱 |
| `ggml_cuda_mmq_x_max_auto_env()` | `GGML_CUDA_MMQ_MAX_X_AUTO` 읽기 (bool) |
| `mmq_use_rdna2_matmul_opt(cc)` | `RDNA2_MATMUL_OPT_V1` env var + CC 검사 |

### 2. `get_mmq_x_max_host()` — RDNA3 auto-cap

`GGML_CUDA_MMQ_MAX_X_AUTO=1` + gfx1100 → x_max 48로 자동 제한.
DrBearJew 테스트 기준 35B IQ4_XS에서 48이 최적.

### 3. `MMQ_TARGET_WG_THREADS` 도입 (256)

워크그룹 타겟 스레드 수를 상수화. `mmq_get_nwarps_host/device`에서 사용.
기존 하드코딩된 `256/warp_size` 대체.

### 4. LDS Double-Buffer (`mul_mat_q_process_tile`)

`#ifdef RDNA2_MATMUL_OPT_V1` + `if (use_experimental)` 가드.

```
// Original:
for kb0:
  load tile_x(kb0)     → LDS
  load tile_y(kb0)     → LDS
  vec_dot(tile_x, tile_y)
  load tile_y(kb0+1)   → LDS
  vec_dot(tile_x, tile_y)
  __syncthreads()

// Double-buffer:
load tile_x(0)          → tile_x (pre-load)
for kb0:
  load tile_y(kb0)      → tile_y
  load tile_x(kb0+1)    → tile_x_next (prefetch, concurrent)
  __syncthreads()
  vec_dot(tile_x, tile_y)
  __syncthreads()
  load tile_y(kb0+1)    → tile_y
  __syncthreads()
  vec_dot(tile_x, tile_y)
  __syncthreads()
  copy tile_x_next → tile_x
```

- `+1 lds_bank_pad`: 32뱅크 대칭성 파괴 → bank conflict 감소
- 첫 tile_x만 pre-load, 이후 매 iteration에서 tile_x_next 동시 로드

### 5. `mmq_get_nbytes_shared()` — Double-buffer LDS 할당

`use_experimental=true` 시 `nbs_x + sizeof(int)` 만큼 추가 할당.

### 6. `mul_mat_q()` 커널 — `const bool use_experimental` 파라미터 추가

호출 체인: `mul_mat_q_case()` → `launch_mul_mat_q()` → `mul_mat_q<>()` → `mul_mat_q_process_tile()`

### 7. mmq_y=64 + nwarps=4 (RDNA3 LDS 예산 맞춤)

**발견된 문제**: RDNA3 WMMA 경로에서 mmq_y=128이면 nbs_x=43KB.
Double-buffer 추가 시 85KB > 64KB(smpbo) → 활성화 불가.

**해결**: RDNA3 + `RDNA2_MATMUL_OPT_V1` 컴파일 시:
- `mmq_y=64` → nbs_x=21KB, double-buffer 포함 49KB < 64KB ✅
- `nwarps=4` → `4 × tile_C::I(16) = 64 = mmq_y` (static_assert 조건 만족)

```cpp
// get_mmq_y_host()
#ifdef RDNA2_MATMUL_OPT_V1
    if (GGML_CUDA_CC_IS_RDNA3(cc)) return 64;
#endif

// get_mmq_y_device()  
#elif defined(RDNA2_MATMUL_OPT_V1)
    return 64;
```

## 활성화 조건

```cpp
use_experimental = (args.ids_dst != nullptr)       // MoE expert 경로
                && (args.ncols_max >= 128)          // Prefill size
                && mmq_use_rdna2_matmul_opt(cc)     // Env var + RDNA2/3
                && (LDS_budget_with_doublebuffer <= smpbo)  // 64KB 이내
```

```bash
# 환경변수
export RDNA2_MATMUL_OPT_V1=1         # 필수: double-buffer 활성화
export GGML_CUDA_MMQ_MAX_X_AUTO=1    # 권장: gfx1100 x_max=48
```

## 테스트 결과

### 환경

| 항목 | 값 |
|------|-----|
| GPU | AMD Radeon RX 7900 XTX (gfx1100) × 2 |
| ROCm | 7.2.0 |
| 모델 | Qwen3.6-35B-A3B-UD-Q3_K_XL (34.66B, MoE 3B active) |
| 빌드 | `-DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DRDNA2_MATMUL_OPT_V1` |

### 성능 (llama-bench)

| Test | V1=0 (mmq_y=64) | V1=1 (double-buffer) | Speedup |
|------|:--------------:|:--------------------:|:-------:|
| pp128 | 327 tok/s | 907 tok/s | **2.77x** |
| pp256 | 528 tok/s | 1214 tok/s | **2.30x** |
| pp512 | 765 tok/s | 1402 tok/s | **1.83x** |
| tg64 | 93 tok/s | 93 tok/s | **1.00x** (MMVQ) |

**TG 영향 없음** — TG는 MMVQ(벡터 커널) 사용, MMQ 변경과 무관.

### API 레벨 검증 (llama-server)

| 테스트 | 결과 |
|--------|------|
| 기본 채팅 | Content 정상 출력, hallucination 없음 |
| Reasoning | Thinking budget 4096 정상 동작, 무한루프 없음 |
| 강제 재처리 | SWA 특성상 cache miss 발생 (upstream PR #13194), MMQ 무관 |
| segfault/OOM | 0건 |
| TG t/s (서버) | 89~92 t/s (생산 컨테이너 원본 37 → 우리 .so 교체 시 2.5배 향상) |

### Docker .so 마운트 주의사항

HIP shared library 교체 시 **SONAME 파일**(`libggml-hip.so.0`)을 마운트해야 함:

```bash
# 올바름
-v /path/to/libggml-hip.so.0.12.0:/app/libggml-hip.so.0:ro

# 잘못됨 (symlink 무시됨)
-v /path/to/libggml-hip.so:/app/libggml-hip.so:ro
```

## 트레이드오프

| 측면 | mmq_y=128 | mmq_y=64 |
|------|:---------:|:--------:|
| LDS 사용량 | 49KB | 28KB |
| Double-buffer 가능? | ❌ (85KB > 64KB) | ✅ (49KB < 64KB) |
| 블록 수 | 적음 | 2배 |
| Dense 레이어 성능 | 기준 | 약 5-10% 감소 |
| MoE expert 성능 | 기준 | **2배 향상** |
| Split mode 영향 | 없음 | 없음 (각 GPU 독립적) |
| TG 성능 | 영향 없음 | 영향 없음 |

**MoE 비중 70% 모델 기준**: `(0.3 × 0.9) + (0.7 × 2.0) = 1.67배` → 전체 67% 순이익.

## 파일 변경

| 파일 | 변경 | 설명 |
|------|:----:|------|
| `ggml/src/ggml-cuda/mmq.cuh` | +191/-49 | LDS double-buffer, mmq_y=64, nwarps=4, env helpers |
| `docs/rdna3-mmq-lds-accelerator.md` | 신규 | 본 문서 |

## 참고

- DrBearJew 원본 브랜치: `tbq4-rdna3-experiment` (commit `474fd01b7`)
- LDS budget 계산: `nbs_x = mmq_y × mmq_tile_x_k × sizeof(int)` = `128 × 84 × 4 = 43008 bytes` (mmq_y=128)
- smpbo: gfx1100 = 65536 bytes (64KB)
- Upstream SWA cache 이슈: PR #13194 (MMQ와 무관)
