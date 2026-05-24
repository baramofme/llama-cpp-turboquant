# Sisyphus 작업 일지 — 2026-05-24

## 작업 개요

DrBearJew의 `tbq4-rdna3-experiment` 브랜치에서 RDNA3 MoE MMQ LDS double-buffer
prefill accelerator를 `feature/turboquant-kv-cache` → `feature/rdna3-mmq-lds-accel`로 포팅.

---

## 상세 내역

### 1. RDNA3 MMQ LDS Accelerator 포팅

**파일**: `ggml/src/ggml-cuda/mmq.cuh`

**변경사항**:
- `getenv`/`strcmp`를 위한 `<cstdlib>`, `<cstring>` include 추가
- `ggml_cuda_mmq_x_max_env()`, `ggml_cuda_mmq_x_max_auto_env()` — env var 헬퍼
- `MMQ_TARGET_WG_THREADS=256` 도입, `mmq_get_nwarps_host/device` 리팩터
- `get_mmq_x_max_host()` — RDNA3 auto-cap (GGML_CUDA_MMQ_MAX_X_AUTO=1 → x_max=48)
- `load_tiles_q1_0` — `get_int_b1()` 헬퍼 사용
- `mul_mat_q_process_tile()` — **LDS double-buffer** 추가
  - `#ifdef RDNA2_MATMUL_OPT_V1` + `use_experimental` 이중 가드
  - tile_x 로드와 vec_dot 연산 오버랩
  - 기존 코드는 `else` 브랜치에 보존 (DRY)
- `mul_mat_q()` — `const bool use_experimental` 파라미터 추가
- `mmq_use_rdna2_matmul_opt(cc)` — env var + RDNA2/3 CC 검사
- `mmq_get_nbytes_shared()` — `use_experimental=true` 시 double-buffer LDS 할당
- `launch_mul_mat_q()` — `use_experimental` 계산 (MoE + ncols_max≥128 + env + LDS 예산)
- `mul_mat_q_case()` — mmq_x 후보군 experimental LDS 적합성 검사
- 모든 4개 kernel launch call에 `use_experimental` 전달

### 2. LDS 예산 문제 해결

**문제**: RDNA3 WMMA 경로에서 `mmq_y=128` → `nbs_x=43KB`. Double-buffer 추가 시
`85KB > 64KB(smpbo)` → accelerator 비활성화.

**해결**: 
- `get_mmq_y_host/device`: RDNA3 + `RDNA2_MATMUL_OPT_V1` 시 `64` 반환
- `mmq_get_nwarps_host/device`: RDNA3 WMMA + `RDNA2_MATMUL_OPT_V1` 시 `4` 반환
  (`4 × tile_C::I(16) = 64 = mmq_y` 조건 만족)
- 결과: nbs_x=21KB, double-buffer 포함 49KB < 64KB ✅

### 3. 빌드

```bash
cmake -B build-hip \
  -DGGML_HIP=ON \
  -DCMAKE_C_COMPILER=/opt/rocm-7.2.0/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-7.2.0/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_FLAGS="-DRDNA2_MATMUL_OPT_V1" \
  -DGGML_CUDA_FA=OFF
make -j$(nproc) ggml-hip llama-bench llama-cli
```

**이슈**: `fattn-mma-f16.cuh` HIP 컴파일 에러 (ROCm 7.2 + gfx1100)
→ `-DGGML_CUDA_FA=OFF`로 우회.

### 4. 성능 측정

**환경**: RX 7900 XTX, Qwen3.6-35B-A3B-UD-Q3_K_XL (15.68 GiB)

| Test | Accel OFF | Accel ON | Speedup |
|------|:---------:|:--------:|:-------:|
| pp128 | 327 | 907 | **2.77x** |
| pp256 | 528 | 1214 | **2.30x** |
| pp512 | 765 | 1402 | **1.83x** |
| tg64 | 93 | 93 | **1.00x** |

### 5. API 레벨 검증 (llama-server)

서버 구동 후 curl로 chat completion 테스트:

| 항목 | 결과 |
|------|------|
| 기본 채팅 | Content 정상 출력, hallucination 없음 |
| Reasoning | Thinking budget 4096 정상, infinite loop 없음 |
| Force recalc | SWA 특성상 cache miss (upstream PR #13194, MMQ 무관) |
| segfault/OOM | 0건 |
| TG t/s (서버) | 89~92 t/s (생산 컨테이너 37 → 우리 .so 교체 시 2.5배 향상) |

### 6. Docker .so 마운트 교훈

HIP shared library 교체 시 SONAME 파일(`libggml-hip.so.0`)을 마운트해야 함.
symlink(`libggml-hip.so`)는 무시됨 → 초기 테스트가 무효화되는 원인.

### 7. 브랜치 정리

| 브랜치 | 설명 |
|--------|------|
| `feature/turboquant-kv-cache` | 원복 완료 (RDNA3 내용 제거) |
| `feature/rdna3-mmq-lds-accel` | **신규**. RDNA3 accelerator 전용 브랜치 |

---

## commit

```
1e6ff165c port RDNA3 MMQ LDS double-buffer accelerator from DrBearJew
```

## 관련 파일

| 파일 | 설명 |
|------|------|
| `ggml/src/ggml-cuda/mmq.cuh` | 본체 (+191/-49) |
| `docs/rdna3-mmq-lds-accelerator.md` | 기술 문서 (신규) |
| `.sisyphus/plans/upstream-merge-build-test-plan.md` | 테스트 계획/결과 (갱신) |
| `.sisyphus/changelog-2026-05-24.md` | 본 작업 일지 |
