# Upstream Merge Plan — Round 2 (2026-07-09)

## 개요

- **브랜치**: `feature/turboquant-kv-cache`
- **Merge Base (이전)**: `1ec44d178` (upstream/master HEAD at last merge)
- **현재 upstream/master HEAD**: `683f0c72e`
- **누적 커밋 수**: 141 commits
- **Conflict 파일**: 14개
- **빌드 인프라**: Docker + ROCm 7.2.1, `-j $(nproc)` (20 cores), `-DGGML_HIP_ROCWMMA_FATTN=ON`

---

## 1. Upstream 변경사항 분석

### 1.1 HIP/ROCm 관련 (7 commits)

| Commit | Summary | Impact |
|--------|---------|--------|
| `ccb0c3422` | ggml-hip: enable -funsafe-math-optimizations | Build flag - safe to take |
| `33ca0dcb9` | ggml-hip: add -fno-finite-math-only alongside -ffast-math | Build flag - safe to take |
| `07e012afd` | Make hip quality check run on all changes | CI only, no code change |
| `d06ddd358` | ggml-hip: enable -ffast-math for HIP builds | Build flag - safe to take |
| `d9df11006` | HIP: use hipBLAS for dense prefill on gfx900, keep MMQ for MoE | **Important** - HIP backend logic change |
| `2f18fe13c` | CUDA: add cublasSgemmBatched mapping for HIP/MUSA vendor headers | Vendor header update - safe |

### 1.2 Server 관련 (중요)

| Commit | Summary | Impact |
|--------|---------|--------|
| `64c8b7db7` | server: respect min-step when splitting prompt batches | Logic change in server-context |
| `f5525f7e7` | server: fix draft model fit vs load inconsistency | Draft model fix - we have process_chunk impl |
| `5eca4e3ca` | server: add timings and progress to /responses API stream | API enhancement |
| `6c487e2f7` | server: enforce prompt cache RAM limit | Memory management |
| `9abce7473` | server: fix deadlock in load_models() when erasing a finished download | Bug fix |
| `bfdf581b8` | server: temporary skip model downloading API test | Test only |
| `2da668617` | Fix stale tensor-split params for draft models | **Important** - affects our draft/speculation code |
| `bbebeec4a` | server-stream: follow-up on SSE Replay Buffer | Stream handling |
| `1a87dcdc4` | server + ui: SSE Replay Buffer | Major SSE refactor |
| `b5315e16e` | server + ui: ping silent SSE streams every 1s | Connection stability |

### 1.3 Core Library (ggml, llama) 관련

| Commit | Summary | Impact |
|--------|---------|--------|
| `57b50e1f6` | ggml: fix A indexing in simd_gemm scalar tail-column path | Bug fix - critical |
| `68a521b59` | ggml: add support for CPU f16->f16 GGML_OP_SET_ROWS | New op support |
| `ed8c26150` | cuda: add support for f16->f16 GGML_OP_SET_ROWS | New op (CUDA) |
| `024c46ae4` | llama: fix quantized kv-cache for dsv4 | **Critical** - KV cache fix |
| `4b2a0cdee` | ggml: fix tensor-parallel + -ncmoe crash on MoE models | Bug fix |
| `ef2d77011` | ggml: fix broken CPU concat implementation for quantized types | Bug fix |
| `a4107133a` | llama: add guard for K/V rotation input when buffer is unallocated | Defensive fix |
| `96183e982` | ggml: bump version to 0.15.3 | Version bump |
| `1a7c25bfd` | ggml: make ggml_time_init idempotent | Robustness fix |
| `3e5036fbf` | abort if we see a multi buffer | Safety check |

### 1.4 Speculative Decoding / DFlash (중요)

| Commit | Summary | Impact |
|--------|---------|--------|
| `d1b34251b` | spec: add DFlash support | **Major new feature** - draft model conversion |
| `fa72bc682` | dflash: refactor draft model conversion | Refactor of above |
| `152d337fa` | spec: support spec-draft-p-min in DFlash | Enhancement |
| `c198af4dc` | spec: fix naming, spacing | Cosmetic |

### 1.5 Model Support

| Commit | Summary | Impact |
|--------|---------|--------|
| `8c146a836` | DeepSeek V4 | New model family |
| `4f31eedb0` | model: register t_layer_inp for qwen3next | Qwen3Next support |
| `9d5d882d8` | model: Add label for LFM2.5-230M | Label only |

### 1.6 UI 관련 (Svelte 5 이주 완료 후)

| Commit | Summary | Impact |
|--------|---------|--------|
| `f1161b15f` | ui: Context usage gauge and panel | New feature |
| `d80e87850` | ui: restore Ctrl+B sidebar toggle shortcut | Bug fix |
| `898b08854` | ui: fake 200 for proxy DELETE req | Bug fix |
| `665892536` | ui: add sync blocks so display/behavior settings can be set via --ui-config-file | Feature |
| `d4cff114c` | ui: Improve performance when streaming | Perf improvement |
| `f113e02d5` | ui: strip path and weight extension from model id in single model mode | UX fix |
| `067de9371` | ui: align persisted config with strict server schema and enable thinking by default | Config sync |
| `94875285e` | ui: Add MCP Servers Opt-In for first time visitors | Feature |
| `ded1561b4` | ui: fix accessibility (then reverted) | Reverted |
| `dbdaece23` | Revert above | Revert |
| `7cb8576e7` | ui: fix stop and reasoning skip in single-model mode | Bug fix |
| `7af4279f4` | ui: Remove PWA navigate fallback to prevent caching API endpoint requests | Bug fix |
| `9d88e7ced` | ui Prevent tool messages from incorrectly appending to other conversations | Bug fix |

---

## 2. Conflict 파일 — 난이도 정렬 (쉬움 -> 어려움)

### Tier 1: upstream 취하면 됨 (ours 변경 최소, 거의 자동)

| # | File | Ours +/- | Upstream +/- | 전략 |
|---|------|----------|--------------|------|
| 1 | `include/llama.h` | **4** | 9 | 버전 번호 + 소량 API. ours TQ 관련선 preserve, upstream 병합 |
| 2 | `src/llama-model-loader.cpp` | **6** | 99 | ours minimal, upstream 대폭 변경. ours 6줄 보존하고 upstream 취함 |
| 3 | `gguf-py/gguf/constants.py` | **8** | 128 | Python GGUF 상수. upstream 취함 |
| 4 | `common/common.cpp` | **10** | 119 | ours 변경 최소. upstream 구조 follow, ours 10줄 preserve |

### Tier 2: backend 전용 (ours 무시하고 upstream 취함)

| # | File | Ours +/- | Upstream +/- | 전략 |
|---|------|----------|--------------|------|
| 5 | `ggml/src/ggml-hexagon/htp/hmx-flash-attn-ops.c` | 9 | DELETE (1840) | **modify/delete conflict**. upstream이 삭제 -> ours도 삭제 |
| 6 | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | 76 | 269 | Vulkan 전용. upstream 취함 |
| 7 | `ggml/src/ggml-cuda/fattn.cu` | 136 | 50 | CUDA FA. HIP에서 자동 컴파일되므로 upstream 취함 |
| 8 | `ggml/src/ggml-metal/ggml-metal.metal` | 2625 | 390 | Metal 전용 (대용량). upstream 취함 |
| 9 | `ggml/src/ggml-opencl/ggml-opencl.cpp` | 780 | 4362 | OpenCL 전용 (대용량). upstream 취함 |

### Tier 3: 양쪽 의미 있는 변경 (분석 후 병합)

| # | File | Ours +/- | Upstream +/- | 전략 |
|---|------|----------|--------------|------|
| 10 | `ggml/include/ggml.h` | **22** | 6 | TQ type 정의(ours) + 버전 bump(upstream). 양쪽 preserve |
| 11 | `tests/test-quantize-fns.cpp` | **29** | 92 | TQ 테스트(ours) + upstream 신규 테스트. 양쪽 preserve |
| 12 | `ggml/src/ggml-cuda/ggml-cuda.cu` | **299** | 1880 | HIP 컴파일 대상. upstream의 HIP 관련 변경(hipBLAS, vendor headers) 확인 후 merge. ours TQ 관련 코드 preserve |

### Tier 4: Critical (TurboQuant 핵심 vs upstream 대대적 변경)

| # | File | Ours +/- | Upstream +/- | 전략 |
|---|------|----------|--------------|------|
| 13 | `src/llama-kv-cache.cpp` | **458** | 56 | **TurboQuant KV cache 핵심**. ours를 기반으로 upstream의 dsv4 fix 등 merge. upstream 변경이 적으므로 비교적 쉬움 |
| 14 | `tools/server/server-context.cpp` | **146** | **305** | **가장 어려움**. ours(process_chunk, mtmd draft) + upstream(SSE Replay Buffer, prompt batch split, SSE ping) 모두 중요. 세심한 수동 merge 필수 |

---

## 3. 통합 계획 (작업 순서: 1 -> 14)

### Phase 1: Merge 실행

```bash
git merge upstream/master --no-commit --no-ff
```

### Phase 2: Conflict 해결 (Tier 1 -> Tier 4 순서)

#### Step 1: Tier 1 — 거의 자동 (4개 파일, #1~#4)

```bash
# include/llama.h: 버전 번호 upstream 취함, TQ type 정의 preserve
# src/llama-model-loader.cpp: ours 6줄 preserve, upstream 취함
# gguf-py/gguf/constants.py: upstream 취함
# common/common.cpp: upstream 구조 follow, ours 10줄 preserve
```

#### Step 2: Tier 2 — backend 전용, upstream 취함 (5개 파일, #5~#9)

```bash
# hmx-flash-attn-ops.c: 삭제
git rm ggml/src/ggml-hexagon/htp/hmx-flash-attn-ops.c

# 나머지는 upstream 내용으로 교체 (-X theirs equivalent)
git checkout --theirs ggml/src/ggml-vulkan/ggml-vulkan.cpp
git checkout --theirs ggml/src/ggml-cuda/fattn.cu
git checkout --theirs ggml/src/ggml-metal/ggml-metal.metal
git checkout --theirs ggml/src/ggml-opencl/ggml-opencl.cpp
```

#### Step 3: Tier 3 — 분석 후 병합 (3개 파일, #10~#12)

각 파일에서 conflict marker 확인 -> ours의 TQ 관련 코드 preserve하면서 upstream 변경 통합.

#### Step 4: Tier 4 — Critical 수동 merge (2개 파일, #13~#14)

- `src/llama-kv-cache.cpp`: TurboQuant KV 코드가 대부분. upstream의 dsv4 fix(56 lines)를 ours에 병합.
- `tools/server/server-context.cpp`: 가장 복잡. process_chunk 구현 보존하면서 SSE Replay Buffer, prompt batch split, SSE ping 통합 필요.

#### Step 5: Commit

```bash
git add -A && git commit -m "merge upstream/master: integrate 141 commits"
```

### Phase 3: 빌드 검증

1. Docker build with current config (`-DGGML_HIP_ROCWMMA_FATTN=ON`, `-j $(nproc)`)
2. Build 성공 시 image tag: `baramofme/llama-cpp-rocm:gfx1100-rocm7.2-tbqplus-pchunk-rocwmma-v2`

### Phase 4: 푸시 + 배포 검증

1. Commit push to `baramofme/feature/turboquant-kv-cache`
2. Docker image push
3. Dokploy deployment update

---

## 4. 리스크 평가

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| KV cache conflict 해결 실패 | Medium | TurboQuant 코드는 ours 기준, upstream 변경은 minimal (56 lines) |
| server-context.cpp SSE replay buffer 충돌 | High | process_chunk 코드 보존 필수, SSE 변경은 upstream follow |
| ggml version bump -> TQ types 영향 | Low | ggml.h conflict는 버전 번호만 바뀔 가능성 높음 |
| DFlash support와의 상호작용 | Medium | Draft model 관련 코드(process_chunk)와 충돌 가능 |
| ggml-cuda.cu HIP 변경 영향 | Low-Medium | upstream의 hipBLAS, vendor header 변경 확인 필요 |
| 빌드 시간 증가 | Low | `-j $(nproc)` 적용됨, 캐시 유효하면 빠름 |

---

## 5. 제외 항목 (이번 통합에서 skip)

- Hexagon backend changes — 우리 환경과 무관
- Metal/Vulkan/WebGPU/OpenCL/SYCL 전용 변경 — upstream 취하지만 검증 대상 아님
- CI/workflow 변경 — `.github/` 폴더
- Binaries release artifacts
- UI-specific Svelte 컴포넌트 추가 (기능적 영향 적음, 자동 merge 될 것)
