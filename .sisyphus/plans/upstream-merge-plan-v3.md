# Upstream Merge — Round 3 (2026-08-16)

## 개요

- **브랜치**: `merge-pr-24785`
- **Merge Base**: `1ec44d178` (2026-06-28 merge, 09ec3d05a)
- **머지된 upstream**: `adb55e514` (ggml-org/llama.cpp master HEAD)
- **커밋 수**: 646 commits (1ec44d178..adb55e514)
- **Conflict 파일**: 34개
- **검증**: 로컬 ROCm HIP 빌드 (gfx1100;gfx942;gfx950) 성공, llama-cli 스모크 테스트 통과

---

## 1. 머지 실행

```bash
git fetch upstream master
git merge upstream/master --no-commit --no-ff
```

## 2. Conflict 해결 전략 (34개 파일)

### Tier A: upstream 취함 (TQ 코드 없음)
| 파일 | 처리 |
|------|------|
| `ggml/src/ggml-hexagon/htp/hmx-flash-attn-ops.c` | **삭제** (upstream deleted) |
| `ggml/src/ggml-opencl/ggml-opencl.cpp` | theirs (TQ=0, upstream 대폭 변경) |
| `common/chat.cpp`, `common/common.cpp` | theirs (ours 변경이 upstream에 이미 반영) |
| `ggml/src/ggml-cuda/mmq.cuh`, `ggml/include/ggml-rpc.h` | theirs |

### Tier B: 양쪽 유지 (TQ + upstream 신규)
| 파일 | 처리 |
|------|------|
| `ggml/include/ggml.h` | TQ type 42-46 유지, **upstream Q2_0를 47로 이동** (GGUF 호환), COUNT=48 |
| `gguf-py/gguf/constants.py` | 동일 - TQ 45/46 유지, Q2_0=47 |
| `include/llama.h` | MOSTLY_TQ3_1S/4_1S 유지 + MOSTLY_Q2_0 추가 |
| `src/llama-model-loader.cpp` | upstream `llama_ftype_name` 구조 채택 + TQ3_1S/TQ4_1S 케이스 추가 |
| `ggml/src/ggml.c` | TURBO_WHT + LIGHTNING_INDEXER/DSV4_HC_* 모두 유지, OP_COUNT=102 |
| `ggml/src/ggml-cpu/*` | turbo_wht + dsv4 dispatch 모두 유지 |
| `ggml/src/ggml-cuda/dequantize.cuh` | TQ dequant 5종 + upstream k-quants 모두 유지 |
| `ggml/src/ggml-cuda/getrows.cu` | TQ4_1S/TQ3_1S 케이스 + upstream Q2_K~IQ2_XXS 케이스 모두 유지 |
| `ggml/src/ggml-cuda/fattn.cu` | helper에 TURBO 타입 지원 추가, WMMA 블록 제거(upstream 제거), RDNA4 fast path 유지 |
| `ggml/src/ggml-metal/*` | TQ kernels + upstream TQ2_0 모두 유지 |
| `ggml/src/ggml-vulkan/*` | TQ 파이프라인 + upstream 3-arg SET_ROWS 매크로 통합 |
| `tests/test-quantize-fns.cpp` | TQ3_1S LOWBIT + Q2_0 TERNARY 모두 유지 |
| `tests/test-backend-ops.cpp` | TURBO_WHT/SET_ROWS 테스트 + upstream 신규 테스트 모두 유지 |
| `tools/server/server-common.cpp` | assistant prefill + reasoning_effort 모두 유지 |
| `tools/server/server-context.cpp` | process_chunk + metrics/stats 모두 유지 |

### Tier C: ours 유지 (TQ 전용, theirs 빈 블록)
| 파일 | 처리 |
|------|------|
| `src/llama-kv-cache.cpp` | mul_mat_aux + InnerQ stubs 유지, DSA arch 목록은 upstream 확장 채택 |
| `src/llama-context.cpp` | TurboQuant FA 자동 활성화 가드 유지 |
| `README.md` | fork 콘텐츠 유지 |
| `scripts/hip/gcn-cdna-vgpr-check.py` | RDNA3 FA 시그니처 유지 |

## 3. 핵심 결정: Q2_0 타입 ID

upstream이 `GGML_TYPE_Q2_0 = 42`를 추가했으나, 우리 fork의 TQ 타입이 42-46을 점유 (이미 배포된 GGUF 포맷).
→ **TQ 42-46 고정, Q2_0를 47로 이동, COUNT=48**. GGUF 타입 ID가 ggml_type enum과 1:1 매핑이므로 ID 변경 시 기존 모델 파일 호환성이 깨질 수 있어 TQ를 지키는 선택.

## 4. upstream 리팩토링 대응

- **CUDA Virtual Devices (#25228)**: `ggml_cuda_copy_across_devices`/host-staged copy 제거됨 → upstream의 `cudaMemcpyPeerAsync` 방식 수용 (단일 GPU 타겟에 무해)
- **WMMA FA 제거**: `BEST_FATTN_KERNEL_WMMA_F16`/`ggml_cuda_should_use_wmma_fattn` 삭제 → 우리 WMMA 블록 제거, RDNA4 VEC fast path 유지
- **`thinking_end_tag` → `thinking_end_tags`** (vector): server-common.cpp 3곳 수정
- **`ctx_dft` raw pointer 전환**: `ctx_dft.get()` → `ctx_dft`
- **`slot.n_decoded` 제거**: `slot.stats.n_gen`으로 대체

## 5. 빌드 검증 (로컬 ROCm HIP)

```bash
cmake --build build -j 20   # GGML_HIP=ON, gfx1100;gfx942;gfx950
```

| 타겟 | 결과 |
|------|------|
| ggml-base | ✅ |
| ggml-cpu | ✅ (ops.cpp 중괄호 수정 1건) |
| ggml-hip | ✅ (dequantize.cuh 중괄호, getrows.cu TQ3_1S case, fattn.cu 잔재 수정) |
| llama | ✅ (llama-kv-cache.cpp 중복 블록 수정) |
| llama-server | ✅ (API 변경 대응 4건) |
| 전체 (tests 포함) | ✅ (test-backend-ops 중괄호/변수명, test-quantize-fns 마커 수정) |
| llama-cli 스모크 | ✅ version 출력 정상 |

## 6. 커밋

```
d937d52de Merge upstream/master (ggml-org/llama.cpp) - 646 commits
```

## 7. 후속 작업 (미수행)

- Docker 이미지 빌드/푸시 (`baramofme/llama-cpp-rocm` 태그)
- dokploy 배포 갱신
- feature/turboquant-kv-cache 브랜치로 머지/푸시 여부 결정
