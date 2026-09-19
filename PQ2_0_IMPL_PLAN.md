# PQ2_0 (Bonsai 2) GPU Port - Implementation Plan

> WORKING NOTES. NOT for PR. Do NOT commit this file.
> Purpose: self-contained reference so any phase can be picked up in a fresh context
> WITHOUT re-deriving the design. Read only the section you are executing.

## 0. Goal (one line)
Get `Ternary-Bonsai-2-27B-PQ2_0.gguf` + `mmproj-BF16` running on **GPU1** (ROCm/gfx1100),
correct output first, then fast. GPU0 is OFF-LIMITS (llm-main runs there).

## 1. Verified facts (DO NOT re-derive - these are confirmed)

### 1.1 The codec: PQ2_0 == Q2_0 with group 64 -> 128 (IDENTICAL otherwise)
```
dequant formula (SAME for both):   y = (q - 1) * d      where q in {0,1,2,3} -> {-d, 0, +d, +2d}
bit layout:                        2 bits/element, packed LSB-first, 4 elems/byte
block_q2_0 : { ggml_half d; uint8_t qs[16]; }   QK2_0    = 64   (WORKS in this fork)
block_pq2_0: { ggml_half d; uint8_t qs[32]; }   QK_PQ2_0 = 128  (target)
```
Confirmed by reading BOTH CPU refs side by side (`ggml/src/ggml-quants.c`):
- `dequantize_row_q2_0`  (line ~473): `y[i*qk+j] = ((int)q - 1) * d;`
- `dequantize_row_pq2_0` (line ~493): IDENTICAL, only `QK_PQ2_0` differs.
=> Any q2_0 GPU kernel mirrored with QK 64->128 is CORRECT for pq2_0.

### 1.2 Hadamard is ALREADY in this fork (NOT a task)
- `src/llama-graph.cpp`: `llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot)` (Q/K/V rotation).
- `src/llama-kv-cache.cpp`: `ggml_gen_hadamard()` + `attn_rot_hadamard` tensors.
- `ggml/src/ggml-cuda/fwht.cu`: `ggml_cuda_op_fwht()` kernel present.
- `ggml-cuda.cu:1905`: dispatches to fwht when `GGML_HINT_SRC0_IS_HADAMARD` set.
=> Do NOT port Hadamard. It is already wired and works (Bonsai 1 Q2_0 runs).

### 1.3 cuBLAS fallback EXISTS -> minimal path only needs dequantize+convert+supports_op
`ggml-cuda.cu:1959`: if no mmvf/mmvq/mmq applies for the type, it calls
`ggml_cuda_mul_mat_cublas` -> `ggml_cuda_mul_mat_cublas_impl` (line 1495) which does:
```cpp
const auto convert_func = traits::convert(src0->type);   // dequantize weight to f16/bf16/f32
GGML_ASSERT(convert_func != nullptr);                     // null => crash. So we MUST register it.
convert_func(src0->data, src0_alloc.get(), ...);          // then cuBLAS GEMM
```
=> Phase A (register dequantize+convert+supports_op) makes ALL matmuls work via this fallback.

### 1.4 Native fast path for Q2_0 ALREADY works on HIP in this fork
- `vecdotq.cuh:821` comment: "HIP path, same identity as the Q2_0 path" -> native decode kernel exists.
- `mmvq.cu`: q2_0 wired at line 15 (vec_dot ptr), 44 (VDR cfg), 1742-1743 (switch_ncols_dst).
=> Phase B = mirror these q2_0 entries for pq2_0.

## 2. Build environment (READY - verified)
- Container `rocm10-dev` is UP (from runtime image, has ROCm clang++ + g++).
- Inside: cmake 4.4.3 (`/usr/local/bin/cmake`), ccache, ninja, g++, python3 all present.
- Mounts: source -> `/src/src`, models -> `/models`, ccache vol -> `/build-ccache`.
- Build script: `beellama-boosts/build-rocm10-0001.sh` (GGML_HIP=ON, AMDGPU_TARGETS=gfx1100).
  NOTE: it does `rm -rf` build dir; ccache gives incremental speed on unchanged files.
- First (cold) build is SLOW (compiles all .cu for gfx1100). Subsequent builds hit ccache = fast.
- Test on GPU1 only: prefix commands with `HIP_VISIBLE_DEVICES=1`.

## 3. Files already edited in Phase 0 (core/CPU - DONE prior session, VERIFY ONLY)
All present; re-read to confirm before building:
- `ggml/include/ggml.h`: `GGML_TYPE_PQ2_0 = 142`, `GGML_TYPE_COUNT = 144`.
- `ggml/src/ggml-common.h`: `QK_PQ2_0 128`, `block_pq2_0 { ggml_half d; uint8_t qs[32]; }`, `QI_PQ2_0`, `QR_PQ2_0`.
- `ggml/src/ggml-quants.c`: `quantize_row_pq2_0_ref`, `dequantize_row_pq2_0` (line ~493), `quantize_pq2_0`, validate case.
- `ggml/src/ggml-quants.h`: 3 declarations.
- `ggml/src/ggml.c`: `type_traits[GGML_TYPE_PQ2_0]`, ftype mapping, quantize dispatch.
- `include/llama.h`: `LLAMA_FTYPE_MOSTLY_PQ2_0 = 141`.

=====================================================================
## PHASE A - Minimal GPU correctness (the gate)
=====================================================================
GOAL: model LOADS on GPU and produces SANE output via the cuBLAS fallback.
REQ: 3 small edits, all mirroring the existing q2_0 entries. NO new algorithms.
PASS CRITERIA:
  [A-build] build exits 0 (no "unsupported type" / no missing-symbol link errors).
  [A-load ] `llama-cli` loads model, offloads tensors to GPU1, no runtime assert/crash.
  [A-sane ] a short prompt yields coherent text (correct language, no garbage/repeat loops).
            (Compare against the same prompt on Bonsai-1 Q2_0 which is known-good.)

### A1. dequantize.cuh - add device dequantize for pq2_0
FILE: `ggml/src/ggml-cuda/dequantize.cuh`
REF : line 26 `dequantize_q2_0(const void * vx, const int64_t ib, const int iqs, float2 & v)`
DO  : add `dequantize_pq2_0(...)` DIRECTLY BELOW it. Copy the q2_0 body verbatim; it reads
      `block_q2_0` -> change to `block_pq2_0`. The formula stays `(q-1)*d` (identical).
      Only difference vs q2_0: block type name + (if the body hardcodes QK2_0) use QK_PQ2_0.
CHECK: read dequantize_q2_0 first; mirror EXACTLY. Do not change bit-packing logic.

### A2. convert.cu - register pq2_0 in the type->kernel dispatch maps
FILE: `ggml/src/ggml-cuda/convert.cu`
REF : line 462-463 pattern (repeats at ~519, ~579, ~638, ~663, ~688):
      `case GGML_TYPE_Q2_0: return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;`
DO  : in EACH of those maps (to_bf16 / to_fp16 / to_fp32 + their `_nc` non-contiguous variants),
      add immediately after the Q2_0 case:
      `case GGML_TYPE_PQ2_0: return dequantize_block_cont_cuda<QK_PQ2_0, QR_PQ2_0, dequantize_pq2_0>;`
CHECK: grep `case GGML_TYPE_Q2_0:` in convert.cu -> there are ~6 sites; add pq2_0 to ALL of them.
      QR_PQ2_0 already defined in ggml-common.h (Phase 0). If a site uses a different helper,
      mirror whatever the q2_0 line at THAT site does.

### A3. ggml-cuda.cu - allow MUL_MAT with pq2_0 weight
FILE: `ggml/src/ggml-cuda/ggml-cuda.cu`
REF : line 5926-5957, the `switch (a->type)` inside the MUL_MAT supports check; Q2_0 at line 5930.
DO  : add `case GGML_TYPE_PQ2_0:` right after `case GGML_TYPE_Q2_0:` (line 5930).
CHECK: this makes `ggml_backend_cuda_supports_op` return true so the tensor offloads to GPU.

### A4. (only if A-load fails) getrows / other op-support switches
If model load complains about GET_ROWS or another op with pq2_0, add `case GGML_TYPE_PQ2_0:`
next to Q2_0 in that specific switch (same file). Embeddings here are BF16 (type 30), so this
is likely NOT needed - only do it if the error names the op.

---------------------------------------------------------------------
BUILD + TEST (Phase A)
---------------------------------------------------------------------
```
# build (cold, slow first time)
docker exec rocm10-dev bash /src/src/beellama-boosts/build-rocm10-0001.sh
#   -> outputs to /src/src/build-rocm10/bin/  (llama-cli, llama-bench, llama-server)

# correctness on GPU1
HIP_VISIBLE_DEVICES=1 /src/src/build-rocm10/bin/llama-cli \
  -m /models/bonsai-2/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --mmproj /models/bonsai-2/Ternary-Bonsai-2-27B-mmproj-BF16.gguf \
  -ngl 99 -c 4096 -p "Explain in one sentence why the sky is blue." -n 64
```
PASS if: loads, offloads to GPU1, prints coherent text. FAIL (garbage) => dequantize bug;
re-check A1 formula against CPU ref `dequantize_row_pq2_0`.

=====================================================================
## PHASE B - Native vecdot/mmvq (performance)
=====================================================================
GOAL: faster decode/prefill using native quantized kernels (instead of dequant->cuBLAS).
REQ: mirror the working q2_0 native path for pq2_0. Group 64 -> 128 only.
PASS CRITERIA:
  [B-build] build exits 0.
  [B-same ] output still sane (no correctness regression vs Phase A).
  [B-fast ] `llama-bench` tg/s >= Phase A for the same model (native path is faster).

### B1. vecdotq.cuh - add native vec-dot kernel for pq2_0
FILE: `ggml/src/ggml-cuda/vecdotq.cuh`
REF : `vec_dot_q2_0_q8_1` (the HIP-adapted Q2_0 kernel, near line 821-904) and the
      config macros at lines 112-113 (`VDR_Q2_0_Q8_1_MMVQ`, `VDR_Q2_0_Q8_1_MMQ`).
DO  : add `vec_dot_pq2_0_q8_1` mirroring `vec_dot_q2_0_q8_1`; change block type to
      block_pq2_0 and any QK2_0 (64) constant to QK_PQ2_0 (128). Add matching VDR_PQ2_0_* macro.
CHECK: read the full q2_0 kernel body first; the dequant math is `(q-1)*d` identical.

### B2. mmvq.cu - register pq2_0 in the 3 dispatch maps
FILE: `ggml/src/ggml-cuda/mmvq.cu`
REF : line 15   `case GGML_TYPE_Q2_0: return vec_dot_q2_0_q8_1;`
      line 44   `case GGML_TYPE_Q2_0: return VDR_Q2_0_Q8_1_MMVQ;`
      line 1742 `case GGML_TYPE_Q2_0:` -> `mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>`
DO  : add a `case GGML_TYPE_PQ2_0:` after each Q2_0 case, pointing at the pq2_0 equivalents
      (vec_dot_pq2_0_q8_1 / VDR_PQ2_0_* / <GGML_TYPE_PQ2_0>).
CHECK: all 3 sites must be added or the native path silently falls back to cuBLAS.

---------------------------------------------------------------------
REBUILD + BENCH (Phase B)
---------------------------------------------------------------------
```
docker exec rocm10-dev bash /src/src/beellama-boosts/build-rocm10-0001.sh   # warm, faster
# benchmark on GPU1 (tg = token generation, pp = prompt processing)
HIP_VISIBLE_DEVICES=1 /src/src/build-rocm10/bin/llama-bench \
  -m /models/bonsai-2/Ternary-Bonsai-2-27B-PQ2_0.gguf -ngl 99 -p 512 -n 128
```

=====================================================================
## PHASE C - Full validation on GPU1
=====================================================================
GOAL: confirm the exact target config the user wants, end to end.
- mmproj-BF16 vision path: run a multimodal prompt (image) if a test image is available; else
  confirm `--mmproj` loads and text-only still works.
- Longer generation (-n 256+) quality check vs Bonsai-1 Q2_0 baseline on the same prompt.
- Record a small benchmark table (Q2_0 vs PQ2_0, pp/tg) for the user's decision.

=====================================================================
## Per-task reference map (what to read -> what to change -> direction)
=====================================================================
| Task | Read first (source of truth)                | Change into            | Direction |
|------|--------------------------------------------|------------------------|-----------|
| A1   | dequantize.cuh:26 dequantize_q2_0          | dequantize_pq2_0       | copy, block_q2_0->block_pq2_0 |
| A2   | convert.cu:462 (and 5 more Q2_0 sites)     | +case PQ2_0 each map   | template<QK_PQ2_0,QR_PQ2_0,dequantize_pq2_0> |
| A3   | ggml-cuda.cu:5930 (MUL_MAT switch)         | +case PQ2_0            | allow offload |
| B1   | vecdotq.cuh:821-904 vec_dot_q2_0_q8_1      | vec_dot_pq2_0_q8_1     | copy, QK 64->128 |
| B2   | mmvq.cu:15,44,1742 (Q2_0 sites)            | +case PQ2_0 each       | point at pq2_0 kernel/cfg |

## Gotchas / risks
- DO NOT mirror q2_0 blindly if its dequant formula differs - it does NOT here (verified identical),
  but always confirm the `(q-1)*d` line in the GPU kernel matches CPU ref.
- Cold build is slow; only pay it once (Phase A). Phase B rebuild is warm/cheap.
- If `convert_func` is null at runtime -> you missed a convert.cu dispatch site (A2).
- If model loads but output is garbage -> dequantize math or block stride wrong (A1), NOT Hadamard
  (Hadamard already works).
- GPU0 must stay free: always `HIP_VISIBLE_DEVICES=1`.
