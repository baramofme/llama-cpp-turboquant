# PTQ1 vs PQ2 MMQ profile (ptq1-mmq-profile-0001)

Date: 2026-09-25. GPU0 (RX 7900 XTX, gfx1100), rocm10-build container.
Tool: rocprofv3 1.1.0 in-container. Graphs disabled at runtime via
GGML_CUDA_DISABLE_GRAPHS=1 (no rebuild needed).
Workload: `llama-bench -m <model> -ngl 99 -fa 1 -pg 512,8`, both models.
DBs: build-rocm10/prof/ptq1_results.db, pq2_results.db (timing),
pmc_ptq1/pmc_pq2/pmc_test_results.db (counters, partial).

## Kernel timing (identical call counts, identical shapes)

| kernel | PTQ1 (143) avg | PQ2 (142) avg | ratio |
|---|---|---|---|
| mul_mat_q<*, 128> (MMQ prefill), 4800 calls | 1326.2 us | 962.4 us | 1.378x |
| mul_mat_vec_q_ksplit (decode), 174592 calls | 92.7 us | 61.7 us | 1.502x |

MMQ share of total: 16.8% (PTQ1) vs 15.6% (PQ2).
No dequantize + rocBLAS fallback kernels in top list: both run native MMQ.

Conclusion: the gap is ISOLATED INSIDE the mul_mat_q kernel
(37.8% per-call slower for PTQ1). Launch geometry, J, shapes, grid all
identical; the MFMA core is shared. Only the tile loader differs.
(Wall-clock bench gap ~20-25% is smaller because non-GEMM work dilutes it.)

## Counters (tooling wall)

Collectible: SQ_WAVES, SQ_BUSY_CYCLES (restate duration only).
SQ_INSTS_VALU, SQ_INSTS_LDS, SQC_LDS_BANK_CONFLICT read exactly 0
(230400 samples), single-counter retry also 0. Legacy rocprof abandoned:
host ROCm is 7.2 vs container ROCm 10 (ABI risk to production GPU).
ISA extraction via clang-offload-bundler blocked on target-triple syntax.

## Consequence for the loader rewrite

Counter split (divergence vs LDS) unavailable. The rewrite addresses
both prime suspects in one pass (uniform control flow for all 8 lanes
+ consecutive LDS stores), since both live in the same loader function
(mmq-load-tiles.cuh:278-357). Geometry (nrows halved by 8 threads/block)
is secondary. LUT patch (+3%) stays underneath.
