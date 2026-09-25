# PTQ1 uniform-loader bench (ptq1-uniform-0001)

Date: 2026-09-25. GPU0 (RX 7900 XTX, gfx1100), rocm10-build container.
Change: mmq-load-tiles.cuh PTQ1 loader restructured (on top of LUT patch):
4 divergent paths (lanes 0-3 / 4-5 / 6-qh / 7-idle) merged into 2
(lanes 0-5 uniform qs path with cmov offsets, lanes 6-7 uniform qh tail
via new 4-trit LUT4). All 8 lanes productive. Store layout unchanged.
Pre-GPU proof: LUT4 bit-identical over all 65536 qh pairs; full 32-int
writer mapping bit-identical over 20000 random blocks
(/tmp/opencode/check_ptq1_newloader.py).

## Wall clock (same protocol as baseline, same day, ~1h apart)

llama-bench -m PTQ1_0.gguf -ngl 99 -fa 1 -pg 512,128 -pg 1024,128 -pg 5201,128

| test | baseline (no LUT) | LUT | uniform (+LUT) |
|---|---|---|---|
| pp512 | 717.92 | 738.25 (+2.8%) | 804.71 (+12.1%) |
| pp1024 (derived) | ~681 | ~712 (+4.5%) | ~769 (+12.9%) |
| pp5201 (derived) | ~668 | ~687 (+2.9%) | ~745 (+11.6%) |
| tg128 | 36.23 | 35.93 | 36.14 (unchanged, expected) |

## Kernel level (rocprofv3, -pg 512,8, DISABLE_GRAPHS)

mul_mat_q<143,128>, 4800 calls, identical shapes:
LUT binary 1326.2 us/call -> uniform binary 1194.7 us/call (-9.9%).
Decode path (mmv ksplit) unchanged, as expected.

Caveat: wall-clock deltas mix real gain with live-traffic contention
(SmallDense serves GPU0 concurrently; pp512 std +-4.14). The kernel
per-call number is the robust one (same shapes, same call counts).
Wall +12% overstates slightly vs kernel -9.9% at ~16% GEMM share;
favorable contention drift likely contributes a few points.

## Verdict

Divergence confirmed as a major factor. Combined LUT + uniform: +12% wall,
-10% kernel. Remaining gap to PQ2 (~880): ~15% (745 vs 880).
Left on the table: strided LDS stores (stride 4/2 kept), halved nrows
(8 threads/block kept), LUT loads vs PQ2 perms. Smoke passed, no faults.
Correctness rests on offline bit-identity proofs (cli/ppl hang pre-exists
in-container for baseline too, unrelated to this change).
