# PTQ1 LUT decode bench (ptq1-lut-0001)

Date: 2026-09-25. GPU0 (RX 7900 XTX, gfx1100), rocm10-build container.
Binary: current WIP tree + LUT patch (mmq-load-tiles.cuh), incremental rebuild 12:19 KST.
Baseline: same tree without LUT, benched same day.
Protocol (both): `llama-bench -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 99 -fa 1
  -pg 512,128 -pg 1024,128 -pg 5201,128`, single invocation.
Pure pp1024/pp5201 derived: pp = N / ((N+128)/combined - 128/tg).

| test | baseline | LUT | delta |
|---|---|---|---|
| pp512 | 717.92 | 738.25 | +2.8% |
| pp1024 (derived) | ~681 | ~712 | +4.5% |
| pp5201 (derived) | ~668 | ~687 | +2.9% |
| tg128 | 36.23 | 35.93 | -0.8% (noise) |

Change: `ggml_cuda_mmq_decode_ptq1_0_qs4` base-3 recurrence
(10x IMUL32 + perms per 4 bytes, serial 5-iteration chain) replaced with
256-entry `__constant__` LUT (byte -> 5 biased trits), 4 loads + shift/OR.
qh lane-6 path untouched. No other file changed.

Correctness:
- LUT proven bit-identical to recurrence: all 256 byte values in every
  lane + 300k random packed words, 0 mismatches (gen script: /tmp/opencode/gen_ptq1_lut.py).
- Smoke: LUT binary loads model, completes prefill+generation, no GPU fault.
- tg128 unchanged, as expected (decode path not touched by this patch).
- Note (pre-existing, unrelated): llama-cli / llama-perplexity hang in this
  container for baseline binary too. Bench path unaffected.

Verdict: decode ALU contributes only ~3%. The "base-3 decode ALU is the
cause" hypothesis is REJECTED as the primary cause (pp5201 687 < 730,
far from PQ2 level ~880). LUT is kept (strictly better, bit-identical).

Revised diagnosis (code-evidenced, mmq-load-tiles.cuh:278-357):
- PTQ1 loader has 4 divergent paths (lanes 0-3 / 4-5 / lane 6 qh / lane 7 idle).
  Diverged wavefront executes paths serially. PQ2 loader is fully uniform.
- Strided LDS scatter stores (stride 4/2) vs PQ2 consecutive stores.
- 8 threads/block doubles threads_per_row vs PQ2 (QI=4 both), halving nrows.

Next: restructure PTQ1 loader to uniform control flow (all lanes same ops),
or rocprof to quantify VALU/LDS/occupancy first.
