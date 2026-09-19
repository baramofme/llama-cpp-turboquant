# Branch Topology

This fork is organized by **build line** (ccache docker build). Each branch = source + build script + bench data, and if you check it out and build it, you get the final working state of that line.

Remote: `baramofme` (github.com:baramofme/llama-cpp-turboquant)

## Build Lines

| Branch | Line | Source Tree | Build Entry |
|---|---|---|---|
| `main` | **rocm10 base** | This repo's tree (upstream sync + q1_0 HIP SWAR optimization) | `beellama-boosts/bench-rocm10.sh`, `beellama-boosts/stock-rocm10-runtime.Dockerfile`, `TEST-PLAN/RESULTS-2026-09-*.md` |
| `bonsai-1` | Bonsai 27B Q1_0 serving | main + Q1/Q2 runtime image + gate proxy v2 + Q1 agentic bench (09-13/14 data) | `beellama-boosts/build-rocm10-runtime.sh`, `rocm10-runtime.Dockerfile` |
| `bonsai-2` | Bonsai 2 PQ2_0 | main + PQ2_0 quant (ggml type 142) + evaluation reports (KR/EN) + adversarial 12 battery | `PQ2_0_IMPL_PLAN.md`; serving image `prism-b10709-pq20-gfx1100` (local registry) |
| `beellama` | beellama.cpp boosts | `beellama-boosts/src` (beellama.cpp + boosts 8+0001, 3765 files) + v0.4.5 clean-5 + bench data | `beellama-boosts/build-rocm10-0001.sh` (runs inside `rocm/dev-ubuntu-26.04:10.0.0-full`), `build-native-0001.sh` |
| `qwen3.8-flash-next` | Qwen3.8-Flash-Next (qwen4exp) expert cache | thecodacus fork final-state snapshot (orphan) + qwen38 bench data + `SESSION-2026-09-09-QWEN4EXP-PERF.md` | — |
| `feature/turboquant-kv-cache` | turboquant KV cache | TheTom fork line (D8.x docs) + tq/kv bench data + bench scripts | `beellama-boosts/bench-tbq-docker.sh`, `bench-kv-vs-tq.sh`, `bench-kvmem.sh` |

## Reference / Working Branches

- `rocm10` — local working branch (the pre-split hybrid of all lines; 5 unpushed commits as of 2026-09-20)
- `rocm10-pure` — pure upstream snapshot (reference)
- `backup/main-tom` — old `main` (TheTom fork line) preserved
- `feature/rdna3-mmq-lds-accel` — RDNA3 MMQ/LDS acceleration
- `codacus-master(-2)`, `pr-*`, `merge-pr-*` — old scratch

## Build Method

Docker base image + ccache.

- **rocm10 base / bonsai-1**: rocm10 image builds (gfx1100, `GGML_HIP=ON`, `GGML_HIP_GRAPHS=ON`, `GGML_HIP_MMQ_MFMA=ON`, `GGML_HIP_NO_VMM=ON`); see `DOCKER-RELEASE-2026-09-10.md` and the runtime Dockerfiles.
- **beellama**: `build-rocm10-0001.sh` runs inside the ROCm 10 image with the workspace mounted; v0.4.5 clean-5 via `beellama-boosts/Dockerfile` + `bench-boosts.sh`.
- **bonsai-2**: PQ2_0 code is in-tree (ggml type 142); the serving image is PrismML's `prism-b10709-pq20-gfx1100` from the local registry (the PrismML fork source itself is not part of this repo).

## History Notes

- `main`, `bonsai-1`, `bonsai-2`, `beellama` share the rocm10 line history; the split is **tree-level** (each branch = one commit adding/removing its line's content), not a rebase.
- `qwen3.8-flash-next` is an **orphan snapshot**: the thecodacus fork history contained committed build artifacts over GitHub's 100 MB limit, so it was re-rooted as a single final-state commit (build artifact trees stripped).
- The beellama boosts (28 files, ~1100 lines) were previously uncommitted in the nested `beellama-boosts/src` working tree; they are now committed on the `beellama` branch.
- `feature/turboquant-kv-cache` tip was rebuilt on top of the remote tip with build artifacts stripped (the old local history had diverged and carried >200 MB binaries).
