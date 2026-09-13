# Bonsai-27B ROCm10 Serving Setup (2026-09-13)

Q1_0 / Ternary-Q2_g64 llama-server runtime images on ROCm 10 (gfx1100),
English-only output enforcement, and Dokploy deployment.

## Final architecture

```
Client (Open WebUI etc.)
  |
  v  Korean input accepted, English-only output enforced by grammar
8082: bonsai-gpu1 (Q2 image, english-only.gbnf builtin)
```

- No gate proxy, no translation layer, no Dense dependency.
- Q2 understands Korean input directly (Qwen-based).
- `english-only.gbnf` physically blocks non-ASCII output
  (Korean/Chinese/Japanese/Cyrillic/Arabic/Hindi/emoji).

## Images (local registry `localhost:5000`)

| Tag | Purpose |
|---|---|
| `rocm10-builder-ccache` | Build base: rocm10 full + cmake 3.28.3 + ccache 4.9.1 + git/g++/ninja |
| `rocm10-gfx1100-rccl-rdnaboosts-mtp-q1` | Q1_0 serving runtime (same binary as q2, tag only) |
| `rocm10-gfx1100-rccl-rdnaboosts-mtp-q2` | Ternary Q2_g64 serving runtime |

## Files

| File | Role |
|---|---|
| `rocm10-runtime.Dockerfile` | Runtime image: rocm10 full + `bin/` + `grammars/`, ENTRYPOINT with `--grammar-file` |
| `build-rocm10-runtime.sh` | Copies `build-rocm10/bin` into context, builds q1/q2 tags, pushes |
| `grammars/english-only.gbnf` | ASCII-printable allowlist (upstream `english.gbnf` pattern) |
| `gate_proxy.py` | Superseded experiment: HTTP gate (translate→reason→backtranslate). Kept for reference, NOT deployed. See "Dead ends" below. |

## Build procedure

```bash
# 1. HIP build inside rocm10-dev (ccache incremental)
docker exec rocm10-dev bash /app/beellama-boosts/build-rocm10-0001.sh
# output: build-rocm10/bin/

# 2. Commit build base once (skip external pull afterwards)
docker commit rocm10-dev localhost:5000/baramofme/llama-cpp-rocm:rocm10-builder-ccache
docker push localhost:5000/baramofme/llama-cpp-rocm:rocm10-builder-ccache

# 3. Runtime images q1/q2
bash beellama-boosts/build-rocm10-runtime.sh
```

## Dokploy deployment

Service `bonsai` (composeId `h5QEsfsllhIBuagdspK0t`), port `8082`:

- Image: `...:rocm10-gfx1100-rccl-rdnaboosts-mtp-q2` (active)
- Q1 image line kept commented directly below for one-line switch.
- Model volume: `/mnt/nvmedata/models:/models:ro`
  - Q2: `/models/ternary-bonsai-27b/Ternary-Bonsai-27B-Q2_g64.gguf`
  - Q1: `/models/bonsai-27b/Bonsai-27B-Q1_0.gguf`
- GPU: `/dev/kfd` + `/dev/dri`, `HIP_VISIBLE_DEVICES=1`,
  `OMP_NUM_THREADS=6`, `MTMD_VISION_BACKEND=hip`,
  `ROC_ENABLE_PREEMPTION=1`, `GPU_MAX_HW_QUEUES=1`
- Server flags: `--n-gpu-layers 99 --ctx-size 32768 --reasoning off -np 1 --no-sched-async-cpu`

Gotchas found during deploy:
- CLI `compose create` defaults `sourceType=github` -> deploy fails
  with "Github Provider not found". Fixed with
  `UPDATE compose SET "sourceType"='raw' WHERE "composeId"=...`.
- CLI `compose import` returns 400; DB direct UPDATE of `composeFile` works.

## Verification (8082, no gate)

| Test | Result |
|---|---|
| Korean greeting | English reply, CJK=0 |
| `23 x 7` in Korean | `23 multiplied by 7 is 161.` |
| Multi-turn name recall | `Minsoo` remembered |
| "answer in Korean" jailbreak | English only, CJK=0 |

## Dead ends (documented so nobody retries them)

1. **Root-context build**: repo root `.dockerignore` has `build*/`,
   so `COPY build-rocm10/bin/` fails with `/build-rocm10/bin: not found`.
   Fix: use `beellama-boosts/` as build context, copy `bin/` there first.
2. **Tag bugs in build script**: `TAG=q...` + `"q1"` produced `-qq1`;
   missing hyphen produced `-mtpq1`. Current script uses `"$TAG-q1"`.
3. **Gate proxy + Dense backtranslate** (`gate_proxy.py`, port 1709):
   Korean→English→Dense→Korean pipeline worked, but Q2 already
   understands Korean and grammar alone enforces English output.
   Extra translation hops add latency and a Dense dependency for no gain.
   Removed from compose.
4. **Request-level `grammar` param**: works for testing without restart,
   but server-level `--grammar-file` in the image is the permanent fix.
