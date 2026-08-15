# TBQ+ Local Build Success - 2026-07-10

## Image
`baramofme/llama-cpp-rocm:gfx1100-rocm7.2-tbqplus-pchunk-rocwmma`

## Build Command
```bash
cd /mnt/ssd-system/opt/llamacpp/llama-cpp
make build-tbq-plus-local-push
```

## Source
- Repo: `/home/baramofme/IdeaProjects/llama-cpp-turboquant`
- Branch: `feature/turboquant-kv-cache`
- Commit: `85de7d5b9` (pre-upstream-merge)
- `git describe`: `gguf-v0.19.0-1488-g85de7d5b9`

## Key Config (Dockerfile.turboquant_plus_local)
- `-DCMAKE_HIP_ARCHITECTURES="gfx1100"` (gfx1100 only)
- `-DLLAMA_BUILD_TESTS=OFF`
- Template patches: removes ncols1_1/2/4/8 (nbatch_fa=0 div by zero)

## Why detached HEAD?
The upstream merge `b1740926c` (2026-07-10) broke:
- `server-context.cpp` (incomplete type `llama_context`)
- `ggml-cuda.cu` (removed `split` buffer types)
- `tests/test-backend-ops.cpp` (type reference)

To build from HEAD, resolve these upstream merge conflicts first.

## Revert After Build
```bash
cd /home/baramofme/IdeaProjects/llama-cpp-turboquant
git checkout feature/tugboquant-kv-cache
```
