#!/usr/bin/env bash
# Native ROCm 7.2.3 build of beellama-boosts/src (8 boosts blocks + 0001 adaptive MTP)
# Mirrors /tmp/beellama-src/build-boosts CMakeCache flags exactly.
set -euo pipefail

SRC=/home/baramofme/IdeaProjects/llama-cpp-turboquant/beellama-boosts/src
BLD=/tmp/be-src-native0001
cd "$SRC"

cmake -S "$SRC" -B "$BLD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.3/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_AR=/opt/rocm-7.2.3/lib/llvm/bin/llvm-ar \
  -DCMAKE_HIP_COMPILER_RANLIB=/opt/rocm-7.2.3/lib/llvm/bin/llvm-ranlib \
  -DCMAKE_HIP_PLATFORM=amd \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_MMQ_MFMA=ON \
  -DGGML_HIP_NO_VMM=ON \
  -DGGML_HIP_RCCL=OFF \
  -DGGML_CUDA=OFF \
  -DCMAKE_C_COMPILER=/opt/rocm-7.2.3/lib/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-7.2.3/lib/llvm/bin/clang++ \
  -DLLAMA_CURL=OFF \
  --graphviz=/tmp/be-src-native0001-graph.dot

cmake --build "$BLD" --config Release -j 16 --target llama-bench llama-server llama-batched-bench 2>&1 | tail -30

echo "=== BUILD DONE ==="
ls -la "$BLD"/bin/llama-bench "$BLD"/bin/llama-server "$BLD"/bin/llama-batched-bench 2>/dev/null
# verify adaptive flag compiled in
strings "$BLD"/bin/llama-server 2>/dev/null | grep -c "draft-mtp-adaptive" | sed 's/^/adaptive_flag_count=/' || true
