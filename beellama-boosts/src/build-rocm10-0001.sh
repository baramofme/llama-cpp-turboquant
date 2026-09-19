#!/usr/bin/env bash
# ROCm 10 rebuild of beellama-boosts/src WITH 8 boosts + 0001 (was clean-5 only)
# Runs INSIDE rocm/dev-ubuntu-26.04:10.0.0-full, workspace mounted at /src/src
set -euo pipefail

export LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib:${LD_LIBRARY_PATH:-}
export HIP_PATH=/opt/rocm/core-10.0
export HIPCXX=/opt/rocm/core-10.0/lib/llvm/bin/clang++
export HIP_DEVICE_LIB_PATH=/opt/rocm/lib/bitcode
export PATH=/opt/rocm/core-10.0/bin:/opt/rocm/lib/llvm/bin:/usr/local/bin:/usr/bin:/bin

# cmake is not shipped; install via pip into a venv
if ! command -v cmake >/dev/null 2>&1; then
  pip3 install --break-system-packages -q cmake 2>/dev/null || pip3 install -q cmake
fi
command -v cmake >/dev/null 2>&1 || { echo "cmake unavailable"; exit 1; }

SRC=/src/src
BLD=/src/src/build-rocm10
rm -rf "$BLD"
cmake -S "$SRC" -B "$BLD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1100 \
  -DCMAKE_HIP_COMPILER="$HIPCXX" \
  -DCMAKE_HIP_PLATFORM=amd \
  -DCMAKE_HIP_FLAGS="--rocm-path=/opt/rocm/core-10.0 --rocm-device-lib-path=/opt/rocm/core-10.0/lib/llvm/amdgcn/bitcode" \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_MMQ_MFMA=ON \
  -DGGML_HIP_NO_VMM=ON \
  -DGGML_HIP_RCCL=OFF \
  -DGGML_CUDA=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF

cmake --build "$BLD" --config Release -j 16 --target llama-bench llama-server llama-batched-bench 2>&1 | tail -25

echo "=== ROCM10 BUILD DONE ==="
ls -la "$BLD"/bin/llama-bench "$BLD"/bin/llama-server "$BLD"/bin/llama-batched-bench
strings "$BLD"/bin/llama-server 2>/dev/null | grep -c "draft-mtp-adaptive" | sed 's/^/adaptive_flag_count=/' || true