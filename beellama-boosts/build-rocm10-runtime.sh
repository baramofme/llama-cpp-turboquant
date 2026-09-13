#!/usr/bin/env bash
# Build rocm10 runtime image for Bonsai Q1_0 / ternary Q2_g64 serving.
# Copies freshly built HIP binaries from build-rocm10/bin into context,
# then builds both q1/q2 tags and pushes to the local registry.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CTX="$ROOT/beellama-boosts"
REG="localhost:5000/baramofme/llama-cpp-rocm"
TAG="rocm10-gfx1100-rccl-rdnaboosts-mtp"

[ -d "$ROOT/build-rocm10/bin" ] || { echo "build-rocm10/bin not found; run build-rocm10-0001.sh first" >&2; exit 1; }

rm -rf "$CTX/bin"
cp -a "$ROOT/build-rocm10/bin" "$CTX/bin"

docker build -f "$CTX/rocm10-runtime.Dockerfile" \
    -t "$REG:$TAG-q1" \
    -t "$REG:$TAG-q2" \
    "$CTX"

docker push "$REG:$TAG-q1"
docker push "$REG:$TAG-q2"

rm -rf "$CTX/bin"
echo "=== PUSHED $REG:$TAG-q{1,2} ==="