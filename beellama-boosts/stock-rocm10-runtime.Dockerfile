# Stock upstream llama.cpp ROCm10 runtime (no fork patches, no builtin grammar).
# Comparison baseline for A/B against rdna-boosts builds.
# Build context is beellama-boosts/ with stock-bin/ copied in first:
#   cp -a /tmp/opencode/llama-stock/llama-b10964 stock-bin
#   docker build -f beellama-boosts/stock-rocm10-runtime.Dockerfile \
#     -t localhost:5000/baramofme/llama-cpp-rocm:stock-b10964-rocm10-gfx1100 beellama-boosts
FROM rocm/dev-ubuntu-24.04:10.0.0-full

ENV ROCM_PATH=/opt/rocm
ENV PATH=/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/app/bin:/app:/opt/rocm/core-10.0/lib:/opt/rocm/lib
ENV OMP_NUM_THREADS=6
ENV MTMD_VISION_BACKEND=hip
ENV GPU_MAX_HW_QUEUES=1
ENV ROC_ENABLE_PREEMPTION=1

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY stock-bin/ /app/bin/
WORKDIR /app
EXPOSE 8080
ENTRYPOINT ["/app/bin/llama-server"]