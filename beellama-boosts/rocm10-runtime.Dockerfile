# rocm10 runtime for Bonsai Q1_0 / ternary Q2_g64 serving.
# Build context is beellama-boosts/ with bin/ copied in first (see build-rocm10-runtime.sh):
FROM rocm/dev-ubuntu-24.04:10.0.0-full

ENV ROCM_PATH=/opt/rocm
ENV PATH=/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/app/bin:/app:/opt/rocm/core-10.0/lib:/opt/rocm/lib
ENV OMP_NUM_THREADS=6
ENV MTMD_VISION_BACKEND=hip
ENV GPU_MAX_HW_QUEUES=1
ENV ROC_ENABLE_PREEMPTION=1

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY bin/ /app/bin/
COPY grammars/ /app/grammars/
WORKDIR /app
EXPOSE 8080
ENTRYPOINT ["/app/bin/llama-server", "--grammar-file", "/app/grammars/english-only.gbnf"]