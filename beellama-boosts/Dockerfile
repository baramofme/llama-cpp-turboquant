# beellama v0.4.5 + rdna-boosts clean blocks (0002/0005/0006/0007/0012)
# build stage: rocm dev container; final: preview-v0.4.5 runtime with replaced binaries/libs
FROM docker.io/rocm/dev-ubuntu-24.04:7.2.1-complete AS build

ENV AMDGPU_TARGETS=gfx1100
RUN apt-get update \
    && apt-get install -y build-essential cmake git libssl-dev curl libgomp1
WORKDIR /src
COPY . .
RUN HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
        -DGGML_BACKEND_DL=ON \
        -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build build --config Release -j$(nproc)

FROM ghcr.io/anbeeld/beellama.cpp:server-rocm-preview-v0.4.5

# replace binaries + backend libs with the patched build
COPY --from=build /src/build/bin/llama-server /app/llama-server
COPY --from=build /src/build/bin/llama-bench /app/llama-bench
COPY --from=build /src/build/bin/llama-batched-bench /app/llama-batched-bench
COPY --from=build /src/build/bin/llama-cli /app/llama-cli
COPY --from=build /src/build/bin/llama-completion /app/llama-completion
COPY --from=build /src/build/lib/*.so* /app/
