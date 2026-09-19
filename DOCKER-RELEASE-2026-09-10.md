# Docker 이미지 릴리스 기록 (2026-09-10)

> Qwen3.8-Flash-Next 속도 개선 이식본(expert cache + async CPU + MTP)을 포함한 ROCm 이미지.
> 로컬 레지스트리 + Docker Hub 양쪽에 푸시 완료.

---

## 1. 이미지 정보

| 항목 | 값 |
|---|---|
| 이미지명 | `baramofme/llama-cpp-rocm` |
| 태그 | `rocm10-gfx1100-rdnaboosts-moecache-mtp-20260910` |
| 이미지 ID | `ebea7221f3c4` |
| 크기 | 21.7GB |
| 베이스 이미지 | `baramofme/llama-cpp-rocm:rocm10-gfx1100-rccl-rdnaboosts-mtp-latest` |
| 빌드 커밋 | `5631d073b` (build 10786) |
| 빌드 방식 | 경량 (기존 ROCm 런타임 이미지에 새 바이너리 `COPY bin/ /app/`) |

### Digest

| 레지스트리 | Digest |
|---|---|
| 로컬 (`localhost:5000`) | `sha256:7a07332ecb574647000af5ccb7144d7326f2c56674bf8a34db1d9bfe793d5e35` |
| Docker Hub | `sha256:6e101b349da8c5c197e2308740e79b7bb68ac076175a6c2985eb2037423a25d8` |

---

## 2. 포함된 기능 (이번 세션 이식분)

| 기능 | 상태 | 성능 |
|---|---|---|
| MoE expert cache (`--moe-cache-profile/slots`) | ✅ | decode +26%, prefill +20% (llama-bench) |
| async CPU/GPU overlap (`--sched-async-cpu`) | ✅ | 기본 on |
| MTP draft head (#28243) | ✅ | draft accept 61~83% |
| mul_mat_id skip-id fix | ✅ | prefill 크래시 해결 |
| host-pin (`GGML_CUDA_REGISTER_HOST`) | ⚠️ HIP 비활성 | SVM 대용량 등록 hang |
| prefetch-experts (`GGML_SCHED_PREFETCH_EXPERTS`) | ⚠️ HIP 자동 off | 두 번째 스트림 race 크래시 |
| `llama-moe-trace` 도구 | ✅ | 라우팅 프로필 캡처 |

---

## 3. 사용법

### Pull
```bash
# 로컬 레지스트리
docker pull localhost:5000/baramofme/llama-cpp-rocm:rocm10-gfx1100-rdnaboosts-moecache-mtp-20260910

# Docker Hub
docker pull baramofme/llama-cpp-rocm:rocm10-gfx1100-rdnaboosts-moecache-mtp-20260910
```

### 실행 (M64 모델, 권장)
```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -v /mnt/nvmedata/models:/models \
  -v /home/baramofme/IdeaProjects/llama-cpp-turboquant:/work \
  -p 12000:8080 \
  baramofme/llama-cpp-rocm:rocm10-gfx1100-rdnaboosts-moecache-mtp-20260910 \
  -m /models/Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64/Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64/Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64-00001-of-00028.gguf \
  -md /models/Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64/MTP/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 1 \
  -ngl 99 -sm layer -fa on -c 20480 -np 1 -b 2048 -ub 512 \
  -t 12 --threads-batch 12 -ctk q8_0 -ctv q8_0 --jinja -fit off \
  --lazy-mode on \
  --moe-cache-profile /work/qwen38-merged.csv --moe-cache-slots 24 \
  -ot 'blk\.(4[1-7])\.ffn_(gate|up|down)_exps\.weight=CPU'
```

### 이미지 기본 설정
- `ENTRYPOINT`: `/app/llama-server`
- `ENV`: `LD_LIBRARY_PATH=/app:/opt/rocm/lib:/opt/rocm/lib64`, `HIP_VISIBLE_DEVICES=0,1`, `OMP_NUM_THREADS=6`, `MTMD_VISION_BACKEND=hip`, `ROC_ENABLE_PREEMPTION=1`

---

## 4. 실측 성능 (이미지 바이너리 기준)

| 모델 | 구성 | decode | prefill |
|---|---|---|---|
| M64 (45.8GB) | expert cache 24슬롯 + MTP | **15.3~17.6 t/s** | 31 t/s |
| UD-IQ3_XXS (82GB) | ncmoe 99 + expert cache | 4.4~4.9 t/s | 106~123 t/s |

상세: `TEST-RESULTS-2026-09-10.md` 참조.

---

## 5. 재빌드 방법

```bash
# 1. 바이너리 빌드
cd /home/baramofme/IdeaProjects/llama-cpp-turboquant
cmake --build build -j$(nproc) llama-server llama-bench llama-moe-trace

# 2. staging
mkdir -p /tmp/opencode/docker-build/bin
cp -a build/bin/. /tmp/opencode/docker-build/bin/

# 3. Dockerfile (기존 이미지 베이스)
cat > /tmp/opencode/docker-build/Dockerfile <<'EOF'
FROM baramofme/llama-cpp-rocm:rocm10-gfx1100-rccl-rdnaboosts-mtp-latest
ENV LD_LIBRARY_PATH=/app:/opt/rocm/lib:/opt/rocm/lib64
ENV HIP_VISIBLE_DEVICES=0,1
ENV OMP_NUM_THREADS=6
ENV MTMD_VISION_BACKEND=hip
ENV ROC_ENABLE_PREEMPTION=1
WORKDIR /app
COPY bin/ /app/
ENTRYPOINT ["/app/llama-server"]
EOF

# 4. 빌드 + 푸시
cd /tmp/opencode/docker-build
TAG=rocm10-gfx1100-rdnaboosts-moecache-mtp-20260910
docker build --network=host -t baramofme/llama-cpp-rocm:$TAG -t localhost:5000/baramofme/llama-cpp-rocm:$TAG .
docker push localhost:5000/baramofme/llama-cpp-rocm:$TAG
docker push baramofme/llama-cpp-rocm:$TAG
```

---

## 6. 주의사항

- **호스트 커널/드라이버 요구**: ROCm 10.0.0, gfx1100 (RDNA3). amdgpu DKMS 버전에 따라 H2D 복사 성능 저하 이슈 존재 (ROCm#6523).
- **host-pin 금지**: `GGML_CUDA_REGISTER_HOST`는 HIP에서 hang. 이미지에서 설정하지 말 것.
- **prefetch-experts**: ROCm 백엔드 감지로 자동 비활성화됨. env로 켜도 안전(무시).
- **expert cache 슬롯**: VRAM 여유에 맞춰 24~32 권장. 초과 시 요청 처리 OOM.
