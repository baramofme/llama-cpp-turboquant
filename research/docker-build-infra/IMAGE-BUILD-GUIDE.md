# 최종 선택 이미지 재생성 가이드: gfx1100-rocm10-tbq-rboosts

> 작성: 2026-09-06 (Asia/Seoul)
> 목적: 선택된 운영 이미지의 구성/제작 과정/적용-비적용 사항을 기록.
> 다음 재생성 시 이 문서만 따라가면 토큰 비용 최소화로 재현 가능.

---

## 1. 최종 이미지 요약

| 항목 | 값 |
|---|---|
| 이미지 | `baramofme/llama-cpp-rocm:gfx1100-rocm10-tbq-rboosts` (21.3GB) |
| 기반 | theTom llama-cpp-turboquant 4a54c52 (`feature/turboquant-kv-cache`) |
| 추가 | rdna-boosts(clean 5 + 수동 3) |
| 런타임 | ROCm 10.0 (`rocm/dev-ubuntu-26.04:10.0.0-full`) |
| ENTRYPOINT | `["/app/llama-server"]` (compose 의 command 가 인자로 붙도록) |
| KV 타겟 | q8_0(K) / turbo3(V), --cache-ram 오프로드 |

## 2. 만드는 과정 (전체 순서)

```
1. theTom 최신 클론
2. rdna-boosts 13개 패치 전수 적용 테스트 -> 클린/충돌 분류
3. 클린 5개 git apply
4. 충돌 3개(000d4/0010/0011) 수동 병합 (파일별)
5. native ROCm 7.2.3 빌드로 컴파일 검증 (빠른 피드백)
6. ROCm 10 docker 빌드 (build-rocm10)
7. 스모크 테스트 (2B turbo3, GPU1)
8. 운영 구성 검증 (27B 140K + adaptive MTP)
9. Docker 이미지 빌드 (build-rocm10/bin -> /app, ENTRYPOINT)
10. 최종 벤치 + needle 정확도 검증
```

## 3. 적용한 것 (8 블록)

### 3.1 클린 적용 5개 (git apply 그대로)
| 패치 | 내용 | 효과 |
|---|---|---|
| 0005 | CPU bit-identical decode/verify | MTP 수용률 정확성 |
| 0006 | host buffer revert (discrete GPU) | prefill 버퍼 안정 |
| 0007 | meta device wrapper skip | offload 안정 |
| 0009 | meta buffer headroom (32) | VRAM 안전 마진 |
| 0012 | hybrid HIP all-reduce (RDNA4 gate) | gfx1100 무해(폴백), 통합 편의 |

### 3.2 수동 병합 3개 (파일별 git apply + 수정)
| 패치 | 핵심 변경 | 효율/필요성 |
|---|---|---|
| **0004** | WMMA FA: `fattn-mma-f16.cuh` 게이트 `DKQ>128 -> DKQ>256` (gfx1100 head_dim256 WMMA 활성) + fattn.cu/mmq-vec-dot.cuh/mmq.cuh | head_dim 256 모델이 WMMA 경로 활성 |
| **0010** | k-quant VDR: `mmvq.cu` Q4_K/Q5_K->vdr4, Q6_K->vdr2 등록 + `vecdotq.cuh` VDR 매크로 + `mmq-vec-dot.cuh` dmA 레지스터 캐시 | tg +7~12% (디코딩 최대 이득) |
| **0011** | prefill CUDA graph skip: `ggml-cuda.cu` graph_compute 에 `is_prefill_graph` 추가 | pp +1~2% (multi-token prefill) |

### 3.3 병합 시 특이사항 (비용 절약 포인트)
- 0004의 `ggml-cuda.cu` op-timing hunk(1/4)는 **진단용(env-gated)이라 스킵** - 성능 무관
- 0010의 `ggml-cuda.cu` op-timing hunk 도 동일 스킵
- tests/test-backend-ops.cpp hunk 는 전부 스킵 (빌드 무관)

## 4. 적용하지 않은 것 (5개) + 이유

| 패치 | 내용 | 제외 이유 |
|---|---|---|
| 0001 | adaptive MTP draft | theTom 자체 MTP 이미 보유 |
| 0002 | fused chunked GDN prefill | GDN 하이브리드 모델 전용 - **dense 무관** |
| 0003 | BF16 KV cache | 사용 시나리오 없음 (q8_0/turbo3 채택) |
| 0008 | (0003 의존) | 0003 제외로 인한 의존성 |
| 0013 | fused MoE gate-up GLU | dense 모델 무관 (Qwen3.8-27B, MoE 아님) |

> 0013은 Qwen3.6-35B-A3B(MoE) 사용 시 재고려 대상 (향후 저도 포함 예정).

## 5. 적용한 것과 안 한 것의 차이 (영향 요약)

| 구분 | 적용 시 | 미적용 시 |
|---|---|---|
| 0004(WMMA 256) | head_dim 256 모델 FA 가 WMMA 경로 | HIP 가 head_dim>128 거부 - 표준 FA 폴백 |
| 0010(k-VDR) | tg +7~12% (Q4_K/VDR4) | tg 기존(-7~12%) |
| 0011(prefill skip) | pp +1~2% (그래프 오버헤드 제거) | ubatch 변동 그래프 캡처 실패 반복 |
| 0002(GDN) | (dense 모델 무변화) | 동일 |
| 0013(fused MoE) | (dense 무변화) | 동일 (MoE에서만 이득) |

## 6. 빌드 절차 (재생성 시 명령)

### 6.1 소스 준비
```bash
git clone --branch feature/turboquant-kv-cache https://github.com/TheTom/llama-cpp-turboquant.git /src/tq
# 패치 다운로드
curl -sL "https://api.github.com/repos/stew675/llama-cpp-rdna-boosts/contents/patches" \
  | grep -oE '"name":\s*"[^"]*"' | sed 's/.*"//;s/"//' > /tmp/patches.txt
# 클린 5개
git apply /tmp/0005.patch /tmp/0006.patch /tmp/0007.patch /tmp/0009.patch /tmp/0012.patch
```

### 6.2 0004 수동 (중요: theTom config 테이블은 유지)
```bash
# fattn-mma-f16.cuh line 2120 근처: DKQ > 128 -> DKQ > 256 (AMD_WMMA 게이트만)
# fattn.cu / mmq-vec-dot.cuh / mmq.cuh 는 클린 (git apply 가능)
awk -v t='a/ggml/src/ggml-cuda/fattn.cu' '/^diff --git/{p=($0~t)}p' /tmp/0004.patch | git apply
awk -v t='a/ggml/src/ggml-cuda/mmq-vec-dot.cuh' '/^diff --git/{p=($0~t)}p' /tmp/0004.patch | git apply
awk -v t='a/ggml/src/ggml-cuda/mmq.cuh' '/^diff --git/{p=($0~t)}p' /tmp/0004.patch | git apply
# fattn-mma-f16.cuh: DKQ>128 -> 256 (게이트만, config 테이블 변경 안 함)
```

### 6.3 0010 수동
```bash
# mmq-vec-dot.cuh, vecdotq.cuh 는 클린 적용
awk -v t='a/ggml/src/ggml-cuda/mmq-vec-dot.cuh' '/^diff --git/{p=($0~t)}p' /tmp/0010.patch | git apply
awk -v t='a/ggml/src/ggml-cuda/vecdotq.cuh' '/^diff --git/{p=($0~t)}p' /tmp/0010.patch | git apply
# mmvq.cu: get_vec_dot_q_cuda 의 Q4_K/Q5_K -> vdr4, Q6_K -> vdr2 등록만 (RDNA3_5 테이블은 gfx1100 무관 - 스킵 가능)
```

### 6.4 0011 수동
```bash
# ggml-cuda.cu graph_compute: `if (graph->is_enabled())` -> `if (graph->is_enabled() && !is_prefill_graph)`
# is_prefill_graph = cgraph->n_nodes > 0 && cgraph->nodes[0]->ne[1] > 1
```

### 6.5 ROCm 10 빌드 (docker)
```bash
docker run --rm -v /src/tq:/src/tq rocm/dev-ubuntu-26.04:10.0.0-full \
  bash -c '
    pip3 install --break-system-packages -q cmake
    export HIPCXX=/opt/rocm/core-10.0/lib/llvm/bin/clang++
    export LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
    cd /src/tq && cmake -S . -B build-rocm10 -DCMAKE_BUILD_TYPE=Release \
      -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 \
      -DCMAKE_HIP_COMPILER=$HIPCXX -DCMAKE_HIP_PLATFORM=amd \
      -DCMAKE_HIP_FLAGS="--rocm-path=/opt/rocm/core-10.0 --rocm-device-lib-path=/opt/rocm/core-10.0/lib/llvm/amdgcn/bitcode" \
      -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON \
      -DGGML_CUDA=OFF -DGGML_CUDA_FA_ALL_QUANTS=ON -DLLAMA_BUILD_TESTS=OFF
    cmake --build build-rocm10 --config Release -j 16 --target llama-bench llama-server'
```

### 6.6 Docker 이미지
```dockerfile
FROM rocm/dev-ubuntu-26.04:10.0.0-full
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates libc6 && rm -rf /var/lib/apt/lists/*
COPY build-rocm10/bin/ /app/
ENV LD_LIBRARY_PATH=/app:/opt/rocm/core-10.0/lib HIP_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=6 MTMD_VISION_BACKEND=hip ROC_ENABLE_PREEMPTION=1
WORKDIR /app
ENTRYPOINT ["/app/llama-server"]
```

## 7. 검증 절차 (재생성 시 최소)

```bash
# 1) 스모크 (터빈 경로 확인)
docker run --rm --device /dev/kfd --device /dev/dri/renderD130 --device /dev/dri/card2 --group-add video \
  -e HIP_VISIBLE_DEVICES=0 --entrypoint /app/llama-bench <img> \
  -m /models/Qwen3.5-2B-MTP-Q4_K_M.gguf -ngl 99 -fa on -ctk q8_0 -ctv turbo3 -p 32 -n 32

# 2) 운영 구성 (27B, 140K, adaptive MTP)
docker run -d --name test --device ... -e HIP_VISIBLE_DEVICES=0 <img> \
  -m /models2/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-MTP-Q3_K_M.gguf -c 140000 -b 4096 -ub 1024 \
  -ngl 99 -fa on -ctk q8_0 -ctv turbo3 --cache-ram 19152 --no-mmap --jinja --reasoning off \
  --spec-type draft-mtp-adaptive --spec-draft-n-max 3 --spec-draft-n-min-adaptive 2 --host 127.0.0.1 --port 8096
# 기대: prefill 4445t ~6s (793t/s), tg ~52t/s, acceptance 50%

# 3) needle 30K 정확도 (필요 시)
python3 /tmp/needle_chat.py 8096 30000 0.78  # 기대 HIT
```

## 8. 재생성 시 토큰 절약 포인트

1. **새 벤치 최소화**: 기대치를 §5/§7의 기준값으로 상정 - 2B 스모크 + 27B 운영 1회만 실행
2. **전체 스윕 불필요**: prefill 최적은 이미 ub=1024, q8_0/turbo3 로 확정 (§12) - 반복 금지
3. **테스트 유의사항 준수** (§5): --reasoning off 없으면 beellama `////` 문제로 정확도 오판
4. **theTom config 테이블 건드리지 말 것**: 0004 의 config 변경이 beellama/base 와 다름 - 게이트만 수정
5. **native 7.2.3 선 검증 옵션**: ROCm 10 빌드 나중으로 미루고 native 로 컴파일 오류부터 잡으면 20분 절약
6. **0002(GDN)/0013(MoE) 는 dense 한정 무변화** - Qwen3.8-27B 재생성에선 제외 유지

## 9. 참조 (원본 파일)

| 파일 | 내용 |
|---|---|
| `/tmp/tq-patch-test/` | 최종 소스 (패치 적용 완료) |
| `/tmp/tq-patch-test/build-rocm10/` | ROCm 10 빌드 산출물 |
| `/tmp/tq-native-build/` | native 7.2.3 검증 빌드 |
| `/tmp/build-tq-rocm10.sh` | ROCm 10 빌드 스크립트 |
| `/tmp/bench-tq-boosts.sh` | S1~S4 벤치 스크립트 |
| `/tmp/needle_chat.py` | long-context 정확도 하니스 |
| `BENCHMARK-FINAL.md` | 최종 비교 문서 |
| `bench-kv-vs-turboquant.md` | 전체 진행 이력 (§1~17) |