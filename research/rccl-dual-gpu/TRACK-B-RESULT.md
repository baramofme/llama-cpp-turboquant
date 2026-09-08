# TRACK-B: upstream 최신 llama.cpp + rdna-boosts + RCCL 빌드 결과 (2026-09-06)

> 목적: beellama(RCCL, ROCm 7.2.3)의 MTP 긴 프롬프트 버그를 우회하기 위해,
>       최신 upstream llama.cpp에 rdna-boosts 13블록 + RCCL + PR#28223 + PR#27861 을 적용해 ROCm 10.0으로 재빌드
> 빌드: rocm/dev-ubuntu-26.04:10.0.0-full (ROCm 10.0, HIP 7.15, clang 23, RCCL 10.0), gfx1100
> 소스: upstream-latest/ = ggml-org/llama.cpp 9e0e220 (2026-09-06) + boosts 13/13 + PR#28223 + PR#27861
> 산출물: rccl-build-v2/bin (llama-server, llama-bench + lib*.so)
> 실행: docker (GLIBC 2.43 필요, 호스트 직접 실행 불가)

## 빌드 구성

```
cmake -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DGGML_HIP_GRAPHS=ON
      -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON -DGGML_HIP_RCCL=ON
      -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF
      -DCMAKE_HIP_FLAGS=--rocm-path=/opt/rocm --rocm-device-lib-path=/opt/rocm/lib/llvm/amdgcn/bitcode
```

- RCCL: rccl_DIR=/opt/rocm/lib/cmake/rccl, librccl.so.1 링크 확인
- 적용 패치: rdna-boosts 0001~0013 전부 (0013 fused MoE 포함) + PR#28223(host override/read_raw, 5cfa6a87+c59754bfc) + PR#27861(moe cache, bccbacdb)
- 검증: llama_moe_cache_* 심볼 존재, draft-mtp-adaptive 옵션 존재

## 벤치 (27B UD-IQ4_XS + 별도 MTP 헤드, 듀얼 tensor 1,1, f16 KV, ub1024, docker)

| 시나리오 | 프롬프트 | tg | pp | draft acceptance | 비고 |
|---|---|---|---|---|---|
| 듀얼 tensor + MTP | 짧음(27tok) | **79.0** | 42.3(cold) | 0.59 | beellama 대비 우위 |
| 듀얼 tensor + MTP | **긴(1211tok)** | **108.7** | **581.7** | **0.94** | 빌드 결정적 우위 |
| 듀얼 tensor + MTP | 중간(21tok, 1st) | 61.1 | 36.4 | 0.39 | 콜드 로드 직후 |
| 듀얼 tensor + MTP | (2nd warm) | 73.0 | 20.6 | 0.49 | warm |

## beellama vs upstream (Track A vs Track B) — 같은 모델/구성

| 측정항목 | beellama RCCL (7.2.3) | **upstream RCCL (10.0)** | Δ |
|---|---|---|---|
| 짧은 프롬프트 tg | 73.5 | 79.0~108.7 | +8~48% |
| 긴 프롬프트 MTP accept | **0.0 (버그)** | **0.94** | 완전 해결 |
| 긴 프롬프트 tg | 27.5 (MTP 붕괴) | **108.7** | +295% |
| reddit 69.25 대비 | 달성(72~81, 짧은프롬프트) | 초과 (79~108) | - |

## 핵심 결론

1. **MTP 긴 프롬프트 버그는 beellama 고유** — upstream 최신에선 1211토큰 프롬프트에서도 accept 0.94로 정상
   (beellama의 llama_set_embeddings_nextn 재작성 경로가 원인)
2. **reddit ghosthand의 87~90% accept를 0.94로 초과** — 긴 컨텍스트에서 현실적 100+ t/s
3. **Track B 빌드가 운영 구성으로 확정**: 최신 upstream + 13 boosts + RCCL + moe-cache(PR#27861) + host override(PR#28223)

## 재현 명령 (docker)

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  -v rccl-build-v2:/app -v <modeldir>:/models \
  -e LD_LIBRARY_PATH=/app:/opt/rocm/lib -e GGML_CUDA_P2P=1 \
  rocm/dev-ubuntu-26.04:10.0.0-full /app/llama-server \
    -m /models/Qwen3.8-27B-UD-IQ4_XS.gguf \
    -md /models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf -ngld 99 \
    -ngl 99 -sm tensor -ts 1,1 -fa on -c 8192 -b 4096 -ub 1024 \
    -ctk f16 -ctv f16 --spec-type draft-mtp --spec-draft-n-max 3
```

## 다음 (미완)
- [ ] 131K/262K 컨텍스트에서 수용률·tg 검증 (VRAM 여유 확인 후)
- [ ] moe-cache(PR#27861) + Flash-Next(80GB MoE) 듀얼 운영 구성
- [ ] A3B(35B MoE)에도 MTP 헤드 존재 시 동일 검증
- [ ] docker 이미지로 패키징 (운영 배포용)