# implement-plan: RCCL + rdna-boosts + Q8/F16 wire (듀얼 GPU)

> 상태: **적용 완료 + 검증 완료** (Track B, 2026-09-06). 아래는 재현용 이식 계획.
> 원본 연구: `rdna-boosts-apply-plan.md`, `TRACK-B-RESULT.md`, `HANDOVER-rocm10-boosts-bench.md`, `BUILD-BEE-QWEN38-NEXT.md`

## 목표

- stew675/llama-cpp-rdna-boosts 패치셋(13블록)을 theTom fork에 순차 이식 (gfx1100/RDNA3).
- RCCL(다중 GPU all-reduce) 활성화 + wire 압축(Q8/F16)으로 듀얼 GPU 성능 확보.

## 하드웨어 상수 (중요)

- 이 보드 7900 XTX x2는 **GPU-GPU P2P 물리 불가** (`hipDeviceCanAccessPeer` can=0).
  - host-staged internal (stew675): RDNA4 전용 게이트 + 실측 실패 -> RDNA3 불가
  - direct-P2P (JohnTDI): can=0 하드웨어 불가
  - **RCCL ncclAllReduce = RDNA3 x2의 유일한 tensor AR 경로** (Q8/F16 wire 포함)
- 결론: **RDNA3 듀얼 GPU = 레이어 split이 정석.** tensor split은 all-reduce 오버헤드로 열위.

## 빌드 구성 (검증됨)

```
cmake -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DGGML_HIP_GRAPHS=ON
      -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON -DGGML_HIP_RCCL=ON
      -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF
      -DCMAKE_HIP_FLAGS=--rocm-path=/opt/rocm --rocm-device-lib-path=/opt/rocm/lib/llvm/amdgcn/bitcode
```
- RCCL: `rccl_DIR=/opt/rocm/lib/cmake/rccl`. 실행은 GLIBC 2.43 -> ROCm 10.0 docker 필수.

## rdna-boosts 이식 순서 (의존성 순)

| 순서 | 블록 | 내용 | 비고 |
|---|---|---|---|
| 1 | 0004 | RDNA3/4 WMMA flash-attn + Q6_K mmq | gfx1100에서 가장 필요 |
| 2 | 0003 | BF16 KV cache + native-BF16 FA | |
| 3 | 0008 | fused-core prefill kernels | 0003+0004 의존 |
| 4 | 0010 | k-quant VDR | greedy 수치 변경 주의 |
| 5 | 0001 | adaptive MTP draft depth | |
| 6 | 0011 | skip CUDA graphs multi-token prefill | |
| 7 | 0009 | meta-buffer compute-container headroom | |
| 8 | 0013 | fused MoE gate+up+GLU MMQ | dense엔 해당 없음 |
- 0012 제외 (RDNA4 전용, gfx1100 RCCL 폴백 이득 없음).
- 충돌 블록(0004/0010/0011)은 수동 병합. 13/13 클린 적용 확인 (Track B).

## Q8/F16 wire (RCCL all-reduce 압축)

| 경로 | env | 사용처 | 성능 | 손실 |
|---|---|---|---|---|
| Q8 | `GGML_CUDA_AR_WIRE_Q8=1` | prefill-heavy | pp +4.3% | 실용 무손실 |
| F16 | `GGML_CUDA_AR_WIRE_F16=1` | decode-heavy | tg +2% | 무손실 |
| BF16 | (기본) | fallback | 기준 | 무손실 |

- Q6/Q5는 ncclSum과 **비트 패킹 비호환** -> 구현 불가. Q7은 어중간 -> 스킵. **Q8이 구조적 하한 + 무손실.**

## 한계 / 얻은 교훈

- tensor split prefill은 레이어 split 대비 -35~62% (half-width matmul 비용, PR#27825).
- host expert PCIe fetch가 prefill 병목 핵심: `late6 host + moe-expert-cache-inserts 1`로 +17%.
- 듀얼 tensor는 MTP 조합 시 RCCL fail + OOM (24GB x2 물리 한계).
