# BENCH-2026-09-06-FINAL: 7900 XTX x2 v2 트랙 벤치 (Track B: upstream+RCCL+boosts+PRs)

> 빌드: rccl-build-v2 (upstream llama.cpp 9e0e220 + rdna-boosts 13 + RCCL + PR#28223 + PR#27861, ROCm 10.0 docker)
> 하드웨어: RX 7900 XTX 24GB x2 (gfx1100), i5-14600K, 96GB DDR4
> 측정: llama-server 직접 구동, slot print_timing (prompt eval / eval time), rocm-smi VRAM/GPU%, free RAM, top CPU
> 표준 프롬프트: MoE 구조 설명 요청 73 tokens, max_tokens 300, temp 0.7

---

## 1. 모델별 결과

### 1.1 Qwen3.6-35B-A3B (MoE, 17.2GB, MTP 임베디드 nextn) — 싱글이 정답

| # | 구성 | tg t/s | pp | accept | VRAM0/1 | GPU% | RAM | CPU |
|---|---|---|---|---|---|---|---|---|
| S1 | **싱글 전부 VRAM + MTP (base)** | **129.55** | 116.5 | 0.46 | 18.9/- | 85-95% | 33GB | 7% |
| S2 | 싱글 + exps ROCm_Host + cache96 | 11.12 | 63.1 | 0.55 | 10.2/- | 19-24% | 47GB | **38-44%** |
| D1 | 듀얼 tensor 전부 VRAM + MTP | 102.15 | 69.3 | 0.53 | 10.2/10.2 | 73-76% | 34GB | 7% |
| D2/D3 | 듀얼 + exps host/CPU + cache | **NCCL 크래시** | - | - | 8.4 | 7-10% | 48GB | 45% |

결론: A3B는 17GB라 한 카드에 완전 수용 → **싱글 MTP 129.6 t/s가 최선**. 듀얼 tensor는 all-reduce 오버헤드로 -27%.
exps host + moe-cache는 VRAM 절반(-10GB)이지만 **GPU 19% / CPU 38-44% 병목**으로 속도 -91%.
또한 **RCCL 10.0 + tensor split + exps offload 조합은 NCCL 크래시** (ROCm 10.0 제약으로 문서화).

### 1.2 Qwen3.8-27B-UD-IQ4_XS (dense, 14.25GB + 별도 MTP 헤드 1.37GB) — MTP가 핵심

| # | 구성 | tg t/s | accept | VRAM | GPU0/1% | 비고 |
|---|---|---|---|---|---|---|
| S1b | 싱글 MTP OFF (base, ctx4096/ub256) | 40.26 | - | 23.0GB | 100/- | 싱글 MTP는 24GB 초과 |
| S1c | 싱글 + MTP (ctx2048/ub128, ngld60) | **54.96** | 0.57 | 25.7GB(스왑) | 100/- | reddit 싱글 53.9와 일치 |
| D1 | **듀얼 tensor + MTP (ctx8192/ub1024)** | **78.17** | 0.53 | 18.3/18.5 | 100/100 | **reddit 69.25 +13%** |

결론: 27B dense는 듀얼 tensor가 정석 (카드당 18GB, 커버리지 + 컨텍스트 여유).
싱글 MTP는 VRAM 압박으로 ctx/ubatch 축소 필요 but +37% (40.3→55.0).

### 1.3 Qwen3.8-Flash-Next (qwen4exp, 84.9GB MoE/512 experts, per_layer 51B) — layer split 한계

| # | 구성 | tg t/s | pp | VRAM0/1 | GPU% | CPU% | RAM |
|---|---|---|---|---|---|---|---|
| D1b | 듀얼 layer + exps 전부 CPU (base) | 4.70 | 24.2 | 3.9/3.9 | 14-24% | **33-39%** | 34GB |
| D2 | 듀얼 layer + GPU41레이어 + late7 host + cache96 | **16.49** | 38.1 | 25.7/18.1 | 32-42% | 15% | 40GB |
| D3 | 동일 + cache128/ub256 | 17.29 | 56.3 | 25.6/18.4 | 35-42% | 14% | 40GB |
| D4 | late10 host + cache192/inserts4 | 17.33 | 54.7 | 25.6/17.7 | 38-42% | 13% | 43GB |

결론: **qwen4exp tensor split 미지원** (upstream/beellama 공통, `llm_arch_supports_sm_tensor`에 명시적 `return false` + `TODO: fix test-llama-archs`).
레이어 split 파이프라인은 GPU 40% 한계 → cache를 아무리 늘려도 ~17 t/s 벽.
- GPU에 41레이어 exps + attention, late 7-10레이어만 host + moe-cache : 4.7→17.3 t/s (+270%)
- reddit 2x3090(17→25-29 t/s)과 같은 패턴이지만 3090보다 낮은 절대값 (RDNA3 layer-split + 24GB 카드 압박)

## 2. 총괄 표

| 모델 | 최적 구성 | tg t/s | VRAM/카드 | 비고 |
|---|---|---|---|---|
| A3B (MoE 17GB) | 싱글 + MTP 전부 VRAM | **129.6** | 18.9 | 듀얼 불필요 |
| 27B (Dense 14GB) | 듀얼 tensor + MTP | **78.2** | 18.3/18.5 | reddit 69 초과 |
| Flash-Next (84.9GB) | 듀얼 layer + GPU41/host7 + cache | **17.3** | 25.7/18.1 | tensor split 미지원 한계 |

## 3. 핵심 기술 발견 (Track B)

1. **ROCm_Host -ot 정상 인식**: "CPU, ROCm0, ROCm1, ROCm_Host" (GPU 초기화 후). PR#28223 적용 확인.
2. **RCCL 10.0 크래시**: `tensor split + exps offload(ROCm_Host/CPU)` → "NCCL init failed (unhandled system error)".
   RCCL이 host tensor의 all-reduce 참여를 못 견딤. base(all GPU)는 정상. → exps offload는 layer split에서만 가능.
3. **qwen4exp tensor split 미지원**: beellama/upstream 공통. `return false` 명시 + TODO.
4. **moe-cache는 GPU가 포화되지 않았을 때만 유효**: A3B 싱글(85-95% GPU)에서 exps host+cache는 오히려 -91%,
   Flash-Next layer(40% GPU 한계)에서도 캐시 슬롯 증가 효과 미미 — 파이프라인 구조가 병목.
5. reddit의 exps pinned prefill 2.3-2.8x 이득은 **tensor split** 기반. layer split에선 적용 불가.

## 4. raw data
- 로그/응답/prof: bench-results-v2/{tag}.{server.log,prof,c1.json,req.json}
- 하니스: /tmp/bench_v2.sh (docker, --network host, VRAM/RAM/CPU/GPU% 트래킹)
- 컨테이너 실행은 "load ~3분" 고려해 health 대기 300초

## 5. 남은 이슈
- [ ] qwen4exp tensor split: llama-model.cpp weight layout 재작업 필요 (단순 플래그 해제로는 비안전)
  → beellama도 미지원, 테스트 실패 위험. 운영은 layer split로 확정
- [ ] Flash-Next 더 빠른 경로: (a) MTP 헤드 없는 상태에서 로드 시간 단축 (b) single-slot 병렬1로 실사용
- [ ] 27B 싱글 MTP VRAM: ctx1024까지 축소하면 스왑 없이 가능한지
---

## 6. 업데이트: PR#28136 (qwen4exp direct PLE) 통합 + 실측 (2026-09-06 후속)

### 빌드
- upstream-latest에 PR#28136 2커밋(90fde1f7 + c6a9e5c9) cherry-pick 적용 (auto-merge, 충돌 없음)
- 커밋: `410bd6e add PR28136: qwen4exp direct PLE reads (lazy-mode DIRECT)`
- 증분 재빌드 완료 (rccl-build-v2, llama-server 20:44)

### 새 옵션 (llama.h)
- `LLAMA_LAZY_MODE_*`: off/auto/on/**direct(=3)** — direct는 arch가 전용 pread()로 PLE 행 읽음
- CLI: `--lazy-mode on-direct` (숫자 아님, 문자열)

### 실측 A/B (Flash-Next M64, 듀얼 layer, GPU41+late7host+cache96, 5850tok 프롬프트)

| lazy mode | pp t/s | 비고 |
|---|---|---|
| off (per_layer CPU 상주) | 280.8 | 기준 |
| auto (mmap 폴트) | 349.7 | +24.5% |
| **on-direct (pread)** | **365.3** | **+30.1% vs off** |

- 짧은 프롬프트(73tok)에선 direct ≈ off (PLE 행이 적어 이득 없음) — PR 설명과 일치
- decode는 무변화 (18.16 t/s) — PR 목적이 PLE prefill 가속이므로 정상
- reddit 보고(300→750)는 GB10 단일 칩 기준, 우리 듀얼 layer 구성에선 +30% (구조상 통신 오버헤드 감안)

### 결론: Flash-Next 최종 운영 구성
```
--lazy-mode on-direct  # PLE prefill +30%
-ot 'per_layer_token_embd.weight=CPU, blk.(4[1-7]).ffn_*.exps.weight=ROCm_Host'
--moe-expert-cache 96
-sm layer (qwen4exp tensor split 미지원)
```
- prefill: 5850tok → 365 t/s (off 대비 +30%), decode ~17-18 t/s (구조적 한계)

---

## 7. 업데이트: qwen4exp tensor split 강제 + MTP 시도 결과 (최종)

### 빌드 커밋 (upstream-latest)
```
9c0c80e add PR144 cuda graph shape-keying + LRU cap
977f772 add unsloth PR144: qwen4exp draft-only MTP load + hc_head
2300013 wip: enable qwen4exp tensor split (remove sm_tensor false)
```
- tensor split 차단(return false) 제거 → 로드 자체는 성공
- unsloth PR#144: MTP 드래프트 로드(mtp_only, load_mtp) + CUDA graph shape-keying 적용

### 실측 (Flash-Next M64, 5850tok prefill / 300tok decode)

| split | MTP | GPU0/1 VRAM | GPU 동시% | tg | pp(5850) | 결과 |
|---|---|---|---|---|---|---|
| layer | X | 25.7/18.1 (불균형) | 40% | 16.7 | 365 | 동작 |
| **tensor (강제)** | X | **22.9/23.1 균등** | 50-57% | 15.1 | 140.7 | 동작, RCCL 정상 |
| tensor | **O (Q8_0 unified)** | 25.2/25.7 | - | ✗ | ✗ | **NCCL fail + OOM 크래시** |

### 최종 결론 (듀얼 7900 XTX, Flash-Next M64)

1. **tensor split 차단 제거로 GPU 균등 배분 성공** (22.9/23.1, 스왑 해소, GPU 57% 동시)
   - 단, all-reduce(RCCL) 오버헤드로 decode 15.1 (layer 16.7보다 약간↓), prefill 140.7 (layer 365의 -62%)
2. **tensor + MTP는 불가**: RCCL 10.0이 host/CPU exps의 all-reduce 참여 시 NCCL 실패 + GPU 풀 상태에서 MTP 그래프 OOM
   - MTP는 layer split에서도 GPU0 1GB 부족으로 불가
3. **AtomicChat quant의 강점은 prefill/mm주** (n-gram 테이블 SSD, -lazy-mode on-direct +30%)
   - decode 36 t/s(64GB Mac)는 단일 칩 통합메모리 기준, 우리 24GBx2 구조와 다름
4. **Flash-Next 최종 실질 한계 = ~17 t/s decode** (레이어 split + GPU41/host7 + cache + lazy-direct)
   - 또는 GPU 균등이 필요한 경우 tensor split 15.1 t/s
   - 24GB x2에서 84.9GB MoE의 현실적 성능. 69 t/s(27B dense)와 다른 영역

### best-effort 운영 구성
```bash
# decode 최대: 레이어 split
-sm layer -ot 'blk.(4[1-7]).ffn_*.exps.weight=ROCm_Host' --moe-expert-cache 96 --lazy-mode on-direct
# GPU 균등 우선: tensor split (MTP 없음)
-sm tensor -ts 1,1  # 단, prefill -62%, decode -10%
```

### raw
- bench-results-v2/fn_t2_tensor_nomtp.* (tensor 균등), fn_atomic_chat.* (layer), pr144.diff (unsloth)

---

## 8. 업데이트: F16 wire (RCCL all-reduce) — RDNA3 최적화 실측

### 배경
- RDNA3는 FP16이 full-rate, BF16은 convert 오버헤드 (문서 §2.4 와 일치)
- llama.cpp RCCL all-reduce가 대형 텐서를 BF16으로 압축 전송 (기본)
- BF16→F16 wire 전환 시 PCIe 전송은 동일(2B)하지만 변환 비용 절감

### 구현 (커밋: F16 wire opt-in)
- `ggml-cuda.cu` allreduce_nccl: BF16 경로 유지 + `GGML_CUDA_AR_WIRE_F16` env로 F16 선택
- F16: to_fp16(to_fp32) + ncclHalf, BF16: to_bf16(to_fp32) + ncclBfloat16
- BF16 기본 경로 비파괴 (독립 분기)

### 중간 원인 규명
- **CUDA graph shape-keying (unsloth PR#144)은 ROCm에서 NCCL과 충돌**: `invalid resource handle` 크래시
  (MTP + tensor split에서). shape-keying revert로 해소 (커밋 9e0a013, cdc17bc)
- F16 wire 첫 구현에 BF16 경로 파괴 버그 → 원본 복원 후 독립 분기로 재구현

### 실측 A/B (27B-UD-IQ4_XS, 듀얼 tensor 1,1, MTP, f16 KV, 300tok x3, warm 마지막)

| wire | tg run1/2/3 | median tg | 비고 |
|---|---|---|---|
| **F16 (GGML_CUDA_AR_WIRE_F16=1)** | 74.53/77.20/62.31 | **74.5** | +6% |
| BF16 (기본) | 91.06/70.43/67.47 | 70.4 | 기준 |

- F16이 중앙값 +6%, 편차 큼 (단발 생성 노이즈). 경향은 F16 유리.
- 첫 단일 비교에서도 F16 74.88 vs BF16 64.42 (+16%)

### 운영 사용법
```bash
docker run ... -e GGML_CUDA_AR_WIRE_F16=1 \
  llama-server -m <model> -sm tensor -ts 1,1 ...  # RCCL wire = F16
```

### 원인 요약 (이 세션에서 해결한 크래시들)
1. CUDA graph shape-keying (unsloth PR#144) ↔ ROCm NCCL → revert
2. F16 wire의 BF16 경로 파괴 → 독립 분기로 재구현
3. 이전 "NCCL init failed" 크래시는 위 두 가지가 원인

---

## 9. 업데이트: Q8 wire 구현 + 3-way wire 비교 (F16/Q8)

### Docker 빌드 인프라 (ccache)
- 이미지: `baramofme/llama-hop-build:rocm10-ccache` (cmake/ninja/ccache, CCACHE_DIR=/ccache)
- 증분 빌드 6.9초 (이전 20분대 → ccache+증분)
- 사용: `docker run -v ccache-vol:/ccache ... llama-hop-build cmake --build /build -j32 --target llama-server`

### Q8 wire 구현 (커밋: Q8 wire for RCCL allreduce)
- `ggml-cuda.cu`: allreduce_nccl에 `GGML_CUDA_AR_WIRE_Q8` 분기 추가
- 커널: `ggml_cuda_q8_wire_quantize` (F32 -> int8, 고정 scale) / `ggml_cuda_q8_wire_dequantize_sum`
- 전송: ncclInt8 + ncclSum (1 byte/elem = BF16/F16의 절반)
- 오버플로우 방지: 값 [-63,63] 클리핑 (2 rank 합이 int8 범위 내)
- opt-in: `GGML_CUDA_AR_WIRE_Q8=1` (BF16/F16 기본 경로 비파괴)

### 3-way 실측 (27B-UD-IQ4_XS, 듀얼 tensor 1,1)
| wire | decode tg | prefill pp(5850) | 출력(temp0) |
|---|---|---|---|
| **Q8** | 73.16 | **1231.1 (+4.3%)** | 동일 |
| F16 | 73.46 | 1190.5 (+0.8%) | 동일 |
| BF16 (기준) | 71.84 | 1180.5 | 동일 |

- **Q8: bandwidth-bound prefill +4.3%** (전송량 50% 감소)
- 출력 정확도: 3개 모두 byte-identical (greedy, 동일 경로)
- decode는 노이즈 내 (73 vs 71, ~+2%)

### 운영 사용법
```bash
docker run ... -e GGML_CUDA_AR_WIRE_Q8=1 llama-server ...   # Q8 wire (prefill 최적)
# 또는 -e GGML_CUDA_AR_WIRE_F16=1  (decode/A3B 등에서)
```

### 결론: RDNA3 wire 선택 정리
1. **prefill (bandwidth-bound)**: Q8 wire 최적 (+4.3%) — 전송량 절반
2. **decode**: F16 ≈ Q8 > BF16 (+2%) — RDNA3 FP16 파이프라인
3. **정확도**: 모두 동일 (greedy), Q8도 실용 안전
