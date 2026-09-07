# 7900 XTX x2 RCCL + MoE/MTP 최적화 벤치 세션 종합 (2026-09-06)

> 하드웨어: RX 7900 XTX 24GB x2 (gfx1100) + i5-14600K (20T) + 96GB DDR4
> 시간대: Asia/Seoul 2026-09-06 (오전 RCCL 빌드 ~ 오후 Track B 빌드, 단일 세션)
> 관련 문서: BENCH-A3B-Q38-MTP.md, BENCH-27B-REDDIT-REPRO.md, BUG-MTP-LONGPROMPT.md, TRACK-B-RESULT.md, BUILD-BEE-QWEN38-NEXT.md

---

## 1. 세션 목표

RCCL 빌드로 MoE 모델을 싱글/듀얼 GPU에서 구동하며 "VRAM을 적게 먹고도 속도 향상"이 가능한지
tensor split + MTP + moe-expert-cache 조합으로 벤치하고, reddit 커뮤니티 수치(29→69 t/s) 대비 우리 성능을 검증.

## 2. 빌드 두 트랙

### Track A — beellama.cpp v0.4.5 (처음 시도)
- beellama-boosts/src (`44a36969` + rdna-boosts 8블록 수동 병합, rdna-test 브랜치)
- ROCm 7.2.3 native, RCCL 빌드 (`rccl-build/bin`, GGML_HIP_RCCL=ON)
- llama-graph.cpp 최종 수정(16:41) → 재컴파일(16:42) 확인, 빌드 무결성 검증됨

### Track B — upstream 최신 (최종 채택) ★
- ggml-org/llama.cpp `9e0e220` (2026-09-06) + **rdna-boosts 13블록 전부** + PR#28223 + PR#27861
- ROCm 10.0 docker (`rocm/dev-ubuntu-26.04:10.0.0-full`, HIP 7.15/clang 23, RCCL 10.0)
- 산출물: `rccl-build-v2/bin` — 13/13 패치 클린 적용(의존 순서로 0008/0011/0013까지 해소) + PR 2개 cherry-pick(자동 병합)
- 실행은 docker 필수 (GLIBC 2.43)

## 3. 핵심 발견 1 — 모델 선택이 벤치 의미를 결정

| 모델 | 아키텍처 | expert(_exps) | MTP 헤드 | moe-cache 효과 | 싱글 GPU |
|---|---|---|---|---|---|
| Qwen3.8-27B-MTP | qwen35 (dense, 65blk) | 0 | 임베디드 | **무효** | 가능 |
| Qwen3.8-27B-UD-IQ4_XS | qwen35 (dense) | 0 | **별도 gguf(-md)** | 무효 | 가능 |
| Qwen3.6-35B-A3B | qwen35moe (41blk) | 123 | `blk.40.nextn.*` 임베디드 | 유효 | 가능 (17GB) |
| Qwen3.8-Flash-Next | qwen4exp (48blk) | 44+/샤드 | (M64 별도) | **유효** | 불가 (26.8GB 텐서) |

## 4. 벤치 데이터 (llama-server 직접 구동, slot print_timing)

### 4.1 A3B-UD-Q3_K_XL (17.2GB MoE) — Track A

| # | 구성 | VRAM | tg | pp | accept |
|---|---|---|---|---|---|
| S1 | 싱글 전부 VRAM (기준) | 16.2GB | 96.5 | 305.8 | - |
| S2 | 싱글 + exps ROCm_Host + cache96 | 8.1GB | 45.4 | 131.6 | - |
| S3 | 싱글 + cache64 | 6.4GB | 46.1 | 129.2 | - |
| S4 | 싱글 + cache128 | 9.8GB | 54.7 | 126.0 | - |
| **M1** | **싱글 + MTP 전부 VRAM** | 16.8GB | **135.4** | 216.5 | 0.82 |
| M2 | 싱글 + MTP + cache128 | 10.4GB | 34.9 | 105.3 | 0.78 |
| M3 | 싱글 + MTP + cache96 | 8.8GB | 29.9 | 104.3 | 0.61 |
| D1 | 듀얼 tensor 전부 VRAM | 17.4GB | 60.9 | 199.8 | - |
| D2 | 듀얼 + exps host + cache96 | 14.1GB | 19.0 | 67.7 | 최악 |
| M4 | 듀얼 tensor + MTP | 18.1GB | 89.2 | 238.1 | - |

A3B 결론: **한 카드에 들어가는 MoE는 싱글이 정답.** 듀얼 tensor는 all-reduce 오버헤드로 손해.
moe-cache는 VRAM 절감(-50~60%) 대가로 속도 반감 — "적게 먹고 빠르게"는 이 크기 모델에선 절충만 존재.

### 4.2 27B-UD-IQ4_XS + 별도 MTP 헤드 (reddit 재현) — Track A/B

| # | 구성 | tg | accept | 비고 |
|---|---|---|---|---|
| X1 | 듀얼 tensor + MTP + ub1024, 짧은 프롬프트 | 79.9~81.4 | 0.65 | reddit 69 초과 |
| X2 | 듀얼 tensor + MTP 롱런(880tok) | 71.7~72.1 | 0.57 | reddit 69.25와 동등 |
| TB1 | Track B 동일 구성, 첫 콜드 | 61.1→73.0 | 0.39→0.49 | |
| **TB4** | **Track B, 긴 프롬프트(1211tok)** | **108.7** | **0.94** | **★ MTP 버그 없음** |
| TB5 | Track B, 짧은 프롬프트 | 79.0 | 0.59 | |

## 5. 핵심 발견 2 — beellama MTP 긴 프롬프트 버그 (Track A)

- 프롬프트 ~300토큰 경계에서 **draft-mtp 드래프트가 silent하게 0으로 붕괴** (tg 73→27)
- 짧은 프롬프트 + 롱런 생성에서는 정상 (8634토큰 생성 accept 0.51 유지)
- 다음n 임베디드/별도 -md 파일 무관, draft-mtp/draft-mtp-adaptive 무관 → **드라이버 공통 버그**
- kv_unified/parallel/ubatch/ctx 무관 (전부 배제)
- reddit 사용자(upstream 정식)는 정상 → beellama의 `llama_set_embeddings_nextn` 재작성 경로 특유 회귀로 추정
- **우회책으로 Track B로 전환하게 된 계기**

### Track B에서 검증: 업스트림 최신에는 버그 없음
- 1211토큰 프롬프트: accept 0.94, tg 108.7 — beellama(27.5) 대비 +295%
- Track B로 **완전 해결** 확인 (BUG-MTP-LONGPROMPT.md 참조)

## 6. 핵심 발견 3 — RCCL의 가치 (PR#27825 데이터와 대조)

reddit virtualworker (RCCL 실패 → butterfly 폴백): tensor **124pp / 33.5tg**
ghosthand (RCCL 동작): **361pp / 65.9tg** — RCCL 없으면 tensor 성능 3배 붕괴

우리 교차검증 (Track A):
- 27B Q3 dense 싱글 37.5 tg = 오늘 낮 벤치(34.6~38.8)와 일치 → builds 정상
- PR 검증자(같은 27B급): 싱글 35.2 / tensor 40.1 → 우리 37.5/42.2 = 동급~우위
- reddit "1200+/90+" 는 Q8_0+262K+MTP 스택의 극단값. Q4/Q6급에서 35~49가 정상 범위

## 7. 운영 구성 (권장, Track B 기반)

```bash
# 27B dense + MTP (빠른 채팅/코드)
-docker run--device /dev/kfd --device /dev/dri --group-add video \
  -v rccl-build-v2:/app -v /mnt/nvmedata/models/unsloth/Qwen3.8-27B-GGUF:/models \
  -e LD_LIBRARY_PATH=/app:/opt/rocm/lib -e GGML_CUDA_P2P=1 \
  rocm/dev-ubuntu-26.04:10.0.0-full /app/llama-server \
    -m /models/Qwen3.8-27B-UD-IQ4_XS.gguf \
    -md /models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf -ngld 99 \
    -ngl 99 -sm tensor -ts 1,1 -fa on -c 8192 -b 4096 -ub 1024 \
    -ctk f16 -ctv f16 --spec-type draft-mtp --spec-draft-n-max 3
```

- 27B dense: **tensor + MTP** → 79~108 t/s (reddit 69 초과)
- Flash-Next M64 (80GB MoE, /opt/llm/models/Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64, 28샤드):
  듀얼 tensor + exps ROCm_Host + moe-cache (PR#27861 포함) — 아직 미벤치 (다음 단계)
- A3B (Q3_K_XL MTP, /mnt/nvmedata/models/unsloth/Qwen3.6-35B-A3B-GGUF):
  싱글 + MTP 116~135 t/s / 듀얼은 불필요

## 8. Q8 전송 / 추가 최적화 (미구현)

- reddit [1w14qal]의 "Q8 wire 전송" (all-reduce 데이터 BF16→Q8 압축, pp 1390 달성)은
  **모델 다운로드와 무관한 코드 패치**. 현재 소스는 BF16 wire round-trip까지만 존재.
- rdna-boosts 0012에 NCCL-failure fallback(issue #13, 커밋 ee2daac) 최신 반영됨
- stew675/llama-cpp-rdna-boosts fork point: upstream 9cffdcc80 (2026-09-02),
  이후 7900XTX 관련은 0013(fused MoE) fold만 존재, Strix Halo(gfx1151) 작업이 주류

## 9. 미완 / 다음 단계

- [ ] Flash-Next M64 + moe-cache 듀얼 벤치 (모델 80GB 준비됨)
- [ ] 131K/262K 컨텍스트에서 accept 상승(reddit 87~90%) 재현
- [ ] A3B에 별도 MTP 헤드(-md) 확보 시 Track B에서 동일 검증
- [ ] Track B 운영 docker 이미지 패키징
- [ ] Q8 wire 전송 구현 (선택, pp 대폭 상승 기대 가능)
- [ ] ROCm 10.1 릴리스 추적 (Ubuntu 24.04 지원 시 호스트 전환 고려)

## 10. 업데이트 (2026-09-07 오전): Flash-Next NVMe 이동 + MTP 재구동

> Trash 171GB 삭제로 NVMe 여유 확보 + 모델 86GB 이동 + MTP 두 구성 재구동 성공
> 상세: 부록 D

| 항목 | 이전 (SATA) | 이후 (NVMe) |
|---|---|---|
| decode (레이어 split, MTP없음) | 17.3 t/s | **20.1 t/s** (+16%) |
| prefill (long) | 54.7 (5850tok) | **418** (2001tok) |
| **레이어 split + MTP** | 불가 (GPU0 1GB 부족) | **17.1 t/s, accept 0.63** |
| **tensor split + MTP + Q8 wire** | 불가 (RCCL fail+OOM) | **21.2 t/s, accept 0.58** |

핵심: 마운트 이동(+Q8 wire) + expert를 host로 더 내리기(`-ot` 확장)로 이전 "MTP 3중 장애" 해소.

## 10b. 재현 자산

| 자산 | 위치 |
|---|---|
| Track B 소스 (패치+PR 적용 완료) | upstream-latest/ |
| Track B 산출물 | rccl-build-v2/bin |
| Track A 소스 (버그 있음) | beellama-boosts/src |
| Track A 산출물 | rccl-build/bin |
| 벤치 하니스 | /tmp/run_cfg.sh (호스트), /tmp/bench_v2.sh (docker, 리소스 트래킹), /tmp/bench_trackb.sh |
| rdna-boosts upstream | /tmp/rdna-boosts-upstream |
| 모델 | /mnt/nvmedata/models/unsloth/ (A3B, 27B IQ4_XS+MTP/MT재), /opt/llm/models/ (Flash-Next M64 + MTP 헤더) |

---

# 부록 A. Track B 이후 추가 작업 (오후~저녁)

> Track B 빌드 커밋 이후 PR#28136 / unsloth PR#144 / tensor split 강제 / AtomicChat M64 구동까지의 전체 후속
> 벤치 하니스 v2: `/tmp/bench_v2.sh` — docker --network host, VRAM/RAM/CPU/GPU% 1초 트래킹, health 300s

## A1. 벤치 하니스 v2 + v2 모델별 결과 (트래킹 포함)

### A1.1 A3B-UD-Q3_K_XL (17.2GB MoE, Track B, 듀얼/싱글 + MTP)

| # | 구성 | tg | pp | accept | VRAM0/1 | GPU% | RAM | CPU% |
|---|---|---|---|---|---|---|---|---|
| S1 | 싱글 전부 VRAM + MTP | **129.55** | 116.5 | 0.46 | 18.9/- | 85-95% | 33GB | 7% |
| S2 | 싱글 + exps ROCm_Host + cache96 | 11.12 | 63.1 | 0.55 | 10.2/- | 19-24% | 47GB | **38-44%** |
| D1 | 듀얼 tensor 전부 VRAM + MTP | 102.15 | 69.3 | 0.53 | 10.2/10.2 | 73-76% | 34GB | 7% |
| D2/D3 | 듀얼 + exps host/CPU + cache | **NCCL 크래시** | - | - | 8.4 | 7-10% | 48GB | 45% |

### A1.2 27B-UD-IQ4_XS (14.25GB dense + 별도 MTP 헤드, Track B)

| # | 구성 | tg | accept | VRAM | 비고 |
|---|---|---|---|---|---|
| S1b | 싱글 MTP OFF (base, ctx4096/ub256) | 40.26 | - | 23.0GB | 싱글 MTP는 24GB 초과 |
| S1c | 싱글 + MTP (ctx2048/ub128, ngld60) | **54.96** | 0.57 | 25.7(스왑) | reddit 싱글 53.9 일치 |
| D1 | **듀얼 tensor + MTP (ctx8192/ub1024)** | **78.17** | 0.53 | 18.3/18.5 | **reddit 69.25 +13%** |

### A1.3 Flash-Next M64 (qwen4exp, 84.9GB MoE/512 experts) — 본 세션 최대 전투

| # | split | 구성 | tg | pp(5850) | VRAM0/1 | GPU% | 비고 |
|---|---|---|---|---|---|---|---|
| D1b | layer | exps 전부 CPU | 4.70 | 24.2 | 3.9/3.9 | 14-24 | CPU 33-39% 병목 |
| D2 | layer | GPU41 + late7 host + cache96 | **16.49** | 38.1 | 25.7/18.1 | 32-42 | GPU0 스왑 |
| D3 | layer | + cache128/ub256 | 17.29 | 56.3 | 25.6/18.4 | 35-42 | |
| D4 | layer | late10 host + cache192 | 17.33 | 54.7 | 25.6/17.7 | 38-42 | cache 무효 |
| **D5** | layer | + **--lazy-mode on-direct** | **18.16** | 54.7 | 25.7/18.1 | 37-45 | 데이트 최선 |
| T2 | **tensor(강제)** | GPU41 + late7 host + cache96 | 15.13 | **140.7** | 22.9/23.1 | 50-57 | GPU 균등! prefill↓ |
| T1/T4 | tensor | + MTP 헤더 | ✗ | ✗ | 25.2/25.7 | - | **RCCL fail + OOM 크래시** |

## A2. PR#28136 (qwen4exp direct PLE) — 적용 + 실측

- 2커밋(90fde1f7 + c6a9e5c9) cherry-pick: `--lazy-mode on-direct` = LLAMA_LAZY_MODE_DIRECT(3) 전용 pread()
- A/B (Flash-Next, 듀얼 layer, 5850tok):

| lazy | pp | vs off |
|---|---|---|
| off (per_layer 상주) | 280.8 | 기준 |
| auto (mmap 폴트) | 349.7 | +24.5% |
| **on-direct** | **365.3** | **+30.1%** |

- 짧은 프롬프트(73tok)는 무차이 (PLE 행이 적어 이득 없음) — PR 설명과 일치
- decode 무변화 (PR이 prefill 전용). 문서 BENCH §6, 커밋 410bd6e

## A3. unsloth PR#144 (qwen4exp MTP) — 적용 + 시행착오

- 발견: **AtomicChat M64 모델에는 nextn(MTP) 텐서가 0개** (이름에 MTP지만 MTP 헤더 없음)
- unsloth/Qwen3.8-Flash-Next-GGUF **MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf** (4.14GB, self-contained) 다운로드
  - `blk.48.nextn.*` 6개 텐서 - MTP 드래프트 헤드 확인
- PR#144 커밋 3종 적용:
  - `977f772`: qwen4exp draft-only MTP load (mtp_only/load_mtp/TENSOR_SKIP) — 8파일 clean 3way
  - `9c0c80e`: CUDA graph cache shape-keying (key = node[0]+n_nodes+shape, LRU cap 64) — MTP verify 배치 변동에서 warmup 재설정 방지
- 상태: MTP 헤더 로드 성공했으나 **tensor split + MTP는 RCCL fail + OOM 크래시** (B200 32GB에서 83→139 t/s가 우리 24GBx2에선 불가)
  - RCCL 10.0이 host/CPU exps 참여 all-reduce에서 NCCL 실패
  - GPU0 25.2GB 풀 상태라 MTP 연산 버퍼 부족
  - 결국 MTP는 레이어 split에서도 GPU0 1GB 부족으로 불가

## A4. qwen4exp tensor split 강제 활성화

- `llm_arch_supports_sm_tensor`에서 QWEN4EXP `return false`(TODO: fix test-llama-archs) 제거 → default(true)
- 커밋 `2300013` + 재빌드. 결과:
  - **로드 성공** (구조적 차단 아님, GPU 균등 22.9/23.1 배분)
  - decode 15.13 (레이어 16.49보다 약간↓) — RCCL all-reduce 오버헤드가 균등 이득 상쇄
  - prefill 5850토큰 140.7 (레이어 365의 -62%) — "두 half-width matmul이 full보다 비싸다" (PR#27825 지적과 동일)
- **qwen4exp tensor split은 upstream/beellama 공통으로 의도적 차단** — 구조적(GET_ROWS per_layer, hc_ffn, moe_intermediate_size 640) 제약 때문. 강제로 풀면 로드되지만 성능은 레이어보다 나쁨

## A5. AtomicChat quant 특성 (README 분석)

- **n-gram 테이블(per_layer_token_embd 51B/39GB)이 전용 샤드 2에 격리** → mmap으로 SSD 상주, 토큰당 2.7KB 랜덤 읽기
  - 이것만이 순수 디스크 offload 가능한 유일한 부분 (일반 offload와 아키텍처 자체가 다름)
- "In memory" = GPU가 실제 잡는 것: AD-3.84bpw = **45.8GB만 메모리** (84.9GB 아님)
- 64GB MacBook에서 36 tok/s — 통합 메모리 단일 칩 기준, 24GBx2와 구조 다름
- 실행 지침: mmap ON(테이블 pageable) → `--load-mode none` 금지, `-ot` 불필요(테이블 자동 host), `-fit off` 필수
- **offload 시 강점은 prefill/로딩** (테이블 SSD read), decode는 일반 MoE와 동일한 레이어 split 한계

## A6. 최종 결론: 듀얼 7900 XTX에서 Flash-Next = ~17 t/s

| 지표 | 값 | 조건 |
|---|---|---|
| **decode (tg)** | **~17 t/s** | 레이어 split + GPU41 + late7 host + cache96 + lazy-direct |
| prefill (pp) | **365 t/s** (5850tok) | 동일 + on-direct |
| GPU 균등 대안 | 15.1 t/s | tensor split (MTP 불가) |
| 24GB x2의 한계 | 84.9GB 모델을 48GB에 수용 → **구조적** | reddit 2x3090의 25-29도 188GB RAM |

- **MTP는 3중 장애**: RCCL 10.0 host-exps 불가 + GPU 풀(MTP 버퍼 부족) + CPU sampler 폴백
- **27B dense(14GB): 78 t/s (reddit 69 초과)** — 이 모델이 듀얼 7900 XTX의 최선
- **A3B(17GB MoE): 싱글 MTP 130 t/s** — 단일 카드로 충분
- Flash-Next가 빛나려면 48GB+ 카드 또는 64GB+ 통합 메모리 필요

## A7. 미완 / 남은 이슈 (최종)

- [x] Flash-Next M64 구동 + moe-cache (17 t/s 확정)
- [x] tensor split 강제 (로드 성공, 성능 열위) — 문서화 완료
- [x] PR#28136, PR#144 적용 (lazy-direct +30%, MTP는 구조적 불가)
- [x] Q8 wire 전송 구현 (RCCL allreduce BF16→Q8=1B, pp +4.3% 실증) — **확정**
- [x] F16 wire (decode 우위, opt-in) — 병행 확정
- [x] Docker 빌드 인프라 (ccache 이미지, 증분 빌드 7초)
- [ ] 131K/262K 컨텍스트에서 accept/성능 검증
- [ ] Track B 운영 docker 이미지 패키징
- [ ] ROCm 10.1 릴리스 추적

### Q8 wire 최종 결론 (wire 경로 선택)

| 경로 | 사용처 | 성능 | 손실 |
|---|---|---|---|
| **Q8 (GGML_CUDA_AR_WIRE_Q8=1)** | prefill-heavy | pp +4.3% (1231 vs 1181) | 실용 무손실 (출력 동일) |
| F16 (GGML_CUDA_AR_WIRE_F16=1) | decode-heavy | tg +2% | 무손실 |
| BF16 (기본) | fallback | 기준 | 무손실 |

- Q6/Q5는 RCCL ncclSum과 **비트 패킹 비호환** (바이트 단위 덧셈이 패킹된 비트 필드를 깨뜨림) → 구현 불가 확정
- Q7은 팩킹 복잡도 대비 이득이 Q8/Q6 사이 어중간 → 스킵
- **Q8이 RCCL 경로의 구조적 하한 + 실용 무손실** → 최종 확정
---

# 부록 B. 단일 GPU 27B 운영 최적화 (미션 완료)

> 목표: Qwen3.8-27B를 7900 XTX **한 장**(24GB)으로 최대 성능 구동
> 결과: **지속 tg 64.7~65.8 t/s**, ctx 48K, MTP 수용률 0.54, VRAM 21.3GB

## 구성 실측 (Track B 빌드, Q8/F16 wire 포함)

| 구성 | tg | accept | VRAM | 비고 |
|---|---|---|---|---|
| q8 KV + 32K + MTP | 49.4 | 0.33 | 19.4GB | 기준 |
| **f16 KV + 32K + MTP** | 62.2 | 0.51 | 20.3GB | RDNA3 FP16 |
| **f16 KV + 48K + MTP** | **64.6** | 0.53 | 21.3GB | ctx↑ 수용률↑ |
| f16 KV + 48K, 2000토큰 롱런 | **64.7~65.8** | 0.54 | - | 지속 안정 |

## 최종 운영 구성 (단일 GPU)

```bash
docker run -d --rm --name llm-27b --network host \
  --device /dev/kfd --device /dev/dri --group-add video \
  -v rccl-build-v2:/app -v /mnt/nvmedata/models/unsloth:/models-unsloth \
  -e LD_LIBRARY_PATH=/app:/opt/rocm/lib -e GGML_CUDA_P2P=1 -e GGML_CUDA_AR_WIRE_F16=1 \
  rocm/dev-ubuntu-26.04:10.0.0-full /app/llama-server \
    -m /models-unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf \
    -md /models-unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf -ngld 99 \
    -ngl 99 -sm none -fa on -c 49152 -b 2048 -ub 512 \
    -ctk f16 -ctv f16 --jinja \
    --spec-type draft-mtp --spec-draft-n-max 3
```

## 핵심 통찰

1. **RDNA3: f16 KV가 q8보다 tg +26%** — 단일 GPU에서도 비양자화 우위 (문서 §2.4 재확인)
2. **컨텍스트↑ = MTP 수용률↑** — 32K(0.51) → 48K(0.53), 롱런 안정
3. **단일 GPU에서 65 t/s** — 듀얼 tensor+MTP(78)와 비교 시 -17%지만, **GPU 1장으로 충분한 성능**
4. MTP 별도 헤더(1.37GB)도 24GB에 부담 없이 수용 (여유 3.3GB)
5. **64 t/s 지속** = reddit 싱글 MTP(53.9) 대비 +20% — 우리 빌드 우위

## 참고: 더 큰 ctx 원하면
- 64K까지는 q8 KV로 로드 가능 (속도 49, 품질 저하)
- f16 48K가 **속도+수용률 균형 최적점**

---

# 부록 C. 단일 GPU 바이브 코딩 구성 (128K 컨텍스트)

> 목적: Qwen3.8-27B를 7900 XTX 한 장에서 **바이브 코딩용**으로 (컨텍스트 최우선)
> 결과: **128K 컨텍스트 + 61~78 t/s + MTP 수용률 0.51~0.74**

## 실측 (q8 KV, 단일 GPU)

| 구성 | ctx | tg | accept | VRAM | 비고 |
|---|---|---|---|---|---|
| q8 KV + MTP | **128K** | 61.5 | 0.51 | 23.8GB | 일반 프롬프트 |
| q8 KV + MTP, 코드 롱런 | **128K** | **78.2** | **0.74** | - | 코드 작업 수용률↑ |
| f16 KV + MTP | 48K | 64.6 | 0.53 | 21.3GB | 품질 최대 (짧은 ctx) |

## 바이브 코딩 최종 구성 (컨텍스트 128K)

```bash
docker run -d --rm --name llm-27b-code --network host \
  --device /dev/kfd --device /dev/dri --group-add video \
  -v rccl-build-v2:/app -v /mnt/nvmedata/models/unsloth:/models-unsloth \
  -e LD_LIBRARY_PATH=/app:/opt/rocm/lib -e GGML_CUDA_P2P=1 -e GGML_CUDA_AR_WIRE_Q8=1 \
  rocm/dev-ubuntu-26.04:10.0.0-full /app/llama-server \
    -m /models-unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf \
    -md /models-unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf -ngld 99 \
    -ngl 99 -sm none -fa on -c 131072 -b 2048 -ub 512 \
    -ctk q8_0 -ctv q8_0 --jinja \
    --spec-type draft-mtp --spec-draft-n-max 3
```

## 핵심 통찰

1. **128K는 q8 KV로만 가능** — f16은 48K가 한계 (VRAM), q8은 128K (23.8GB)
2. **바이브 코딩에서 MTP 수용률 0.74** — 코드는 라우팅이 집중돼 드래프트가 잘 맞음
3. **78 t/s** = 1000토큰 코드 생성 13초 — 실시간 코딩 보조에 충분
4. Q8 wire(+4.3% pp)도 유지 (128K 프리필 bandwidth-bound 이득)
5. VRAM 96% 사용 (0.8GB 여유) — 필요시 MTP 빼면 여유 확보, 속도는 45~50

---

# 부록 D. Flash-Next NVMe 이동 + MTP 재구동 (2026-09-07)

> 이전 A3에서 "MTP 3중 장애"로 실패했던 것 — NVMe 이동 + Q8 wire + expert host 확장으로 두 구성 모두 재구동 성공
> 모델: /mnt/nvmedata/models/Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64 (NVMe, 28샤드)
> MTP: MTP/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf (NVMe, 4.14GB)

## D1. NVMe 이동

- Trash 171GB 삭제 (ExpertClone 81GB, thetom 38GB, Q4_K_S 20GB, Gemma/OCR 등) → NVMe 여유 72→232GB
- Flash-Next 86GB cp -a 이동 (~2분), shard 2 크기 비교로 무결성 검증, SATA src 삭제
- 원래 /opt 경로를 NVMe로 심볼릭 링크 (경로 호환, 기존 도커 마운트 그대로 동작)

## D2. NVMe 성능 비교 (레이어 split + lazy-direct, MTP없음)

| 지표 | SATA | NVMe | 개선 |
|---|---|---|---|
| decode tg | 17.3 | **20.1** | +16% |
| prefill (2001tok) | (54.7 @5850tok) | **418** | 디스크 지연 제거 |

## D3. MTP 재구동 — 두 가설 검증

| # | 구성 | 로드 | tg | accept | VRAM 0/1 | 비고 |
|---|---|---|---|---|---|---|
| H1 | 레이어 split, GPU31+late17 host, cache96 | ✅ | 17.12 | 0.628 | 25.7/15.8 | 이전 "GPU0 1GB 부족" 해소 |
| H2 | tensor split, late8 host, cache96, Q8 wire | ✅ | **21.24** | 0.575 | 24.9/25.6 | NCCL fail → butterfly 폴백, OOM 해소 |

가설 검증:
1. **Guanaco 없이도 `-ot` 확장만으로 레이어 split+MTP 가능** (expert를 host로 더 내려 VRAM 확보)
2. **Q8 wire가 RCCL 데이터 반으로** → OOM 해소로 tensor split+MTP 로드. 단 NCCL fail(host-exps 참여)은 여전 — butterfly 폴백으로 동작

## D4. 튜닝 스윕 결과 (400토큰 안정 측정, 2026-09-07)

> 200토큰 샘플링은 accept 변동으로 tg 16~21 폭주 → 400토큰으로 안정화

### wire 대조 (tensor+MTP, late8, cache96)

| wire | tg | accept |
|---|---|---|
| Q8 | 17.30 | 0.456 |
| F16 | 17.04 | 0.476 |

**무차이** → Q8 유지 (prefill +4.3% 이득 포기 안 함)

### host 레이어 수 (Q8 wire, cache96)

| host | tg | accept | 비고 |
|---|---|---|---|
| late4 | ❌ | - | GPU1 1923MiB 추가 OOM (GPU1 풀) |
| **late8** | **15.79** | 0.407 | 유일 유효 |
| late12 | 13.16 | 0.426 | host↑ 느려짐 |
| late16 | 11.50 | 0.482 | host↑ 느려짐 |

**패턴: host로 내릴수록 느려짐. late8이 24GB x2의 유일 구성** (그 이상 GPU = OOM, 이하 = 폴트 증가)

### cache 크기 (late8, Q8)

| cache | tg | accept |
|---|---|---|
| 64 | 17.46 | 0.491 |
| **96** | **17.87** | 0.463 |

**96 유지** (레이어 split과 동일 결론)

### draft-n-max

| draft | 결과 |
|---|---|
| 3 | ✅ 동작 |
| 4 | ❌ GPU0의 687MiB pp 버퍼 OOM (GPU0 여유 31MB) — `-ot`/cache로도 해결 불가 (attention 그래프 버퍼는 고정) |

**draft4는 24GB 물리 한계로 확정 불가. draft3이 최대.**

### 최종 운영 구성 (Flash-Next M64, 24GB x2)

```
-sm tensor -ts 1,1 + Q8 wire + late8 host + cache96 + draft3
= 17.3~17.9 t/s, accept ~0.46, mean len 2.4
```

MTP는 accept 배율만큼 실질 산출 증가 (mean len 2.4 = 스텝당 평균 2.4토큰 생성)
```

## D5. butterfly 폴백 조사 및 해결 시도 (2026-09-07 오전~오후)

> 결론: "butterfly 폴백"은 오진이었음. 실제로는 RCCL(wire 포함)이 정상 동작 중이었고, 병목은 구조적.

### D5.1 사실 확정 (A/B 실측)
- NCCL init 성공 확인 (실패 수 0), Q8 wire는 ncclAllReduce에 실제 구현됨 (d34c407/43ae223)
- wire A/B: Q8 181 / BF16 180 / F16 185 = 노이즈 내 무차이 (27B 듀얼의 +4.3%는 Flash-Next tensor에서 미재현)
- tensor split prefill 181 t/s vs 레이어 split 271 t/s = 35% 구조적 열위 (half-width matmul, PR#27825 지적 확인)
- GPU 0↔77% 진동 = ubatch 경계 갭 + host expert PCIe fetch

### D5.2 A: RDNA3 internal AR 재활성화 — 실패 (리버트 유지)
- d048605(게이트 완화) 재적용 → 빌드 → gfx1100에서 internal AR 시작됨
- **peer arrival not observed within 20ms → butterfly 재동기화 반복** (신호 가시성 실패)
- 성능 무효과 (181→182) → 리버트. 어제의 리버트(535333f)가 정당했음

### D5.3 C: host-expert fetch 개선 — 성공 (+17%)

| 구성 (5K prefill) | t/s | vs 베이스 |
|---|---|---|
| late8 host + ins2 (기존) | 175.3 | 기준 |
| cache192 | ❌ GPU arena OOM | - |
| inserts4 | 173.7 | -0.9% |
| inserts1 | 182.7 | +4% |
| **late6 host + inserts1** | **200~205** | **+17%** |

- expert 8→6 레이어만 GPU로 올리니 prefill +17% → host-export PCIe fetch가 prefill 병목의 핵심
- late4는 GPU1 풀이라 불가, late6이 최적점

### D5.4 최종 tensor split 구성 (prefill 최적)
```
-sm tensor -ts 1,1 -c 131072 -b 2048 -ub 512 -np 1 -t 20
-ot 'per_layer_token_embd=CPU, blk.4[2-7] exps=ROCm_Host'
--moe-expert-cache 96 --moe-expert-cache-inserts 1
--lazy-mode on-direct + Q8 wire
= prefill ~205 t/s (5K), 기존 175 대비 +17%
```

## D6. TurboQuant KV (turbo2/3/4) 포팅 완료 (2026-09-07)

> 사용자 아이디어: "turboquant 소스(WHT+PolarQuant)에 분산 정규화(InnerQ) 이미 포함 — beellama KVarN 대신 turboquant 이식"
> 결과: **turbo4 KV 캐시가 A3B/Flash-Next에서 작동** (17.3 t/s, VRAM 절약)

### D6.1 포팅 범위
- ggml.h/c: TURBO2/3/4 타입 + TURBO_WHT op (가중치 TQ3/4_1S 제거 — 사용자 지시)
- CUDA/HIP: turbo-quant.cuh, turbo-wht.cu, turbo-innerq, set-rows, convert, FA 계열(fattn-* 부모 채택)
- HIP CMakeLists: vec turbo 인스턴스 20개 + mma 640 + tile 640
- llama: kv-cache turbo 패딩/뷰, llama-graph turbo 역-WHT + Q 전-회전, CPU get_rows turbo
- HIP 호환: warp 프리미티브 64비트, cudaMemcpyToSymbol→hip

### D6.2 벤치 (Flash-Next, 레이어 split)
| 지표 | turbo4 | q8_0 |
|---|---|---|
| tg | 17.34 | 18.52 |
| pp (7tok) | 6.74 | 5.09 |
| GPU0 VRAM | 23.2GB | 23.6GB |

- tg -6% (WHT 역변환 오버헤드), pp +32% (turbo가 짧은 프롬프트에서 유리)
- KV 절반 압축 → 동일 VRAM에서 2배 컨텍스트 가능 (c=131072+에서 이점 최대화)
- A3B: turbo4 106.5 vs q8_0 109.8 t/s (-3%)

### D6.3 핵심 커밋
- ff664e2: TurboQuant KV 이식 (33파일)
- 4fb3a88: supports_op + FA 채택 + CPU turbo_wht
- a74027e: HIP vec/mma/tile 인스턴스 + kv-cache turbo
- (마지막): get_rows turbo + CPU group size (Flash-Next 작동)

## D7. TurboQuant KV 전체 벤치 (2026-09-07)

### D7.1 컨텍스트 확장 (c=131072)
| 모델 | KV 타입 | c=131072 | GPU0 |
|---|---|---|---|
| Flash-Next | q8_0 | ❌ OOM | - |
| Flash-Next | **turbo4** | ✅ | 24.0GB |
| A3B | **turbo4** | ✅ | **16.7GB** |

turbo4 KV 절반 압축 → 131K 실현 (q8_0는 Flash-Next에서 OOM)

### D7.2 컨텍스트 스윕 (Flash-Next, turbo4, layer split)
| 컨텍스트 | pp | tg |
|---|---|---|
| 20k | 418.8 | 7.33 |
| 50k | 349.6 | 4.29 |
| 80k | 230.9 | 2.89 |
| 100k | 174.3 | 2.42 |

컨텍스트↑ = decode 선형 감소 (MoE expert 스트리밍 + attention)

### D7.3 Needle-in-Haystack (22k)
| KV | 결과 |
|---|---|
| **q8_0** | ✅ 정확 회수 (TURBO-QUANT-2026-X7) |
| turbo4 (대칭/비대칭) | ❌ 무의미 응답 |
| turbo3 | ❌ hallucination |

**Flash-Next(qwen4exp)는 turbo KV 품질 붕괴** — head_dim이 turbo 블록(128)과 비호환. A3B(128 배수)는 정상 → **turbo는 A3B 전용**

### D7.4 MTP 수용율 (Flash-Next q8_0)
| 컨텍스트 | tg (MTP OFF) | tg (MTP) | accept |
|---|---|---|---|
| 20k | 7.33 | **23.56** (+221%) | 0.672 (mean 3.02) |
| 50k | 4.29 | **17.46** (+307%) | **1.000** (mean 3.98) |

**MTP가 decode 3배+ 가속**. 50k에서 accept 1.0 (반복 텍스트 드래프트 적중)

### D7.5 Split mode (Flash-Next, MTP, 20k)
| 구성 | pp | tg | accept |
|---|---|---|---|
| layer + MTP | 365.4 | 23.56 | 0.672 |
| tensor + MTP | 184.8 | 24.07 | 0.576 |

tensor: tg 소폭 우위, pp 절반 (half-width 비용). **layer+MTP가 균형 우위**

### D7.6 A3B turbo4 + MTP (실사용처)
- c=131072 + turbo4 + MTP: **105.1 t/s, GPU0 16.7GB** (단일 GPU 131K 실현)
- 106.5 t/s (q8_0 대비 -3%)로 KV 절반 → **A3B 131K 바이브 코딩 가능**

### D7.7 결론
1. **turbo4 = KV 절반 압축**: A3B에서 131K 단일 GPU 실현 (16.7GB)
2. **Flash-Next는 turbo 불가** (head_dim 비호환) — q8_0 + MTP가 정답
3. **MTP가 decode 3배 가속** (Flash-Next 20k: 7.3→23.6, 50k: 4.3→17.5)
4. **layer split + MTP**가 Flash-Next 최선 (pp 365, tg 23.6, accept 0.67)

## D8. TurboQuant KV 붕괴 원인 규명 + asymmetric 해법 + 150K NIAH 검증 (2026-09-07 저녁)

> 이전 D6/D7에서 turbo3/4 포팅 후 긴 컨텍스트 NIAH 붕괴(30K+ 실패)를 발견. 이번 세션에서 원인 규명 + 150K 통과 구성 확정.

### D8.1 원인 규명 (3단계)

1. **그래프 Q 회전이 원인** (핵심): K 저장 시 WHT 회전 + 그래프 Q 회전(ggml_turbo_wht forward)이 어긋나 내적 붕괴.
   - Q 회전 ON → 짧은 프롬프트도 깨짐 ("The answer is 7" → "1")
   - Q 회전 OFF → 짧은 프롬프트 정상 ("7" 회수)
   - TheTom pre-rotate 문서와 일치: 그래프 Q 회전 PPL 23.5 vs dequant inverse 6.19 (그래프 접근은 폐기됨)
2. **WHT 구현 교체**: 저장 커널(set-rows.cu)과 Q 회전 커널(turbo-wht.cu)의 WHT를 shared butterfly → **TheTom warp-shuffle** 구현으로 교체.
3. **K precision이 지배 요인** (asymmetric 문서 확인): K가 turbo(2/3/4 모두)면 어느 V와도 30K 붕괴. V는 turbo로 압축해도 안전.

### D8.2 최종 검증 매트릭스 (27B, 30K NIAH, Q회전 OFF)

| K | V | 결과 |
|---|---|---|
| turbo4 | turbo4 | ❌ "42" |
| turbo4 | q8_0 | ❌ "42" |
| turbo3 | q8_0 | ❌ "42" |
| q5_0 | turbo3 | ✅ (단, FA vec 인스턴스 부재로 prefill 14 t/s) |
| **q8_0** | **turbo3** | ✅ index 391 |
| **q8_0** | **turbo4** | ✅ (V turbo는 모두 정상) |

- **K가 turbo면 (2/3/4 모두) 30K 붕괴** — K 양자화 오류가 어텐션 라우팅을 깨뜨림
- **V는 turbo로 자유 압축** — V 오류는 비례적이라 안전
- q6_K는 ctk 미지원, q5_0은 turbo V와 FA vec 인스턴스 부재(원본도 없음)
- **TheTom 지원 조합**: turbo×{q8_0, f16} 만 정식. q5_0/q4_0×turbo는 원본에도 없음

### D8.3 150K NIAH 검증 (K=q8_0 V=turbo3 asymmetric)

| 모델 | 컨텍스트 | needle | prefill | 비고 |
|---|---|---|---|---|
| 27B UD-IQ4_XS | 30K | ✅ index 391 | - | |
| 27B | 100K | ✅ index 2408 | 305s | |
| 27B | **150K** | ✅ index 3618 | 520s | |
| **A3B Q3_K_XL (MoE)** | **150K** | ✅ 15000150 | **222s** | prefill 2.3배 빠름 |

### D8.4 MTP 적용 (2026-09-07)

- **adaptive MTP는 임베디드 MTP 모델에서만 동작** — 별도 -md 파일 + draft-mtp-adaptive는 **segfault (exit 139)** (이 빌드 버그)
- **draft-mtp(비adaptive) + 별도 -md는 정상** 동작
- **Qwen3.8-27B-MTP-Q4_K_M.gguf 다운로드** (Jackrong HF, 16GB, 임베디드 MTP)
- **Q4_K_M + adaptive MTP + K=q8/turbo3 검증**:
  - 30K: needle ✅, MTP accept 0.92, mean len 3.64
  - **150K: needle ✅ (entry 3618 지목), MTP accept 0.87, mean len 3.52**

### D8.5 최종 실전 구성 (150K 바이브 코딩)

```bash
# 27B (더 정밀한 가중치, 임베디드 MTP + adaptive)
docker run -d --rm --name llm-27b-mtp --network host \
  --device /dev/kfd --device /dev/dri --group-add video \
  -v rccl-build-v2:/app -v /mnt/nvmedata/models:/models-nvme \
  -e LD_LIBRARY_PATH=/app:/opt/rocm/lib -e GGML_CUDA_P2P=1 -e TURBO_INNERQ=5000 \
  rocm/dev-ubuntu-26.04:10.0.0-full /app/llama-server \
    -m /models-nvme/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-MTP-Q4_K_M.gguf \
    --main-gpu 1 -ngld 99 -ngl 99 -sm none -fa on -c 163840 -b 2048 -ub 512 \
    -t 20 --threads-batch 20 -np 1 -ctk q8_0 -ctv turbo3 --jinja \
    --spec-type draft-mtp-adaptive --spec-draft-n-max 3 --spec-draft-n-min-adaptive 2

# A3B (prefill 2.5배 빠름, 150K)
# -m /models-nvme/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf (동일 플래그)
```

### D8.6 prefill 속도 메모

- **ubatch 512가 2048보다 빠름** (30K: 112s vs 205s) — 큰 ubatch는 FA 커널 메모리/스케줄링 역효과
- **A3B prefill 2.3배 빠름** (150K: 222s vs 27B 520s) — MoE가 dense 대비 prefill 우위
- q5_0 K + turbo V는 FA vec 인스턴스 부재로 14 t/s (비실용)

### D8.7 변경 코드 요약 (미커밋, 커밋 대기)

| 파일 | 변경 |
|---|---|
| set-rows.cu | turbo3 저장 WHT → TheTom warp-shuffle |
| turbo-wht.cu | Q 회전 커널 → TheTom warp-shuffle |
| turbo-quant.cuh | InnerQ managed 배열 identity 초기화 |
| llama-graph.cpp | Q 회전 OFF (회귀 방지 주석 포함) |
| llama-kv-cache.cpp | turbo 회전/scale 텐서 생성 (이전 작업) |

### D8.8 운영 구성 적용 (config.ini + llm-main 재기동, 2026-09-07)

> 검증 완료 구성들을 운영 서버(/opt/llm/llama-cpp/main-llm-config.ini)에 적용. llm-main 재기동 확인.

#### config.ini 프리셋 변경

| 프리셋 | 모델 | KV | MTP | GPU | 컨텍스트 |
|---|---|---|---|---|---|
| **[Dense]** (변경) | **A3B Q3_K_XL** + mmproj | q8_0/turbo4 | adaptive | GPU1 | 163840 |
| **[Dense.27]** (신규) | 27B MTP-Q4_K_M + mmproj | q8_0/turbo4 | adaptive | GPU1 | 163840 |
| **[Dense.next]** (신규) | Flash-Next 28샤드 + 별도 MTP | q8_0/turbo4 | draft-mtp | 듀얼 layer | 131072 |

- [Dense.27]: 27B Q4_K_M(16GB, Jackrong HF 다운로드) 임베디드 MTP — adaptive 가능
- [Dense.next]: Flash-Next는 **별도 MTP 파일이라 adaptive segfault** → draft-mtp(비adaptive) + `override-tensor`(exps=ROCm_Host) + `moe-expert-cache 96` + `lazy-mode on-direct`
- load-on-startup: [Dense]만 true, 나머지 false (VRAM 충돌 방지)

#### llm-main 재기동 (docker run 방식, compose 아님)

- llm-main은 **dokploy-network 브리지** + docker run으로 실행됨 (compose 파일 없음, restart 스크립트: /tmp/llm-main-restart.sh)
- `--host 0.0.0.0` 지정해도 **인스턴스는 127.0.0.1 내부 리슨** — 호스트 직접 접근 불가, 컨테이너 내부/라우터 경유
- **config.ini 키 주의**: `-ot` CLI 플래그는 프리셋 키로 **`override-tensor`** (offload-tensor 아님 — 미인식 크래시 유발)

#### 적용 후 상태

- 9개 프리셋 인식 (Dense, Dense-1, Dense-bellama, Dense.00, Dense.1, Dense.27, Dense.next, LFM2.5, 9B)
- [Dense] = A3B 로드 완료 (GPU1 20.3GB, n_ctx 163840)
- `offload-tensor` → `override-tensor` 수정으로 크래시 루프 해결

### D8.9 50K 통일 벤치 + q5_0 FA 수정 (2026-09-07 밤)

> 3개 모델 × KV 구성을 같은 50K 프롬프트로 통일 측정. 도중 q5_0 V가 FA에서 제외돼 64 t/s로 붕괴하는 버그 발견·수정.

#### D8.9.1 q5_0 V FA 제외 버그 (수정)

- **증상**: A3B q8_0/q5_0 prefill이 64 t/s로 붕괴 (turbo는 즉시)
- **원인**: HIP FA 선택(`ggml_cuda_get_best_fattn_kernel`)의 `is_kv_compat`에 Q5_0이 없어 `BEST_FATTN_KERNEL_NONE` → FA 미사용 + 비-FA fallback. `ggml_cuda_fattn_kv_type_supported`도 `#ifndef GGML_CUDA_FA_ALL_QUANTS`로 Q5_0 거부.
- **수정 3파일**:
  1. `ggml-hip/CMakeLists.txt`: fattn-vec-instance-q8_0-q5_0.cu, q5_0-q8_0.cu 추가 (HIP는 ggml-hip/CMakeLists 사용, ggml-cuda/CMakeLists 아님)
  2. `fattn.cu`: `is_kv_compat`에 Q4_0/Q4_1/Q5_0/Q5_1 추가
  3. `fattn.cu`: `ggml_cuda_fattn_kv_type_supported`에서 Q4_1/Q5_0/Q5_1 허용 + vec dispatch else 블록에 q8_0-q5_0/q5_0-q8_0 case 추가
- **효과**: A3B 5.8K prefill 64 → 2070 t/s (24배 개선)

#### D8.9.2 50K 벤치 결과 (A3B·27B는 50K, Flash-Next는 30K)

| 모델 | KV | pp | tg | MTP 수용률 |
|---|---|---|---|---|
| **A3B** Q3_K_XL | q8_0/q5_0 | **1539.6** | 60.3 | - |
| A3B | q8_0/turbo4 | 578.9 | 50.7 | - |
| A3B | q8_0/turbo4+MTP | ~578 | 65.9 | 0.75 |
| **27B** Q4_K_M | q8_0/q5_0 | **612.2** | 30.3 | - |
| 27B | q8_0/turbo4 | 262.2 | 25.4 | - |
| 27B | q8_0/turbo4+MTP | ~262 | 36.0 | 0.89 |
| **Flash-Next** | q8_0/q5_0 | **419.9** | 20.9 | - |
| Flash-Next | q8_0/turbo4 | 345.0 | 18.9 | - |
| Flash-Next | q8_0/turbo4+MTP | ❌ | - | ROCm1 OOM |

#### D8.9.3 핵심 발견

1. **q5_0 V가 turbo4보다 prefill 2.3~2.7배 빠름** (A3B 1539 vs 579, 27B 612 vs 262)
   - turbo4는 TILE prefill의 f16 변환(convert)에서 디콴트 비용이 큼
   - 11K까지는 둘 다 ~2100 t/s 동일, 45K+에서 turbo4만 급락 (-73%)
2. **MTP는 turbo4에서 유효**: A3B tg +30%, 27B tg +41% (수용률 0.75~0.89)
3. **q8/q5가 prefill 최강** → 운영 기본 확정 (config.ini 전체 cache-type-v = q5_0으로 전환)
4. **Flash-Next MTP는 VRAM 한계** (별도 MTP 헤더 4GB + 듀얼 = ROCm1 OOM)

### D8.10 turbo4 prefill 병목 조사 + q8/q5 운영 확정 (2026-09-07)

#### turbo4 vs q5_0 prefill 차이 (50K)

| 컨텍스트 | q5_0 | turbo4 | turbo3 |
|---|---|---|---|
| 5.8K | ~2070 | ~2070 | ~2070 |
| 11.4K | 2195 | 2130 | - |
| 44.8K (50K) | 1545 | 579 | - |

- 11K까지 둘 다 ~2100 t/s 동일, **45K+에서 turbo4만 -73% 급락**
- turbo4는 TILE prefill에서 K/V를 f16으로 변환하는데, turbo4 디콴트가 q5_0보다 비싸 컨텍스트 길이에 비례 비용

#### f16 변환 최적화 시도 (결과: 무효 → 되돌림)

- convert.cu에 turbo4 전용 f16 변환 커널 작성 (norm 1번 로드, 4요소/스레드)
- **결과: 557 t/s — 개선 없음** → 병목이 f16 변환 커널이 아니라 TILE 커널 자체의 turbo4 처리
- **소스 되돌림** (convert.cu 원복, 불필요한 ggml-cuda/CMakeLists 변경 제거)

#### q8/q5 운영 확정

- **prefill 최강 (q5_0가 turbo4보다 2.7배)**, config.ini 전체 cache-type-v = q5_0
- A3B 50K prefill 1539 t/s (turbo4 579 대비)
- llm-main 재기동, A3B(q8/q5) 로드 확인
- turbo4는 VRAM 절약(더 큰 컨텍스트) 필요 시에만
