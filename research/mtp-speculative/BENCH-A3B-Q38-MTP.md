# BENCH-A3B: Qwen3.6-35B-A3B-UD-Q3_K_XL MTP — single vs dual vs moe-cache 스윕

> 작성: 2026-09-06 (Asia/Seoul)
> 빌드: RCCL (gfx1100, ROCm 7.2.3, GGML_HIP_RCCL=ON, PR#28223 + PR#27861)
> 하드웨어: RX 7900 XTX 24GB x2 + i5-14600K + 96GB DDR4
> 모델: Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf (17.2GB, qwen35moe 41층, MoE 35B/3B active, MTP 헤드 내장 `blk.40.nextn.*`)
> 측정: llama-server 직접 구동 + 임의 프롬프트(123 tok) + max_tokens 300 x2회, slot print_timing + rocm-smi VRAM
> 배치: `-b 2048 -ub 512 -t 12 -ctk q8_0 -ctv q8_0 -c 8192 -fa on --jinja`

---

## 요약 (핵심 판정)

1. **이 모델은 한 카드(24GB)에 완전히 들어간다** (17GB) → **싱글 GPU가 최선**.
   듀얼 tensor split은 all-reduce(RCCL) 오버헤드만 추가해 **싱글보다 느림**.
2. **최고 속도 = 싱글 + MTP + 전부 VRAM: tg 116.8~135.4 t/s** (기준 대비 +21~40%).
   특히 **draft acceptance 0.81** — ROCm에서 MTP가 이렇게 잘 도는 건 이례적 (문서 §5.4의 기존 ~0.36 지식과 반대).
3. **exps 호스트 이동(-ot ROCm_Host) + moe-expert-cache**는 VRAM을 절반(16.2→8.1GB)으로 줄이지만
   **속도도 절반(96.5→45)** — 싱글에서 VRAM 절감 대가가 큼.
4. **듀얼 + exps 호스트 + cache = 최악**(17-19 t/s): CPU→GPU expert 로딩 + RCCL all-reduce 이중 병목.
5. VRAM을 아끼면서 속도를 유지하려면 **MTP가 정답**이 아니라, **각 구성의 VRAM↔속도 tradeoff**를 명확히:
   전부 VRAM이 항상 이김 (VRAM이 허락하는 한).

## 전체 데이터

| # | 구성 | VRAM | tg cold | tg warm | pp | 비고 |
|---|---|---|---|---|---|---|
| S1 | 싱글 전부 VRAM (기준) | 16.2GB | 96.5 | 96.3 | 305.8 | GPU1 단독 |
| S2 | 싱글 + exps ROCm_Host + cache96 | 8.1GB | 44.5 | 45.4 | 131.6 | VRAM -50% |
| S3 | 싱글 + exps ROCm_Host + cache64 | 6.4GB | 44.1 | 46.1 | 129.2 | VRAM -60% |
| S4 | 싱글 + exps ROCm_Host + cache128 | 9.8GB | 48.7 | 54.7 | 126.0 | cache↑속도↑ |
| **M1** | **싱글 + MTP + 전부 VRAM** | 16.8GB | **135.4** | **116.8** | 216.5 | accept 0.82 |
| M2 | 싱글 + MTP + cache128 | 10.4GB | 27.4 | 34.9 | 105.3 | accept 0.78 |
| M3 | 싱글 + MTP + cache96 | 8.8GB | 36.1 | 29.9 | 104.3 | accept 0.87/0.61 |
| D1 | 듀얼 tensor 전부 VRAM | 17.4GB | 61.4 | 60.9 | 199.8 | 싱글보다 ↓ |
| D2 | 듀얼 + exps 호스트 + cache96 | 14.1GB | 17.5 | 19.0 | 67.7 | 최악 |
| D3 | 듀얼 + exps 호스트 + cache128 | 17.5GB | 20.2 | 19.5 | 62.5 | |
| M4 | 듀얼 tensor + MTP 전부 | 18.1GB | 97.0 | 89.2 | 238.1 | 듀얼 최선, 싱글 MTP와 동급 |

VRAM은 두 GPU 합산 (듀얼은 카드당 절반).
tg cold/warm: slot print_timing eval time 기준 (1차/2차 completion).

## 해석

### MTP는 이 모델에서 무조건 켜라
- 싱글 전부 VRAM: 96.5 → 135.4 (MTP) = **+40%**
- draft acceptance 0.82는 ROCm MTP 평균(0.36)의 2.3배 — A3B 구조상 nextn 헤드가 잘 맞음

### exps 호스트 + moe-cache는 VRAM이 절실할 때만
- VRAM 절감은 확실 (-50~60%), 속도는 절반 이하로 떨어짐
- 캐시 슬롯 늘리면 개선: 64(46) < 96(45) ≈ 128(55) — 슬롯 128에서 점프
- MTP와 결합 시 오히려 악화 (업로드 작업과 MTP 검증 동시 경합)

### 듀얼은 이 모델에 불필요
- 모델이 한 카드에 들어가니 tensor split의 all-reduce가 순손실
- 듀얼의 유일한 이점(더 큰 컨텍스트/더 큰 quant)은 이번 벤치 범위 밖

## raw 데이터
- 서버 로그/응답: bench-results-a3b/ (cfg_*.log, run_cfg.sh)
- 커맨드 재현: /tmp/run_cfg.sh <tag> <server flags...> (HIP_DEV 환경변수로 GPU 선택)

## 다음 실험 (권장)
- Q4_K_S (20.9GB, 더 정확)로 M1 반복 — 같은 결론인지
- 더 긴 컨텍스트(32K+)에서 exps 호스트 + cache128 vs 전부 VRAM A/B — KV가 VRAM 압박할 때만 moe-cache 유효
- MTP + cache 64~192 슬롯 스윕 (VRAM 6~12GB 영역에서 최적점)

---

## 부록: RCCL 빌드 정상성 교차검증 (2026-09-06, 사용자 제기 의문)

### 빌드 무결성
- rccl-build 소스 = beellama-boosts/src (rdna-test 브랜치, 미커밋 boosts 변경 포함)
- `llama-graph.cpp` 수정(16:41:56) → 오브젝트 재컴파일(16:42:08) → libllama.so(16:42:31) — 최종 소스 반영 확인
- moe_cache 심볼 + kvarn SWA 옵션 + host buffer override(-ot) 전부 포함
- 27B dense 싱글 기준: 37.5 t/s = 오늘 낮(BENCHMARK-FINAL 34.6~38.8) 재현 → **정상**

### PR #27825 (HIP AllReduce) 검증 데이터와 대조
| 구성 | PR검증자(패치후, Q4/X6급) | 우리 RCCL (Q3급) |
|---|---|---|
| 27B 싱글 | 35.0~35.2 | 37.5 |
| 27B tensor | 39.9~40.2 | 42.2 |
| gemma-31B tensor | 35.0 | - |
| Muse-30B tensor | 48.9 | - |
| gpt-oss-20b tensor (Q8 MoE) | ~150 | - |

- PR 검증자 "tensor가 싱글 이김(패치 후) / 용량 작으면 싱글이 이김" = 우리 A3B 결론과 일치
- ub1024/--load-mode none은 27B tensor에서 무변화 (tg는 memory-bound)

### reddit 1200+/90+ 수치의 정체
- Q8_0 + 262K ctx + MTP + 특정 ubatch 스택의 극단값. Q4/Q6급에서는 35~49 t/s가 정상 범위.
- 우리 27B tensor+MTP(48.8~55.8)는 PR 검증자의 40.1 초과 → 정상 이상의 성능.
