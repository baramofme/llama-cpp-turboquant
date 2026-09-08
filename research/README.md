# research/ — 이식(feature port) 연구 문서 목차

7900 XTX x2 (gfx1100) + RCCL + rdna-boosts + TurboQuant + MTP 이식/최적화 작업의
연구 문서를 **이식할 feature 단위 + 날짜** 로 정리한 폴더.

> ⚠️ `주요 운영 구성` 표의 "사용중" 여부는 마지막 확인(2026-09-08) 기준.
> 하드웨어/빌드 상수: RX 7900 XTX 24GB x2 (gfx1100), i5-14600K(20T), 96GB DDR4.
> GPU-GPU P2P 물리 불가 → RCCL이 유일한 tensor AR 경로. 기반: theTom llama-cpp-turboquant fork.

---

## 1. 하위 폴더 목차 (feature 단위)

| # | 폴더 | 설명 | 사용중? | 사용 방법 |
|---|---|---|---|---|
| 1 | [turboquant-kv/](turboquant-kv/) | TurboQuant KV cache (turbo2/3/4, WHT+polar quant). K는 끝 V만 압축. | ✅ 사용중 (ctk=q8_0, ctv=turbo3/4) | `-ctk q8_0 -ctv turbo3` (A3B), `-ctv turbo4` (더 큰 ctx). K는 절대 turbo 금지 |
| 2 | [rccl-dual-gpu/](rccl-dual-gpu/) | RCCL all-reduce + rdna-boosts 13블록 + Q8/F16 wire + 듀얼 GPU. | ⚠️ 듀얼은 레이어 split 위주 / RCCL 상시 활성 | `-sm tensor`는 RDNA3에서 열위. wire: `GGML_CUDA_AR_WIRE_Q8=1` |
| 3 | [moe-expert-cache/](moe-expert-cache/) | moe-expert-cache (PR#27861): expert를 host offload + cache. | ✅ 사용중 (override-tensor exps=ROCm_Host, cache 96) | `--moe-expert-cache 96 --moe-expert-cache-inserts 1` |
| 4 | [mtp-speculative/](mtp-speculative/) | MTP speculative decoding + 긴 프롬프트 버그(beellama 한정). | ✅ 사용중 (draft-mtp / adaptive) | `--spec-type draft-mtp[-adaptive] --spec-draft-n-max 3` |
| 5 | [docker-build-infra/](docker-build-infra/) | 운영 이미지 빌드 + 벤치 하니스 + **SSE ping 튜닝**. | ✅ 사용중 (llm-main 컨테이너) | `IMAGE-BUILD-GUIDE.md`로 재빌드, `bench-beellama.sh`로 벤치 |
| 6 | [session-overview/](session-overview/) | 세션 종합/마스터 (위 feature들의 교차 정리 + 최종판). | — (참고용) | 개별 feature 계획 참조 시 우선 열람 |

---

## 2. 날짜별 변경/발견 표

> 날짜: Asia/Seoul. 각 항목은 해당 일에 발생/기록된 주요 작업·발견·결정.

### 2026-09-05

| 작업/발견 | 링크 | 간단 설명 |
|---|---|---|
| rdna-boosts 이식 계획 수립 | [rccl-dual-gpu/rdna-boosts-apply-plan.md](rccl-dual-gpu/rdna-boosts-apply-plan.md) | stew675 rdna-boosts 8블록 순차 이식 계획. RDNA3(gfx1100)에 0012 제외. |
| beellama v0.4.4/0.4.5 벤치 계획·결과 | [docker-build-infra/bench-beellama-plan.md](docker-build-infra/bench-beellama-plan.md) 외 | beellama vs tbqplus 대조 벤치. GPU1 단독. |
| 벤치 하니스 작성 | [docker-build-infra/bench-beellama.sh](docker-build-infra/bench-beellama.sh) | smoke/bench/server/batched 하위명령 하니스 스크립트. |

### 2026-09-06 (Track A → Track B 전환)

| 작업/발견 | 링크 | 간단 설명 |
|---|---|---|
| A3B MTP 벤치 (싱글/듀얼/moe-cache) | [mtp-speculative/BENCH-A3B-Q38-MTP.md](mtp-speculative/BENCH-A3B-Q38-MTP.md) | A3B는 한 카드에 들어감 → 싱글+MTP가 최선(135 t/s, accept 0.81). |
| reddit 69 t/s 재현 | [mtp-speculative/BENCH-27B-REDDIT-REPRO.md](mtp-speculative/BENCH-27B-REDDIT-REPRO.md) | 듀얼 tensor + 별도 MTP 헤드 = 67~72 t/s 재현 성공. |
| **MTP 긴 프롬프트 버그 발견** | [mtp-speculative/BUG-MTP-LONGPROMPT.md](mtp-speculative/BUG-MTP-LONGPROMPT.md) | ~300tok 넘으면 draft accept 0으로 붕괴. beellama 회귀. |
| Track B 결과 (upstream 최신 + boosts + RCCL) | [rccl-dual-gpu/TRACK-B-RESULT.md](rccl-dual-gpu/TRACK-B-RESULT.md) | ROCm 10.0 재빌드, 13/13 패치 클린, MTP 버그 없음(1211tok accept 0.94). |
| 초기 벤치 통합 (Track A/B) | [session-overview/BENCH-2026-09-06-FINAL-v2.md](session-overview/BENCH-2026-09-06-FINAL-v2.md), [BENCHMARK-FINAL.md](session-overview/BENCHMARK-FINAL.md) | 벤치 최종 정리. |
| ROCm 10 빌드 가이드 | [docker-build-infra/IMAGE-BUILD-GUIDE.md](docker-build-infra/IMAGE-BUILD-GUIDE.md) | 최종 이미지 `gfx1100-rocm10-tbq-rboosts` 재생성 절차. |

### 2026-09-07

| 작업/발견 | 링크 | 간단 설명 |
|---|---|---|
| **TurboQuant KV 포팅** (turbo2/3/4) | [turboquant-kv/bench-kv-vs-turboquant.md](turboquant-kv/bench-kv-vs-turboquant.md) | TurboQuant+ 코드c 포팅 + 전벤치. 33파일. 131K 실현(A3B). |
| TurboQuant 라이브 운영 적용 | [turboquant-kv/implement-plan.md](turboquant-kv/implement-plan.md) | K=q8_0 / V=turbo3, WHT warp-shuffle, 그래프 Q회전 OFF. |
| 나이들/150K NIAH 검증 | [session-overview/SESSION](session-overview/SESSION-2026-09-06-MASTER.md) §D8 | 150K needle 회수 성공, A3B prefill 2.3x 빠름. |
| 운영 구성 config.ini 적용 | [session-overview/SESSION](session-overview/SESSION-2026-09-06-MASTER.md) §D8.8 | Dense/Dense.27/Dense.next 프리셋, llm-main 재기동. |

### 2026-09-08

| 작업/발견 | 링크 | 간단 설명 |
|---|---|---|
| **P2P 하드웨어 불가 확정** (JohnTDI 시험) | [session-overview/SESSION](session-overview/SESSION-2026-09-06-MASTER.md) 부록 E | `hipDeviceCanAccessPeer` can=0 → RCCL이 유일 경로, 레이어 split 정석. |
| 서브에이전트 KV 메모리 운영 지식 | [session-overview/SESSION](session-overview/SESSION-2026-09-06-MASTER.md) 부록 F | unified KV 풀, 슬롯, 에빅션, parallel 분리 전략. |
| **SSE ping 튜닝 (중간 드랍 방지)** | [docker-build-infra/sse-ping-config.md](docker-build-infra/sse-ping-config.md) | "http client error: Connection handling canceled" 원인+해결. `sse-ping-interval=10` |

---

## 3. 주요 운영 구성 (현재 사용중)

| 항목 | 값 | 문서 |
|---|---|---|
| 라우터 컨테이너 | `llm-main` (`baramofme/llama-cpp-rocm:...mtp-latest`) | [docker-build-infra/](docker-build-infra/) |
| 서버 명령 | `llama-server --models-preset /app/config.ini --swa-full --host 0.0.0.0 --port 8080` | — |
| 설정 파일 | `/opt/llm/llama-cpp/main-llm-config.ini` → `/app/config.ini` | [sse-ping-config.md](docker-build-infra/sse-ping-config.md) |
| KV 조합 | `-ctk q8_0 -ctv turbo3/4` | [turboquant-kv/](turboquant-kv/) |
| MTP | 임베디드: adaptive / 별도 -md: draft-mtp | [mtp-speculative/](mtp-speculative/) |
| MoE cache | `--moe-expert-cache 96 --moe-expert-cache-inserts 1` | [moe-expert-cache/](moe-expert-cache/) |
| SSE ping | `sse-ping-interval=10`, `timeout=3600` | [sse-ping-config.md](docker-build-infra/sse-ping-config.md) |

## 4. 사용 방법 요약 (quick start)

```bash
# 이미지 재빌드 (운영 이미지 재생성)
#   docker-build-infra/IMAGE-BUILD-GUIDE.md 참고

# llm-main 라우터 재기동 (config.ini 변경 반영)
/tmp/llm-main-restart.sh            # 또는 docker restart llm-main

# 벤치 실행
research/docker-build-infra/bench-beellama.sh smoke   # 스모크
research/docker-build-infra/bench-beellama.sh bench   # llama-bench
research/docker-build-infra/bench-beellama.sh server  # 서버 수용률
research/docker-build-infra/bench-beellama.sh all     # 전부

# 실제 추론 호출 (라우터 via 8081)
curl http://localhost:8081/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Dense","messages":[{"role":"user","content":"hi"}]}'
```

> 각 feature의 상세 플래그/조합은 해당 폴더의 `implement-plan.md` 와 `session-overview/SESSION`.
