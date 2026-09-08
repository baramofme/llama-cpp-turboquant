# implement-plan: MTP speculative decoding + 긴 프롬프트 버그

> 상태: **동작 확인 + 버그 원인 분석 완료** (2026-09-06). 
> 버그는 beellama-boosts 특유 회귀로, upstream(Track B) 최신에선 없음.
> 원본 연구: `BUG-MTP-LONGPROMPT.md`, `BENCH-A3B-Q38-MTP.md`, `BENCH-27B-REDDIT-REPRO.md`

## 목표

- MTP(Multi-Token Prediction) speculative decoding을 llama.cpp에 이식.
  - `--spec-type draft-mtp` / `draft-mtp-adaptive`, `--spec-draft-n-max`
- **긴 프롬프트에서 draft가 붕괴하는 버그**를 해결/회피.

## 핵심 발견 (버그, beellama-boosts 한정)

- 프롬프트가 ~300토큰을 넘으면 **draft acceptance가 silent하게 0으로 붕괴** (tg 73 -> 27).
  - 프롬프트 296/339/420/1352tok -> accept 0.0. 75/123tok -> 0.66/0.57.
  - 짧은 프롬프트 후 **긴 생성**에서는 안 죽음 (8634tok, accept 0.51 유지).
  - 원인: **긴 프롬프트를 먼저 먹인 뒤 첫 생성에서 드래프트 시드가 초기화 안 되는 패턴**.
  - kv_unified/parallel/ubatch/ctx 무관. 임베디드/별도 -md 무관. draft-mtp/adaptive 무관.
  - beellama의 `llama_set_embeddings_nextn` 재작성 경로 특유 회귀.
- **해결책: upstream(Track B, 9e0e220)으로 전환하면 버그 없음.** 1211tok 프롬프트 accept 0.94, tg 108.7.

## 운영 가이드 (검증된 조합)

- **임베디드 MTP 모델** (e.g. Qwen3.8-27B-MTP-Q4_K_M) + `--spec-type draft-mtp-adaptive`
- **별도 -md 파일** + `draft-mtp-adaptive` 는 **segfault (exit 139)** — 이 빌드 버그.
  별도 -md + `draft-mtp`(비adaptive)는 정상.
- MTP 수용률 & 성능 (20K, 2026-09-07): A3B tg ~142~147 (MTP 없음 대비 +60%), 27B ~60, 수용률 0.84~0.92.

## 재현 데이터

- `BENCH-A3B-Q38-MTP.md`: A3B 싱글+MTP 전부 VRAM = **135.4 t/s, accept 0.81** (최고).
  exps host + cache는 절반 성능.
- `BENCH-27B-REDDIT-REPRO.md`: 듀얼 tensor + 별도 MTP 헤드 = **67~72 t/s (reddit 69 재현)**.
  단일 GPU MTP ~62 (~reddit 53.9 초과).
- `bench-kv-vs-turboquant.md`: MTP가 decode 3배+ 가속 (Flash-Next 20k: 7.3->23.6, 50k: 4.3->17.5).

## 비고

- draft-n-max: 24GB x2에서 **draft3이 최대** (draft4는 pp 버퍼 OOM).
- `--sse-ping-interval`은 MTP와 무관 (라우터 HTTP 유지보수용).
