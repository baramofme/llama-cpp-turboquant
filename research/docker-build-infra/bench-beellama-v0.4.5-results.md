# beellama v0.4.5 (preview) 성능 벤치 결과

> 날짜: 2026-09-05 (Asia/Seoul)
> 대상: `ghcr.io/anbeeld/beellama.cpp:server-rocm-preview-v0.4.5` (build 11636, commit 44a369699)
> 비교 기준: v0.4.4 (`server-rocm-v0.4.4`, build 11573, commit f8cd4e6dd) 벤치 결과 [`bench-beellama-results.md`](bench-beellama-results.md)
> 모델: Qwen3.8-27B-MTP-Q4_K_M (16.8GB), 스모크: Qwen3.5-2B-MTP-Q4_K_M
> GPU: RX 7900 XTX (gfx1100) 1개 단독 사용 (renderD130/card2), `HIP_VISIBLE_DEVICES=0`
> 공통: `-c 32768 -b 4096 -ub 1024 -ngl 99 -fa on`, llama-bench `-r 3`

## 1. 결과 (v0.4.5 vs v0.4.4)

| # | 시나리오 | 구성 | v0.4.4 | v0.4.5 | 변화 |
|---|---|---|---|---|---|
| S1 | pp/tg | f16 KV | pp 917.2 / tg 34.52 | pp 922.5 / tg 34.58 | +0.6% / +0.2% |
| S2 | pp/tg | q8_0/q5_0 | pp 921.8 / tg 33.17 | pp 920.7 / tg 33.19 | -0.1% / +0.1% |
| S3 | pp/tg | kvarn5/kvarn4 (tail 1024) | pp 919.4 / tg 34.54 | pp 923.3 / tg 34.53 | +0.4% / -0.03% |
| S4 | tg (server) | draft-mtp, n-max 3 | tg 33.75, 수용률 34.7% | tg 42.29, 수용률 55.4% | **tg +25.3%, 수용률 +20.7%p** |
| S5 | tg (server) | draft-dflash | 실패 | 실패 | 동일 (드래프트 모델 부재) |
| S6 | tg (server) | MTP + KVarN | tg 32.61, 수용률 40.3% | tg 35.27, 수용률 50.8% | tg +8.2%, 수용률 +10.5%p |
| S7 | pp/tg (4병렬) | batched-bench -npl 4 | pp 944.6 / tg 77.64 | pp 899.9 / tg 77.59 | pp -4.7% / -0.1% |

(단위: t/s. S4~S6은 llama-server `/v1/completions` max_tokens=256 기준)

## 2. 핵심 관찰

1. **MTP 시뮬레이션 디코딩이 크게 개선됨 (S4)**: v0.4.5가 draft 수용률 55.4%(mean len 2.66)로 v0.4.4(34.7%, mean len 2.04)보다 20.7%p 높고, tg 42.29 t/s로 25.3% 가속. v0.4.5는 llm-main(tbqplus)의 40.23 t/s(수용률 56.7%)를 사실상 추월.
2. **S6(MTP + KVarN)도 개선**: 수용률 40.3% -> 50.8%, tg 32.61 -> 35.27 t/s. 다만 KVarN 구성이 여전히 S4(f16)보다 tg가 낮음 (35.27 vs 42.29) - 캐시 압축의 검증 오버헤드 영향은 유지.
3. **S1~S3(pp/tg 단일 스트림)은 v0.4.4와 동급** (±1% 이내, 실행 노이즈 범위).
4. **S7(4병렬) pp가 -4.7% 하락** (944.6 -> 899.9), tg는 동일. 단일 실행이라 노이즈일 수 있으나, v0.4.5의 batched 경로에서 회귀 가능성 확인 필요.
5. **S5(draft-dflash)는 여전히 실행 불가**: 별도 DFlash 드래프트 모델 파일이 모델 세트에 없음.

## 3. 결론

- **v0.4.5의 MTP speculative decoding 수용률 개선이 실측으로 확인**: 34.7% -> 55.4%, tg 33.75 -> 42.29 t/s.
- 이로써 v0.4.4 기준 "tbqplus 대비 MTP 디코딩 16% 저하" 문제가 **v0.4.5에서는 tbqplus 대비 +5% 우위**로 역전됨 (42.29 vs 40.23 t/s).
- pp/단일 tg/멀티슬롯 디코딩은 v0.4.4와 동일 수준.
- S7의 pp 하락(-4.7%)은 재측정(다중 실행)으로 확인 권장.
- DFlash(draft-dflash)는 드래프트 모델 파일 확보 후 재평가 필요.

## 4. 실행 환경 메모

- 스크립트: [`bench-beellama.sh`](bench-beellama.sh) (환경변수 오버라이드 지원)
  ```
  BEELLAMA_IMG=ghcr.io/anbeeld/beellama.cpp:server-rocm-preview-v0.4.5 \
  OUT=bench-results-v045 BUILDS=beellama ./bench-beellama.sh smoke|bench|server|batched
  ```
- 원시 출력: `bench-results-v045/` (jsonl, 로그, metrics 스냅샷, completion 응답)
- v0.4.5 플래그 호환성: `--spec-type`, `--spec-draft-n-max`, `--spec-dm-controller`, KVarN, `--kv-tail-tokens` 모두 v0.4.4와 동일 지원.
