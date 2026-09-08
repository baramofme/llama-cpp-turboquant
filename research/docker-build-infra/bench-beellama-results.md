# beellama v0.4.4 성능 벤치 결과

> 날짜: 2026-09-05 (Asia/Seoul)
> 대상: `ghcr.io/anbeeld/beellama.cpp:server-rocm-v0.4.4` (build 11573, commit f8cd4e6dd)
> 비교: `baramofme/llama-cpp-rocm:gfx1100-rocm7.2-tbqplus-rebuild` (build 11198, commit a8ec5c276)
> 모델: Qwen3.8-27B-MTP-Q4_K_M (16.8GB), 스모크: Qwen3.5-2B-MTP-Q4_K_M
> GPU: RX 7900 XTX (gfx1100) 1개 단독 사용, `HIP_VISIBLE_DEVICES=0`
> 공통: `-c 32768 -b 4096 -ub 1024 -ngl 99 -fa on`, llama-bench `-r 3`

## 1. 결과 요약

| # | 시나리오 | 구성 | beellama | llm-main(tbqplus) | 비고 |
|---|---|---|---|---|---|
| S1 | pp/tg | f16 KV | pp 917.2 / tg 34.52 | pp 921.2 / tg 34.56 | baseline, 거의 동일 |
| S2 | pp/tg | q8_0/q5_0 | pp 921.8 / tg 33.17 | pp 921.6 / tg 33.54 | TheTom 비대칭 KV |
| S3 | pp/tg | kvarn5/kvarn4 (tail 1024) vs q8_0/turbo3 | pp 919.4 / tg 34.54 | pp 911.3 / tg 33.59 | KVarN vs turbo3 |
| S4 | tg (server) | draft-mtp, n-max 3 | tg 33.75, 수용률 34.7% | tg 40.23, 수용률 56.7% | MTP 시뮬레이션 디코딩 |
| S5 | tg (server) | draft-dflash + profit | 실패 (드래프트 모델 부재) | 실패 (`--spec-dm-controller` 미지원) | 별도 드래프트 모델 필요 |
| S6 | tg (server) | S4 + KVarN / turbo3 | tg 32.61, 수용률 40.3% | tg 34.70, 수용률 39.0% | MTP + KV 캐시 조합 |
| S7 | pp/tg (4병렬) | batched-bench -npl 4 | pp 944.6 / tg 77.64 | pp 945.6 / tg 77.55 | 멀티 슬롯 동시 디코딩 |

(단위: t/s. S4~S6은 llama-server `/v1/completions` max_tokens=256 기준, S1~S3/S7은 llama-bench/batched-bench)

## 2. 핵심 관찰

1. **pp(프롬프트 처리)는 양쪽 빌드가 사실상 동일** (911~922 t/s). KVarN/turbo3 KV 타입이 pp에 미치는 영향이 미미.
2. **S1/S2/S3의 tg(단일 스트림 디코딩)도 거의 동일** (33~34.5 t/s). tbqplus의 VDot/turbo3 최적화가 단일 디코딩에서는 표준 릴리스 대비 우위를 보이지 않음.
3. **S4 MTP 시뮬레이션 디코딩에서 큰 차이**: llm-main(tbqplus) 40.23 t/s(수용률 56.7%, mean len 2.68) vs beellama 33.75 t/s(수용률 34.7%, mean len 2.04). tbqplus가 draft 수용률을 ~22%p 높여 tg를 ~19% 가속.
4. **S6(MTP + KV 캐시)에서는 beellama가 S4 대비 수용률이 34.7%->40.3%로 상승** (KVarN이 MTP 수용률에 긍정적), 그러나 tg는 32.61 t/s로 오히려 S4(33.75)보다 낮음. 캐시 압축의 검증 단계 오버헤드가 tg를 낮추는 것으로 보임.
5. **S7 멀티 슬롯(4병렬)은 양쪽 동일** (tg ~77.6 t/s). 병렬 디코딩 성능은 빌드 무관.
6. **S5(draft-dflash)는 양쪽 모두 실행 불가**: beellama는 별도 DFlash 드래프트 모델 파일 요구, llm-main은 `--spec-dm-controller` 플래그 미지원. 현 모델 세트에 DFlash 드래프트가 없어 미측정.

## 3. 결론

- **단일 스트림 pp/tg, 멀티 슬롯 디코딩**: beellama 표준 릴리스와 tbqplus 커스텀 빌드 간 성능 차이 없음.
- **MTP 시뮬레이션 디코딩**: tbqplus가 draft 수용률(56.7% vs 34.7%)에서 압도적으로 우위, tg 40.23 vs 33.75 t/s. tbqplus의 VDot/터보 최적화가 시뮬레이션 디코딩 경로에서 실익을 제공.
- **KVarN은 pp/tg에는 무해, MTP 수용률에는 긍정적**이지만 tg 가속으로는 이어지지 않음.
- beellama 표준 릴리스로 마이그레이션 시 **MTP 시뮬레이션 디코딩 성능이 ~16% 저하**될 수 있음. tbqplus 유지 또는 beellama의 draft 수용률 개선이 필요.

## 4. 실행 환경 메모

- 계획서 GPU 매핑 오류: `renderD129`(03:00.0)가 llm-main 점유 GPU였고, 유휴 GPU는 `renderD130`(06:00.0). 벤치는 `renderD130`/`card2`로 수행.
- `llama-bench` jsonl 스키마: `n_prompt>0`=pp, `n_gen>0`=tg, `avg_ts`=t/s (`.test`/`.result.mean` 아님).
- `llama-batched-bench`는 `-p/-n/-r` 대신 `-npp/-ntg/-npl` 사용.
- 로딩 시간: 모델 로드 ~3.2~3.8s (양쪽 유사).
- 원시 출력: `bench-results/` (jsonl, 로그, metrics 스냅샷, completion 응답).
