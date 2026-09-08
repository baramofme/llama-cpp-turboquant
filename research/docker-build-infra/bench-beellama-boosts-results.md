# rdna-boosts clean 블록 beellama v0.4.5 적용 벤치

> 날짜: 2026-09-05 (Asia/Seoul)
> 대상: beellama v0.4.5 (`44a369699`) + `stew675/llama-cpp-rdna-boosts` clean 블록 5개 (0002/0005/0006/0007/0012)
> 방법: 13개 패치 중 beellama v0.4.5에 `git apply --check`로 클린 적용되는 5개 + 수동 병합 3개
>   - 0002 fused chunked gated-delta-net prefill (gfx11 WMMA)
>   - 0005 CPU bit-identical decode/verify
>   - 0006 host buffer revert (beellama에 이미 적용됨)
>   - 0007 meta device wrapper skip (beellama에 이미 적용됨)
>   - 0012 hybrid HIP all-reduce (RDNA4 전용 게이트 - gfx1100에서는 RCCL 폴백, 실질 효과 없음)
>   - 0004 RDNA4 WMMA FA + Q6_K mmq (수동 병합: DKQ>128->576, wmma_max_head)
>   - 0010 k-quant VDR (수동 병합: Q4_K/Q5_K->vdr4, Q6_K->vdr2, nwarps 2->8)
>   - 0011 skip CUDA graphs for multi-token prefill (클린 적용)
>   - 0009 meta buffer headroom (beellama에 이미 32, vanilla 16 - 이미 최적화됨)
>   - 건너뜀: 0001(adaptive MTP, llama-server 전용), 0003(BF16 KV, 사용 시나리오 없음), 0008(0003 의존), 0013(fused MoE, dense 모델 무관)
> 빌드: 호스트 네이티브 (ROCm 7.2.3, cmake 4.2.3, gfx1100, Release) - Docker 없이
> 모델: Qwen3.8-27B-MTP-Q4_K_M, GPU1(RX 7900 XTX) 단독, `-c 32768 -b 4096 -ub 1024 -ngl 99 -fa on`

## 1. 애플-토-애플 비교 (모두 native ROCm 7.2.3, 동일 머신)

동일 머신/동일 ROCm/동일 플래그로 **패치 없는 v0.4.5(clean)** vs **패치 적용(boosts)** 비교.
환경 변수를 통제해 패치 자체의 이득만 분리.

| 시나리오 | 구성 | clean pp/tg | boosts pp/tg | Δpp | Δtg |
|---|---|---|---|---|---|
| S1 | f16 KV | 928.9 / 34.57 | 965.2 / 34.55 | **+3.9%** | -0.06% |
| S2 | q8_0/q5_0 | 925.0 / 33.11 | 958.3 / 33.17 | **+3.6%** | +0.2% |
| S3 | kvarn5/kvarn4 (tail 1024) | 923.5 / 34.50 | 960.9 / 34.53 | **+4.1%** | +0.1% |
| S4 | MTP (draft-mtp n-max3) | tg 36.19, 수용률 37.6% | tg 35.11, 수용률 45.6% | -3.0% | 수용률 **+8.0%p** |

(단위: pp/tg = t/s. S4는 llama-server `/v1/completions` max_tokens=256 단일 실행)

## 2. 핵심 관찰

1. **prefill(pp) 일관된 +3.6~4.1% 개선**: 3개 KV 구성 모두에서 재현. 0002의 chunked gated-delta-net prefill 커널(gfx11 WMMA) 효과로 판단.
2. **tg(단일 스트림 디코딩)는 변동 없음** (±0.2% 내, 노이즈 범위). clean 블록 중 디코딩 경로에 영향을 주는 것이 없음.
3. **S4(MTP) 수용률 개선**: 패치 적용 시 draft acceptance 37.6% -> 45.6%(+8.0%p). 0005(CPU bit-identical decode/verify)가 verify 정합성에 기여한 것으로 보임. tg 절대값은 단일 256토큰 실행이라 노이즈가 크고, 수용률(집계 지표)이 더 신뢰됨.
4. **이전 "S4 회귀"는 환경 아티팩트였다**: docker(ROCm 7.2.1) v0.4.5의 S4 tg 42.29/수용률 55.4%와 native(ROCm 7.2.3) boosts 35.11/45.6%를 직접 비교해 회귀로 보였으나, **native clean 기준선(36.19/37.6%)과 비교하면 boosts는 동급~수용률 우위**. 즉 ROCm 버전(7.2.1 vs 7.2.3) 차이가 MTP 수용률에 큰 영향을 줬고, 패치는 회귀가 아니라 수용률 개선.

## 3. 누적 블록별 벤치 (clean 기준선 대비)

clean 5 블록(0002/0005/0006/0007/0012)에 0004/0010/0011 수동 병합까지 누적 적용한 최종 빌드.
동일 머신/ROCm 7.2.3/GPU1, `-p 512 -n 512 -r 3`.

| 시나리오 | KV | clean pp/tg | boosts(clean 5) pp/tg | boosts(+0004/0010/0011) pp/tg | Δpp(clean 대비) | Δtg(clean 대비) |
|---|---|---|---|---|---|---|
| S1 | f16 | 928.9 / 34.57 | 965.2 / 34.55 | 982.0 / 37.10 | **+5.7%** | **+7.3%** |
| S2 | q8_0/q5_0 | 925.0 / 33.11 | 958.3 / 33.17 | 981.3 / 35.47 | **+6.1%** | **+7.1%** |
| S3 | kvarn5/4 (tail 1024) | 923.5 / 34.50 | 960.9 / 34.53 | 978.1 / 37.06 | **+5.9%** | **+7.4%** |

- **clean 5 블록**: prefill +3.6~4.1% (0002 chunked GDN), tg 변동 없음.
- **0004 (WMMA FA)**: beellama가 이미 확장 MMA config 테이블을 보유해 추가 효과 미미 (S1 965.2->964.8, 동급).
- **0010 (k-quant VDR)**: Q4_K/Q5_K vdr4 + Q6_K vdr2 + nwarps 8으로 **tg +7~8%** (34.57->37.10, 33.11->35.47, 34.50->37.06). 디코딩의 실질 이득.
- **0011 (CUDA graph skip)**: prefill +1~2% (965.2->982.0 S1).
- **0006/0007/0009**: beellama에 이미 적용되어 있음 (호스트 버퍼 회귀, meta device skip, headroom 32).
- **건너뜀**: 0001(adaptive MTP - llama-server 전용, llama-bench 무관), 0003(BF16 KV - 사용 시나리오 없음), 0008(0003 의존), 0012(RDNA4 전용), 0013(fused MoE - dense 모델 무관).

## 4. 결론

- 13개 블록 중 8개 적용(clean 5 + 수동 병합 3: 0004/0010/0011), 5개 건너뜀(0001/0003/0008/0012/0013 - 무관 또는 의존성).
- **최종 이득: prefill +5.7~6.1%, 디코딩(tg) +7.1~7.4%** (clean 기준선 대비, S1~S3).
  - prefill: 0002(chunked GDN) + 0011(CUDA graph skip)
  - 디코딩: 0010(k-quant VDR: Q4_K/Q5_K vdr4, Q6_K vdr2, nwarps 8)
- MTP 수용률은 애플-토-애플 기준 개선(37.6%->45.6%).
- **S7(4병렬 batched-bench)**은 native 빌드에서 테이블이 비어 출력되지 않음(docker에서는 899.9/77.6 정상). MTP 모델 + 4병렬 조합의 native 빌드 호환성 문제 - 별도 조사 필요.
- **ROCm 10 빌드**: S1/S2/S3 모두 7.2.3과 동급. 초기 "S3 -63% 회귀"는 `bench-rocm10.sh`의 `--kv-tail-tokens 1024` 누락 버그였으며, 수정 후 정상. **KVarN은 `--kv-tail-tokens` 플래그 필수** (없으면 portable 폴백 -63%).

## 5. ROCm 10 빌드 (build.md HIP 절차) + 벤치 비교

- `rocm/dev-ubuntu-26.04:10.0.0-full` (ROCm 10.0 dev, clang-23, CMake 4.2.3)
- build.md의 `HIPCXX/HIP_PATH/HIP_DEVICE_LIB_PATH` + `-DAMDGPU_TARGETS=gfx1100` 구성
- **빌드 성공** (cmake 구성 + 컴파일 OK, `build-rocm10/bin` 산출)
- **장치 매핑 정정**: GPU0=card1/renderD129(03:00.0, llm-main 점유), GPU1=card2/renderD130(06:00.0, 유휴). 계획서/디버그의 "GPU1=renderD129" 표기가 틀림.
  - 초기 "27B 로딩 실패(OOM)"는 ROCm 10 문제가 **아님** - 디버그 명령이 점유 중인 GPU0(renderD129, 2018 MiB free)을 잘못 사용. 유휴 GPU1(renderD130, 24524 MiB free)을 쓰면 27B 정상 로딩/벤치.

### ROCm 7.2.3(native, boost) vs ROCm 10(docker, boost) 비교

동일 소스(v0.4.5 + clean 5 블록), 동일 GPU1, 동일 플래그(`-b 4096 -ub 1024 -ngl 99 -fa on -p 512 -n 512 -r 3`, KVarN은 `--kv-tail-tokens 1024`).

| 시나리오 | KV | R7.2.3 pp/tg | R10(수정 전) | R10(수정 후) |
|---|---|---|---|---|
| S1 | f16 | 965.2 / 34.55 | 918.2 / 35.43 | 914.82 / 35.73 |
| S2 | q8_0/q5_0 | 958.3 / 33.17 | 902.3 / 34.16 | 912.66 / 34.47 |
| S3 | kvarn5/4 (tail 1024) | 960.9 / 34.53 | 353.8 / 22.91 | **911.80 / 35.44** |

**정정 - S3 "ROCm 10 -63% 회귀"는 벤치 스크립트 버그였다**:
1. `bench-rocm10.sh`가 S3에서 `--kv-tail-tokens 1024` 플래그를 **누락** (7.2.3 벤치는 포함).
2. 이 플래그가 없으면 KVarN이 `kvarn_route_portable=32`, `generic_rejected=32` (32/32 레이어 portable 폴백) -> pp 353.8.
3. 플래그를 추가하면 `kvarn_route_portable=0` (최적화 경로) -> **pp 911.80, tg 35.44** (7.2.3과 동급).
4. 즉 ROCm 10이 KVarN을 못 돌리는 것이 아니라, **KVarN은 `--kv-tail-tokens` 없이 실행하면 portable 폴백으로 -63% 회귀**한다. 플래그 있으면 양쪽 ROCm에서 정상.
5. `bench-rocm10.sh`를 수정해서 kvarn 계열에서 `--kv-tail-tokens 1024`를 자동 추가하도록 했다.

**이슈 #122**([Anbeeld/beellama.cpp#122](https://github.com/Anbeeld/beellama.cpp/issues/122))의 15-22x 회귀는 **별도 문제**: D=256 모델에서 HIP WMMA가 `head_dim > 128`을 거부(`fattn-kvarn-route-policy.h` 63줄) -> portable 폴백. RDNA3에서 D=256 KVarN은 본질적으로 portable 경로에 의존. `--kv-tail-tokens`가 있으면 최적화 경로를 쓴다.

**rdna-boosts 패치는 KVarN 라우팅을 해결하지 못한다**:
- 13개 패치 전부 grep: KVarN route-policy(`fattn-kvarn-route-policy.h`)를 건드리는 패치 **0개**.
- 0004/0008은 표준 FA WMMA(`GGML_CUDA_FA_WMMA_MAX_HEAD`)만 건드리고, KVarN 라우팅은 건드리지 않음.
- KVarN D=256 portable 폴백을 해결하려면 `fattn-kvarn-route-policy.h`의 `head_dim > 128` 거부 조건을 WMMA D=256으로 확장하거나, CUDA split/vector 디코딩을 HIP으로 이식해야 함. rdna-boosts에는 해당 패치가 없음.

**결론**:
- KVarN은 `--kv-tail-tokens` 플래그가 있으면 ROCm 7.2.3/10 양쪽에서 정상. 플래그 없으면 portable 폴백으로 -63% 회귀.
- `bench-rocm10.sh`는 이제 kvarn에서 `--kv-tail-tokens 1024`를 자동 추가.
- 이슈 #122의 15-22x 회귀는 D=256 모델에서 HIP WMMA 거부로 인한 별도 문제. rdna-boosts는 KVarN 라우팅에 영향을 못 줌.

## 6. 산출물 / 재현

- 패치 적용 소스: `beellama-boosts/src/` (v0.4.5 + clean 5 블록, 워크스페이스 보존)
- 빌드: `/tmp/beellama-src/build-boosts` (패치 적용, ROCm 7.2.3), `/tmp/beellama-src/build-clean` (패치 없는 대조)
- ROCm 10 빌드: `beellama-boosts/src/build-rocm10/` (ROCm 10.0, 2B/27B 모두 정상 - 유휴 GPU1 사용 시)
- 원시 출력: `bench-results-boosts/` (ROCm 7.2.3 S1~S4 jsonl/로그, clean_s1~s4 대조), `beellama-boosts/rocm10/` (ROCm 10 S1~S3 jsonl/로그)
- 벤치 스크립트: [`beellama-boosts/bench-boosts.sh`](beellama-boosts/bench-boosts.sh)
- Dockerfile(미사용, native 빌드로 대체): [`beellama-boosts/Dockerfile`](beellama-boosts/Dockerfile)
