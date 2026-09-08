# rdna-boosts 순차 이식 계획 (gfx1100 / RX 7900 XTX)

> 날짜: 2026-09-05 (Asia/Seoul)
> 대상: `stew675/llama-cpp-rdna-boosts` 패치 8개 (0001/0003/0004/0008/0009/0010/0011/0013)
> 방법: **경로 B** - beellama v0.4.5(clean 5 블록 적용 상태)에 **하나씩** 이식-빌드-테스트
> 0012 제외 (RDNA4 전용, gfx1100에서 RCCL 폴백 - 이득 없음)
> 0002/0005/0006/0007/0012는 이미 적용 완료 (clean 5 블록)
> 하드웨어: RX 7900 XTX (gfx1100), GPU1(renderD130/card2) 단독
> 모델: Qwen3.8-27B-MTP-Q4_K_M (head_dim 256, MTP dense 27B)

---

## 1. 이식 순서 (우선순위 + 의존성)

0004가 7900XTX(RDNA3)에서 가장 필요(WMMA FA, head_dim 256 해당). 0008은 0003+0004 의존.

| 순서 | 블록 | 내용 | 의존 | 기대 효과 |
|---|---|---|---|---|
| 1 | 0004 | RDNA3/4 WMMA flash-attn + Q6_K mmq | - | **prefill WMMA** (head 576 limit, RDNA3.0 해당) |
| 2 | 0003 | BF16 KV cache + native-BF16 FA | - | FA 정밀도/성능 |
| 3 | 0008 | fused-core prefill kernels | 0003+0004 | prefill融合, GPU bit-identical |
| 4 | 0010 | k-quant VDR (Q4_K/Q5_K/Q6_K/Q8_0) | - | mmvq 디코딩/버리파이 (greedy 수치 변경 주의) |
| 5 | 0001 | adaptive MTP draft depth | - | MTP 수용률/속도 |
| 6 | 0011 | skip CUDA graphs for multi-token prefill | - | prefill 그래프 오버헤드 제거 |
| 7 | 0009 | meta-buffer compute-container headroom | - | meta/호스트 버퍼 |
| 8 | 0013 | fused MoE gate+up+GLU MMQ | - | **MoE 전용 - dense 27B에 해당 없음** (테스트만) |

---

## 2. 단일 블록 이식 워크플로 (반복)

베이스 소스: `beellama-boosts/src/` (v0.4.5 + clean 5 블록). 각 블록마다 아래 루프.

### 2.1 패치 다운로드 + 클린 적용 확인

```
cd beellama-boosts
curl -sL "https://raw.githubusercontent.com/stew675/llama-cpp-rdna-boosts/main/patches/<NNNN>-*.patch" -o /tmp/<NNNN>.patch
cd src
git apply --check /tmp/<NNNN>.patch        # 클린 적용 여부 (에러 나면 수동 병합)
git apply /tmp/<NNNN>.patch                 # 클린이면 적용
```

- **클린 적용 실패 시**: 패치의 변경 파일을 beellama 소스에서 대조, 수동 병합
  - beellama 커스텀 코드(KVarN/MTP/FA/quant)와 충돌 지점만 수동 조정
  - 패치의 신규 파일(커널 .cu 등)은 그대로 복사
  - 기존 파일 수정은 beellama 코드를 깨지 않는 선에서 병합

### 2.2 빌드

```
cd /tmp/beellama-src (또는 beellama-boosts/src/build-boosts 재사용)
cmake --build build-boosts -j20            # 증분 빌드 (변경 파일만)
```

- 빌드 실패 시: 컴파일 에러를 패치 변경 파일 중심으로 디버깅

### 2.3 스모크 테스트 (2B 모델)

```
build-boosts/bin/llama-bench -m /opt/llm/models/Qwen3.5-2B-MTP-Q4_K_M.gguf \
  -ngl 99 -fa on -ctk f16 -ctv f16 -p 512 -n 512 -r 1   # 파이프라인 정상 확인
```

### 2.4 27B 벤치 (회귀 여부)

```
build-boosts/bin/llama-bench -m /opt/llm/models/Qwen3.8-27B-MTP-Q4_K_M.gguf \
  -ngl 99 -fa on -ctk f16 -ctv f16 -b 4096 -ub 1024 -p 512 -n 512 -r 3 -o jsonl
# KVarN 시나리오: -ctk kvarn5 -ctv kvarn4 --kv-tail-tokens 1024
```

- 기준선: S1 f16 pp 965.2 / tg 34.55 (clean 5 블록 상태)
- **회귀(pp/tg 5% 이상 저하) 또는 크래시면**: 해당 블록 revert, 원인 기록
- **개선이면**: 유지, 다음 블록 진행

### 2.5 결과 기록

- `bench-results-boosts/block-<NNNN>_s1.jsonl` 등 원시 출력 보관
- `bench-beellama-boosts-results.md`에 블록별 pp/tg/수용률 누적 표 업데이트

---

## 3. 벤치 매트릭스 (블록마다 실행)

- 모델: `/opt/llm/models/Qwen3.8-27B-MTP-Q4_K_M.gguf`
- GPU1(renderD130/card2) 단독
- 공통 플래그: `-b 4096 -ub 1024 -ngl 99 -fa on -p 512 -n 512 -r 3 -o jsonl`

| # | 시나리오 | 플래그 |
|---|---|---|
| S1 | f16 KV | `-ctk f16 -ctv f16` |
| S2 | q8_0/q5_0 | `-ctk q8_0 -ctv q5_0` |
| S3 | KVarN | `-ctk kvarn5 -ctv kvarn4 --kv-tail-tokens 1024` |
| S4 | MTP | llama-server `--spec-type draft-mtp --spec-draft-n-max 3` (0001/0013 관련 시) |

---

## 4. 주의 / 리스크

1. **0010 디코드 수치 변경**: fp32 리덕션 순서 변경, greedy 출력 비트-동일 보장 없음 (max logit diff 0.184). GREEDY-PURITY.md 참조.
2. **0008은 0003+0004 필요**: 순서대로(0004 -> 0003 -> 0008) 적용.
3. **0013은 MoE 전용**: Qwen3.8-27B는 dense - 실질 효과 0. 테스트만.
4. **순차 이식**: 한 블록 적용 -> 빌드 -> 스모크 -> 27B 벤치 -> 회귀 확인 -> 다음 블록.
   - 전체 동시 적용 금지 (회귀 원인 특정 불가)
5. **revert 가능**: 각 블록 적용 전 `beellama-boosts/src` 스냅샷(또는 git init)으로 회귀 시 되돌림
6. **`git apply` 사용**: 단일 패치는 `git apply` 가능 (연속 13패치 일괄 `git apply`만 금지)

---

## 5. 산출물

- `rdna-boosts-apply-plan.md` (본 문서)
- `beellama-boosts/src/` - 순차 이식된 소스 (블록 누적)
- `bench-results-boosts/block-<NNNN>_s*.jsonl` - 블록별 원시 출력
- `bench-beellama-boosts-results.md` - 블록별 누적 결과 표
