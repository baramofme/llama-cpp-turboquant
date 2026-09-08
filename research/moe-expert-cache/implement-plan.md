# implement-plan: moe-expert-cache (PR#27861)

> 상태: **적용 완료 + 검증 완료** (Track B). 아래는 재현용 이식 계획.
> 참고: 본문 데이터는 상위 문서들과 공유 — `session-overview/SESSION` §D, `rccl-dual-gpu/TRACK-B-RESULT.md`, `bench-kv-vs-turboquant.md`.

## 목표

MoE 모델에서 expert 가중치를 GPU가 아닌 **host(CPU) 메모리로 offload**하고, 활성 expert만
cache로 유지해 **VRAM을 크게 줄이면서** 속도 저하를 완화하는 기능을 이식.

- upstream PR#27861 (`bccbacdb`): `--moe-expert-cache N`, `--moe-expert-cache-inserts N`

## 이식 범위

| 항목 | 내용 |
|---|---|
| 소스 | PR#27861 cherry-pick (upstream-latest, Track B에서 적용됨) |
| 플래그 | `--moe-expert-cache N`, `--moe-expert-cache-inserts N` |
| 연동 | `-ot <exps>=ROCm_Host` (expert 텐서를 host 버퍼로 override) |
| 검증 | `llama_moe_cache_*` 심볼 존재 (TRACK-B-RESULT) |

## 측정 (핵심)

- **VRAM 절감 성공**: 16.2GB -> 8.1GB (A3B 싱글, exps host + cache96). Flash-Next에서도 GPU41+late7 host + cache96 로 24GB x2 수용.
- **속도는 trade-off**: exps host + cache로 VRAM 절반 -> 속도도 절반 (A3B 96.5 -> 45 t/s).
  host로 내릴수록 느려짐 (late8 < late12 < late16).
- **inserts가 관건 (prefill)**: `moe-expert-cache-inserts 1` 이 최적 (4는 오히려 -0.9%).
  late6 host + inserts1 = prefill +17% (200~205 t/s).
- cache 크기: **96이 최적점** (64/128/192 무효하거나 OOM).

## 운영 조합 (검증 완료)

```
# VRAM 절감형 MoE (Flash-Next 24GB x2)
-oct 'per_layer_token_embd=CPU, blk.4[2-7] exps=ROCm_Host'   # (주의: preset 키는 override-tensor)
--moe-expert-cache 96 --moe-expert-cache-inserts 1
--lazy-mode on-direct
```

## 비고

- preset(`config.ini`)에서 이 플래그 키는: `moe-expert-cache`, `moe-expert-cache-inserts`,
  그리고 `-ot`는 반드시 **`override-tensor`** (offload-tensor 아님 — 미인식 크래시).
- 단일 카드에 통째로 들어가는 크기(A3B 17GB)면 **cache 없이 전부 VRAM이 최선**. cache는
  VRAM 한계를 넘는 큰 MoE(Flash-Next 84.9GB)의 실용화에만 의미.
