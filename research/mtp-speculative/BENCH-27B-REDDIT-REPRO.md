# BENCH-27B-REDDIT-REPRO: reddit "29→69 tok/s" 구성 재현 검증 (2026-09-06)

> 목적: reddit [1vtoux9] (dual 7900 XTX, Qwen3.8-27B, tensor+MTP, 29→69 tok/s) 사례가
>       우리 RCCL 빌드(gfx1100, ROCm 7.2.3, boosts 포함)에서 재현되는지 검증
> 모델: unsloth/Qwen3.8-27B-GGUF
>       - Qwen3.8-27B-UD-IQ4_XS.gguf (14.25GB, 4.25bpw)  ← reddit 동일
>       - MTP/mtp-Qwen3.8-27B-Q4_0.gguf (1.37GB)          ← reddit 동일 (별도 MTP 헤드)
> 빌드: rccl-build/bin (RCCL + PR#28223 + PR#27861 + kvarn boosts)

## 결과 대조 (reddit vs 우리 재현)

| 구성 | reddit (86,675-tok prompt) | 우리 (8.6K 충전) | 판정 |
|---|---|---|---|
| 1 GPU baseline | 29.28 | (IQ4_XS 단독은 123-tok에서 37.5 Q3급; IQ4+unMTP는 미측정) | - |
| 1 GPU + MTP | 53.90 | 61.4~62.5 | ✓ 우위 |
| 2 GPU tensor | 34.66 | A3B에서 60.9(다른모델)/27B Q3에서 41~42 | 개별비교 |
| **2 GPU tensor + MTP** | **69.25** | **67~72 (long-run 67, peak 72)** | **✓ 재현** |
| 2 GPU tensor + MTP 262K | 69.41 | - | ctx확장은 VRAM 여유시 |

## 요약

1. **reddit 69 t/s 구성 재현 성공**: 듀얼 tensor + 별도 MTP 헤드(-md) + f16 KV + IQ4_XS
   - long-run 지속 생성 tg 67.06~68.95, 최종 300토큰 72.0 (8.6K 컨텍스트 충전)
   - reddit의 69.25와 실질 동일 (우리는 오히려 소폭 우위)
2. **핵심 포인트**: 
   - MTP 헤드는 **별도 GGUF로 로드** (`-md mtp.gguf -ngld 99`) — 임베디드 nextn 아님
   - **MOE 캐시 불필요** (dense 모델) - reddit도 tensor split만 사용
   - IQ4_XS(4.25bpw)가 Q3_K_M 대비 수용률·속도 우수
   - 컨텍스트 충전 시 수용률 상승: reddit 73%→87%(86K), 우리 8.6K에서 51~56%
3. **우리 VRAM 우위**: GPU0 9.4 / GPU1 9.4GB (reddit 13.8/13.8) — IQ4_XS가 작아서.

## 실행 명령 (정확 재현)

```bash
export LD_LIBRARY_PATH=rccl-build/bin:/opt/rocm-7.2.3/lib GGML_CUDA_P2P=1
HIP_VISIBLE_DEVICES=0,1 rccl-build/bin/llama-server \
  -m /mnt/nvmedata/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf \
  -md /mnt/nvmedata/models/unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf -ngld 99 \
  -ngl 99 -sm tensor -ts 1,1 -fa on -c 8192 -b 2048 -ub 512 -t 12 --threads-batch 12 \
  -ctk f16 -ctv f16 --jinja --spec-type draft-mtp --spec-draft-n-max 3
# 컨텍스트 확장: -c 262144 도달 가능 (VRAM 여유 확인 후), VRA 대칭 -ts 1,1 유지
```

## 남은 검증
- [ ] 86K+ 프롬프트, 131K/262K ctx에서 수용률 87% 재현 (VRAM 여유: 9.4GB/카드 사용 중, 15GB 여유 → 131K f16 KV 가능)
- [ ] -ts 45,55 비대칭 (reddit이 데스크톱 카드 부하 보정으로 사용)
- [ ] A3B MoE에 동일 방법 적용: A3B도 별도 MTP 헤드 존재 시 (현재는 nextn 임베디드)
