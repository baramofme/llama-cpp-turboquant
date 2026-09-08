# implement-plan: TurboQuant KV cache 이식 (turbo2/3/4)

> 상태: **적용 완료 + 검증 완료** (2026-09-07). 아래는 재현/유지보수용 이식 계획.
> 원본 연구: `bench-kv-vs-turboquant.md`

## 목표

llama.cpp에 TurboQuant+ 코드c(이기: Walsh-Hadamard 회전 + polar codebook) KV 캐시 양자화
타입 `turbo2`/`turbo3`/`turbo4` 를 이식. KV를 최대 ~4.6x 압축.

## 이식 범위 (파일)

| 영역 | 파일 | 내용 |
|---|---|---|
| ggml 타입 | `ggml/include/ggml.h`, `ggml/src/ggml.c` | TURBO2/3/4 타입 + `GGML_OP_TURBO_WHT` |
| CUDA/HIP 커널 | `ggml/src/ggml-cuda/turbo-quant.cuh`, `turbo-wht.cu`, `turbo-innerq`, `set-rows.cu`, `convert.cu`, FA 계열(fattn-*) | WHT 회전 + polar quant/dequant |
| HIP 인스턴스 | `ggml-hip/CMakeLists.txt` | vec 20 + mma 640 + tile 640 |
| llama 통합 | `src/llama-kv-cache.cpp`, `src/llama-graph.cpp`, `src/llama-context.cpp` | KV 캐시 turbo 패딩/뷰, 역-WHT, CPU get_rows |

## 검증된 필수 조건 (얻어낸 교훈 — 꼭 지킬 것)

1. **K는 turbo로 압축 금지**. K가 turbo(2/3/4 모두)면 30K+ 컨텍스트에서 attention 라우팅 붕괴.
   **V만 turbo로 압축** (V 오류는 비례적이라 안전). K는 `q8_0` 이상 유지.
2. **그래프 Q 회전 OFF** (`llama-graph.cpp`). 그래프 Q 회전(WHT forward)은 내적 붕괴 유발
   (짧은 프롬프트도 깨짐). 저장 시 dequant inverse 방식만 사용. 커밋에 회귀 방지 주석 포함.
3. **WHT 구현은 TheTom warp-shuffle 사용** (`set-rows.cu` / `turbo-wht.cu`). shared butterfly는 지양.
4. **지원 조합**: `turbo × {q8_0, f16}`만 정식. `q5_0/q4_0 × turbo`는 원본에도 없음.
   `q4_0`/`q5_0` V + turbo는 FA vec 인스턴스 부재로 비실용.

## 운영 조합 (검증 완료)

```
-ctk q8_0 -ctv turbo3/4   (K=정밀, V=turbo)  ← 권장
-ctk f16 -ctv turbo4                           ← 더 보수적
```

- A3B (head_dim 128 배수): turbo4 정상, 131K 단일 GPU 실현 (16.7GB)
- Flash-Next (qwen4exp, head_dim != 128 배수): **turbo 불가** — turbo 블록(128)과 비호환, 품질 붕괴

## 성능 노트

- prefill: turbo4가 q8_0/q4_0 대비 **2.7x 느림** (TILE prefill의 f16 변환 디콴트 비용).
  prefill 최강은 `q8_0(K)/q4_0|q5_0(V)`. turbo는 **VRAM 절약(더 큰 컨텍스트)** 필요할 때만.
- decode: turbo4는 q8 대비 -3~-6% (WHT 역변환 오버헤드).

## 비고 / 미해결

- `TURBO_INNERQ=5000` env (InnerQ managed 배열 identity 초기화) — 설정 필요.
- 131K/262K에서 NIAH 검증은 `session-overview/SESSION` §D8 참고 (150K 통과).
