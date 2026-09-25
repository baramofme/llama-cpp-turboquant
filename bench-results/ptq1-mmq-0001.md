# PTQ1_0 네이티브 MMQ 이식 - 벤치 결과 (2026-09-22)

GPU: RX 7900 XTX (gfx1100) GPU0 / ROCm 10 / container rocm10-build
빌드: WIP (8500a04ee + PTQ1_0 MMQ 9사이트 언가드, 2026-09-21 16:22 클린 리빌드)
모델: Ternary-Bonsai-2-27B-PTQ1_0 (5.53 GiB, 1.75 bpw, group-128 ternary), Ternary-Bonsai-2-27B-PQ2_0 (6.70 GiB, 2.0 bpw)
벤치 명령: `llama-bench -m <model> -ngl 99 -fa 1 -pg 512,128` / `-pg 1024,1` / `-pg 5201,1`
(이 포크의 llama-bench는 -c 옵션이 없고 n_ctx = prompt+gen+depth 로 자동 설정, llama-bench.cpp:1343)

## 결과 (t/s)

| 모델 | 경로 | pp512 | pp1024 | pp5201 | tg128 |
|---|---|---:|---:|---:|---:|
| PTQ1_0 이식 전 (dequant+hipBLAS 폴백) | 744.45 | - | (686-697)* | 36.29 |
| PTQ1_0 이식 후 (네이티브 MMQ) | 741.29 | 722.43 | 703.70 | 36.07 |
| PQ2_0 (네이티브 MMQ, 회귀 컨트롤) | 921.50 | 901.63 | 880.37 | 47.36 |

- "이식 전" 기준: 09-21 이 포크 이식 전 빌드로 실측 (pp512/tg128만 측정).
- *pp5201 "이식 전" 값: 본 벤치에서 재측정 안 함. KV-CACHE-VRAM-PQ2-PTQ1.md (09-21, pq20had 이미지, q4_0 KV, ctx 131072) 기록 686-697.
- PQ2_0 이식 전/후: 폴백/MMQ 무관하게 921.5 vs 09-21 기록 871-888 (동일 밴드, 회귀 없음).
- 변동폭: pp512 ±2-3, pp1024 ±1, pp5201 ±0.5, tg ±0.2 (llama-bench std).

## 결론

1. **속도 불변.** PTQ1_0 네이티브 MMQ = 기존 dequant+hipBLAS 폴백과 통계적으로 무차별
   (pp512: 741 vs 744, pp5201: 704 vs 686-697). 목표 ~850 t/s 미달.
2. **PQ2 vs PTQ1 격차 (~20%)는 경로 무관.** 이식 전 697/886 ≈ 0.79, 이식 후 741/921 ≈ 0.80.
   GEMM 경로를 MMQ로 갈아타도 격차가 안 줄어듦 => 격차의 원인은 GEMM 경로가 아니라
   **base-3 trit 디코드 ALU 비용** (5 trit당 5회귀 반복: 32bit mul+AND+byte_perm+vsub4,
   PQ2_0 2-bit unpack의 2-4배)으로 판단. 두 경로(폴백/MMQ) 모두 이 디코드 비용 때문에
   같은 천장에 붙음.
3. **정합성 OK.** llama-cli -n 48: thinking 모드 코히어런트 출력, 사실 관계 정확
   (레이리 산란 답변). "We need..." 는 thinking 접두어이지 퇴행이 아님 (09-21 프로브에서
   오인했던 것).
4. **PQ2_0 회귀 없음** (921.5/47.4 vs 926.9/47.5). **decode(tg) 불변** (36.1 vs 36.3,
   mmvq 경로라 MMQ와 무관 - 설계대로).
5. **경로 확인:** GGML_CUDA_FORCE_CUBLAS는 컴파일타임-only (런타임 env 없음)라 계획서의
   A/B 대신 임시 디버그 프린트 사용 => MMQ 경로 실행 확인, 폴백 0회. (프린트는 제거됨)

## 결정

- **이식 유지.** 속도 무차별이지만 네이티브 경로(폴백 소거, CUDA/HIP 타입 정합,
  향후 디코드 개선의 기반)이므로 유지.
- WIP는 미커밋 유지 (사용자 결정 대기).
- PTQ1을 PQ2 속도에 맞추려면 디코드 방식 자체 개선 (repack/재양자화)이 필요한 별개
  프로젝트. MMQ 경로만으로는 한계.

## 배포 (2026-09-22, 전과정 Dokploy `bonsai-sghcma` = h5QEsfsllhIBuagdspK0t)

1. 런타임 이미지: `localhost:5000/baramofme/llama-cpp-rocm:rocm10-gfx1100-rccl-rdnaboosts-mtp-ptq10`
   (digest sha256:d54cdc25, 16:22 클린 빌드 bin, rocm10-runtime.Dockerfile). `--version` 스모크 OK.
2. SmallDense(8082): 이미지/모델 전환 PQ2_0 -> PTQ1_0 + **KV q8_0/q5_0(구, 붕괴 조합) -> q4_0/q4_0**
   (DB compose가 실행 컨테이너보다 뒤처져 있던 것 수정).
3. gate-proxy(1709/1710) 제거 - Bonsai-1 전용(Bonsai-2는 게이트 불필요).
4. hymt(8083) 제거 - 한국어 번역 불필요.
5. `-fa 1` 명시적 추가: 이 빌드의 기본값 AUTO도 gfx1100에서 WMMA FA를 이미 선택
   (fattn.cu:650-675, RDNA3_0 검증 주석, head cap 576)이라 성능 변화 없음 - 결정성/자기문서화 목적.

최종 상태: 앱 = bonsai 단일 서비스. 8082 direct (게이트 없음), completion 정상,
GPU0 VRAM 11.1/24 GB (hymt 제거로 13.6 -> 11.1), GPU1 Dense 원상(23.9/24).

## 환경 노트

- 빌드 컨테이너는 `rocm10-build` (rocm10-dev는 이미 삭제됨).
- bin 실행 시 `LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib` 필수 (빠트리면 libhipblas 로드 실패).
- 사용자 지시대로 전부 GPU0 측정 (GPU1 Dense 건드리지 말 것).
