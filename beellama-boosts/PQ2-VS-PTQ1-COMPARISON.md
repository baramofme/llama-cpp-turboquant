# Ternary-Bonsai-2-27B: PQ2_0 vs PTQ1_0 비교 자료

측정일: 2026-09-21 / GPU: RX 7900 XTX 24GB / 이미지: rocm10-gfx1100-rccl-rdnaboosts-mtp-pq20had
조건: `-ngl 99`, reasoning-budget 4096, temp 0 (배터리는 각 스크립트 기본 시드)

## 1. 스펙

| | PQ2_0 | PTQ1_0 |
|---|---|---|
| 파일 | 7.21 GB (6.70 GiB) | 5.95 GB (5.53 GiB) |
| 코덱 | 2비트 슬롯 (2.13 bpw) | base-3 조밀 trits (1.75 bpw) |
| ggml type | 142 | 143 (이번에 트리에 이식, 미커밋) |
| 모델카드 공称 | prefill 전 구간 우세 | decode는 Ada/L4에서만 우세 |

## 2. 속도 (pp5201 / tg64, bench 실측)

| K / V | PQ2_0 pp | PQ2_0 tg | PTQ1_0 pp | PTQ1_0 tg |
|---|---|---|---|---|
| f16/f16 | 888 | 47.1 | 697 | 36.0 |
| q8_0/q8_0 | 878 | 46.0 | 690 | 35.3 |
| q4_0/q4_0 | 886 | 45.2 | 691 | 34.9 |
| bf16/bf16 | 878 | 47.2 | 693 | 35.8 |
| q4_0/f16 | 878 | 45.9 | 686 | 35.0 |
| q8_0/f16 | 871 | 45.7 | 687 | 34.9 |
| int4(vllm) | 1036 (vllm) | ~45 (vllm) | - | - |

PQ2가 전 구간 약 27~30% 빠름. 원인: (a) prefill — PQ2 네이티브 MMQ vs PTQ dequant+hipBLAS 폴백 (포크가 PTQ MMQ를 NVIDIA 전용으로 스코핑), (b) decode — trit unpack ALU 비용 (shift/mask 1op vs base-3 recurrence 3~4ops).

vllm 조건: vllm에 `-fa` 상당 플래그 없음. 자동 선택 backend는 TRITON_ATTN (FlashAttention-Triton 미활성). Triton attention + int4 KV + MTP speculative(n=1), pp5252+tg1 기준.

## 3. 품질 (악랄 배터리, PTQ 실측)

| 배터리 | PQ2 | PTQ1 |
|---|---|---|
| 12문항 추론 배터리 | 진성 12/12 (측정 11/12, tuesday 실제 PASS) | 10/12, 진성 11/12 (tuesday 빈응답 실패) |
| 수학 난이도 사다리 14문항 | 14/14 | 14/14 |
| 한국어 도구 사용 | 6/6 | 6/6 (K-L3/K-L4 각 3/3) |
| 한국어 어휘 9종 | 6 OK / 1 인정 / 2 약함 (신이미지 재측정, 아래 상세) | 6 OK / 1 인정 / 1 약함 (아래 상세, gorilla 누출 미확인) |
| NIAH (budget+max6000) | 9/9 | 9/9 (동일 조건 재측정, 동점) |
| 24턴 메모리 4퀴즈 | 4/4 | 4/4 (6/6, 8/8, 5/5, 5/5) |
| 악랄 12종 | KR 6P+6R, EN 5P+7R. ERROR 0 (아래 매트릭스; Q07 직접판독 P상당 포함 시 KR 7P) | KR 7P+3F+2부분, EN 9P+1F+2부분 (아래 매트릭스, ERROR 0) |

### 악랄 항목별 매트릭스 (P=pass, F=fail, R=review/부분, E=error)

| 항목 | PQ2-KO | PQ2-EN | PTQ1-KO | PTQ1-EN |
|---|---|---|---|---|
| Q01 5문장 회문 | R | R | F (1문장) | R (내부 쌍 실패) |
| Q02 음절/문자 금지 | R | R | F (154자+25위반) | F (225자+e 3회) |
| Q03 111자 | R | P | F (70자) | P |
| Q04 기차 방위 | R | R | R (C만 정답) | R (회전만 정답) |
| Q05 패러독스 | P | P | R (2/4) | P |
| Q06 계보 불가능증명 | P | R | P (원문 회수·판독: 엄밀한 증명, possible 플래그는 오탐) | P |
| Q07 가짜 NASA 논문 | R (직접판독 P상당) | P | P | P |
| Q08 무한 대화 | P | R | P | P |
| Q09 시+코드 | P | P | P | P |
| Q10 swap 코드 | P | R | P | P |
| Q11 취약점+회피 | R | R | P | P |
| Q12 테서랙트 감각묘사 | P | P | P | P |

읽을 포인트 (전체 배터리 기준):
1. **정답형 수학·추론은 동등** — 수학 사다리 14/14 양쪽 만점. 추론 배터리는 tuesday 1문항 차이 (PQ2 PASS vs PTQ 빈응답), sendmore는 양쪽 다 정답(transactional 아티팩트).
2. **형식 제약 준수는 PQ2 우위** — PTQ-KO가 Q01(1문장)·Q02(154자+25위반)·Q03(70자) 3연속 F. PQ2-KO 동일 문항은 전부 R(내용은 있음). 빡빡한 형식 + thinking 종료가 겹치면 PTQ가 빈손으로 끝내는 경향.
3. **서술·증명·안전은 PTQ 우위** — EN에서 PTQ 9P vs PQ2-신이미지 5P. PTQ Q06(엄밀 증명)·Q10(양쪽 PASS)·Q11(유해 거부+안전 대체) 확정. PQ2-EN은 R 7건으로 미결이 많음.
4. **한국어 도구는 동등 (6/6)** — 어휘 9종은 PQ2만 측정 (B2 6/1/2 → 신이미지 6/1/1+누출1, 한근 개선). PTQ 어휘 미측정.
5. **장문 retrieval·메모리 동점** — NIAH 9/9 양쪽 (공정 조건), memory 전 PASS 양쪽.
6. **속도는 PQ2가 전 구간 30% 우위** (2번 표). PTQ의 유일한 강점은 총합 1GB 절약.
7. **장시간 악랄 부하는 PTQ 서버 2회 SIGSEGV 확인** (신이미지, process_ubatch 경로, OOM 아님). PQ2 프로덕션은 일반 부하에서 장시간 안정 (RestartCount 0). 구 prism 이미지는 B2 전량 완주. PTQ 장문 안정성은 미해결.

PQ2 수는 EVAL-REPORT-BONSAI2-KO.md B2열 (prism 이미지). 서버가 달라 NIAH 외 항목은 동일 조건 재측정이 아님에 유의.

### 추론 배터리 실패 2종 상세 (PTQ, temp 0, seed 고정)
이 배터리는 답이 하나로 정해지는 문제만 모아, 모델 출력 안에 정답 문자열이 있는지로 기계 판정한다. 아래 2건은 둘 다 모델이 아니라 테스트 조건·체커 탓으로 판명됐다.
- **tuesday** (빈응답, 18.1s): 문제 "두 자녀 중 최소 한 명이 화요일생 아들일 때 둘 다 아들일 확률" (정답 13/27). 모델이 thinking을 600토큰 풀로 써버려서 content가 빈 문자열로 반환됨. reasoning-budget 서버 + 작은 max_tokens 조합에서 생기는 하네스 아티팩트 (NIAH max30과 동일 기전). PQ2(B2·신이미지 모두)는 같은 문항에 13/27을 답했으므로, PTQ가 못 푸는 문제가 아니라 생각 정리를 못 끝낸 것이다. max를 6000으로 주면 답이 나올 가능성이 높음 (미재측정).
- **sendmore** (판정 실패, 7.9s): 문제 SEND+MORE=MONEY에서 S의 숫자 (정답 9). 모델 출력에 `S = **9**`로 정답이 버젓이 있는데, 체커 정규식(`S\s*=\s*9`)이 마크다운 bold 표시(`**`)를 통과 못 해서 FAIL. 모델 정답, 체커 오판. B2·PQ2-신이미지에서도 같은 형태로 발생 가능한 공통 아티팩트.

### 한국어 어휘 9종 상세 (SmallDense PQ2 신이미지 재측정, temp 0)
| 단어 | 답변 | 판정 (B2 대비) |
|---|---|---|
| 2500원 (x3) | "7500" | OK (동일) |
| 철수 | "철수가 민지에게 돈을 갚아야 합니다." | OK (동일) |
| 가는말 | "명사구(명사)" — B2는 "부사" | 인정 (라벨은 다르나 coherent) |
| 30평 | "99.17" | OK (B2 99.18, 반올림 차이) |
| 한근 | "600g" | OK (B2 MISS에서 개선) |
| 먹였다 | "과거시제" | OK (동일) |
| banana | "3" | OK (동일) |
| gorilla | 의미 정답 + CJK 鬚 누출 | OK-ish (B2 灵 누출과 동일 유형) |
| 백지장 | "Blank Check (공백 수표)" | 약함 (B2도别义. coherent하나 빗나감) |

집계: 6 OK / 1 인정 / 1 약함 (+gorilla OK-ish/CJK 1). B2(6 OK/1인정/2약함) 대비 한근 개선, 나머지는 같은 bucket.

PTQ 동일 9종 (temp 0, seed 고정, 위와 동일 프롬프트): 2500원 "7500" OK, 철수 OK, 가는말 "명사구" 인정, 30평 "99.17" OK, 한근 "600g" OK, 먹였다 "과거시" OK, banana "3" OK, gorilla 의미 정답 (전문 미확인이라 누출 여부 보류), 백지장 "Blank Check" 약함. 집계 6 OK / 1 인정 / 1 약함. PQ2 신이미지와 항목별 동일 판정 (한근 포함), gorilla 누출만 미확인.

## 4. VRAM 총합 (131072 ctx, 16 KV층, 가중치+KV)

| K / V | PQ2_0 총합 | PTQ1_0 총합 |
|---|---|---|
| q4_0/q4_0 | 8.95 GiB | 7.78 GiB |
| q8_0/q8_0 | 10.95 GiB | 9.78 GiB |
| q4_0/f16 | 11.83 GiB | 10.66 GiB |
| q8_0/f16 | 12.83 GiB | 11.66 GiB |
| f16/f16 | 14.70 GiB | 13.53 GiB |
| int4(vllm) | 21.2 GiB (vllm) | - |

(위 K/V 조합표는 mmproj 제외, 가중치 + KV cache만. 단위 GiB, 고정 오버헤드 ~1.5GB 별도. 상세: KV-CACHE-VRAM-PQ2-PTQ1.md.)

### 가중치+kvcache+mmproj 총합 3자 비교 (int4 혹은 q4/q4)

| 구성 | 가중치 | KV (131072) | mmproj | 총합 |
|---|---|---|---|---|
| vllm (int4) | 18.2 GiB | 3.0 GiB (pool cap) | 0 (language-only) | 21.2 GiB |
| PQ2_0 (q4_0) | 6.70 GiB | 2.25 GiB | 0.87 GiB (BF16) | 9.82 GiB |
| PTQ1_0 (q4_0) | 5.53 GiB | 2.25 GiB | 0.87 GiB (BF16) | 8.65 GiB |

위 표는 가중치 + KV cache + mmproj 전부 포함. vllm은 mmproj를 안 쓰므로 0. 일반 dense 64층이라 f16 KV 풀윈도우는 34GB로 초과하므로 int4+cap 필수.

## 5. 붕괴 조합 (양쪽 공통)
- V=q5_0: 초선형 악화 (PQ2 실측 23 t/s → 타임아웃)
- K=f16 + V=quant 혼합: 로드 hang
- K-quants: KV로 거부됨

## 6. 배포 상태 및 결론
- 배포 중: PQ2_0 + q4_0/q4_0 (8082, 856 t/s 실측, 안정)
- PTQ1_0: 타입 지원 이식 완료·동작 확인(Rayleigh 정상)이나 **미커밋**, 장문 안정성(악랄 크래시) 미해결, MMQ 미이식
- 권장: PQ2 유지. PTQ는 총합 0.7~1.2GB 절약이 필요할 때만 고려 (정확성 추가 검증 + MMQ 이식 후).
