# CMP 170HX 구매 결정 문서 (2026-09-07 확정)

> 하드웨어 구매 최종 결정 기록. 주문 1 (8GB × 2, 64GB 언락 목표) PayPal 결제 완료.
> 관련: SESSION-2026-09-06-MASTER.md (벤치 세션), IMAGE-BUILD-GUIDE.md (운영 이미지)

---

## 1. 최종 결정 요약

| 항목 | 값 |
|---|---|
| 상품 | NVIDIA CMP 170HX 8GB × 2 (GA100, 0x20C2) |
| 목표 | 64GB × 2 = 128GB HBM2e (cmpunlocker 언락) |
| 주문 번호 | #28782233001038157 |
| 판매자 | Chongqing Yuqian Trading Co., Ltd. |
| 주문액 | USD 4,070 (프로모션 -$215.79 + 쿠폰 -$30) |
| 결제 | PayPal (할인 -$150, 수수료 +$117) |
| PayPal 총액 | USD 4,037 |
| 인코텀스 | DAP (관세 0%, 부가세 10% 구매자 부담) |
| 부가세 (한국 납부) | ~USD 404 |
| **총 실지출** | **USD 4,441 (≈ 603만원, 환율 1,357)** |
| 카드당 실효가 | USD 2,220 |
| KS 마크 | 있음 (통관 폐기 리스크 없음) |
| 보호 | Trade Assurance + PayPal Buyer Protection (이중) |
| 언락 | 구매자 직접 (셀러 책임 아님 — 계약 명시) |

---

## 2. 왜 8GB(64GB) 2장인가

### 2.1 니 워크로드 기준 (세션 전체 결론)

```
카드 A (64GB) ── vLLM 서빙 (Flash-Next M64 45.8GB / Qwen3.8-27B, DFlash2)
카드 B (64GB) ── MiniMax H3 (int8/fp16 고품질) ── 쇼츠 공장
7900 XTX ×2   ── A3B 131K (105 t/s) 감시/코딩 + SDXL 이미지
```

- 40GB(10GB 카드)로는 Flash-Next M64(45.8GB) 구조적 불가 → 64GB 필수
- "동시 생산"(LLM 서빙 + 영상 생성 병렬) = 카드 2장 필요
- 74만원 차이(주문2 vs 주문1)로 +48GB = GB당 1.54만원 — 시장 평균(4.69만/GB)의 1/3

### 2.2 검토된 대안 (기각 사유)

| 대안 | 사유 |
|---|---|
| 10GB × 2 (40GB, $3,605 DAP) | 40GB는 45.8GB Flash-Next 불가. 안전하지만 목표 미달 |
| 오늘 딜 ($4,325 DDP 64GB×2) | **"Alibaba online transactions 미지원" → 먹튀 패턴 → 기각** |
| 주문 1 대신 주문 2 | unlock 보장 없음에도 +48GB 가치가 리스크보다 큼 (unlock은 직접 가능) |

---

## 3. 핵심 기술 사실 (의사결정 근거)

### 3.1 170HX 언락 현황

- 8GB (0x20C2, Hynix) → 64GB: **가장 안정 경로** (커뮤니티 수백 장 검증)
- 10GB (0x2082, Samsung) → 40GB: 안정. **80GB는 구조적 불가** (GSP-RM 40GB cap, fuse/SKU 제어 불가)
- cmpunlocker (amoghmunikote): 드라이버 레벨 PLM/레지스터 언락, 콜드 부팅 필요, Secure Boot off
- 5년 채굴 카드 메모리 불량률: **약 6~7%** (채굴업자 80장 중 5~6장)

### 3.2 성능 수치 (1장 기준, Qwen3.8-27B)

| 지표 | 값 |
|---|---|
| raw decode | ~60~70 tok/s (래틀 취합) |
| prefill | ~800~1,300 pp/s (구성에 따라 2배 편차) |
| vLLM + DFlash2 (27B) | ~200 tok/s (포도나무 레시피) |
| Flash-Next M64 (llama.cpp) | 32~38 tok/s @ 262K 풀컨텍 (flur 실측) |

### 3.3 대역폭 (정정 기록)

- 언락 후 실측: **1,263~1,600 GB/s** (clpeak 1355 / STH copy 1263 / dev-forum 1600)
- 뉴스의 "700-800 GB/s"는 오류 → 1차 실측 기준 (7900 XTX 960 GB/s의 1.4~1.7배)

---

## 4. 구매 리스크와 방어

| 리스크 | 확률 | 방어 |
|---|---|---|
| 카드 사망/미인식 | <1% | Trade Assurance (불량품 환불) |
| unlock 후 셀 불량 (56~62GB) | 6~7% | 셀러 책임 밖 — 부분 사용 (Flash-Next 45.8GB는 그래도 동작) |
| unlock 실패 | 매우 낮음 | 8GB→64GB 안정 경로, cmpunlocker 직접 |
| 통관 폐기 | 낮음 | KS 마크 있음 |
| PayPal 환율/수수료 | 확정 | USD 청구 선택 (원화결제 DCC 금지, 환율 +1% 손해) |

**셀러 계약 범위 (문서화됨):**
> "우리가 파는 건 CMP170HX 8GB다. 64GB 언락 이후 메모리는 책임지지 않는다."

---

## 5. 도착 후 실행 절차

```
1. 부가세 ~$404 납부 (특송 통관대행)
2. cmpunlocker 설치 → 64GB 언락 (nvidia-smi 65536 MiB 확인)
3. memtest로 64GB 검증 (불량 셀 발견 시 부분 사용 or Trade Assurance 분쟁)
4. 카드 A = vLLM (Flash-Next W4A16/NVFP4 + DFlash2, 포도나무 레시피)
5. 카드 B = MiniMax H3 (int8)
6. 7900 XTX ×2 = A3B 131K (turbo4 KV, 105 t/s) + SDXL
```

### vLLM 레시피 (Flash-Next/27B 서빙용, 포도나무 기준)

```bash
vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --max-model-len 262144 \
  --kv-cache-dtype bfloat16 \
  --gpu-memory-utilization 0.88 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --speculative-config '{"method":"dflash","model":"$DRAFT","num_speculative_tokens":7}'
```

---

## 6. 참고 자료 (소스)

| 주제 | 소스 |
|---|---|
| 언락 도구 | https://github.com/amoghmunikote/cmpunlocker |
| 10GB→80G 연구 (40GB cap 증명) | https://github.com/ggualerz/cmp170hx10g-to-80g |
| 언락 가이드 | https://github.com/abobasixseven/unlock-cmp-170hx |
| HBM 대역폭 실측 | clpeak 1355 / STH 1263 / dev-forum 1600 GB/s |
| 64GB 1장 Flash-Next 262K (32-38 t/s) | arca 181231860 (flur) |
| 27B vLLM 200 t/s (DFlash2) | arca 181529007 (포도나무) |
| 가성비 비교 (4×170HX pp 4.5-5.3K) | arca 181050917 (래틀) |
| GPU 후기 (300만원=VRAM 가성비 1위) | arca 182246899 (참치맛사탕) |

---

## 7. 비용 총정리

```
주문액                 $4,070
PayPal 할인            -$150
PayPal 수수료          +$117
부가세 10% (DAP)      +$404
────────────────────────────
총 실지출             $4,441 ≈ 603만원 (환율 1,357)
카드당 실효            $2,220
```

---

## 8. 미완 / 대기 항목

- [ ] 15일 발송 대기 (지연 시 Trade Assurance 보상)
- [ ] 도착 → 부가세 납부 → 언락 → memtest
- [ ] 검사 서비스($48) 커버 여부 미확정 — 커버되면 추가 고려
- [ ] 카드 도착 후 니 구성(q8_0/q5_0 + draft-mtp-adaptive) prefill 실측