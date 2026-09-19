# Bonsai 27B 서빙 & 평가 리포트

Date: 2026-09-13 ~ 09-14. Stack: 7900XTX 2장 (gfx1100), ROCm 10, llama-cpp-turboquant (rocm10).

## 1. 아키텍처 (배포 기준)

| 서비스 | 컨테이너 | GPU | 포트 | 모델 |
|---|---|---|---|---|
| llm-main | llm-main | 0 (단독, 이전엔 0,1) | 8081 | Dense Qwen3.8-27B-MTP-Q4_K_M, ctx 163840 |
| bonsai | bonsai | 1 | 8082 | Bonsai-27B-Q1_0, ctx 65536 |
| gate-proxy | gate-proxy | - | 1709 | gate_proxy_v2.py -> bonsai:8080 |

- 레지스트리 (`localhost:5000/baramofme/llama-cpp-rocm`): `rocm10-builder-ccache`,
  `rocm10-gfx1100-rccl-rdnaboosts-mtp-q1` / `-q2` (바이너리 동일, 태그만 구분),
  `stock-b10964-rocm10-gfx1100` (upstream 비교용),
  공식 `ggml-org/llama.cpp:server-rocm-b10951` (22.8 GB) 로컬 미러 완료.
- Dokploy 컴포즈 `bonsai-sghcma` (`h5QEsfsllhIBuagdspK0t`): Q1 활성, MTP-Q2_K 주석.
- OpenWebUI: `http://gate-proxy:1709/v1` (Dokploy망) 또는 `http://bonsai:8080/v1` (도커망).
  커스텀 모델 `보조` (id `e2b--`, base `bonsai`)에 시스템 프롬프트 + temp 0.7 설정.

## 2. 모델 팩트 (추정 아님, 실측)

- Bonsai-27B-Q1_0.gguf: 3.8 GB. 가중치는 **Q1_0 (type 41)**,
  1비트 이진 {-1,+1}, 128개당 FP16 scale 1개 = **실효 1.125 bpw** (코드+PrismML 문서 일치).
- Ternary 파일은 **Q2_0 (type 42)**: {-1,0,+1} 2비트 패킹, 64개당 = **2.25 bpw**
  (TQ2_0 (type 35, base-3 패킹)과 다름! ).
- 아키텍처 `qwen35`는 **하이브리드**: full_attn_interval=4라서 64층 중 **16층만
  full-attention**이고, 나머지는 고정 크기 state의 linear (GDN) 층.
  결과: KV 캐시가 순진한 full-GQA 계산의 약 1/4.
- 토크나이저/vocab 248320은 Q1/Q2/MTP 변종 공통. embd 5120.

## 3. 영어 출력 강제 스택

Q1/Q2는 한국어 생성이 약함 (다국어 붕괴: 힌디어/한자/아랍어 조각, 로마자 한글,
반복 루프). 4층으로 영어를 강제:

1. **서버 문법** (`grammars/english-only.gbnf`, 이미지 ENTRYPOINT 내장):
   ASCII-printable 허용 목록. 요청에 도구 없을 때 동작. CJK=0 실증済
   ("한국어로 답해" 유도 공격 포함).
2. **게이트웨이 v2** (`gate_proxy_v2.py`, 1709포트): 도구는 그대로 전달하고,
   메시지 content만 non-ASCII 검사, 위반 시 재시도
   (단문 지시 + temp 0 + 초안 미인용 — 장문 지시는 앵무새처럼 복사됨).
   클라이언트가 안 보내면 SYSTEM_EN 주입, 끝문장 "Respond in English
   only." 주입 (recency 효과, 5항 참조). 버퍼 후 SSE 방출. ThreadingHTTPServer
   (단일스레드가 94초 head-of-line 블로킹 유발 — 수정済).
   요청 로그에 `retries=`/`wall=`/`tools=` 기록 (지연 회계용).
3. **클라이언트 설정** (OpenWebUI 보조 모델): 시스템 프롬프트 (영어 전용 문구)
   + temperature 0.7. temp는 서버가 강제 못 함 (요청이 덮어씀) — 클라이언트에서 설정 필수.
4. **온도 규율**: 1.0은 grammar 재시도 폭증 유발. 0.7 이하.

### 도구 예외 (핵심 구조 문제)
- 요청에 `tools`가 있으면 서버가 `--grammar-file`을 **버리고**
  자동 생성 lazy tool-call 문법으로 교체 (`server-common.cpp:1342`).
  lazy = tool 호출 트리거 전까지 휴면, 자유 텍스트는 무제한 통과.
- 코드 + 재현으로 확정 (도구 있음 CJK=27, 없음 CJK=0).
- Upstream Discussion #22408 확인: grammar 슬롯은 하나, 공존 불가.
  공식 권고는 "도구 있으면 grammar 끄라".
- 관련 upstream 취약점 기록: maxLength 큰 tool 스키마 400 에러,
  ~58개 누적 문법 상한, 일부 템플릿에서 tool_choice:required 무시됨.

### 끝문장 지시 발견
- 10k 도구 토큰 밑에 깔린 시스템 프롬프트는 조향력 상실 (6/6 위반).
- 동일 조건 + 끝문장 "Respond in English only." → 0/3 위반,
  도구 호출도 3/3 정상. 게이트 주입으로 구현 (템플릿 파일 수정 아님 —
  Qwen 네이티브 tool 포맷 포크 회피).

## 4. VRAM 발견 (도중 정정됨)

- 순진한 full-GQA KV 계산은 틀렸음 (12~13 GB 주장 철회). 하이브리드 실측:
  Q1 @65k 약 1.7 GB, Q2 @122k 약 3.2 GB.
- GPU1 내역 (Q1): 가중치 3.8 + KV 1.7 + mmproj 없음 + compute/graph ≈ 6~11 GB.
- KV 타입 속도 매트릭스 (llama-bench pp512, Q1, 통제됨, 편차 작음):

| KV | PP | TG | 판정 |
|---|---|---|---|
| f16 | 1184 | 80.4 | 최고속, +2.5 GB |
| q8_0/q8_0 | 1182 | 76.5 | **채택급: 전속도, -1.9 GB** |
| q4_0/q4_0 | 1180 | 73.4 | **채택: 전속도, 최소 VRAM** |
| q5_0/q5_0 | 302 | 69 | 저속 경로, 사용 금지 |
| q8_0/q4_0 | 356 | ~72 | 저속 경로 (혼합 K/V가 fast FA 빗나감), 사용 금지 |

- 규칙: fast 패스는 균일 조합만 (f16, q8/q8, q4/q4). 혼합·q5는 폴백.
- 프로덕션: KV q4_0/q4_0 (K 축소 금지는 데이터 확인 후 사용자가 해제).
- 참고: ternary에서 q4_0 K 품질 우려 ("파국") 있었으나 Dense가 운용 중.
  --kv-mean-center (이슈 #85) 기록만 하고 미적용.

## 5. 속도 발견

- 벤치 상한 (7900XTX, Q1): pp512 1184, tg 80 — published ROCm 수치
  (523~1030) 상회. 하드웨어/커널 무죄 입증.
- 서버 PP는 프롬프트 길이에 따라 감쇠 (O(n2) 성향 + 슬롯 상태). 단문 서버
  턴은 경합에 따라 300~1100 오락가락. 공유 슬롯 상태 통제 없이 수치 비교 금지.
- 실전 지배 비용 (실측): -np 1 대기열 (거대 턴 뒤 수 분), 5~20k 히스토리
  prefill, temp 1.0에서 grammar 재시도.
- ubatch 512 vs 1024: 차이 없음 실측. mlock 유지. ngram-simple 유지
  (효과 미분리, 무해).
- SWAR q1_0 패치 (300ddf528)가 배포 바이너리에 빠져 있었음
  (커밋 25분 전 빌드) → 리빌드 + 재배포. TG 62~64.
- 하이브리드 디스패치 실험 (Q1_0 decode를 cublas로): TG 64 -> 23 + VRAM +8 GB.
  롤백済. 교훈: 이전 CUBLAS A/B는 교란됨 (ngram+슬롯 차이). vecdot 유지.
- FORCE_CUBLAS 전체 리빌드 테스트: PP 1/4 토막. 기각.
- 게이트웨이 비용: 첫 바이트에 생성 1회분 (버퍼 SSE). 전체 wall 시간은
  거의 동일. 수정済: 단일스레드 블로킹 (스레딩), 크래시 루프 (import),
  앵무새 유출 (단문 지시), 폴링 강건성 (try/except).

## 6. 품질 벤치마크 (규칙 채점, temp 0)

배터리 (`scored_battery.py`, 12문항: 사칙, 퍼센트, 방정식, 자릿수,
팩토리얼, 화요일, 암호식, 고양이 오류, 회문, 소수, 상자, 몬티홀):

| 모델 | 점수 | 탈락 |
|---|---|---|
| Q1 | 10/12 | 방정식 (11→11/2), 화요일 (→1/3) |
| Q2-MTP | 11/12 | 화요일 (→1/3). 암호식은 S=9 정답이나 장황 서술로 체커 탈락 |

- 공통 사각지대: 화요일 소년 모호성 (둘 다 1/3, 정답 13/27).
- 어려운 추론은 Q2 우위 (방정식), 나머지는 동등.

NIAH (`niah_probe.py`, needle 73921):
- Q1: 9/9 (8k/24k/48k × 10/50/90% 깊이)
- Q2: 6/6 (8k/24k × 깊이)

멀티턴 기억 (`en_memory_probe.py` 12턴 + `en_memory_24.py` 24턴):
- Q1: 24턴까지 4/4. Q2: 24턴까지 4/4. 열화 없음.

## 7. 모델 추천

- **서브에이전트용 (OpenWebUI, bounded)**: Q1. 상한 **20턴** + 턴당 ~2k 토큰
  예산 (합 40k < 65k ctx, 마진 확보). 24턴까지 품질 벼랑 없음 확인済.
  상한 근거는 품질이 아니라 지연/VRAM.
- **일상 hermes형 에이전트** (수 단계 도구 호출): 게이트웨이 경유 Q1
  (도구 통과, 내용 강제). Q2-MTP는 추론 난이도가 요구하고 +8 GB VRAM +
  느린 응답을 감수할 때만.
- Q1을 기본 배포 모델로 유지.

## 8. MTP 실험 (판정済)

- vinpix `Ternary-Bonsai-27B-MTP-Q2_K.gguf` (9.77 GB, SHA 검증):
  로드됨, MTP 초기화됨, **수용률 0.43~0.51 일반 / 0.58~0.87 반복문**,
  그러나 TG 42 < Q1의 63, PP 31 (RDNA3에서 Q2_K MMQ 느림). 롤백.
  파일은 디스크 보관.
- dspark draft: 로드 불가 (우리 Q2_0 대비 type-42 인코딩 불일치). 보관.
- Q1+MTP 병합 (`merge-q1-mtp.py`): 기술적 성공 (866텐서,
  block_count=65 교훈 — trunk 수 + MTP, 아니면 로더가 blk.63.nextn 탐색),
  로드됨, MTP 초기화됨, 그러나 **수용률 0.00** (Q1 이진 vs Q2 삼진
  hidden 분포 불일치). 폐기, 스크립트 보관.
- MTP는 trunk 분포가 맞아야 함. 재학습은 미착수.

## 9. 배포 인벤토리 (현재)

- bonsai (Q1): ctx 122768, KV q4_0/q4_0, ubatch 1024, mlock, alias bonsai,
  mmproj 복원済 (vision 실측 통과), 문법 내장, 8082, GPU1. MTP-Q2_K 주석 보관.
- gate-proxy (v2): 1709, BACKEND http://bonsai:8080. 클라이언트는
  `http://gate-proxy:1709/v1` (Dokploy망) — 직접 8082는 도구 끌 때만.
- llm-main: GPU0 단독 (기존 0,1). Dense ctx 163840. 공유 GPU 분리済.
- OpenWebUI 보조 모델: base bonsai, 시스템 프롬프트, temp 0.7.
- 기지 취약점 목록: -np 1 직렬화 (거대 턴이 전부 막음). Anytype MCP
  (37개 도구, 약 10k 토큰) 전역 자동 장착 — 채팅별 정리 안 하면 매턴
  5~20k prefill. Inlet 압축 필터 가동 중 (임계 12000, Dense 요약).
  error 상태 메시지 (done:false)는 전송 버튼을 벽돌로 만듦 (새 채팅 쓸 것).
  유령 모델 id (e2b--/Moe 패턴) 조용히 라우팅 불가.

## 부록. 과제 전문 (문제지)

### A. scored_battery.py (12문항, temp 0, seed 고정, 규칙 채점)
1. **arith**: "What is 17 multiplied by 23?" → 정답 391
2. **percent**: 80000원 25% 할인 후 추가 10% → 정답 54000
3. **equation**: (2x+5)/3-(x-1)/2=4 → 정답 x=11
4. **digit**: 자릿수 합 11, 뒤집으면 27 작음 → 정답 74
5. **factorial**: 1,2,6,24,120 다음 → 정답 720
6. **tuesday**: 화요일생 아들 최소 1명 조건 둘 다 아들 확률 → 정답 13/27
7. **sendmore**: SEND+MORE=MONEY → 정답 S=9 (9567+1085=10652)
8. **cats**: "고양이는 네 발, 내 개도 네 발, 그러므로 개는 고양이" 오류 지적 → 후건긍정(affirming the consequent) 포함 여부
9. **palindrome**: 'babad' 최장 회문 → bab 또는 aba
10. **prime**: 100보다 큰 최소 소수 → 101
11. **boxes**: 라벨 전부 오기재 3상자, 1회 열람으로 전수 판별 → mixed 상자에서 꺼내기 포함 여부
12. **monty**: 3문 중 변경 여부 → switch 포함 여부

### B. niah_probe.py (NIAH)
- needle: "The special access code for the Atlas project is 73921."
- haystack: 시의회 정례문단 반복. 깊이 10/50/90%, 총량 Q1 8k/24k/48k, Q2 8k/24k.
- 질문: 번호만 답. 정답 문자열 "73921" 포함 여부로 채점.

### C. en_memory_24.py (24턴 기억)
- intro: 이름 Alex, 파리 거주, 개 Coco, 여자친구 Mia, 예산 300달러, 바닐라 선호/초콜릿 기피, 행운숫자 7, 개발자, 베를린 여행 예정, 주말 수영.
- f1~f9: 일상 filler 9턴 (언어, 프로젝트, 산책, 선물, 주말계획, 날씨, 커피, 방문, 데이트).
- quiz (10문항 회상), final (예산+아이스크림+7+5+베를린 종합).
- f10~f17 추가 팩트: 형 Tom(리옹 셰프), 빨간 자전거, 영화 Inception, Coco 목욕 기피, 이사 후보 Marseille, 피아노 학습.
- quiz2 (8문항), final2 (종합). 채점은 키워드 포함 개수.

## 10. Dead ends 목록 (새 정보 없이 재시도 금지)

- Gate v1 + Dense 역번역 (느림, 불필요 — 출력은 영어 유지).
- dspark draft (type-42 불일치).
- Q1+MTP 병합 (수용률 0%).
- 하이브리드 디스패치 패치 (TG 3배 악화, 롤백. 교란 A/B 교훈).
- FORCE_CUBLAS 전체 전환 (PP 1/4 토막).
- 템플릿 파일 수정 (리스크 > 이득. 게이트 주입이 recency 커버).
- 서버 문법 + 도구 공존 (구조적 불가. upstream 확인).
- 템플릿/문법 레벨 CJK 도구 인자 차단 (게이트가 인자 통과).
- ROCm에서 ubatch를 PP 레버로 (실측 무효과. 큰 배치는 저속 FA 경로 위험).
- ctx 크기를 PP 레버로 (8192 vs 65536 동일).
- 스톡 이미지 비교 (ghcr `server-rocm-b10951` 풀+로컬 푸시 완료).

## 11. 게이트웨이 하드닝 로그 (배포 후 수정분)

- 단일스레드 head-of-line 블로킹 (단독 94초) → ThreadingHTTPServer. 동시 2요청 1.1초 검증.
- 재시작 크래시 루프 (import 참조 누락) 1건 발생·수정. 교훈: 재시작 전 syntax + 즉시 health 확인.
- 앵무새 유출 (장문 재시도 지시가 출력에 복사됨) → 단문 지시 + 초안 미인용 + 재시도 temp 0.
- ASCII-only 검사로 강화 (이모지까지 차단. CJK 정규식만으로는 부족).
- SYSTEM_EN 자동 주입 (클라이언트 미전송 시) + 끝문장 주입 (도구 있어도 recency primes 영어).
- 요청 로그 `retries=/wall=/tools=` (게이트 지연 회계용).
- 폴링 중단 BrokenPipeError try/except 처리.

## 12. Q1 수학 사다리 (14문항, 풀이 전문 판독)

- 통과: 나눗셈·백분율·소수·제곱근·1차·연립·이차·베이즈 장문·피보나치 (풀이 정확).
- 답 맞고 풀이 틀림이 아니라 반대 케이스 다수: 1234×5678 과정 정확·합계 오기,
  베이즈 장문 정확→단답 강제 시 기저율 무시(1%).
- 진성 탈락: 분수 방정식 (11→11/2), 화요일 문제 (→1/3).
- 디코딩 병리 (수학 문제가 아님): 번호 리스트 "1. 1. 1..." 무한루프 3건,
  타 턴 지시문 에코 (슬롯 KV 오염).
- 결론: 중~고1 수준 풀이 정확. 단답 강제 금지, 번호 매기기 프롬프트 금지,
  분수대수·조건부확률은 불신. (채점자 본인도 연립 1회 오채점 후 정정.)

## 13. OpenWebUI 장애 사가 (교훈 위주)

- Anytype MCP 37개 도구 전역 자동 장착 → 매턴 10~20k prefill. 채팅별 정리 필수.
- Inlet 압축 필터: 임계 12000, 요약은 느린 Dense 경유. 'coroutine' 에러 로그는 무해.
- done:false 에러 메시지가 전송 버튼을 벽돌로 만듦. 새 채팅으로 회피.
- 유령 모델 id (e2b--/Moe 패턴): 라우팅 불가인데 조용히 실패. base 확인 필수.
- 14k 토큰 폭주 생성 1건 (단일 슬롯 점유) → e2b--에 num_predict 2048 상한 설정.
- 브라우저 미전송 이슈 다수: 서버 무흔적이면 브라우저 측 (SW/확장/묵은 탭). 시크릿 창이 판별기.
- 재시작 창(~2분) 전송분은 에러 행으로 쌓임. 재시작 예고 + ready 후 전송 규칙.

## 14. 미결 항목

- Hermes (`~/.hermes/hermes-agent` 발견): 기본 모델 Q1 변경 + 턴 제한 미적용.
- llm-main on-demand 모델 공존 시 단일 GPU OOM 가능 (미검증).
- ngram-simple 효과 미분리 (무해 추정, 제거 여부 미결).

## 15. 장문 래더 + 게이트 500 재시도 (2026-09-14 심야)

- Q1 장문: 11.9k→PP 1045, 12.4k→855, 21.4k→644 (캐시 어시스트 포함 실전 수치).
- Q2-MTP 장문: 12k→PP 816/수용 0.87, 25k→627/0.63, 100k→460/0.58.
  수용률은 반복문에서 높음. Q2 사이드카는 사용 후 종료, rocm10-dev 좀비
  정리겸 재시작 (GPU1 5.9GB로 회복).
- 도구 호출 JSON 파손 500 (parse error col 58, 중첩 브레이스): 모델이 뱉은
  인자가 깨지면 서버가 500. 게이트웨이에 백엔드 500/502/503 1회 재시도 추가
  (seed 변경, 로그 기록). 처방 병행: 도구 수 축소, 도구 턴 temp↓.
- Anytype MCP 37개 전역 자동 장착 확인 (매턴 10k+ 토큰). 채팅별 정리 필요.
- 14k 토큰 폭주 생성 1건 → e2b--에 num_predict 2048 상한.

## 16. 게이트웨이 진화 2차 (500 대응)

- 백엔드 500/502/503 1회 재시도 (seed 변경). 빈 응답 방지용.
- 재시도는 temp 0 확정 재생성 (유효 JSON 확률 최대).
- JSON 수리층: 이어붙음 분리·trailing 정리·추출 후 검증. 단위 테스트 통과.
  수리/불가 로그 (`repaired=`/`unrepairable=`).
- $ref 포함 도구는 사전 제거 (서버 문법 빌드 400 방지). 제거 목록 로그.
  근본은 OpenWebUI 변환기가 components를 떨굼. mcpo 스펙 자체는 정상 확인済.
- 무제한 요청에 max_tokens 8192 기본 삽입 (어느 경로로 와도 폭주 불가).
- 재시도 시간 예산 60초 (느린 턴 무한 재시도 방지).
- 상태 표시: 도구 검사·재시도 발생 시 reasoning_content로 전달 (본문 무오염).
- 미적용 보류: CJK 위반 재시도에서 tools 제거안 (에이전트 정합성 저하 우려).

## 17. 장문 래더 실측 (Q1/Q2)

- Q1: 11.9k→PP 1045, 12.4k→855, 21.4k→644 (동일 파일 재측정 시 111↔1083 진동 —
  서버 수치 단독 비교 금지. 슬롯 경합 탓. bench만 신뢰).
- Q2-MTP: 12k→815/수용 0.87, 25k→627/0.63, 100k→460/0.58.
- prefix 캐시 주의: 동일 문단 반복 프롬프트는 2회차부터 캐시 적중 (측정 오염원).

## 18. OpenWebUI 운영 이슈 모음

- WEBUI_SECRET_KEY 미설정 → 재배포마다 세션 전멸 (401). 고정값 박음. 이후
  재배포 시 1회 재로그인으로 종결.
- Anytype MCP 37개 전역 자동 장착 (매턴 10k+). 채팅별 정리 필요 (미해결).
- Inlet 압축 필터 summary는 느린 Dense 경유. 'coroutine' 에러 로그 무해.
- done:false 에러 방은 전송 버튼 벽돌. 새 방 사용.
- e2b--/Moe 유령 모델 사건: base를 bonsai로 고쳐 해결. num_predict 8192.
- 401 이후 무응답 패턴 다수는 브라우저 미전송 (서버 무흔적) — 시크릿 창 판별.
- Brook: 단일 슬롯 14k 폭주 2건. num_predict 상한 + 게이트 기본값으로 봉인.
