# BUG: MTP draft가 긴 프롬프트(>~300 tok)에서 silent하게 죽음

> 발견: 2026-09-06, RCCL 빌드 (beellama-boosts/src, rdna-test, MTP 재작성 포함)
> 영향: --spec-type draft-mtp / draft-mtp-adaptive 사용 시, 프롬프트가 ~300토큰 이상이면
>        생성 직후 draft acceptance가 0으로 떨어짐 (mean len=1.00 = 드래프트 자체가 안 생성됨)

## 재현 데이터 (27B UD-IQ4_XS + 별도 mtp 헤드, 듀얼 tensor, f16, n-max 3)

| 프롬프트 | draft acceptance | tg t/s |
|---|---|---|
| 75 tok | 0.66 | 73.6 |
| 123 tok | 0.57 | 73.5 |
| 296 tok | 0.0 | 27.5 |
| 339 tok | 0.0 | 27.4 |
| 420 tok | 0.0 | 28.1 |
| 1352 tok | 0.0 | 27.5 |
| (30 tok prompt + 8634생성) | 0.51 유지 | 67.1 |

## 관찰
- 드래프트가 죽으면 tg가 ~73 → ~27 (MTP 이득 전체 소실, MTP 그래프 오버헤드만 남음)
- 짧은 프롬프트 후 **긴 생성**에서는 죽지 않음 (8634 토큰 롱런 accept 0.51 안정)
- 문제는 **긴 프롬프트를 먼저 먹인 뒤 첫 생성**에서 드래프트 시드가 초기화 안 되는 패턴
- kv_unified, parallel, ubatch, ctx 크기는 무관 (전부 배제 확인)
- reddit 사용자(86K 프롬프트, upstream/beellama 정식)는 정상 동작 → beellama-boosts의
  MTP 재작성(lama_set_embeddings_nextn 경로) 특유 회귀로 추정

## 원인 추정 위치
common/speculative.cpp draft-mtp: `pending_h`(n_past 캐리오버)가 긴 프롬프트 후
첫 draft() 호출에서 이전 process()의 hidden row와 짝이 안 맞는 경로.
0172 draft()/0289 draft() 진입, pending_h/verify_h 캐리오버 로직 대조 필요.

## 우회책 (당장 운영 가능)
1. 긴 프롬프트는 **두 번에 나눠** 보내 첫 요청을 300토큰 미만으로 유지
2. 또는 긴 프롬프트 세션은 --spec-type none (tensor split 단독: 27B에서 34.7~42)
3. draft-mtp 대신 draft-mtp-adaptive (자체 depth 튜닝이 경계를 피할 수 있는지 미확인)

## 상태: 코드 수정 필요 (이번 세션 범위 밖) - 다음 세션에서 pending_h 초기화 경로 조사

## 추가 검증 (2026-09-06 후속)
- 별도 MTP 파일(-md mtp-Qwen3.8-27B-Q4_0.gguf)로도 동일 발생 — nextn 임베디드와 무관, draft-mtp driver 버그 확정
- draft-mtp-adaptive(--spec-draft-n-min-adaptive 2)도 826토큰 프롬프트에서 accept 0.0 → 전체 draft-mtp 계열 해당
- Q8 관련: "Q8 transmission"(all-reduce wire를 Q8로 압축, pp 1390달성)은 **모델 다운로드와 무관한 코드 패치**.
  현재 소스의 wire는 BF16까지만 (allreduce.cu T_wire/BF16 round-trip). Q8 wire는 미구현 상태.
