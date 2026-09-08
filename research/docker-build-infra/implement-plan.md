# implement-plan: 운영 이미지 빌드 + 벤치 하니스

> 상태: **적용 완료** (운영 이미지 + 벤치 스크립트). 아래는 재현용 계획.
> 원본 연구: `IMAGE-BUILD-GUIDE.md`, `bench-beellama.sh`, `bench-beellama-plan.md`, `bench-beellama-*.results.md`

## 목표

- theTom llama-cpp-turboquant + rdna-boosts + RCCL 을 ROCm 10.0 docker 이미지로 패키징.
- 재현 가능한 벤치 하니스(스모크/벤치/서버/batched) 제공.

## 빌드 이미지 (최종)

| 항목 | 값 |
|---|---|
| 이미지 | `baramofme/llama-cpp-rocm:gfx1100-rocm10-tbq-rboosts` (21.3GB) |
| 기반 | theTom llama-cpp-turboquant (`feature/turboquant-kv-cache`) |
| 추가 | rdna-boosts (clean 5 + 수동 3) |
| 런타임 | ROCm 10.0 (`rocm/dev-ubuntu-26.04:10.0.0-full`) |
| ENTRYPOINT | `["/app/llama-server"]` |

빌드 순서: 1) theTom 클론 2) rdna-boosts 13개 클린/충돌 분류 3) clean 5 git apply
4) 충돌 3개(0004/0010/0011) 수동 병합 5) ROCm 7.2.3 native로 컴파일 검증(빠른 피드백)
6) ROCm 10 docker 빌드 7) 2B turbo3 스모크 8) 운영 구성(27B 140K + adaptive MTP) 검증

**증분 빌드**: ccache 이미지 활용해 **7초** 증분 컴파일 (Track B 세션에서 확립).

## 벤치 하니스 `bench-beellama.sh`

- GPU1(renderD129/card2) 단독 사용, llm-main(GPU0) 간섭 없음.
- 하위명령: `smoke`(2B 파이프라인), `bench`(27B S1~S3 llama-bench), `server`(S4~S6 수용률),
  `batched`(S7 llama-batched-bench), `all`.
- 오버라이드 env: `BEELLAMA_IMG`, `MAIN_IMG`, `BUILDS`, `OUT`, `PORT`, `PROMPT`.
- 원시 출력: `bench-results*/`.

## 현재 운영 관련 (llm-main)

- llm-main 컨테이너: `baramofme/llama-cpp-rocm:rocm10-gfx1100-rccl-rdnaboosts-mtp-latest`
- Cmd: `--models-preset /app/config.ini --swa-full --host 0.0.0.0 --port 8080`
- 설정: `/opt/llm/llama-cpp/main-llm-config.ini` (host) -> `/app/config.ini` (container)
- 재기동: `/tmp/llm-main-restart.sh` (docker run 방식, compose 없음)

## 참고

- 이미지 태그는 세션마다 명시적 패턴(`gfx1100-rocm10-tbq-rboosts`, `...mtp-latest`)으로 관리.
- `bench-beellama-plan.md`/`*-results`는 beellama v0.4.4/0.4.5 대비 측정 (tbqplus/rdna-boosts 대조용).
