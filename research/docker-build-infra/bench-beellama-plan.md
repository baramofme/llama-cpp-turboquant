# beellama v0.4.4 성능 테스트 계획

> 대상: `ghcr.io/anbeeld/beellama.cpp:server-rocm-v0.4.4`
> 날짜: 2026-09-05 (Asia/Seoul)
> 목적: 현재 하드웨어에서 beellama 빌드(0.4.4, build 11573, commit f8cd4e6dd)의 실제 성능 측정
> 비교 대상: 기존 운영 빌드 `baramofme/llama-cpp-rocm:gfx1100-rocm7.2-tbqplus-rebuild` (llm-main 컨테이너)

---

## 1. 테스트 환경 (탐색 결과)

### 하드웨어

| 항목 | 값 |
|---|---|
| GPU | AMD Radeon RX 7900 XTX x2 (gfx1100, VRAM 24GB/개) |
| GPU0 | `llm-main` 컨테이너 점유 중 (VRAM 23.4GB/24GB 사용) |
| GPU1 | 유휴 (VRAM 28MB 사용) |
| CPU | 20 코어 |
| RAM | 94GB (사용 27GB, 가용 66GB) |
| 디스크 | /opt 1.2TB 여유, / 185GB 여유 |
| ROCm | rocm-smi 정상 동작, gfx1100 |

### 도커 장치 매핑

| GPU | /dev/dri 장치 |
|---|---|
| GPU0 | card0, renderD130 |
| GPU1 | card2, renderD129 |

### beellama 이미지 검증 (완료)

- `docker pull ghcr.io/anbeeld/beellama.cpp:server-rocm-v0.4.4` 성공
- `llama-server --version`: 0.4.4 (build 11573, commit f8cd4e6dd), GNU 11.4.0
- 백엔드 로드 확인: ROCm(hip) / RPC / CPU(alderlake)
- 바이너리: `llama-server`, `llama-bench`, `llama-cli`, `llama-batched-bench`, `llama-completion`, `llama-quantize`
- 주요 플래그 지원 확인: `--spec-type`(none, draft-simple, draft-eagle3, draft-mtp, draft-dflash, draft-dspark, ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache), KVarN(`kvarn2..kvarn8`), `--kv-tail-tokens`, `--spec-dm-controller off|profit`, `--swa-full`, `--cache-ram`, `--ctx-checkpoints`
- 주의: 이미지 엔트리포인트가 `/app/llama-server`로 고정. 다른 바이너리 실행 시 `--entrypoint /app/llama-bench` 등 오버라이드 필요
- 주의: `llama-bench`는 `--spec-type` 플래그가 없음. 시뮬레이션 디코딩(MTP/DFlash) 벤치는 `llama-server` + `/metrics`로 측정

### 모델 (벤치 후보)

| 모델 | 경로 | 용량 | 비고 |
|---|---|---|---|
| Qwen3.5-2B-MTP-Q4_K_M | /opt/llm/models/Qwen3.5-2B-MTP-Q4_K_M.gguf | 1.3GB | 빠른 검증/스모크 테스트 |
| Qwen3.8-27B-MTP-Q4_K_M | /opt/llm/models/Qwen3.8-27B-MTP-Q4_K_M.gguf | 16.8GB | 메인 벤치 모델 (MTP 텐서 포함) |
| Qwen3.8-27B-MTP-Q5_K_M | /opt/llm/models/Qwen3.8-27B-MTP-Q5_K_M.gguf | 19.5GB | 정밀도 민감 비교용 (선택) |
| Qwen3.8-27B-MTP-Q3_K_M | /mnt/nvmedata/models/unsloth/Qwen3.8-27B-GGUF/ | - | llm-main [Dense] 섹션 현행 모델 |

### 기존 운영 구성 (비교 기준)

`/opt/llm/llama-cpp/main-llm-config.ini` [Dense] 섹션:
- 모델: Qwen3.8-27B-MTP-Q3_K_M, `spec-type=draft-mtp`, `spec-draft-n-max=3`
- KV 캐시: `cache-type-k=q8_0`, `cache-type-v=turbo3` (TheTom 비대칭 방식)
- `flash-attn=on`, `ctx-size=140000`, `batch-size/ubatch-size=4096`
- `main-gpu=0`, `n-gpu-layers=99`
- 참고: tbqplus 커스텀 빌드는 `turbo3` KV 타입을 지원. beellama 표준 릴리스에는 `turbo3`가 없고 KVarN 계열(`kvarn2..kvarn8`)이 있으므로, 동일 조건 비교는 KVarN 계열로 대체 가능

---

## 2. 실행 전제 조건

1. **GPU1 단독 사용**: `llm-main`이 GPU0을 점유 중이므로, 벤치는 GPU1(`renderD129`/`card2`)로 수행. 간섭 없는 순수 비교.
   - GPU0 사용이 필요하면 `docker stop llm-main` 후 재실행 (벤치 완료 후 `docker start llm-main`)
2. **장치 패스스루**: `--device /dev/kfd --device /dev/dri/renderD129 --device /dev/dri/card2 --group-add video`
3. **HIP_VISIBLE_DEVICES=0** (패스스루된 GPU가 컨테이너 내 0번으로 매핑)
4. **포트**: llama-server 벤치용 8090 (llm-main 8081, open-webui 등과의 충돌 회피)
5. **모델 볼륨 마운트**: `/opt/llm/models:/models`, `/mnt/nvmedata/models:/models2`
6. **재현성**: 동일 플래그, 동일 컨텍스트, 동일 반복 횟수. VRAM/클럭 상태는 `rocm-smi`로 기록
7. **시리즈 분리**: 각 시나리오마다 컨테이너를 새로 띄워 실행 (VRAM 잔여 상태 제거)

---

## 3. 시나리오 매트릭스

공통 파라미터 (27B 모델 기준):
- 모델: `/models/Qwen3.8-27B-MTP-Q4_K_M.gguf` (Qwen3.5-2B-MTP로 스모크 테스트 먼저)
- `-c 32768 -b 4096 -ub 1024 -ngl 99 --jinja --metrics`
- 스모크 테스트: 2B 모델 + `-c 4096`

| # | 도구 | 구성 | 목적 |
|---|------|------|------|
| S1 | llama-bench | `-fa on -ctk f16 -ctv f16` (pp512/16, tg512/16) | 기본 pp/tg baseline |
| S2 | llama-bench | `-fa on -ctk q8_0 -ctv q5_0` | TheTom 비대칭 정밀 KV 비교 |
| S3 | llama-bench | `-fa on -ctk kvarn5 -ctv kvarn4 --kv-tail-tokens 1024` | KVarN 캐시 효과 (현행 구성의 표준 릴리스 대응) |
| S4 | llama-server | `--spec-type draft-mtp --spec-draft-n-max 3` | MTP 시뮬레이션 (기존 운영 구성과 동일) |
| S5 | llama-server | `--spec-type draft-dflash --spec-dm-controller profit` | DFlash 비교. 드래프트 모델은 MTP 헤더 기반이므로 동일 모델 지정 |
| S6 | llama-server | S4 + `-ctk kvarn5 -ctv kvarn4 --kv-tail-tokens 1024` | MTP + KVarN 조합 (현행 운영 구성 대응) |
| S7 | llama-batched-bench | `-np 4` (pp512/16, tg512/16) | 멀티 슬롯(동시 4) 성능 |

비교 대상 실행:
- S1~S3, S7은 `llm-main` 이미지의 `llama-bench`/`llama-batched-bench`로 동일 플래그 재실행
- S4~S6은 `llm-main` 이미지 `llama-server`로 동일 플래그 재실행 (tbqplus 빌드는 `turbo3` 지원이므로 S3/S6은 `turbo3`와 `kvarn` 버전 각각 측정)

---

## 4. 측정 지표

| 지표 | 출처 |
|---|---|
| pp t/s (프롬프트 처리) | llama-bench 출력 / server `/metrics` |
| tg t/s (토큰 생성) | llama-bench 출력 / server `/metrics` |
| MTP 수용률 (accept rate) | server 로그 `spec: accepted/total` + `/metrics` spec 계열 카운터 |
| VRAM 사용량 | `rocm-smi --showmeminfo vram` (실행 전/후) |
| 스로틀링/오버클럭 | `rocm-smi` 온도, 클럭 |
| 로딩 시간 | server 로그 timestamp |

---

## 5. 실행 방식

### 5.1 스크립트 구성 (`bench-beellama.sh`)

- 각 시나리오를 `docker run --rm`으로 독립 실행 (시리즈 간 상태 격리)
- llama-bench 계열:
  ```
  docker run --rm \
    --device /dev/kfd --device /dev/dri/renderD129 --device /dev/dri/card2 \
    --group-add video \
    -v /opt/llm/models:/models -v /mnt/nvmedata/models:/models2 \
    --entrypoint /app/llama-bench \
    ghcr.io/anbeeld/beellama.cpp:server-rocm-v0.4.4 \
    -m /models/Qwen3.8-27B-MTP-Q4_K_M.gguf \
    -ngl 99 -fa on -ctk <K> -ctv <V> \
    -c 32768 -b 4096 -ub 1024 \
    -p 512 -n 512 -r 3 -o jsonl
  ```
- llama-server 계열 (S4~S6):
  ```
  docker run -d --name bench-server <장치/볼륨 옵션> \
    --entrypoint /app/llama-server \
    ghcr.io/anbeeld/beellama.cpp:server-rocm-v0.4.4 \
    -m /models/Qwen3.8-27B-MTP-Q4_K_M.gguf \
    -c 32768 -b 4096 -ub 1024 -ngl 99 -fa on --jinja --metrics \
    --spec-type draft-mtp --spec-draft-n-max 3 \
    --host 0.0.0.0 --port 8090
  # ready 확인 후:
  curl -s http://localhost:8090/v1/completions -d '{"model":"x","max_tokens":256,"prompt":"..."}'
  # 수용률: docker logs bench-server | grep spec, /metrics 스크래핑
  # 종료: docker stop bench-server
  ```
- 결과 누적: `bench-beellama-results.md` (표: 시나리오 / 빌드 / pp / tg / 수용률 / VRAM / 비고)

### 5.2 실행 순서

1. 스모크: 2B 모델 + S1 (llama-bench) 로 파이프라인 검증 (장치 매핑, 모델 로드, 출력 파싱)
2. 27B S1~S3 (llama-bench, beellama)
3. 27B S1~S3 (llm-main, 동일 플래그)
4. S4~S6 (llama-server, beellama) + 수용률 측정
5. S4~S6 (llm-main, 동일 플래그. S3/S6은 turbo3 버전 추가)
6. S7 (llama-batched-bench, 양쪽 빌드)
7. 결과 표 정리 + 분석 (KVarN vs q8_0/q5_0, MTP 수용률, DFlash profit 컨트롤러 효과)

### 5.3 예상 소요

- 27B 로딩 + 벤치: 시나리오당 1~3분
- 전체: 약 20~30분 (llama-bench -r 3 기준)

---

## 6. 주의사항 / 리스크

1. **llama-bench에는 시뮬레이션 옵션이 없음**: MTP/DFlash 수용률 기반 tg는 반드시 llama-server로 측정
2. **DFlash 드래프트 모델**: beellama의 draft-dflash는 별도 드래프트 모델 파일을 요구할 수 있음. 현 모델 세트에 DFlash 드래프트가 없으므로 S5는 draft-mtp 헤더 기반 MTP 모델로 대체 실행하고, 수용률/속도만 확인. (별도 드래프트 모델이 있으면 별도 실행)
3. **KVarN tail**: KVarN은 항상 128토큰 exact suffix를 내부적으로 유지 (help 문서 확인)
4. **GPU0 간섭**: llm-main이 GPU0을 점유 중. GPU1 단독 벤치로 진행. GPU0 비교 실행이 필요하면 llm-main 일시 중지
5. **tbqplus vs 표준 릴리스**: tbqplus는 `turbo3` KV 타입과 VDot 계열 최적화를 포함. KVarN 기준 비교에서는 beellama가 불리할 수 있음. 비교 시 이 점을 명시
6. **시리즈 간 VRAM 잔류**: 각 실행 후 컨테이너 제거(`--rm`)로 정리
7. **결과 기록**: 모든 실행의 원시 출력(jsonl/로그)을 `bench-results/`에 보관

---

## 7. 산출물

- `bench-beellama.sh` - 통합 벤치 스크립트
- `bench-beellama-results.md` - 결과 표 + 분석
- `bench-results/` - 원시 출력(로그, jsonl, metrics 스냅샷)
