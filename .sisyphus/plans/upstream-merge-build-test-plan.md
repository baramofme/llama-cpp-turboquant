# Upstream Merge — Build & Test Plan

## 개요

- **목표**: `feature/turboquant-kv-cache` 브랜치에 `upstream/master` 머지 후, 로컬 소스 기반으로 Docker 이미지를 빌드하고 배포하여 정상 동작을 검증
- **대상 저장소**: `/home/baramofme/IdeaProjects/llama-cpp-turboquant` (로컬)
- **빌드 인프라**: `/opt/llamacpp/llama-cpp` (Makefile + docker-bake.hcl + Dockerfile)
- **배포 인프라**: `/mnt/ssd-work/dokploy-etc/compose/hometool-vllmqwen3535b-wlnzww/code` (docker-compose)
- **타겟 GPU**: AMD Radeon 7900 XTX (gfx1100), ROCm 7.2.1

---

## Part 1 — 빌드 시스템 변경

### 1.1 신규 Dockerfile: `Dockerfile.turboquant_plus_local`

**위치**: `/opt/llamacpp/llama-cpp/Dockerfile.turboquant_plus_local`

기존 `Dockerfile.turboquant_plus`에서 `git clone` 부분을 로컬 COPY로 대체:

```dockerfile
# AS-IS (기존):
# RUN git clone --branch ${TBQ_PLUS_VERSION} https://github.com/${GITHUB_OWNER}/${GITHUB_REPO} .

# TO-BE:
# (Docker build context가 로컬 repo를 가리키도록 bake.hcl에서 설정)
COPY . /app
```

나머지 빌드 옵션(template-instances 패치, 컴파일러 경로, cmake HIP flags, Multi-arch Fat Binary)은 `Dockerfile.turboquant_plus`와 동일하게 유지.

**변경사항 상세**:
- `ARG GITHUB_OWNER`, `ARG GITHUB_REPO`, `ARG TBQ_PLUS_VERSION` 제거 (로컬 복사이므로 불필요)
- `git clone` 명령어를 `COPY . /app`으로 대체
- `WORKDIR /app` 다음에 `COPY` 실행
- `template-instances` 패치(RUN rm -f ...)는 유지

### 1.2 docker-bake.hcl: `turboquant-plus-local` 타겟 추가

```hcl
variable "TBQ_PLUS_LOCAL_TAG" {
  default = "baramofme/llama-cpp-rocm:gfx1100-rocm7.2-tbqplus-local"
}

target "turboquant-plus-local" {
  dockerfile = "Dockerfile.turboquant_plus_local"
  context    = "/home/baramofme/IdeaProjects/llama-cpp-turboquant"
  target     = "turboquant_plus-server"
  tags       = ["${TBQ_PLUS_LOCAL_TAG}"]
  platforms  = ["linux/amd64"]
  args = {
    ROCM_DOCKER_ARCH = "gfx1100"
  }
}
```

**핵심**: `context`를 로컬 repo 절대경로로 지정하여 Docker build가 로컬 파일을 직접 사용하도록 함.

### 1.3 Makefile: `build-tbq-plus-upstream-local` 타겟 추가

```makefile
# === TurboQuant Plus Local Build ===
LOCAL_TBQ_REPO := /home/baramofme/IdeaProjects/llama-cpp-turboquant

build-tbq-plus-upstream-local: prepare-log
	@echo "🚀 Building TBQ+ from local source with upstream merge..."
	@echo "   Source dir : $(LOCAL_TBQ_REPO)"
	@echo "   TheTom hash: $$(git -C $(LOCAL_TBQ_REPO) rev-parse --short origin/feature/turboquant-kv-cache)"
	@echo "   Upstream hash: $$(git -C $(LOCAL_TBQ_REPO) rev-parse --short upstream/master)"
	THE_TOM_HASH=$$(git -C $(LOCAL_TBQ_REPO) rev-parse --short origin/feature/turboquant-kv-cache) \
	UPSTREAM_HASH=$$(git -C $(LOCAL_TBQ_REPO) rev-parse --short upstream/master) \
	TBQ_PLUS_LOCAL_TAG=baramofme/llama-cpp-rocm:gfx1100-rocm7.2-tbqplus-$${THE_TOM_HASH}-$${UPSTREAM_HASH} \
	docker buildx bake -f $(BAKE_FILE) turboquant-plus-local --progress=plain --load 2>&1 | tee $(BUILD_LOG)
	@echo "✅ Build complete: baramofme/llama-cpp-rocm:gfx1100-rocm7.2-tbqplus-$$(git -C $(LOCAL_TBQ_REPO) rev-parse --short origin/feature/turboquant-kv-cache)-$$(git -C $(LOCAL_TBQ_REPO) rev-parse --short upstream/master)"
```

**버전명 규칙**: `tbqplus-{thetom_commit_hash}-{upstream_commit_hash}`
- `thetom_hash`: `origin/feature/turboquant-kv-cache`의 HEAD (merge base)
- `upstream_hash`: `upstream/master`의 HEAD (merged commit)

---

## Part 2 — 전체 워크플로우

### Step 0 — Upstream Merge

```bash
cd /home/baramofme/IdeaProjects/llama-cpp-turboquant

# upstream 최신 내용 가져오기
git fetch upstream master

# 머지 실행
git merge upstream/master
# → 충돌 발생시 해결 후 git commit

# (선택) origin에 푸시
git push origin feature/turboquant-kv-cache
```

### Step 1 — Docker 이미지 빌드

```bash
cd /opt/llamacpp/llama-cpp
make build-tbq-plus-upstream-local
```

### Step 2 — docker-compose 이미지 태그 교체

```bash
# 자동 태그 교체
cd /home/baramofme/IdeaProjects/llama-cpp-turboquant
THE_TOM_HASH=$(git rev-parse --short origin/feature/turboquant-kv-cache)
UPSTREAM_HASH=$(git rev-parse --short upstream/master)

cd /mnt/ssd-work/dokploy-etc/compose/hometool-vllmqwen3535b-wlnzww/code
sed -i "s|image: baramofme/llama-cpp-rocm:gfx1100-rocm7.2-tbqplus-.*|image: baramofme/llama-cpp-rocm:gfx1100-rocm7.2-tbqplus-${THE_TOM_HASH}-${UPSTREAM_HASH}|" docker-compose.yml
```

### Step 3 — 컨테이너 구동

```bash
cd /mnt/ssd-work/dokploy-etc/compose/hometool-vllmqwen3535b-wlnzww/code
docker compose up -d
sleep 15  # 모델 로딩 대기
docker compose logs --tail=50 llm-main
```

### Step 4 — 테스트 실행

아래 Part 3의 테스트 계획을 단계별로 수행.

---

## Part 3 — 테스트 계획

### Phase 1 — 🟢 1차 게이트 (통과 못하면 STOP)

| # | 테스트 | 명령어/방법 | Pass 조건 |
|---|--------|-----------|----------|
| **1.1** | 빌드 성공 | `make build-tbq-plus-upstream-local` exit code 확인 | `0` |
| **1.2** | 컨테이너 기동 | `docker ps --filter name=llm-main --format '{{.Status}}'` | `Up (healthy)` |
| **1.3** | Health API | `curl -s -o /dev/null -w "%{http_code}" http://localhost:8081/health` | `200` |
| **1.4** | 모델 로드 | `curl -s http://localhost:8081/v1/models \| python3 -c "import sys,json; [print(m['id']) for m in json.load(sys.stdin)['data']]"` | 5개 모델 모두 표시 |
| **1.5** | 서버 로그 에러 | `docker logs llm-main 2>&1 \| grep -ciE "error\|fatal\|abort\|segfault\|SIG"` | `0` |
| **1.6** | GPU 인식 | `docker exec llm-main rocm-smi --showproductname \| grep "AMD Radeon"` | GPU 정보 정상 출력 |

### Phase 2 — 🟡 프롬프트 캐시 검증 (최우선)

**준비**: 테스트용 프롬프트 파일 (~1800 tokens)
```bash
cat > /tmp/test_prompt.txt << 'EOF'
Write a comprehensive Python script for data analysis that includes:
1. Loading CSV and JSON data
2. Data cleaning and preprocessing
3. Statistical analysis (mean, median, standard deviation, correlation)
4. Data visualization using matplotlib and seaborn
5. Machine learning model training using scikit-learn
6. Exporting results to Excel
... (충분히 길게 ~1800 tokens)
EOF
PROMPT=$(cat /tmp/test_prompt.txt)
```

| # | 테스트 | 방법 | Pass 조건 |
|---|--------|------|----------|
| **2.1** | 캐시 활성화 확인 | `docker logs llm-main 2>&1 \| grep "prompt cache"` | `prompt cache is enabled, size limit: 8192 MiB` |
| **2.2** | Cold PP 측정 | 첫 요청 → `timings.prompt_per_second` 저장 | 기준값 확보 |
| **2.3** | Warm PP (캐시 적중) | 동일 프롬프트 재요청 → PP tok/s | 첫 대비 **≥ 5x 향상** |
| **2.4** | cached_tokens | warm response → `prompt_tokens_details.cached_tokens` | `cached ≈ total - echo_delta` |
| **2.5** | 🚫 재처리 방지 | `docker logs llm-main 2>&1 \| grep -c "forcing full prompt re-processing"` | **0건 ← 하드 게이트** |
| **2.6** | 연속 적중 | 동일 프롬프트 3회 반복, 2~3회차 PP < 1초 | 지속적 캐시 적중 |

**캐시 검증 스크립트**:
```python
#!/usr/bin/env python3
"""prompt_cache_test.py - Verify prompt cache works after upstream merge"""
import requests, json, sys, time

API = "http://localhost:8081/v1/chat/completions"
MODEL = "qwen-3.6-coder"
PROMPT = open("/tmp/test_prompt.txt").read()

def request(desc):
    t0 = time.time()
    r = requests.post(API, json={"model": MODEL, "messages": [{"role": "user", "content": PROMPT}]})
    elapsed = time.time() - t0
    data = r.json()
    pp = data.get("timings", {}).get("prompt_per_second", 0)
    cached = data.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    total = data.get("usage", {}).get("prompt_tokens", 0)
    print(f"{desc}: {elapsed:.2f}s, PP={pp:.1f} tok/s, cached={cached}/{total}")
    return cached, total, elapsed

# Cold
c1, t1, _ = request("[1/4] Cold")
time.sleep(1)

# Warm
c2, t2, _ = request("[2/4] Warm-1")
time.sleep(0.5)

# Warm x2
c3, t3, e3 = request("[3/4] Warm-2")

# Check: forcing full re-processing in logs
import subprocess
logs = subprocess.run(["docker", "logs", "llm-main"], capture_output=True, text=True)
reproc = logs.stdout.count("forcing full prompt re-processing")

print(f"\n--- Results ---")
print(f"Cached tokens (cold→warm): {c1} → {c2}")
print(f"Warm PP speed: {100*c2/max(t1,1):.0f}% of cold (expect >500%)")
print(f"[{'PASS' if reproc == 0 else 'FAIL'}] Full re-processing count: {reproc}")
print(f"[{'PASS' if c2 > 0 else 'FAIL'}] Cache hit detected")

if reproc > 0 or c2 == 0:
    sys.exit(1)
```

### Phase 3 — 🟡 Batch-size 성능 검증

| # | 테스트 | 대상 | 방법 | Pass 조건 |
|---|--------|------|------|----------|
| **3.1** | PP batch sweep | Qwen-3.6-Coder | batch=512/1024/2048/4096 각각 ~1500tok cold PP 측정 | batch=2048이 512 대비 **≥ 30% 향상** |
| **3.2** | PP batch sweep | Qwen-3.5-9B-Sub | 동일, batch=512/1024/2048 | 이상 없음 |
| **3.3** | TG 영향 | 양 모델 | batch 변경시 TG(tok/s) 변화 | ±5% 이내 |
| **3.4** | VRAM 체크 | Qwen-3.6-Coder | 각 batch 후 `rocm-smi --showmeminfo vram` | OOM 없음, < 24GB |

### Phase 4 — 🔵 모델별 기능 검증

| # | 테스트 | 대상 | Pass 조건 |
|---|--------|------|----------|
| **4.1** | 기본 채팅 | Qwen-3.6-Coder | 200 OK, 유효한 응답 |
| **4.2** | Reasoning | Qwen-3.6-Coder | `reasoning_content` 필드 존재 |
| **4.3** | Reasoning budget | Qwen-3.6-Coder | budget 4096 제한 동작 확인 |
| **4.4** | Sub 모델 | Qwen-3.5-9B-Sub | 정상 응답 |
| **4.5** | 소형 모델 | Qwen-3.5-2B | GPU 1에서 정상 동작 |
| **4.6** | 임베딩 | BGE-M3 | float 벡터 배열 반환 |

### Phase 5 — 🔴 회귀 검증

| # | 테스트 | 방법 | Pass 조건 |
|---|--------|------|----------|
| **5.1** | GPU 분리 | Qwen-3.6-Coder → `rocm-smi` 확인 | GPU 0만 VRAM 증가, GPU 1 변화 없음 |
| **5.2** | Sub GPU 분리 | Qwen-3.5-9B-Sub → `rocm-smi` | GPU 1 VRAM 증가 |
| **5.3** | Compose 복원력 | `down && up -d` 재시작 | 모든 모델 정상 로드 |
| **5.4** | SWA (8000+ tok) | 긴 프롬프트 요청 | OOM 없음, attention 정상 |
| **5.5** | Config 변경 | `config.ini` 수정 후 재시작 | 변경 반영 확인 |

### Phase 6 — 💀 스트레스 테스트 (선택)

| # | 테스트 | Pass 조건 |
|---|--------|----------|
| **6.1** | 10회 연속 요청 | 모두 200 OK, 응답시간 편차 ±20% |
| **6.2** | 동시 요청 3건 | 데드락/행 없음 |
| **6.3** | 메모리 누수 | Phase 5 후 VRAM, 초기 대비 +100MB 미만 |

---

## Part 4 — 최종 Pass 기준

```
🟢 MUST PASS (Phase 1 + 2) = 11/11  ← Hard Gate
🟡 SHOULD PASS (Phase 3 + 4) = 10/11
🔵 NICE TO PASS (Phase 5) = 5/6
💀 INFO ONLY (Phase 6)

HARD FAIL 조건:
- Phase 2.5: "forcing full prompt re-processing" 로그 1건 이상 → FAIL
- Phase 1: 1개라도 실패 → 즉시 STOP (하위 테스트 무의미)
```

---

## Part 5 — 문제 발생시 대응 매뉴얼

| 증상 | 예상 원인 | 1차 대응 | 2차 대응 |
|------|---------|---------|---------|
| cmake 빌드 실패 | upstream에서 cmake 변수 삭제/변경 | `Dockerfile.turboquant_plus_local` cmake 플래그 확인 | `git diff upstream/master~5..HEAD -- CMakeLists.txt` |
| HIP compile error | ROCm/HIP API 변경 | `ggml-cuda` → `ggml-hip` 네임스페이스 변경 확인 | 컴파일러 플래그에 `-DGGML_HIP=ON` 존재 확인 |
| 서버 기동 실패 | config.ini 파라미터 key 변경/삭제 | `docker logs llm-main` 에러 메시지 확인 | upstream의 `common/common.cpp` 파라미터 파싱 diff |
| 캐시 미동작 | cache key 해싱, storage 로직 변경 | `git diff upstream/master..HEAD -- ggml/src/ggml-cache* ggml/src/ggml-compute*` | `LLAMA_CACHE` 관련 코드 diff 전수조사 |
| VRAM OOM | 메모리 풀/버퍼 크기 변경 | `--cache-ram 4096`으로 축소 재시도 | `ggml-backend` 메모리 할당 로직 diff |
| 특정 모델만 실패 | 모델 아키텍처별 로직 회귀 | 해당 모델 config 섹션 확인 | `llama-model-loader` diff 확인 |
| TG 속도 저하 | CUDA/HIP kernel 레지스트리 영향 | `--no-mmap` 시도 | kernel selection 로직 diff |
| Prompt 재처리 발생 | cache key 구성 변경 | cache 관련 소스 전수 비교 | `git bisect`로 원인 커밋 탐색 |

---

## Part 6 — 파일 변경 요약

| 파일 | 상태 | 설명 |
|------|------|------|
| `/opt/llamacpp/llama-cpp/Dockerfile.turboquant_plus_local` | **신규 생성** | git clone → COPY . /app 교체 |
| `/opt/llamacpp/llama-cpp/docker-bake.hcl` | **수정** | `turboquant-plus-local` 타겟 + `TBQ_PLUS_LOCAL_TAG` 변수 추가 |
| `/opt/llamacpp/llama-cpp/Makefile` | **수정** | `build-tbq-plus-upstream-local` 타겟 + `LOCAL_TBQ_REPO` 변수 추가 |
| `.sisyphus/plans/upstream-merge-build-test-plan.md` | **생성** | 본 문서 |

---

## Part 7 — Build Execution Log (2026-06-28)

### Used: `.devops/rocm.Dockerfile` (project-local, not `/opt/llamacpp/llama-cpp/`)

빌드는 `/opt/llamacpp/llama-cpp/` 시스템 대신 프로젝트 루트의 `.devops/rocm.Dockerfile`을 사용하여 `--target server`로 진행. Docker layer cache 재사용을 위해 `--target server` 로 빌드 (web stage 및 cmake configure 레이어 캐시).

### Compilation Errors Fixed (server-context.cpp)

| # | Line | Symptom | Root Cause | Fix |
|---|------|---------|-----------|-----|
| 1 | 2475 | `'id_slot' was not declared` | Upstream에서 `slot::task` 필드명 또는 접근 방식 변경 | `id_slot` → `task.id_slot` |
| 2 | 3036 | `'continue' not within a loop` | `continue`가 람다 내부에서 사용됨 (람다는 loop가 아님) | `continue` → `return` |
| 3 | 3481 | `'continue' not within a loop` | 동일 원인 | `continue` → `return` |
| 4 | 17 | `fatal: llama-ext.h: No such file or directory` | `#include "llama-ext.h"` → `src/` 디렉토리가 include path에 없음 | `#include "../../src/llama-ext.h"` (다른 파일들의 convention과 일치) |
| 5 | 3561 | `'input_tokens.process_chunk' was not declared` | 함수가 우리 fork에서는 존재하지 않음 (mtmd draft 처리 미구현) | `SLT_WRN` skip + TODO 주석으로 대체 (코드 경로 무효화) |
| 6 | 3690 | `'break' not within a loop` | `break`가 람다 내부에서 사용됨 | `break` → `return` |
| 7 | 3561 | `'SLT_WARN' was not declared` | 매크로명은 `SLT_WRN` (WARN 아님) | `SLT_WRN` + `"%s","msg"` 패턴으로 수정 (공백 `__VA_ARGS__` 방지) |

### Compilation Errors Fixed (server-models.cpp)

| # | Line | Symptom | Root Cause | Fix |
|---|------|---------|-----------|-----|
| 8 | 943 | `no matching function for call to 'update_status(name, SERVER_MODEL_STATUS_UNLOADED, 1)'` | 시그니처가 `update_status_args` 구조체 기반으로 변경 | `update_status(name, {SERVER_MODEL_STATUS_UNLOADED, 1})` |

### Compilation Errors Fixed (web build — dialogs)

| # | File | Symptom | Root Cause | Fix |
|---|------|---------|-----------|-----|
| 9 | `tools/ui/src/lib/dialogs/index.ts` | `esbuild: Multiple exports with the same name "DialogExportSettings"` | `DialogExportSettings`가 중복 export | 중복된 export 제거 |

### Docker Build Fix (web stage: log.h)

| # | File | Symptom | Root Cause | Fix |
|---|------|---------|-----------|-----|
| 10 | `tools/server/server-context.cpp` | `'SLT_WARN' → 'SLT_WRN'` 후에도 `log.h:107: error: expected primary-expression before ')' token` | `SLT_WRN` 매크로가 empty `__VA_ARGS__`로 호출되어 trailing comma 발생 | `"%s","message"` 패턴 적용 (전체 코드베이스의 convention) |

### Build Result

| 항목 | 상태 |
|------|------|
| Docker build (`docker build ... --target server`) | ✅ **PASS** (exit 0) |
| Image tag | `baramofme/llama-cpp-rocm:gfx1100-rocm7.2-tbqplus-upstream-rda3-lds` |
| Docker Hub push | ✅ **PUSHED** |
| Digest | `sha256:0aac82c3f6aa84ba8ee64a7dfeb91126db70978fd8c6f55910b9d25936cbf171` |
| Image size | 22.3 GB |

### Phase 1 Gate Test Results

| # | Test | Status | Note |
|---|------|--------|------|
| 1.1 | Build success | ✅ PASS | Docker build exit 0 |
| 1.2–1.6 | Container deploy | ⏸️ SKIP | llama-swap 이미지 기반 배포 병렬 — 별도 빌드 필요 |
| Push to registry | ✅ DONE | Docker Hub 업로드 완료 |

### Notes

- **llama-swap 은 건드리지 않음.** 배포 인프라(`llama-swap-rocm` 이미지)는 별도 Dockerfile(`/opt/llamacpp/llama-swap/Dockerfile`)로 관리되며, 본 빌드는 순수 `llama-cpp-rocm` (llama-server only) 이미지.
- 실제 서비스 배포를 위해서는 `llama-swap/Dockerfile`에서 llama.cpp 바이너리 소스를 릴리스 다운로드 대신 로컬 빌드 결과로 변경하는 추가 작업 필요.
