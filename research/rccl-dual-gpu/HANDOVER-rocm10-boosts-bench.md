# 핸드오프: ROCm 10 + rdna-boosts 벤치마킹 (다음 LLM용)

> 작성: 2026-09-05 (Asia/Seoul)
> 목적: 이전 세션의 작업 내용(참조 링크/핵심 결론/산출 소스)을 전달하고, 다음 작업을 정의
> 다음 작업: **ROCm 10 + rdna3 boost를 순정 llama.cpp와 beellama 양쪽에 적용해 벤치 + 최적화 + KVarN KV 캐시 양자화(pp/tg/메모리) 조사**

---

## 1. 하드웨어 / 환경

| 항목 | 값 |
|---|---|
| GPU | AMD Radeon RX 7900 XTX x2 (gfx1100, RDNA3, VRAM 24GB/개) |
| GPU0 | card1/renderD129 (PCI 03:00.0) - `llm-main` 컨테이너 점유 중 |
| GPU1 | card2/renderD130 (PCI 06:00.0) - **벤치 전용 (유휴)** |
| CPU/RAM | 20 코어 / 94GB |
| ROCm (호스트) | 7.2.3 (native, clang-20 계열) |
| ROCm 10 | docker `rocm/dev-ubuntu-26.04:10.0.0-full` (ROCm 10.0 dev, clang-23, CMake 4.2.3) |
| 운영 컨테이너 | `llm-main` = `baramofme/llama-cpp-rocm:gfx1100-rocm7.2-tbqplus-rebuild` (tbqplus 커스텀 빌드, ROCm 7.2.1) |
| beellama 이미지 | `ghcr.io/anbeeld/beellama.cpp:server-rocm-v0.4.4` (docker), v0.4.5 소스 (commit 44a369699)는 native 빌드 사용 |
| 모델 | `/opt/llm/models/Qwen3.8-27B-MTP-Q4_K_M.gguf` (16.8GB, head_dim 256, MTP dense 27B) |
| 스모크 모델 | `/opt/llm/models/Qwen3.5-2B-MTP-Q4_K_M.gguf` (1.3GB) |

**벤치 공통 플래그**: `-b 4096 -ub 1024 -ngl 99 -fa on -p 512 -n 512 -r 3 -o jsonl`
**KVarN 필수**: `--kv-tail-tokens 1024` (누락 시 portable 폴백, pp -63% 회귀)
**GPU 선택**: native는 `HIP_VISIBLE_DEVICES=1`, docker(패스스루)는 `HIP_VISIBLE_DEVICES=0`

---

## 2. 참조 링크 / 소스

| 항목 | 위치 |
|---|---|
| rdna-boosts 패치 (13개) | https://github.com/stew675/llama-cpp-rdna-boosts (`patches/0001..0013-*.patch`) |
| beellama.cpp | https://github.com/Anbeeld/beellama.cpp |
| KVarN D=256 회귀 이슈 | https://github.com/Anbeeld/beellama.cpp/issues/122 (15-22x 회귀, RDNA3 WMMA head_dim>128 거부) |
| KVarN 라우팅 정책 | `beellama-boosts/src/ggml/src/ggml-cuda/fattn-kvarn-route-policy.h` (63줄: RDNA_WMMA + head_dim>128 -> portable 거부) |
| beellama boosts 소스 | `beellama-boosts/src/` (v0.4.5 + clean 5 블록 + 0004/0010/0011 수동 병합, 0001 부분 적용 중) |
| 순정 llama.cpp (tbqplus 기반) | 이 저장소 루트 (`/home/baramofme/IdeaProjects/llama-cpp-turboquant`) |
| 빌드 디렉터리 | `/tmp/beellama-src/build-boosts` (ROCm 7.2.3 boosts), `/tmp/beellama-src/build-clean` (clean 대조), `beellama-boosts/src/build-rocm10/` (ROCm 10) |
| 벤치 스크립트 | `beellama-boosts/bench-boosts.sh` (native S1~S4/S7), `beellama-boosts/bench-rocm10.sh` (ROCm 10 docker), `beellama-boosts/Dockerfile` |
| 원시 결과 | `bench-results/` (v0.4.4 docker), `bench-results-boosts/` (boosts 블록별), `bench-results-v045/` (v0.4.5 clean) |
| 결과 문서 | `bench-beellama-results.md`, `bench-beellama-boosts-results.md`, `bench-beellama-v0.4.5-results.md`, `rdna-boosts-apply-plan.md`, `bench-results/results.csv` |
| tbqplus fixes | `FIXES.md` (Moltes94 fork, recurrent shrink/expand, PR #24785) |

---

## 3. 핵심 결론 (이전 세션에서 확인된 것)

### 3.1 성능 (Qwen3.8-27B Q4_K_M, GPU1, pp/tg t/s)

| 구성 | S1 f16 | S2 q8_0/q5_0 | S3 kvarn5/4 (tail 1024) | S4 MTP | S7 4병렬 |
|---|---|---|---|---|---|
| llm-main (tbqplus, ROCm 7.2.1 docker) | 921.2 / 34.56 | 921.6 / 33.54 | 911.3 / 33.59 (q8_0/turbo3) | 40.23 (56.7%) | 945.6 / 77.55 |
| beellama v0.4.5 clean (ROCm 7.2.3) | 928.9 / 34.57 | 925.0 / 33.11 | 923.5 / 34.50 | 36.19 (37.6%) | 944.6 / 77.64 |
| beellama boosts (clean 5, ROCm 7.2.3) | 965.2 / 34.55 | 958.3 / 33.17 | 960.9 / 34.53 | 35.11 (45.6%) | - |
| **beellama boosts 최종 (+0004/0010/0011)** | **982.0 / 37.10** | **981.3 / 35.47** | **978.1 / 37.06** | 미측정 | - |
| beellama boosts (ROCm 10 docker) | 914.8 / 35.73 | 912.7 / 34.47 | 911.8 / 35.44 | - | - |

### 3.2 핵심 발견

1. **rdna-boosts 13개 블록 중 8개 적용 완료** (beellama v0.4.5 기준):
   - clean 5: 0002(chunked GDN prefill), 0005(CPU decode verify), 0006(host buffer, 이미 적용), 0007(meta device skip, 이미 적용), 0012(RDNA4 전용 - gfx1100 무효)
   - 수동 병합 3: 0004(WMMA FA, DKQ>128->576), 0010(k-quant VDR: Q4_K/Q5_K vdr4, Q6_K vdr2, nwarps 8), 0011(prefill CUDA graph skip)
   - 건너뜀: 0001(adaptive MTP), 0003(BF16 KV), 0008(0003 의존), 0009(이미 headroom 32), 0013(fused MoE - dense 무관)

2. **0010(k-quant VDR)가 디코딩 tg +7~8%의 실질 이득** (34.57->37.10, 33.11->35.47, 34.50->37.06). 0011이 pp +1~2%. clean 5 블록이 pp +3.6~4.1%.

3. **KVarN은 `--kv-tail-tokens` 플래그 필수**: 없으면 `kvarn_route_portable=32/32` (전 레이어 portable 폴백) -> pp -63%. 플래그 있으면 최적화 경로. `bench-rocm10.sh`는 이제 자동 추가.

4. **KVarN D=256 portable 폴백은 이슈 #122**: `fattn-kvarn-route-policy.h` 63줄에서 RDNA3 WMMA가 `head_dim > 128` 거부. rdna-boosts 13개 패치 중 KVarN 라우팅을 건드리는 패치 0개 (0004/0008은 표준 FA WMMA만). 해결하려면 route-policy의 head_dim 제한을 WMMA D=256으로 확장하거나 CUDA split/vector 디코딩을 HIP으로 이식.

5. **MTP 수용률**: tbqplus가 여전히 우위 (56.7% vs beellama boosts 45.6%). 0001(adaptive MTP)이 수용률 개선 후보이나 **beellama speculative.cpp 커스텀 코드와 16 hunk 충돌, tbqplus도 base 클래스 시그니처 차이로 충돌**. 0001 수동 병합 진행 중 (아래 3.4).

6. **ROCm 10 vs 7.2.3**: S1~S3 동급 (R10 911~915 pp, 7.2.3 958~982 pp - boosts 최종 기준). R10은 디코딩 tg가 약간 높음(35.73 vs 34.55 S1).

7. **S7(4병렬 batched-bench)**: native boosts 빌드에서 테이블 비어 출력 안 됨 (docker에서는 899.9/77.6 정상). MTP 모델 + 4병렬 native 빌드 호환성 문제 - 별도 조사 필요.

### 3.3 패치 적용 방법 (재현 가능)

```bash
# 패치 다운로드
curl -sL "https://raw.githubusercontent.com/stew675/llama-cpp-rdna-boosts/main/patches/<NNNN>-<name>.patch" -o /tmp/<NNNN>.patch
# 파일명 확인 (API)
curl -sL "https://api.github.com/repos/stew675/llama-cpp-rdna-boosts/contents/patches" | grep -oE '"name":\s*"[^"]*"'
# 적용 (파일별로 분리 - 일괄 적용 금지)
cd beellama-boosts/src
awk -v target="a/<file>" '/^diff --git/ { p = ($0 ~ target) } p { print }' /tmp/<NNNN>.patch > /tmp/single.patch
git apply --check /tmp/single.patch   # 클린 여부
git apply /tmp/single.patch          # 적용 (실패 시 --reject + .rej 수동 병합)
```

### 3.4 진행 중 작업: 0001(adaptive MTP) beellama 수동 병합

- beellama에 `git apply --reject /tmp/0001.patch` 실행:
  - **적용 성공**: common.cpp(1), common.h(3), speculative-adaptive.h(1), delta-net-base.cpp(1), server-context.cpp(1), test-speculative-adaptive.cpp(1), arg.cpp(2/3 hunk - hunk#2는 빈 줄 1개, 무해), speculative.cpp(14/16 hunk)
  - **수동 병합 필요 (.rej)**:
    1. `common/speculative.cpp` hunk#7 (draft_mtp::begin() 내, `auto * ctx_dft = this->params.ctx_dft;` **직전**에 삽입 - beellama의 begin()는 571줄 `auto * ctx_dft` 앞):
       ```cpp
       // new generation: the depth learned for the previous content is stale,
       // so the controller starts from the floor again
       if (adaptive) {
           adaptive_ctrl[seq_id].reset(this->params.n_max, this->params.n_min_adaptive);
       }
       ```
       (정확 위치: `void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {` 내, `if (N <= 0) { return; }` 블록 뒤, `auto * ctx_dft = this->params.ctx_dft;` 앞. 570~571줄 사이)
    2. `common/speculative.cpp` 2885줄 `spec_mtp` 체크에 DRAFT_MTP_ADAPTIVE 추가:
       ```cpp
       const bool spec_mtp = std::find(params.speculative.types.begin(),
                                       params.speculative.types.end(),
                                       COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end() ||
                             std::find(params.speculative.types.begin(),
                                       params.speculative.types.end(),
                                       COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE) != params.speculative.types.end();
       ```
  - 적용된 플래그: `--spec-type draft-mtp-adaptive`, `--spec-draft-n-min-adaptive <N>` (arg.cpp에 추가됨)
  - **다음 단계**: 위 2개 .rej 수동 삽입 -> `llama-server` 빌드 -> S4 벤치 `--spec-type draft-mtp-adaptive --spec-draft-n-max 3 --spec-draft-n-min-adaptive 1`으로 수용률 비교 (현재 45.6% 기준, tbqplus 56.7% 목표)
  - **주의**: 0001은 llama-server 전용(llama-bench 미지원). `speculative-adaptive.h`의 `common_speculative_adaptive` 컨트롤러가 draft 루프에 n_cap/n_last를 씀.

---

## 4. 다음 LLM의 작업 (명시된 목표)

### 4.1 ROCm 10 + rdna3 boost를 순정 llama.cpp와 beellama 양쪽에 적용

- **beellama**: `beellama-boosts/src/`에 boosts 8 블록 적용된 상태. ROCm 10 docker 빌드(`beellama-boosts/src/build-rocm10/`, `bench-rocm10.sh`)는 **clean 5 블록 기준** - 0004/0010/0011 수동 병합 포함 상태로 재빌드 후 S1~S4 벤치.
- **순정 llama.cpp (tbqplus)**: 이 저장소 루트. rdna-boosts 13개 패치 적용 시도:
  - 0004/0010/0011(GGML 커널)은 tbqplus의 `ggml/src/ggml-cuda/`에 `git apply --check` 후 적용 (beellama와 파일 구조 동일 계열 - tbqplus도 ggml-cuda 사용)
  - 0002/0005/0006/0007/0012(clean 5)도 적용 가능 여부 확인
  - 0001(adaptive MTP)은 tbqplus speculative.cpp와 **충돌**(base 생성자 시그니처: tbqplus는 `common_speculative_impl(DRAFT_MTP, n_seq)`, 패치는 `..., n_seq, params.draft.n_max`). tbqplus는 이미 n_max/n_min block-size 클램핑 보유(973-977줄). 0001 적용은 tbqplus draft 루프에 맞춰 재적응 필요 - **우선순위 낮춤** (사용자 지시: 기존 llama.cpp는 건드리지 말 것)
  - **사용자 지시: 기존 순정 llama.cpp는 건드리지 말 것** - tbqplus 빌드/소스는 read-only 참조, 벤치는 docker `llm-main` 이미지 사용

### 4.2 ROCm 10 + boosts 벤치 매트릭스 (양쪽 빌드)

| # | 시나리오 | 플래그 |
|---|---|---|
| S1 | f16 KV | `-ctk f16 -ctv f16` |
| S2 | q8_0/q5_0 | `-ctk q8_0 -ctv q5_0` |
| S3 | KVarN | `-ctk kvarn5 -ctv kvarn4 --kv-tail-tokens 1024` |
| S3' | KVarN 다른 조합 | `-ctk kvarn7 -ctv kvarn5 --kv-tail-tokens 1024` (선택) |
| S4 | MTP | llama-server `--spec-type draft-mtp --spec-draft-n-max 3` (0001 적용 시 `draft-mtp-adaptive` 추가) |
| S7 | 4병렬 | llama-batched-bench `-np 4` |

- 비교 대상: (a) 순정 llama.cpp docker(llm-main, ROCm 7.2.1), (b) beellama boosts ROCm 7.2.3 native, (c) beellama boosts ROCm 10, (d) 순정 llama.cpp + boosts ROCm 10 (신규)
- 지표: pp/tg(llama-bench jsonl), MTP 수용률(server 로그 `draft acceptance` + /metrics), VRAM(`rocm-smi --showmeminfo vram`)

### 4.3 최적화

- 0010(k-quant VDR)이 tg +7~8% - 순정 llama.cpp(tbqplus)에도 이식 가능 여부 확인 (tbqplus는 VDot/turbo3 커스텀 - 양립 확인 필요)
- 0011(CUDA graph skip)은 ggml-cuda.cu 단일 hunk - 양쪽 모두 적용 가능
- 0004(WMMA FA)은 beellama가 이미 확장 MMA 테이블 보유 - tbqplus에 적용 시 추가 이득 확인
- KVarN D=256 WMMA 라우팅(`fattn-kvarn-route-policy.h` 63줄 head_dim>128 거부) 확장 = 최대 개선 여지 (이슈 #122)

### 4.4 KVarN KV 캐시 양자화 조사 (사용자 명시)

- **pp/tg 영향**: kvarn2/3/4/5/6/7/8 x tail-tokens(256/512/1024/2048) 조합별 S1~S3 벤치
- **메모리 절약량**: 각 KV 타입의 캐시 바이트/토큰 비교
  - f16: K=V=256B/헤드/토큰 (head_dim 256 x f16 2B)
  - q8_0: K=V=33B (1+256/32 scale)
  - q5_0: V=18.5B
  - kvarnN: N에 따라 128/256/512-bit 양자화 (bit/요소 = N*8/128?) - `ggml/src/ggml-cuda/kvarn.cu`/`kvarn.h`에서 bit 폭 확인
  - 측정법: `rocm-smi --showmeminfo vram` 실행 전/후, 또는 llama-server 로그의 `KV self size` / `KV cross size` (GB) 출력
  - 32K 컨텍스트, 27B 모델(32 레이어, 8 KV 헤드, head_dim 256) 기준: f16 KV = 2 x 32 x 8 x 256 x 2B x 32768 토큰 = 10.7GB
- **정밀도**: KVarN은 128토큰 exact suffix 내부 유지(tail-tokens로 확장). PPL/수용률로 정밀도 확인 (MTP 수용률이 정밀도 프록시)

---

## 5. 주의사항

1. **llama-bench는 시뮬레이션 옵션 없음** - MTP/DFlash tg는 llama-server + /metrics로 측정
2. **KVarN은 `--kv-tail-tokens` 필수** (없으면 -63% 회귀)
3. **GPU1 전용 벤치** (GPU0은 llm-main 점유). native: `HIP_VISIBLE_DEVICES=1`, docker: `HIP_VISIBLE_DEVICES=0` + `--device /dev/kfd --device /dev/dri/renderD130 --device /dev/dri/card2 --group-add video`
4. **시리즈 간 컨테이너/프로세스 제거** (VRAM 잔여 제거)
5. **0010 디코드 수치 변경**: fp32 리덕션 순서 변경, greedy 출력 비트-동일 보장 없음
6. **ROCm 10 빌드**: `LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib`, `HIPCXX/HIP_PATH/HIP_DEVICE_LIB_PATH` 설정 (build.md HIP 절차)
7. **시나리오마다 새 프로세스** (VRAM/클럭 상태 재현성)
8. **llama.cpp 기여 규칙**: AGENTS.md 준수 - AI 생성 코드 허용이지만 이해 못 하는 코드 제출 금지, PR/커밋 자동 제출 금지

---

## 6. 산출물 인벤토리

| 파일 | 내용 |
|---|---|
| `rdna-boosts-apply-plan.md` | 순차 이식 계획 + 워크플로 |
| `bench-beellama-boosts-results.md` | boosts 블록별 누적 결과 + ROCm 10 비교 + KVarN 라우팅 분석 |
| `bench-beellama-results.md` | v0.4.4 docker S1~S7 beellama vs llm-main |
| `bench-beellama-v0.4.5-results.md` | v0.4.5 clean vs boosts (S1~S4) |
| `bench-beellama-plan.md` | 원본 벤치 계획 |
| `bench-results/results.csv` | v0.4.4 docker 원시 결과 |
| `bench-results-boosts/block0004_*.jsonl, block0010_*.jsonl, block0011_*.jsonl` | 블록별 원시 출력 |
| `beellama-boosts/src/` | boosts 적용 beellama 소스 |
| `beellama-boosts/bench-boosts.sh` | native boosts 벤치 스크립트 |
| `beellama-boosts/bench-rocm10.sh` | ROCm 10 docker 벤치 스크립트 (kvarn tail 자동 추가) |
| `beellama-boosts/Dockerfile` | ROCm 10 빌드 이미지 (미사용 - native로 대체) |
| `beellama-boosts/src/build-rocm10/` | ROCm 10 빌드 |
| `FIXES.md` | tbqplus fork fixes (recurrent shrink/expand) |
| `beellama-boosts/src/common/speculative.cpp.rej` | 0001 수동 병합 대기 hunk |

---

## 7. (추가) 2026-09-05 세션 완료 사항

### 완료 내역
1. **0001 adaptive MTP 수동 병합 완료** (`common/speculative.cpp` 2개 hunk, `.rej` 2개 삭제). 단, hunk#7 삽입 위치가 dflash begin()에 잘못 들어간 것을 MTP begin()(1619줄)으로 수정.
2. **컴파일 블로커 2개 제거** (이전 세션 미완성 병합 잔재):
   - `fattn-tile.cuh`: `launch_fattn_tile_switch_ncols1` 내 n_q<=8 verify 블록 (미존재 API type_KV/need_f16_K 참조) 제거
   - `unary.cu/.cuh`: `unary_gated_q8_1_*` q8_1 캐시 블록 (미존재 `ctx.q8_1_cache_get` 참조, 호출처 없음) 제거
3. **ROCm10 빌드 성공** (ROCM 10.0 clang-23): `--rocm-path=/opt/rocm/core-10.0 --rocm-device-lib-path=/opt/rocm/core-10.0/lib/llvm/amdgcn/bitcode` 필수. cmake는 pip 설치. 동시(병렬) 빌드 시 clang-23 segfault → 단독 빌드로 해결.
4. **벤치 완료** (자세한 결과: `bench-kv-vs-turboquant.md`):
   - beellama 8boosts+0001: tbq 대비 tg +9~11% (37.0 vs 33.6), kvarn8/8 최고 (948.6/37.29)
   - ROCm10 + boosts: S1 948.2/36.65, S2 949.0/35.58, S3 943.4/36.94 (native +5~7% pp)
   - 0001 adaptive MTP: 수용률 41.2→52.2%(+11pp), tg 43.6→47.9
   - KV 메모리: kvarn5/4 30.5% of f16 vs tbq q8/turbo3 35.5% (계산 기준)
5. **빌드 산출물**: native `/tmp/be-src-native0001/bin/`, ROCm10 `beellama-boosts/src/build-rocm10/bin/`
6. **알려진 잔여 문제**: S7(4병렬 native) empty table 아직 미조사. turboquant 메모리 실측은 prefill 비대칭으로 근사치만.

### 수정된 파일 (beellama-boosts/src)
- `common/speculative.cpp` (0001 hunk#7/#2)
- `ggml/src/ggml-cuda/fattn-tile.cuh` (고아 블록 제거)
- `ggml/src/ggml-cuda/unary.cu`, `unary.cuh` (고아 q8_1 제거)
- `common/speculative.cpp.rej`, `common/arg.cpp.rej` (삭제)
