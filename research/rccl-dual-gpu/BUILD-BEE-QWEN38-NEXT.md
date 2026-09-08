# BUILD-BEE-QWEN38-NEXT: dual GPU llama.cpp(beellama) 빌드 — Qwen3.8-Flash-Next 구동 가이드

> 작성: 2026-09-06 (Asia/Seoul)
> 목적: 듀얼 GPU(7900 XTX 24GB x2)에서 88GB MoE(Qwen3.8-Flash-Next UD-IQ4_XS)를 최대 속도로 구동하기 위한
>       RCCL 빌드 + 커뮤니티 패치(PR #28223, PR #27861) 적용 + 파라미터 튜닝의 전체 과정과 근거를 정리한다.
> 하드웨어: RX 7900 XTX x2 (gfx1100, 24GB/개) + i5-14600K(20T) + 96GB DDR4
> 비교 참조 환경: RTX 3090 24GB x2 (사용자 성공 사례), 2x 3090 + DDR4 (reddit expert cache 글)

---

## 1. 배경: 왜 이 빌드가 필요한가

Qwen3.8-Flash-Next는 **활성 파라미터가 작은 MoE**라 단일 GPU로도 "돌릴 수" 있지만,
전체 가중치는 UD-IQ4_XS 기준 **85.6 GiB** (파일 88GB)에 달한다.

GGUF 내부 구조 (3 shard 파싱 결과):

| 구성 요소 | 크기 | 비중 | 배치 특성 |
|---|---|---|---|
| per_layer_token_embd.weight | 26.8 GiB | 31% | **GGML_OP_GET_ROWS → 열 분할 불가(MIRRORED)** |
| FFN exps (gate/up/down) | 54.5 GiB | 64% | GGML_OP_MUL_MAT_ID → axis1 분할 가능 |
| 기타(attn/dense/output/embd) | 4.3 GiB | 5% | 분할 가능 |

- 레이어 수: **48** (per_layer_token_embd가 [160, 320001536] — 아, 160이 아니라 48 레이어 구조)
  - 정정: GGUF의 blk은 0~47. 레이어당 exps 평균 1.13 GiB
- **핵심 문제 1**: per_layer_token_embd 26.8GB 단일 텐서는 24GB 카드에 못 들어감 → CPU 필수
- **핵심 문제 2**: exps 54.5GB는 split 가능하지만 GPU당 27.2GB라 24GB 초과 → 일부 host 필요
- **핵심 문제 3**: host로 보낸 exps를 그냥 CPU에서 계산하면 디코딩이 CPU GEMM 병목 (10 t/s 수준)

이 세 문제를 해결하기 위해 커뮤니티의 3가지 빌드 기법을 적용한다:
1. **RCCL 빌드** — 듀얼 GPU tensor split의 all-reduce 고속화 (§3)
2. **PR #28223** — host 버퍼 오버라이드 + mmap read_raw 로딩 고속화 (§4)
3. **PR #27861** — GPU-resident LRU expert cache (§5, 핵심)

---

## 2. 커뮤니티 데이터 종합 (인용)

### 2.1 RCCL의 필요성 (reddit, dual 7900 XTX)

> "Build llamacpp with rccl enabled, the default pre builds do not have this. So make sure your build has that.
> Then tensor split the xtx cards and you will get decent performance. I get 1200+ on dual xtx with this setup"

- stock llama.cpp b10362 / 4801e3c, ROCm 7.2.4 + RCCL 2.27.7 + HIP 7.2.53211 + AMD clang 22
- 벤치(5회 반복): **pp4096 1251.2 / tg512 90.4 / tg2048 104.9**

### 2.2 tensor + MTP = 29→69 tok/s (reddit 1vtoux9, dual 7900 XTX)

| config | gen tok/s | vs baseline |
|---|---|---|
| 1 GPU (baseline) | 29.28 | - |
| 1 GPU + MTP | 53.90 | +84% |
| 2 GPU -sm tensor | 34.66 | +18% |
| **2 GPU tensor + MTP** | **69.25** | **+137%** |
| 2 GPU tensor + MTP (262144 ctx) | 69.41 | +137% |

- **RCCL 미사용 빌드(10481)** 로도 "peer access enabled"면 69 t/s 달성
- `-sm tensor`는 KV 캐시를 양 카드로 분산 → MTP 헤드(1.3GB) + 256K 풀 윈도우 동시 수용
- **MTP 수용률은 컨텍스트가 길수록 상승**: 벤치 73% → 86K 차면 87% → 실사용 64%

### 2.3 Vulkan은 느리고 ROCm이 정상 (reddit mbrodie)

> "on 2 x 7900 xtx you can run full 262k context and get 1200pp and 50 - 70tps with a q8 on long token runs
> using rocm for whatever reason Vulkan is incredibly slower for qwen 3.8 and tensor parallelism is broken
> but it seems to be working fine in rocm"

27B dense ROCm 설정 (결과 1100 pp / 55-70 tps):
```
--split-mode tensor --tensor-split 1,1 --ctx-size 204800
--cache-type-k f16 --cache-type-v f16
--cache-type-k-draft q8_0 --cache-type-v-draft q8_0
--spec-type draft-mtp --spec-draft-n-max 4
--kv-unified -b 2048 -ub 512
```

### 2.4 RDNA3에서 비양자화 KV가 오히려 빠름 (reddit johnchristianson, dual 7900 XTX 72K)

| KV 타입 | pp (72K) | tg (1000t+) |
|---|---|---|
| **BF16/BF16** | **967.21** | **51.53** |
| Q8_0/Q8_0 | 923.66 | 48.00 |
| 양자화 비용 | -4.5% | **-6.8%** |

> "the xtx and rdna3 seems to actually just be 16 bit floating point under the hood, so quantizing slows it down"
> "longer the context, the bigger the hit"

- RDNA3는 내부적으로 FP16 파이프라인 → KV를 q8로 양자화하면 디양자화 오버헤드가 메모리 이득 상쇄
- 결론: **컨텍스트가 길수록, VRAM 여유가 있으면 비양자화(f16/bf16) KV가 유리**

### 2.5 MoE expert cache 가 이 모델의 정답 (reddit, 2x 3090 24GB + DDR4 188GB)

> "Now as for what I did: PR #27861, the GPU-resident LRU expert cache. Instead of parking whole expert layers
> in VRAM, it caches recently used experts per layer. The experts this model picks for one token are mostly the
> same ones it picked for the last few dozen tokens. so the hit rate is 80-85% on code and higher on prose."

- UD-Q6_K_XL (88GB급) + exps 전체 host + `--moe-expert-cache 135` → **decode 17 → 25-29 t/s**
- exps 48 expert 레이어 전부 host(RAM)에 고정, 나머지는 GPU. 261K full context f16 KV
- 핵심 파라미터: `-ot "ffn_(gate|up|down)_exps\.weight=CUDA_Host,per_layer_token_embd\.weight=CPU" --moe-expert-cache 135`
- **ubatch 2048→512로 내리면 GPU당 ~5GB 확보** (compute buffer가 ubatch에 비례) → cache slot 80→135개
- cache slot 크기: Q6 기준 **~100MB/slot/GPU**
- 실패한 것들: thread count, poll, CPU masks, q8 KV, lazy PLE, n-gram draft(prose), MTP(temp 0.7), cache upload 2개 초과

### 2.55 PR #28223 독립 검증 (RTX 5090 32GB, 9800X3D, 192GB — reddit 1w5vjp6 댓글 A/B)

> "[PR #27861] gives a real gain ... Real-prose corpus, medians of 3 at matched depths"
> setup: 5090 32GB + 9800X3D + 192GB, Flash-Next AD-4.27bpw Q4_K_M(~88GB, 33-shard), expert cache 192 slots / inserts 4, q8 KV, ub 512

**중요: 이 데이터는 두 가지를 분리해 보여준다**

A) **expert cache 단독 효과** (master b10775 + #27861, 캐시만):
| depth | Before(무캐시) | After(캐시) | Δ |
|---|---|---|---|
| 50K | 21.3 | 46.9 | +121% |
| 100K | 15.5 | 38.3 | +148% |
| 154K | 14.2 | 34.5 | +143% |
- 심층에서 캐시 자체 순효과: ~+40% (캐시 off 새 master 24.7 → 캐시 on 34.5 @154K)
- K=192에서 LRU 시뮬 히트율 95.9%

B) **PR #28223 (pinned experts) 단독 효과** (같은 하드웨어, A/B 단일 변수):
| 메트릭 | Run A 무PR (-ot CPU) | Run B PR (-ot CUDA_Host) | Δ |
|---|---|---|---|
| load_tensors | experts in **CPU_Mapped** | **CUDA_Host 48300 MiB** | pinned ✓ |
| Load time | 29.5s | 26.8s | -9% |
| Prefill 47K (cold) | 227 t/s | **627 t/s** | **2.8×** |
| Prefill 101K | 237 | 575 | 2.4× |
| Prefill 154K | 226 | 515 | 2.3× |
| Decode 47/101/154K | 44.5/37.1/32.3 | 44.1/37.7/33.1 | flat ~+2.5% |

> "Result: pinned experts give a consistent **2.3-2.8× prefill gain** across all depths, with **no decode loss**."

**핵심 통찰 (우리 빌드 적용)**:
- PR #28223은 **decode가 아니라 prefill**을 극대화 (2.3-2.8×) — mmap에서 host 텐서를 pinned로 유지
- **--load-mode none으로도 같은 문제 회피 가능**: "You ran --load-mode none so mmap was never in play, which is the other way to dodge the same problem"
- → 우리 구성은 **--load-mode none (§11.6) + read_raw**로 이미 이득 상당부분 확보. mmap+`-ot ROCm_Host` 조합이면 PR #28223 커밋1로 추가 prefill 이득.

**Windows/다른 플랫폼 이슈 (이식 시 주의)**:
1. `GGML_MOE_LOG` 블록의 `flockfile/funlockfile`은 POSIX-only → MSVC 빌드 실패, `#ifndef _WIN32` 가드 필요
2. **잠재 버그(중요)**: mmq scatter 경로는 토큰의 라우팅 id가 서로 다르다고 가정. 캐시는 **모든 uncached expert를 같은 dummy slot에 매핑** → id가 충돌 → `mm_ids_helper`가 중복 제거하며 `ids_src1`에 초기화 안 된 구멍이 생김 → scatter 커널이 오염된 행 인덱스 읽음. 5090에서 ~149GB OOB 쓰기 감지. 단일 토큰 decode는 풀 재사용으로 우연히 통과.
   - fix: `ids_src1`을 -1(scatter)/0(forward-map)으로 memset + scatter 커널에 skip-negative 가드
   - **우리는 단일 토큰 decode가 주 타깃이라 실사용엔 무해하지만, 배치 확장 시 반드시 수정 필요**

> 5090 full config:
> ```
> llama-server -m <model>-00001-of-00033.gguf
>   --jinja -c 192512 -t 8 --parallel 1
>   --ubatch-size 512 --load-mode none -fa on
>   --cache-type-k q8_0 --cache-type-v q8_0
>   -fitt 17250
>   --moe-expert-cache 192 --moe-expert-cache-inserts 4
>   --temp 0.8 --top-p 0.95 --top-k 20
> ```

### 2.6 안 되는 조합 (우리 실측)

| 구성 | 결과 |
|---|---|
| tensor split + exps 전부 GPU | GPU당 30.6GB 요구 → OOM |
| tensor split + exps 전부 host(CPU) | GPU 유휴, CPU 730-820%, tg 10 t/s |
| layer split + 17레이어 exps CPU | GPU 사용률 30%→0%, tg 10-11 t/s |
| tensor + exps ROCm_Host + cache 135 | **진행 중 (본 문서 대상)** |

---

## 3. RCCL 빌드

### 3.1 왜 RCCL인가 (코드 분석)

beellama의 텐서 병렬 all-reduce는 **하이브리드**:
- **RCCL**: 대형 텐서 (P2P + BF16 왕복) — bandwidth-bound
- **내부 HIP 파이프라인**: 소형 텐서 (호스트 스테이징) — latency-bound

그런데 내부 파이프라인은 `ggml/src/ggml-cuda/allreduce-hip.cu:743-766`에 **RDNA4-only 게이트**가 있다:

```cpp
// RDNA4-only gate: the internal/hybrid all-reduce is verified ONLY on
// RDNA4 (gfx1200/gfx1201).  On any other architecture the pipeline is
// not created and the allreduce dispatch falls back to the default (RCCL) path
```

- gfx1100(7900 XTX)에서는 `return nullptr` → **무조건 폴백**
- 폴백 대상: `ggml_backend_cuda_comm_init_nccl()` → RCCL이 컴파일되어 있고 `ncclCommInitAll`이 성공하면 RCCL 사용
- RCCL 미빌드면 butterfly(느림) → 이전 세션 §19의 "tensor split tg -6~12%" 원인
- 런타임 로그 확인:
  `internal all-reduce is RDNA4-only (gfx1200/gfx1201); device 0 reports gfx1100 -- falling back to the default path`

**교훈**: RCCL 없이 내부 파이프라인을 gfx1100에서 쓰려면 Stastez의 HIP AllReduce 패치(PR #27825)나
RDNA4 게이트 제거가 필요하지만, **RCCL이 정상 동작하면 RCCL이 더 빠르다** (llama.cpp 리뷰어 IMbackK:
"rccl should also be faster anyhow").

### 3.2 소스 세팅

```
SRC=/home/baramofme/IdeaProjects/llama-cpp-turboquant/beellama-boosts/src
BLD=/tmp/be-src-native0001

cmake -S $SRC -B $BLD \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.3/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_AR=/opt/rocm-7.2.3/lib/llvm/bin/llvm-ar \
  -DCMAKE_HIP_COMPILER_RANLIB=/opt/rocm-7.2.3/lib/llvm/bin/llvm-ranlib \
  -DCMAKE_HIP_PLATFORM=amd \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_MMQ_MFMA=ON \
  -DGGML_HIP_NO_VMM=ON \
  -DGGML_HIP_RCCL=ON \     # ← 핵심 (기본 OFF)
  -DCMAKE_C_COMPILER=/opt/rocm-7.2.3/lib/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-7.2.3/lib/llvm/bin/clang++
```

- RCCL 옵션: `ggml/CMakeLists.txt:326` (`option(GGML_HIP_RCCL ... OFF)`)
- `ggml/src/ggml-hip/CMakeLists.txt:50-52`: RCCL=ON이면 `find_package(rccl REQUIRED)` + `GGML_USE_NCCL` 정의
- `:139-141` `add_compile_definitions(GGML_USE_NCCL)` (RCCL은 NCCL과 인터페이스 동일)
- `:167-169` `target_link_libraries(ggml-hip PRIVATE ggml-base roc::rccl)`
- 런타임 초기화: `ggml-cuda.cu:1267` `ncclCommInitAll` → 성공 시 `try_allreduce = nccl`

### 3.3 만난 오류와 해결

| 오류 | 원인 | 해결 |
|---|---|---|
| **clang-22 optimizer segfault** (`fattn-mma-kvarn-instance-ncols1_16-ncols2_8.cu.o`, `llvm::simplifyCall` crash) | -j 16 동시 컴파일 시 리소스 경합 (이전 ROCm10 clang-23과 동일 패턴) | 해당 오브젝트만 **-j 1 단독 재빌드** → 통과 → 이후 -j 8 전체 빌드 정상 |
| `llama_params_fit is not implemented for SPLIT_MODE_TENSOR` | `--fit`이 tensor split 미지원 | `--fit off` 필수 |

### 3.4 검증

```
ldd bin/llama-server | grep rccl
# librccl.so.1 => /opt/rocm-7.2.3/lib/librccl.so.1   ← 링크 확인
```

- 27B tensor split 1,1 스모크: 로드 성공, **VRAM 정확히 균등** (GPU0 7.09 / GPU1 7.09 GiB, 차이 7KB 미만)
- `llama-bench -p 2048 -n 128 -sm tensor -ts 1,1`: pp 800 / tg 31
- "NCCL init failed" 로그가 없음 = `ncclCommInitAll` 성공

> 주의: `internal all-reduce is RDNA4-only` 경고는 내부 파이프라인 비활성일 뿐이고 RCCL 경로로 폴백되어 정상 동작.

---

## 4. PR #28223 — host buffer override + read_raw 로딩

### 4.1 왜 필요한가

1. `-ot '...=ROCm_Host'`가 **"unknown buffer type"** 으로 거부됨:
   `parse_tensor_buffer_overrides()`(common/arg.cpp)가 `ggml_backend_dev_buffer_type()`만 수집하고
   **host buffer type**을 목록에서 누락했다.
2. mmap 로드 모드에서 host 버퍼로 지정된 텐서를 **CPU 버퍼로 강등** (create_tensor의 안전장치) → pinned 메모리를 못 씀
3. host 버퍼로 복사하는 로딩이 **페이지 단위 faulting** (readahead 없음) → 44GB exps 로딩이 극도로 느림

### 4.2 커밋 1 (5cfa6a87): host buffer override 허용 + 강등 방지

- `common/arg.cpp`: buft_list에 `ggml_backend_dev_host_buffer_type(dev)` 추가
  → HIP에서 이름은 `"ROCm_Host"` (GGML_CUDA_NAME="ROCm" + "_Host")
- `src/llama-model-loader.cpp create_tensor()`: `buft_overridden` 플래그 추가,
  override가 지정되면 mmap의 host→CPU 강등 로직 건너뜀

reddit 2x3090 실측 (전 이 커밋 내용):
> "Testing revealed that a host buffer type, specifically CUDA_Host was being replaced with CPU buffer type under
> mmap. Patch makes it so that ... the tensor is then copied into pinned memory ... Prefill of a 26k prompt went
> from 166 t/s to 379 t/s on 2 x RTX 3090 with 40 expert layers on the host."

### 4.3 커밋 2 (c59754bfc): host 대상 텐서를 read_raw로 로드

- `src/llama-model-loader.cpp load_all_data()`: 목적지가 host 버퍼(ROCm_Host/CPU)면
  mmap memcpy 대신 `file->read_raw(cur->data, n_size)` 직접 읽기
- 효과 (reddit 2x3090, 101.75 GiB exps on CUDA_Host): **cold load 512s → 168s** (copy 450s → 104s)
- 커밋 메시지 인용:
> "The mmap carries POSIX_MADV_RANDOM under --numa distribute, so the memcpy faulted one 4 KiB page at a time
> with no readahead."

---

## 5. PR #27861 — MoE expert cache (이 빌드의 핵심)

### 5.1 왜 필요한가

exps를 host로 보내는 것만으로는 **디코딩이 CPU GEMM 병목**:
- 우리 실측: GPU 사용률 0%, CPU 730-820%, tg 10 t/s
- MoE 라우팅은 시간적 국소성(temporal locality)이 강함: 이번 토큰이 고른 expert는 수십 토큰 전과 대부분 동일
- → **자주 쓰는 expert만 VRAM에 LRU 캐시하면 대부분의 expert 연산을 GPU에서 처리**

### 5.2 동작 원리 (커밋 bccbacdb8, 12파일/645줄)

```
[CPU mul_mat_id]                      [device cache chain]
  experts → host RAM                   up_c/gate_c/down_c (VRAM, slot k+1)
     │                                     │
  src[3] 테이블로 캐시된 id 스킵          dev_table로 slot remap → mul_mat_id
  (캐시된 expert는 dst 0으로)              → swiglu → down → 결과
                                           ↓
                                     experts = CPU결과 + GPU결과 (합산)
```

- **신규 파일**: `src/llama-moecache.cpp/h`
  - 레이어별 companion 텐서 `up_c/gate_c/down_c` [ne0, ne1, n_slots+1] — **slot n_slots는 영구 0 (dummy)**
  - I32 `dev_table[512]`: expert id → slot (device copy는 get_rows로 cache chain에, host copy는 src[3]으로 스킵)
  - async 업로드 워커: decode 스레드에서 분리, CPU 테이블만 step()에서 갱신 → 실행 중 그래프는 torn 상태 안 봄
  - 관찰 콜백: CPU mul_mat_id가 ffn_gate_exps 라우팅 id 기록 → LRU 배치
- **ggml-cpu.c mul_mat_id**: `dst->src[3]` 존재하면 캐시된 expert는 dst 행 zeroing + 스킵;
  `op_params[0]` = dummy slot 값. 라우팅 관찰 콜백 + GGML_MOE_LOG(진단용)
- **ggml.h/ggml.c**: `ggml_moe_obs_cb_t` 콜백 전역 등록/조회
- **llama-graph.cpp build_moe_ffn**: n_tokens==1 && 조건(LLM_FFN_SILU 등) 충족 시
  `mcache = llama_moe_cache_lookup(up_exps)` → slot remap → up/gate/down에 src[3] 연결 →
  device 체인(mul_mat_id) 실행 → `experts = ggml_add(experts, down_g)`
- **파라미터**: `--moe-expert-cache N` (0=비활성), `--moe-expert-cache-inserts N` (기본 2)
- 호출점: context 생성 시 `llama_moe_cache_init()`, `decode()` 끝 `llama_moe_cache_step()`

### 5.3 reddit 실측 (2x 3090 + DDR4 188GB, UD-Q6_K_XL 88GB급)

> "Before: ~17 t/s decode ... Now: 25-29 t/s decode short and mid context, ~17 at 131k"
> "hit rate is 80-85% on code and higher on prose"
> "dropping ubatch from 2048 to 512 frees ~5 GB per GPU (compute buffers scale with ubatch),
> which went from 80 to 135 slots per layer at full context"
> "The cost here is slower prefill on long prompts"

핵심 명령:
```bash
llama-server -m Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf \
  -ngl 99 -c 261888 --parallel 1 -fa on \
  -ot "ffn_(gate|up|down)_exps\.weight=CUDA_Host,per_layer_token_embd\.weight=CPU" \
  -t 16 -tb 44 -b 4096 -ub 512 -ctk f16 -ctv f16 \
  --moe-expert-cache 135
```

### 5.4 llama.cpp issue 참조 (왜 RCCL 없이도 되는가)

- **llama.cpp PR #27825 (Stastez "HIP: Enable AllReduce for ROCm")**: RCCL 없이는 내부 HIP AllReduce로 폴백.
  `cudaHostAlloc*` → `hipHostAlloc*`, `__nanosleep` → `__builtin_amdgcn_s_sleep` 치환으로 활성화.
  gfx1030+gfx1201에서 pp2048 +16%. 단, 리뷰어 IMbackK:
  > "This changes the fallback path used when rccl is not available/enabled so you should not see anything
  > with it enabled. rccl should also be faster anyhow"
- **ROCm issue #6074**: RCCL 2.27.7 + dual 7900 XTX에서 ROCm 7.2.1의 dual-GPU collective 실패 사례.
  7.2.0에서는 동작, 7.2.1에서 회귀 후 복구. 우리는 native 7.2.3 → 정상.
- **llama.cpp #27124 / #26750**: ROCm에서 MTP draft acceptance가 낮음 (`draft acceptance = 0.35929`).
  CUDA도 0.36 수준, Vulkan은 ~0.92. → MTP 대신 tensor split/expert cache로 tg 이득을 얻는 게 ROCm에서 유효.

---

## 6. 최종 구성과 파라미터 근거

### 6.1 사용자 성공 사례 (동급 환경: 3090 24GB x2 + DDR4 64GB, UD-IQ4_XS 88GB, 131K)

> "UD-IQ4_XS 약 88GB 모델을 131K context로 구동 성공, 실제 생성 속도 약 30 tokens/s"
> "per_layer_token_embd + 일부 FFN expert를 CPU로 offload"
> "ublch 512 → 256으로 줄였지만 compute buffer가 2GB → 1.85GB로밖에 안 줄었다. 그래서 GPU weight를
> 조금 더 CPU로 뺐다. blk 0~8, 25~31 → blk 0~9, 25~31 (한 레이어만 추가) 그리고 성공."

```bash
llama-server -m Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 -sm layer -fit off \
  -c 131072 -fa on -ctk q8_0 -ctv q8_0 \
  -b 2048 -ub 256 -t 8 --threads-batch 8 \
  --jinja --load-mode mmap --lazy-mode on \
  -ot 'per_layer_token_embd.weight=CPU,blk\.([0-9]|2[5-9]|3[01])\.ffn_(up|down|gate|gate_up)_(ch|)exps=CPU'
```

핵심 통찰:
- 이 모델은 **-sm layer** (파이프라인)와 CPU 오프로드를 조합해야 함 (tensor split은 per_layer 26.8GB 때문에 OOM)
- **ub 512→256**으로 compute buffer 축소, 그래도 부족하면 **GPU weight를 1레이어씩 CPU로 추가 이동**
- `-ot` 패턴: `blk\.([0-9]|2[5-9]|3[01])` = blk.0-9 + blk.25-31 (17레이어 exps CPU)

### 6.2 우리 빌드의 최적 구성 (7900 XTX + RCCL + expert cache)

```
llama-server -m /opt/llm/models/Qwen3.8-Flash-Next-UD-IQ4_XS/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --host 0.0.0.0 --port 8323 \
  -ngl 99 -sm tensor -ts 1,1 -fit off \            # RCCL 빌드라 tensor split 가능
  -c 8192 -fa on -ctk f16 -ctv f16 \               # RDNA3: 비양자화 KV 유리 (§2.4)
  -b 2048 -ub 512 -t 12 --threads-batch 12 \
  --jinja --load-mode none \                       # read_raw 로딩 (§4.3)
  -ot 'ffn_(up|down|gate)_exps\.weight=ROCm_Host,per_layer_token_embd\.weight=CPU' \
  --moe-expert-cache 135                           # 핵심 (§5)
```

- exps 전체 → ROCm_Host (pinned, GPU가 직접 접근) — CUDA의 CUDA_Host에 대응
- per_layer_token_embd → CPU (26.8GB는 어차피 분할 불가)
- `--moe-expert-cache 135`: hot expert를 VRAM 캐시 → 디코딩의 expert 연산을 GPU에서 처리
- `--load-mode none`: mmap 없이 순차 read → 페이지 캐시 경합 제거 (스왑 정체 시 특히 중요)

### 6.3 VRAM 분배 시나리오 (tensor split + exps host)

- GPU0/GPU1: dense 4.3GB split (각 2.1GB) + KV/compute ≈ 3.4GB (실측)
- host: exps 54.5GB (ROCm_Host) + per_layer 26.8GB (CPU) = 81.3GB
- cache slot: GPU당 여유 ~20GB / 100MB-per-slot(Q6, IQ4_XS는 더 작음) → **135 slot 이상 가능**

---

## 7. 빌드/테스트 과정에서 만난 오류 일지

| 단계 | 오류 | 해결 |
|---|---|---|
| RCCL 빌드 | clang-22 segfault (kvarn MMA 템플릿) | -j 1 단독 재빌드 |
| 실행 | `--fit is not implemented for SPLIT_MODE_TENSOR` | `--fit off` |
| -ot | `unknown buffer type` (ROCm_Host) | PR #28223 커밋1 (host type 수집) |
| 로딩 | mmap → host 복사가 페이지 단위 faulting (느림) | PR #28223 커밋2 (read_raw) |
| build_moe_ffn | `mc_inp` undeclared | `cur = ggml_reshape_3d(...)` 직후 `ggml_tensor * mc_inp = cur;` 삽입 |
| 시스템 | swap 8GB 100% 고갈 + mmap 캐시 56GB | `--load-mode none`으로 캐시 경합 제거 |
| 디코딩 | exps host 전부 → GPU 0%, CPU 730% (tg 10) | `--moe-expert-cache`로 hot expert VRAM 캐시 |

---

## 8. 참고 자료

- beellama: https://github.com/Anbeeld/beellama.cpp (다중 GPU: docs/multi-gpu.md)
- tensor parallelism: https://github.com/ggml-org/llama.cpp/pull/19378 (tg +41~127%, pp -29%)
- HIP AllReduce: https://github.com/ggml-org/llama.cpp/pull/27825
- host override + read_raw: https://github.com/ggml-org/llama.cpp/pull/28223
- MoE expert cache: https://github.com/ggml-org/llama.cpp/pull/27861
- reddit [1vtoux9] Tensor+MTP dual 7900 XTX: 29→69 tok/s
- reddit r/ROCm 1vyfisx + 후속 (PCIe x4, HIP AllReduce, Q8 전송)
- llama.cpp issues: #27124, #26750 (ROCm MTP acceptance), ROCm #6074 (RCCL 7.2.1 회귀)
- -ot 관련: PR #11397 (최초 도입), PR #14990 (CPU extra bufts + --no-repack), PR #15191 (draft overrides),
  issue #17274 (CUDA_Host 비호환 텐서), discussion #13154 (부족한 문서 논의), #20642 (expert spread)
- surgical MoE offload 레시피: https://github.com/Forge-the-Kingdom/inference-serving-recipes/blob/main/recipes/llama.cpp/surgical-moe-ot-offload.md
- DocShotgun CPU+GPU 튜닝 가이드: https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0

---

## 9. 재현 방법

```bash
# 1) 소스 (RCCL 패치 + PR#28223 + PR#27861 포함)
#    beellama-boosts/src  (본 프로젝트 내)

# 2) 빌드 (native ROCm 7.2.3)
cmake -S beellama-boosts/src -B /tmp/be-src-native0001 -DGGML_HIP_RCCL=ON ... (위 §3.2)
cmake --build /tmp/be-src-native0001 --config Release -j 8 --target llama-server

# 3) 산출물
#    rccl-build/bin/ (본 프로젝트 내) — llama-server/llama-bench + lib*.so

# 4) 실행 (§6.2 구성)
rccl-build/bin/llama-server ... 

# 5) 진단
#    GGML_MOE_LOG=/tmp/moe.log 로 expert 라우팅 dump (히트율 분석)
#    rocm-smi --showuse : 생성 중 GPU 사용률 (cache 히트 시 올라가야 정상)
```

---

## 부록 A. 우리 실측 벤치 (2026-09-06)

| 구성 | pp | tg | 비고 |
|---|---|---|---|
| RCCL 27B tensor split 1,1 (llama-bench p2048) | 800 | 31 | dense 비교 |
| Flash-Next tensor + exps ROCm_Host + cache 135 | 로딩 중 | - | 본 가이드 최종 대상 |
| (과거) Flash-Next exps 전부 CPU | - | 10 | GPU 0%, CPU 병목 |
---

## 10. reddit/커뮤니티 출처 통합 (URL 포함)

아래 reddit 글들은 각 섹션의 결론 근거. reddit 접근이 캡차로 차단되어 있어, 사용자가 제공한 원문/스냅샷을 기준으로 정리.

### 10.1 dual 7900 XTX Qwen3.8-27B Q8 설정 (1vxh7gq)
- https://www.reddit.com/r/ROCm/comments/1vxh7gq/help_setting_up_dual_7900_xtx_for_qwen3827bq8/
- 관련: dual 7900 XTX에서 qwen3.8-27B Q8_0 구동 설정 문의. §2.2/§2.3의 결론(비양자화 KV + tensor split + f16)의 배경이 되는 글.
- 핵심 교훈: **같은 하드웨어에서도 RCCL 유무 + KV 타입 + 빌드 버전이 tg를 29~69로 갈라놓음** (29→69는 1vtoux9 참조).

### 10.2 tensor + MTP dual 7900 XTX 29→69 tok/s (1vtoux9)
- https://www.reddit.com/r/ROCm/comments/1vtoux9/qwen3827b_on_llamacpp_tensor_mtp_on_dual_7900_xtx/
- 전체 내용 §2.2 참조 (이 문서).

### 10.3 dual 7900 XTX PCIe X4 tensor parallel (1w14qal) — RCCL/HIP AllReduce
- https://www.reddit.com/r/ROCm/comments/1w14qal/dual_7900xtx_tensor_parallel_on_limited_pcie_x4/
- **핵심**: 칩셋 뒤 PCIe x4 슬롯의 카드에서 RCCL AllReduce가 동작 안 함 → llama 내부에서 CPU 경유 통신으로 폴백 (매우 느림)
- 해결: CUDA 내부 AllReduce 대신 **HIP 버전으로 교체**:
  - `cudaHostAlloc` / `cudaHostAllocMapped` / `cudaHostAllocPortable` → `hipHostAlloc*`
  - `cudaHostGetDevicePointer` → `hipHostGetDevicePointer`
  - `__nanosleep(100)` → `__builtin_amdgcn_s_sleep(1)` (HIP)
- 결과: **PP 470~500 → 1100 tk/s** (P2P 통신 복구)
- 이후 rocprofv3로 프로파일 → bandwidth-starved 확인 → **Q8 전송 구현** (BF16 대신, 전송량 반으로) → **PP8192 1390~1400 tk/s**
- rdna-boosts 적용 + adaptive MTP로 prose 60-65, code 85-90 t/s sustained
- 상향 반영된 공식 PR: **llama.cpp #27825** "HIP: Enable AllReduce for ROCm" (이 버전에는 높은 우선)
- 우리 빌드 함의: 우리는 PCIe가 8x/8x 이상이라 P2P 동작, RCCL 빌드로 해결 (§3). HIP AllReduce 패치(#27825)는 RCCL 없이도 필요한 대체 경로.

### 10.4 Qwen3.8-Flash-Next 2x3090 + DDR4 17→25-29 t/s (1w5vjp6) — MoE expert cache
- https://www.reddit.com/r/LocalLLaMA/comments/1w5vjp6/qwen38flashnext_on_2x3090_ddr4_17_2529_ts_decode/
- 전체 내용 §2.5 + §2.55 참조 (PR #27861 expert cache 원본 실측 + 5090 독립 A/B)
- 핵심: expert cache는 decode +40%(심층), PR #28223 pinned는 prefill 2.3-2.8× — 두 이득 분리 확인

---

## 11. 각 과정별 RDNA3 / 7900 XTX 최적화 포인트

이 빌드의 모든 결정은 gfx1100(RDNA3) 특성에 근거. 무엇을 왜 바꿨는지 정리.

### 11.1 RCCL (gfx1100 all-reduce 제약)

**RDNA3의 문제**: 내부 HIP all-reduce 파이프라인이 RDNA4-only.
- gfx1100에서 `ggml_cuda_ar_pipeline_init`이 `return nullptr` → all-reduce 폴백
- RCCL을 컴파일하지 않으면 butterfly(CPU 경유) → tensor split tg 대폭 손해

**변경**: `-DGGML_HIP_RCCL=ON` 빌드
- gfx1100 듀얼에서 RCCL(P2P) 사용 → all-reduce 고속화
- 기대: reddit 1200+ pp, tg 90+ (§2.1) 수준

### 11.2 비양자화 KV (RDNA3 FP16 파이프라인)

**RDNA3 특성**: 내부적으로 FP16 연산. KV를 q8로 양자화하면 디양자화 오버헤드.
- 실측(§2.4 johnchristianson): BF16 KV가 Q8_0보다 pp +4.5%, tg +6.8%
- 27B 운영은 원래 q8_0/q5_0이지만, MoE Next는 **f16 KV** (컨텍스트 짧을 때)

**변경 (Next 구성)**: `-ctk f16 -ctv f16`
- RDNA3 FP16 특성 활용
- VRAM 여유 있을 때만 (컨텍스트/슬롯 수 고려)

### 11.3 -ot ROCm_Host (GPU가 접근 가능한 pinned host)

**RDNA3/듀얼 특성**: exps를 host로 보내되 GPU가 직접 읽으려면 pinned memory 필요.
- 기본 `-ot ...=CPU`는 일반 host (GPU 직접 접근 불가)
- `ROCm_Host`(CUDA_Host의 HIP 대응) = hipHostAlloc pinned → GPU 접근 가능

**변경**: PR #28223 커밋1로 `ROCm_Host`를 -ot 타겟으로 인식
- exps는 ROCm_Host, per_layer_token_embd만 CPU

### 11.4 read_raw 로딩 (mmap 페이지 fault 회피)

**RDNA3 환경 문제**: mmap(AUTO/`--numa distribute`)로 host 버퍼 복사 시 페이지 단위 fault.
- RDNA3 호스트(DDR4)에서 특히 느림

**변경**: PR #28223 커밋2 — host 대상 텐서는 `read_raw` 직접 읽기
- cold load 512s → 168s (reddit 실측, 101GiB exps)

### 11.5 MoE expert cache (hot expert만 VRAM)

**MoE 특성**: 라우팅의 시간적 국소성 → 최근 expert 재사용률 80-85%.
**RDNA3 함의**: CPU만으론 expert GEMM 병목(GPU 0%, tg 10) → VRAM 캐시로 GPU 활용.

**변경**: PR #27861 — `--moe-expert-cache 135`
- hot expert를 GPU VRAM에 LRU 캐시 → expert 연산을 GPU로
- 기대: 2x3090 실측 17→25-29 t/s, prose/code에서 hit 80-85%

### 11.6 --load-mode none (캐시 경합 제거)

**RDNA3 호스트 문제**: 88GB 모델 mmap + read_raw 복사가 페이지 캐시/스왑과 경합.
- 우리 환경: swap 8GB 100% 고갈 + 캐시 56GB → 로딩 정체

**변경**: `--load-mode none` — 순차 read로 캐시 경합 없이 host 텐서 로드.
- read_raw(§4.3)와 조합 시 최적.

### 11.7 -sm tensor 채택 (RCCL 빌드 후)

**RDNA3 + 듀얼 특성**: 
- RCCL 없이 tensor는 tg -6~12% (§19) → RCCL 빌드 후 tensor 채택
- RCCL + tensor는 tg 특화 (PR #19378: layer 대비 +41~127%)
- per_layer_token_embd 26.8GB는 24GB 초과 → CPU offload 필요 (§11.3과 결합)

---

## 11A. -ot (--override-tensor) 심층 사용법 — 웹 문서/이슈 종합

### 11A.1 문법과 동작 (PR #11397, llama.cpp docs, Debian manpage)

```
-ot, --override-tensor <tensor name pattern>=<buffer type>,...
override tensor buffer type
(env: LLAMA_ARG_OVERRIDE_TENSOR)
```

- **패턴은 C++ 정규식(regex)** — 단순 문자열 검색이 아님 (PR #11397 원래 string search였다가 regex로 변경)
- **여러 개는 콤마로 묶거나 -ot 반복** 지정 가능
- **주의(중요)**: 최근 빌드는 **다중 -ot 플래그 사용이 deprecated** — 두 번째 -ot가 **모든 매칭 텐서를 마지막 타겟으로 조용히 재전송**함. → 반드시 **하나의 콤마 구분 regex**로 통합
- 사용 가능한 buffer type 확인: 잘못된 타입 지정 시 에러에 목록 출력
  ```
  Available buffer types:
    CPU
    ROCm0
    ROCm1
    ROCm_Host   ← (PR #28223 패치 적용 시 표시됨)
  ```
- **매칭 확인**: `-v` (verbosity)로 "tensor X buffer type overridden to Y" 로그 확인
- **적용 순서**: `--tensor-split`/`-ngl`이 먼저 할당되고 `-ot`가 마지막에 적용

### 11A.2 기본 사용 예시

```bash
# 1) 모든 exps를 CPU로 (MoE 기본)
-ot "exps=CPU"

# 2) 특정 레이어 범위만 exps CPU로 (regex)
-ot "blk\.(4[0-7])\.ffn_(gate|up|down)_exps\.weight=CPU"

# 3) 레이어별 GPU 지정 (다중 GPU)
-ot "blk\.([0-9])\.=CUDA0,blk\.(1[0-9])\.=CUDA1,exps=CPU"

# 4) 특정 텐서만 CPU (질문 예시)
-ot "output\.weight=CPU"
```

### 11A.3 MoE expert offload 우선순위 (PR #11397 실험)

> "Try to offload as many `ffn_down_exps` tensors as possible on GPU... 1) offload ffn_down_exps to GPU
> 2) then ffn_up_exps 3) finally ffn_gate_exps"

Mixtral 8x22 실측: **GPU offload 우선순위**
| offload | pp | tg |
|---|---|---|
| (exps 전부 CPU) | 65.36 | 1.84 |
| only gate_exps | 74.69 | 2.31 |
| only up_exps | 80.29 | 2.34 |
| **only down_exps** | **93.64** | **2.54** |

- **ffn_down_exps가 VRAM에 있으면 가장 큰 이득** — 라우팅이 gate/up보다 down에 집중

### 11A.4 surgical MoE offload 레시피 (Qwen3.6-35B-A3B 실측, late-layer만 CPU)

> "Push ONLY layers 40-47's expert FFNs to CPU (8 of 48 layers). Everything else — attention, KV,
> router, shared experts, early-layer experts — stays on GPU."

- **원리**: late 레이어 expert는 토큰당 활성도가 낮음 → DDR-read 패널티를 최소 영향 텐서에 집중
- 실측 (24GB GPU, qwen35moe 48레이어):

| Split (GPU/CPU 레이어) | Decode t/s | VRAM | 비고 |
|---|---|---|---|
| `--cpu-moe` (전부 CPU) | 5.3 | 3.7GB | GPU 유휴 (최악) |
| 32/16 | 30.0 | 18.6GB | 보수적 |
| **40/8** | **75.9** | 22.4GB | **최적** (2GB 여유) |
| 전부 GPU | OOM | - | 24GB에 안 맞음 |

- 레시피 명령:
```bash
OT_REGEX='blk\.(4[0-7])\.ffn_(gate|up|down)_exps\.weight=CPU'
llama-server -m model.gguf -c 98304 --parallel 1   --n-gpu-layers 999 -ot "$OT_REGEX"   --batch-size 2048 --ubatch-size 512   --flash-attn on --cache-type-k q4_0 --cache-type-v q4_0
```
- **튜닝**: OOM이면 다음 밴드로 (40/8 → 32/16 → 24/24). 최초 로드 후 **nvidia-smi/rocm-smi로 VRAM이 목표보다 몇 GB 여유**인지 확인 → 여유 크면 더 많이 GPU로
- **주의**: offload 레이어 decode는 RAM 대역폭 제약. 시스템 메모리가 느리면 offload 레이어당 패널티 증가

### 11A.5 CPU 오버라이드와 extra bufts (PR #14990)

- `-ot ...=CPU` 지정 시 기본적으로 CPU extra bufts(ACCEL/repack) 고려
- `--no-repack`(`-nr`) 옵션 추가됨 — weight repacking 비활성화
- CPU 오버라이드는 `select_weight_buft()`가 hparams/op 기반으로 최적 buft 선택

### 11A.6 -ot 관련 알려진 이슈

- **token_embd/q5_0 등과 CUDA_Host 비호환** (issue #17274):
  `tensor 'token_embd.weight' (q6_K) cannot be used with preferred buffer type CUDA_Host, using CPU instead`
  → 모든 텐서가 host buft를 지원하진 않음 (GET_ROWS 계열은 안 될 수 있음)
- **KV 캐시는 -ot로 못 옮김**: "Using -ot this does not change where the KV cache is allocated"
  (KV 위치는 -ts/-dev로)
- **RPC와 결합 시**: `--rpc`를 `-ot`보다 먼저 지정해야 RPC buft가 목록에 나타남
- **-fit과 충돌**: 오버라이드 튜닝 시 `-fit off`로 auto-fit 비활성화 권장 (DocShotgun 가이드)
- **tensor split모드 그래프 스케줄링**: -smgs는 오버라이드 사용 시 자동 비활성

### 11A.7 우리 구성 적용 (7900 XTX + Flash-Next)

```bash
# exps 전부 ROCm_Host (GPU 접근 가능한 pinned) + per_layer만 CPU
-ot 'ffn_(up|down|gate)_exps\.weight=ROCm_Host,per_layer_token_embd\.weight=CPU'

# 더 세밀하게: late 15레이어 exps만 host, 나머지 GPU (surgical 접근)
-ot 'blk\.(3[3-9]|4[0-7])\.ffn_(gate|up|down)_exps\.weight=ROCm_Host,per_layer_token_embd\.weight=CPU'
```

- **우리 하드웨어 7900 XTX 24GB x2 + 96GB RAM** — Q8/Q6에선 ffn_down_exps GPU 우선 전략 적용
- Flash-Next는 **exps가 GPU당 27.2GB라 전체 GPU 불가** → surgical split + expert cache(§5) 조합이 정답
- `-v`로 "buffer type overridden" 로그 확인 후 VRAM 균등 분배 검증 (사용자 지적: 1,1이면 양 카드 균등해야 함)

---

## 12. Docker compose + 구동 명령어

### 12.1 운영 이미지 빌드 (Dockerfile, beellama-boosts/Dockerfile 패턴)

```dockerfile
# beellama v0.4.5 + RCCL + PR#28223 + PR#27861
# build stage: rocm dev 컨테이너에서 RCCL 빌드
FROM docker.io/rocm/dev-ubuntu-24.04:7.2.3-complete AS build

ENV AMDGPU_TARGETS=gfx1100
RUN apt-get update \
    && apt-get install -y build-essential cmake git libssl-dev curl libgomp1
WORKDIR /src
COPY . .
RUN HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
        -DGGML_HIP_RCCL=ON \              # ← RCCL
        -DGGML_BACKEND_DL=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build build --config Release -j$(nproc)

# final stage: beellama runtime에 교체
FROM ghcr.io/anbeeld/beellama.cpp:server-rocm-preview-v0.4.5
COPY --from=build /src/build/bin/llama-server /app/llama-server
COPY --from=build /src/build/bin/llama-bench /app/llama-bench
COPY --from=build /src/build/lib/*.so* /app/
ENV ROCM_PATH=/opt/rocm LD_LIBRARY_PATH=/app:/opt/rocm/lib
```

빌드:
```bash
docker build -t baramofme/beellama-rocm:gfx1100-rocm10-boosts-rccl -f beellama-boosts/Dockerfile beellama-boosts/
```

### 12.2 docker-compose.yml (듀얼 GPU, 7900 XTX)

```yaml
version: "3.8"
services:
  llm-main:
    image: baramofme/beellama-rocm:gfx1100-rocm10-boosts-rccl  # RCCL+방식 빌드
    container_name: llm-main
    privileged: true
    restart: unless-stopped
    ipc: host
    shm_size: 16gb
    network_mode: dokploy-network   # 또는 bridge + 포트
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri/card0:/dev/dri/card0
      - /dev/dri/renderD130:/dev/dri/renderD130   # GPU0 (7900XTX)
      - /dev/dri/card2:/dev/dri/card2
      - /dev/dri/renderD129:/dev/dri/renderD129   # GPU1 (7900XTX)
    volumes:
      - /opt/llm/models:/models
      - /mnt/nvmedata/models:/models2
      - /opt/llm/llama-cpp/main-llm-config.ini:/app/config.ini
    environment:
      - ROCM_PATH=/opt/rocm
      - HIP_VISIBLE_DEVICES=0,1
      - ROC_ENABLE_PREEMPTION=1
      - OMP_NUM_THREADS=12
      - LD_LIBRARY_PATH=/app:/opt/rocm/lib
      - GGML_CUDA_P2P=1
    entrypoint: ["/app/llama-server"]
```

devices 매핑 주의: 위는 컨테이너로부터 실제 llm-main의 내부 device 이름(`card0`, `card2` 등). docker 구성 시 `rocm/rocm` 대신 직접 매핑하면 됨.

### 12.3 config.ini (Qwen3.8-27B Q4 — 운영)

config.ini `[Dense-bellama]` (beellama q8_0/q5_0 검증 구성). Q4 가중치는 Q3_K_M을 Q4_K_M으로 교체 시 §11.2에 따라 KV를 f16으로:

```ini
[Dense-q4]
model = /models2/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-MTP-Q4_K_M.gguf
mmproj = /models2/unsloth/Qwen3.8-27B-GGUF/mmproj-BF16.gguf
image-min-tokens = 1024

ctx-size = 131072
batch-size = 4096
ubatch-size = 1024
flash-attn = on

# RDNA3 비양자화 KV (Q4 가중치는 VRAM 여유 -> f16)
cache-type-k = f16
cache-type-v = f16

spec-type = draft-mtp-adaptive
spec-draft-n-max = 4
spec-draft-n-min-adaptive = 2

temp = 0.6
top-p = 0.95
min-p = 0.0
top-k = 20
repeat-penalty = 1.0
reasoning = on
reasoning-budget = 4096

# tensor split (RCCL 빌드 + 듀얼) — 27B dense는 단일 가능 but reddit 사례 따라 tensor 권장
split-mode = tensor
tensor-split = 1,1
n-gpu-layers = 99
main-gpu = 0

jinja = true
chat-template-kwargs = "{reasoning_effort: medium}"
load-on-startup = false
```

27B Q4 구동 CLI:
```bash
docker exec llm-main /app/llama-server \
  -m /models2/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-MTP-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 -fa on -c 131072 -ctk f16 -ctv f16 \
  -b 4096 -ub 1024 -s 1 -np 1 \
  --split-mode tensor --tensor-split 1,1 --fit off \
  --spec-type draft-mtp-adaptive --spec-draft-n-max 4 --spec-draft-n-min-adaptive 2 \
  --jinja --reasoning on --reasoning-budget 4096
```

### 12.4 config.ini (Qwen3.8-Flash-Next — dual + expert cache)

```ini
[FlashNext]
model = /models/Qwen3.8-Flash-Next-UD-IQ4_XS/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
# MoE, 텍스트 주력. 필요시 mmproj 추가

ctx-size = 8192        # ↑ VRAM 여유 생기면 확장 (131072 목표)
batch-size = 2048
ubatch-size = 512
flash-attn = on

# RDNA3 비양자화 KV (컨텍스트 짧을 때)
cache-type-k = f16
cache-type-v = f16

# tensor split (RCCL) + exps host + expert cache
split-mode = tensor
tensor-split = 1,1
n-gpu-layers = 99
main-gpu = 0

# ⚠️ 핵심: exps는 ROCm_Host(pinned), per_layer는 CPU, hot expert는 VRAM 캐시
override-tensor = ffn_(up|down|gate)_exps\.weight=ROCm_Host,per_layer_token_embd\.weight=CPU
moe-expert-cache = 135
moe-expert-cache-inserts = 2

# CPU MoE ACCEL
threads = 12
threads-batch = 12

jinja = true
load-on-startup = false

# read_raw 로딩 (mmap 캐시 경합 방지)
no-mmap = true
```

Flash-Next 구동 CLI(직접):
```bash
export GGML_CUDA_P2P=1 ROCM_PATH=/opt/rocm LD_LIBRARY_PATH=/app:/opt/rocm/lib

docker exec -e GGML_CUDA_P2P=1 llm-main /app/llama-server \
  -m /models/Qwen3.8-Flash-Next-UD-IQ4_XS/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 -sm tensor -ts 1,1 -fit off \
  -c 8192 -fa on -ctk f16 -ctv f16 \
  -b 2048 -ub 512 -t 12 --threads-batch 12 \
  --jinja --load-mode none \
  -ot 'ffn_(up|down|gate)_exps\.weight=ROCm_Host,per_layer_token_embd\.weight=CPU' \
  --moe-expert-cache 135 --moe-expert-cache-inserts 2
```

### 12.5 구동 확인

```bash
# GPU 전부 사용 확인
rocm-smi --showuse   # 생성 중 GPU0/GPU1 사용률 (expert cache hit 시 올라감)
rocm-smi --showmeminfo vram  # VRAM 분배

# 성능
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":128}'

# expert 라우팅 dump (진단)
GGML_MOE_LOG=/tmp/moe.log
```

---

## 13. 향후 갱신 로그

- [ ] Flash-Next + expert cache 실측 tg 확정 (현재: 로딩 중)
- [ ] --moe-expert-cache slot 수 스윕 (135 → VRAM 여유 따라)
- [ ] -sm layer vs tensor 비교 (RCCL 빌드 후)
- [ ] 131K 컨텍스트에서 q8 KV vs f16 KV A/B (RDNA3 비양자화 확인)
- [ ] Docker 이미지 재빌드 후 compose 반영
- [ ] -ot 세밀 튜닝: ffn_down_exps GPU 우선 + late-layer surgical split (§11A.3/§11A.4 실측 비교)
- [ ] -ot + --moe-expert-cache 조합: surgical split 후 남는 VRAM을 cache slot으로
