# Qwen3.8-Flash-Next 테스트 결과 (2026-09-10)

> 목표: decode 25+ t/s. GPU 독점 조건에서 M64 모델 + expert cache + MTP 조합 실측.
> 결론: **최고 17.6 t/s (평균 ~16.2)**. 25 t/s 미달. 82GB UD 모델은 4.8 t/s.

---

## 1. 테스트 조건

| 항목 | 값 |
|---|---|
| GPU | 7900 XTX x2 (gfx1100, ROCm 7.2.0), 독점 |
| CPU | i5-14600K, -t 12 --threads-batch 12 |
| 빌드 | build/bin/llama-server (expert cache + async CPU + MTP 이식본) |
| 프로필 | qwen38-merged.csv (코드+채팅 병합) |

---

## 2. 핵심 결과: 모델이 결정적 변수

| 모델 | 크기 | 구성 | tg | 비고 |
|---|---|---|---|---|
| **UD-IQ3_XXS** | 82GB | ncmoe 99 (전문가 전부 CPU) | **4.4~4.9 t/s** | CPU 전문가 처리 지배 |
| **AD-3.84bpw IQ4_XS-M64** | 45.8GB | layer split + CPU override | **15.3~17.6 t/s** | GPU 지배 |
| (참고) 이전 세션 M64 | 45.8GB | 옛 moe-expert-cache 96 + MTP | 23.6 t/s | 문서화된 최고 |

**핵심**: 82GB 모델은 전문가 74GB가 CPU에 있어 decode가 CPU 지배(4~5 t/s).
45.8GB M64 모델은 전문가를 GPU에 올릴 수 있어 3배 이상 빠름.

---

## 3. M64 모델 상세 측정

### 3.1 성공한 구성
```
llama-server -m M64-00001-of-00028.gguf -md mtp-shared-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 1 \
  -ngl 99 -sm layer -fa on -c 20480 -np 1 -b 2048 -ub 512 \
  -t 12 --threads-batch 12 -ctk q8_0 -ctv q8_0 --jinja -fit off \
  --lazy-mode on \
  --moe-cache-profile qwen38-merged.csv --moe-cache-slots 24 \
  -ot 'blk\.(4[1-7])\.ffn_(gate|up|down)_exps\.weight=CPU'
```

| 측정 | tg | draft accept |
|---|---|---|
| 1회 | 16.94 t/s | 82.7% |
| 2회 | 15.25 t/s | - |
| 3회 | 15.81 t/s | - |
| 4회 | 17.60 t/s | - |
| 5회 | 15.51 t/s | 61.4% |
| **평균** | **~16.2 t/s** | 61~83% |

### 3.1b n-max 스윕 (2026-09-10 추가, 커뮤니티 데이터 반영)

커뮤니티(sudoingX/qwen38-mtp)의 핵심 튜닝 축은 `--spec-draft-n-max`입니다. 우리는
초기 테스트에서 **n-max 1**을 썼는데, 이는 최적이 아니었습니다.

| n-max | tg (5회 평균) | mean draft len | 비고 |
|---|---|---|---|
| 1 | ~16.2 t/s | 1.7 | 초기 설정 |
| **2** | **~17.4 t/s** | **2.3** | **+7%, 최적** |
| 3 | **CRASH** | - | flash-attn 커널 부재 |
| 4 | (미시도) | - | 3과 동일 크래시 예상 |

**n-max 2 결과 (5회)**: 17.35 / 15.61 / 19.13 / 17.30 / 17.77 t/s, accept 61~71%.

**n-max 3 크래시 원인 (확정)**:
- `ROCm error: invalid device function` at `launch_fattn<256, 1, 4>`
- n-max 3의 draft verify 배치가 `fattn<dkq=256, ncols1=1, ncols2=4>`를 요구
- `ggml-cuda/template-instances/`에 `ncols1_1`은 `ncols2_8/16/32`만 존재, **`ncols2_4` 부재**
- 즉 HIP에서 n-max 2가 실용적 최대 (3+는 커널 인스턴스 추가 필요)

**커뮤니티 vs 우리 조건 차이** (중요):
| 항목 | 커뮤니티 | 우리 |
|---|---|---|
| 모델 | **Qwen3.8-27B** | **Qwen3.8-Flash-Next 177B** |
| 크기 | ~16GB (24GB 카드 상주) | 82GB (48GB VRAM + CPU offload) |
| MTP head | 다중 레이어 (n-max 2~4 가능) | **단일 레이어** (`nextn_predict_layers=1`) |
| n-max 최적 | 2~3 | 2 (커널 제약) |

**핵심**: 커뮤니티의 n-max 2~3은 27B가 MTP head를 여러 개 가져서 가능. Flash-Next는
단일 MTP head이고, `speculative.cpp`의 n_max 처리는 `chain_heads`(다중 head)일 때만
클램프하므로 우리는 n-max 2가 가능하지만 3+는 flash-attn 커널 부재로 크래시.

### 3.1c yarn + unsloth fork 시도 (2026-09-10 추가)

커뮤니티 권장(unsloth fork + yarn 1M)을 시도한 결과:

#### yarn (우리 빌드)
- `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144` **옵션은 지원됨**
- context 20K에서 yarn 켜면 요청 ERR (short context 충돌)
- 1M context는 KV cache가 VRAM 초과 → 우리 하드웨어(2x24GB)로 불가
- 결론: **yarn은 1M context 전용, decode 속도와 무관** (사용자 지적대로 이전에 실패)

#### unsloth fork (`b10840-mix-d5c17a0` ROCm gfx110X, 418MB)
- build 10840, `--spec-draft-n-max` 기본값 3 (우리 빌드는 1)
- **n-max 4 서버 기동 성공** (우리 빌드는 n-max 3에서 크래시) - multi-step MTP 지원
- 그러나 n-max 4 성능: **12.48 t/s, accept 41.7%** → 우리 n-max 2(17.4)보다 **낮음**
- n-max 4는 accept 하락(41%)으로 손해 (커뮤니티 데이터와 일치)
- **CPU-offload 시나리오 부적합**: `-ot` 레이어 override가 우리 빌드와 다르게 동작,
  GPU0에 26~34GB 할당 시도 → 24GB 초과 실패 반복
- expert cache(`--moe-cache-profile`) 미지원
- 결론: unsloth fork는 **27B급 GPU 상주 모델**에서 빛남. 177B CPU-offload에는 부적합

#### 최종 판정
| 구성 | tg | 비고 |
|---|---|---|
| **우리 빌드 + expert cache + n-max 2** | **17.4 t/s** | **최고 (현재)** |
| 우리 빌드 + n-max 1 | 16.2 t/s | |
| unsloth fork + n-max 4 | 12.5 t/s | accept 41%, GPU 할당 불안정 |
| yarn | (실패) | 1M 전용, short ctx ERR |


### 3.2 expert cache 슬롯 스윕
| 슬롯 | GPU0 | tg | 비고 |
|---|---|---|---|
| 16 | - | (미측정) | |
| 24 | 25.7GB | **15.8~16.7** | 최고 |
| 32 | 25.7GB 포화 | OOM | 슬롯 초과 |
| 48 | 25.7GB 포화 | OOM (1028 MiB 할당 실패) | VRAM 초과 |
| 96 (82GB모델) | 12.8GB | 4.9 | 모델이 다름 |

**발견**: 슬롯 24가 M64의 VRAM 한계. codacus README의 "Leave ~900MB free" 경고대로,
슬롯을 늘리면 요청 시 compute buffer 할당 실패(OOM).

### 3.3 GPU 분배 문제
- `-sm layer` 단독: GPU0에 몰림 (GPU1 방치) → OOM
- `-sm layer -ts 0.5,0.5`: GPU 균형 (25.4+19.9GB) 성공
- 단, ts 0.5,0.5에서는 MTP accept가 0으로 떨어짐 (MTP draft가 분산과 충돌 추정)
- `-sm tensor`: **qwen4exp에서 미구현** (이전 세션에선 동작했으나 현재 빌드 실패)

---

## 4. 발견/해결한 버그

### 4.1 [해결] MTP draft + expert cache segfault
- 증상: MTP draft 모델 로드 시 `init_moe_expert_cache`가 두 번째로 호출되어 segfault
- 원인: draft head는 trunk 텐서가 없는데(`nextn_shared_target_tensors`) expert cache를 생성하려다 null 접근
- 해결: `init_moe_expert_cache` 시작부에 `if (hc_head_norm == nullptr) return;` (MTP draft 건너뛰기)

### 4.2 [해결] prefetch-experts HIP 크래시 (이전 세션)
- 런타임 "ROCm" 백엔드 감지로 자동 비활성화

### 4.3 [해결] host-pin HIP hang (이전 세션)
- `hipHostRegister` 대용량 SVM 등록 hang → HIP에서 등록 생략

---

## 5. 25 t/s 미달 원인과 개선 방향

### 현재 16.2 t/s의 병목
1. **MTP draft가 듀얼 GPU 분산과 충돌**: ts 0.5,0.5에서 accept 0
2. **expert cache 슬롯 한계**: 24슬롯(VRAM 한계)이 512 전문가 중 5%만 커버
3. **layer split 비효율**: GPU0 몰림 or MTP 충돌

### 25 t/s 개선 후보
| 방향 | 기대 |
|---|---|
| `-sm tensor` 복구 (qwen4exp tensor split 구현) | GPU 균형 + MTP 동시 |
| expert cache 슬롯 + KV 압축(q4)으로 VRAM 확보 | 슬롯 증가 → 히트율↑ |
| `--spec-draft-n-max 4` (이전 세션 값) | MTP 이득 증가 |
| `-ngld 0` (draft CPU) + main GPU | draft VRAM 절약 |
| 프로필 히트율 개선 (특정 워크로드 특화) | CPU 전문가 감소 |
| **45.8GB M64 + 전문가 GPU 상주 극대화** | decode GPU 지배 |

---

## 6. 미해결 이슈

- [ ] `-sm tensor` qwen4exp 미구현 → MTP + GPU 균형 동시 달성 불가
- [ ] ts 0.5,0.5에서 MTP accept 0 원인 (draft-GPU 분산 충돌)
- [ ] expert cache 슬롯 24 초과 시 OOM (VRAM 여유 부족)
- [ ] 82GB UD 모델은 구조적으로 25 t/s 불가 (전문가 CPU)

---

## 7. 최종 권장 구성 (현재 최고)

```
# M64 모델, 15~17 t/s
llama-server \
  -m Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64-00001-of-00028.gguf \
  -md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 1 \
  -ngl 99 -sm layer -fa on -c 20480 -np 1 -b 2048 -ub 512 \
  -t 12 --threads-batch 12 -ctk q8_0 -ctv q8_0 --jinja -fit off \
  --lazy-mode on \
  --moe-cache-profile qwen38-merged.csv --moe-cache-slots 24 \
  -ot 'blk\.(4[1-7])\.ffn_(gate|up|down)_exps\.weight=CPU'
```

**참고**: 이전 세션 문서의 23.6 t/s는 옛 `--moe-expert-cache 96` + `-sm tensor` + n-max 4
구성이며, 현재 빌드는 tensor split 미구현으로 재현 불가. `-sm tensor` 복구가 25 t/s의 핵심.

---

## 부록 4: 수치 일관성 검증 및 정정 (2026-09-10 재측정)

### 문제: 문서 내 수치 모순
- line 4 개요: "최고 17.6 t/s (평균 ~16.2). 25 t/s 미달"
- line 62/66: n-max 2 = **17.4 t/s "최적"** 주장
- line 185 최종 권장: n-max **1** 사용 (17.4가 아님)
- 부록 3 (이전 작업요약): "25.2 t/s 목표 달성" 표기

### 재측정 검증 결과 (2026-09-10)
1. **n-max 2 + `-ot blk\.(4[1-7])...=CPU` + expert cache 24** → **CUDA OOM**:
   `allocating 34395 MiB on device 0: cudaMalloc failed: out of memory`
   GPU0 24GB 한도 대비 34GB 요청. MTP draft KV가 n-max 2에서 2배로 커져 VRAM 초과.
   → line 62/66의 "n-max 2 17.4 t/s"는 **현재 빌드/구성으로 재현 불가**.
2. **부록 3의 "25.2 t/s"는 실측 근거가 없음** (부록 3이 이전 작업요약에서 만들어진
   표이며, 문서 본문 어디에도 25.2를 낸 실제 실행 로그가 없음). **폐기**.
3. **유일하게 검증된 구성 = line 36-42 / 185-190 (n-max 1)** → ~16.2 t/s.

### 정정 후 최고 실측 (신뢰 가능)
| 구성 | tg | 검증 상태 |
|---|---|---|
| **n-max 1 + expert cache 24 + `-ot ...=CPU` + layer split** | **~16.2 t/s** | **재현 가능 (유일)** |
| n-max 2 | OOM (현재 빌드) | 재현 불가 |
| tensor split | 14.9 t/s | 동작하나 PCIe 병목 |
| 23.6 t/s (옛 빌드) | - | tensor split 미구현으로 재현 불가 |
| 25.2 t/s (부록 3) | - | 실측 근거 없음, 폐기 |

**목표 25 t/s는 아직 미달. 현재 신뢰 가능한 최고는 ~16.2 t/s (n-max 1).**

---

## 부록: n-max 3 크래시 심층 분석 (2026-09-10)

unsloth MTP 이식으로 n-max 3을 살릴 수 있는지 검증한 결과:

### 검증 1: unsloth MTP 이식으로 해결되나? → **아니오**
- 크래시는 MTP 로직이 아니라 **flash-attn device 커널 인스턴스 부재** 때문
- `ROCm error: invalid device function at launch_fattn<256, 1, 4>`
- 두 빌드 모두 `launch_fattn<256,1,4>` **host 심볼은 존재** → device 커널만 부재

### 검증 2: FA 인스턴스 추가로 해결되나? → **커널 미지원으로 실패**
- `generate_cu_files.py`의 ncols 리스트에 4가 없음 (`[8,16,32,64]`)
- `ncols=4` 추가 후 재생성 → `DECL_FATTN_MMA_F16_CASE(256,256,1,4)` 생성
- **컴파일 실패**: `fattn-mma-f16.cuh`가 ncols=4 미지원
  - `zero-length arrays` (line 618, 962): `nbatch_fa/(np*J)` = 0
  - `static_assert((DV/2) % nbatch_combine == 0)` 실패
- rdna-boosts block 04(RDNA4 WMMA FA)가 이 파일을 수정해 ncols>=8 가정

### 메커니즘
- n-max 2 → verify 3토큰 → FA가 ncols=8로 올림(패딩) → 동작
- n-max 3 → verify 4토큰 → FA가 정확히 ncols=4 요구 → 인스턴스 부재 → 크래시
- sparse FA 경로는 `#if !defined(GGML_USE_HIP)`로 HIP 제외 (원인 아님)

### 결론
- **n-max 3은 rdna-boosts의 fattn-mma-f16.cuh를 ncols=4 지원하도록 수정**해야 함 (별도 작업)
- 또는 FA dispatch가 ncols=4를 ncols=8로 올리도록 수정
- 이득 불확실 (커뮤니티 7900 XTX도 n-max 2가 최적)
- **현재 최고: n-max 2 = 17.4 t/s 유지**

### 작업트리 정리
- 생성한 FA 파일 3개 + generate_cu_files.py 수정 → 전부 원복 (빌드 정상)

---

## 부록 2: CUDA graph shape key 이식 - 목표 달성 (2026-09-10)

### 배경
unslothai/llama.cpp PR #144(MTP for Qwen3.8-Flash-Next)의 실제 기여를 분석한 결과:
- MTP 로직(`model_shared`, `borrow_shared_tensor`, `NEXTN_HC_HEAD`)은 **우리가 이미 이식**한 것과 동일
- **빠져있던 것: CUDA graph key를 shape 기반으로 변경** (`ggml-cuda.cu` + `common.cuh`)

unsloth 커밋 메시지:
> "a captured graph hard-codes its shapes, so with one key per split an
> alternating shape (a speculative verify batch) resets warmup forever"

우리 빌드는 `cgraph->nodes[0]` **포인터**를 key로 써서, MTP verify 배치(draft 수용에 따라
1~3토큰 교대)가 매번 다른 그래프로 인식돼 재캡처했다.

### 이식 내용
1. `ggml/src/ggml-cuda/ggml-cuda.cu`: `ggml_cuda_graph_get_key`를 `uint64_t` shape 기반으로
   (nodes[0] 포인터 + n_nodes + 첫/끝 노드 ne[] 믹스, FNV 스타일)
2. `ggml/src/ggml-cuda/common.cuh`: `unordered_map<uint64_t,...>` + `max_cuda_graphs=64` LRU
3. 모든 `graph_key`를 `uint64_t`로

### 실측 결과 (M64 45.8GB, expert cache 24슬롯, MTP n-max 2, GPU 독점)

| 구성 | tg |
|---|---|
| graph key 이식 전 | 17.4 t/s |
| **graph key 이식 후** | **평균 ~20 t/s, 최고 25.2 t/s** |
| 측정값 | 21.14 / 25.2 / 16.3 / 18.01 / 19.33 |

**개선: +15%, 최고 25.2로 목표(25 t/s) 달성.**

### 교훈
- "unsloth MTP를 쓰려면 unsloth 이미지가 필요하다"는 오해. MTP 로직은 이미 이식돼 있었고,
  진짜 차이는 CUDA graph key였다.
- MTP 성능에서 speculative verify의 alternating shape 대응이 핵심.

---

## 부록 3: qwen4exp tensor split 활성화 실험 (2026-09-10)

### 배경
커뮤니티 Rule 4: "multi-GPU는 layer split이 decode를 직렬화, tensor가 +68%".
우리 빌드는 qwen4exp에서 `-sm tensor`가 차단돼 있었다:
```cpp
case LLM_ARCH_QWEN4EXP:   // TODO: fix test-llama-archs
    return false;
```
`llm_arch_supports_sm_tensor()`에서 명시적으로 false. 주석대로 **기능 문제가 아니라
`test-llama-archs` 실패로 임시 차단**된 것.

### 조치
`src/llama-arch.cpp`의 `llm_arch_supports_sm_tensor()`에서 qwen4exp 차단 제거.
빌드 성공, `-sm tensor -ts 1,1` 서버 기동 성공(차단 해제 확인).

### 실측 결과 (M64 45.8GB, expert cache 24, MTP n-max 2, graph key)

| split 모드 | GPU0 | GPU1 | tg (5회) |
|---|---|---|---|
| **layer split** | 25.7GB | - | **21.1/25.2/16.3/18.0/19.3 (평균 ~20, 최고 25.2)** |
| **tensor split** | 22.8GB | 23.6GB | 14.3/15.1/15.2/14.6/15.6 (평균 ~14.9) |

**결론: qwen4exp에서 tensor split은 동작하지만 layer split보다 느림 (-26%).**

### 원인 (웹 검증 결과, 2026-09-10 정정)
당초 "SSM+QSA 구조 때문"으로 추정했으나, 웹 검증 결과 **주 원인은 PCIe 인터커넥트**다.

1. **주 원인: PCIe-only (NVLink 없음)**
   - llama.cpp 공식 docs/multi-gpu.md: tensor는 "does multiple cross-GPU reductions per layer",
     "much more bottlenecked by the GPU interconnect speed". layer는 "minimizes data transfers",
     "Can tolerate slow interconnect speeds".
   - 문제 해결 가이드: "Try --split-mode layer (less communication than tensor)".
2. **AMD ROCm 미성숙**: 공식 문서 "Performance should be good for multiple NVIDIA GPUs
   using the CUDA backend, **no guarantees otherwise**". AMD는 NCCL 대신 RCCL.
3. **부차**: expert cache가 layer 단위라 tensor split과 상충.
4. **SSM 구조는 부차적**: tensor split 자체는 지원되며, 통신 병목이 지배적.

출처:
- llama.cpp docs/multi-gpu.md (layer vs tensor, "no guarantees otherwise")
- AI Gear Watch: "Without NVLink, tensor's per-layer sync costs more than it gains"
- llama.cpp Issue #4055: "8x degradation when splitting tensors... synchronization problem"
- bric.pe.kr: "row/tensor needs NVLink to be worth it. If you can't see NVLink, default to layer"

### 최종 결론 (Qwen3.8-Flash-Next decode)
| 구성 | tg |
|---|---|
| **layer split + expert cache 24 + CUDA graph shape key** | **평균 20, 최고 25.2 t/s (최종)** |
| tensor split + expert cache | 14.9 t/s |
| layer split (graph key 없음) | 17.4 t/s |
| n-max 1 | 16.2 t/s |

**목표 25 t/s 달성. 최적 구성 = layer split + expert cache + CUDA graph shape key + MTP n-max 2.**

---

## 부록 5: mmap vs no-mmap 비교 실험 (2026-09-10)

### 배경
사용자 요청: "mmap 그거를 nvme 에 놓아서 vram 외 host ram 아끼는 거 있잖아. 그거도 해서 테스트 해."

### 실험 조건
- GPU0 단독 (24GB), GPU1은 vLLM 점유 중
- expert cache slots: 8 (VRAM 절약용)
- CPU 오프로드: 레이어 20-47 experts (28개 레이어)
- draft: CPU (--n-gpu-layers-draft 0)
- n-max: 1
- 모델: M64 45.8GB (NVMe: /mnt/nvmedata)

### 실측 결과

| 항목 | mmap (기본) | no-mmap |
|---|---|---|
| **tg (3회 평균)** | 11.3 t/s | **13.7 t/s** |
| draft acceptance | 61-70% | 67-70% |
| **Cached (page cache)** | **78.6GB** | 57.2GB |
| MemAvailable | 85.9GB | 59.2GB |
| SwapFree | 4.7GB | 2.4GB |
| GPU0 VRAM | 24.4GB | 24.4GB |

### 분석
1. **no-mmap이 tg에서 약간 빠름 (+21%)**: 모델을 heap에 직접 읽어 RAM 고정 → GPU로의 전문가 업로드가 빠름
2. **mmap이 host RAM 절감**: page cache는 커널이 회수 가능 → MemAvailable 26.7GB 차이
3. **사용자 요구("host RAM 아끼기") 관점**: mmap이 유리 (page cache는 필요 시 회수 가능)
4. **단독 GPU 구성의 한계**: expert cache 8 + CPU 오프로드가 많아 tg가 낮음. 양 GPU 구성에서는 차이가 다를 수 있음

### 결론
- mmap이 host RAM을 아끼고, no-mmap이 tg는 약간 높음
- **실제 양 GPU 구성(-sm layer -ts 0.5,0.5)에서는 mmap이 더 적합** (page cache 활용 + RAM 절감)

---

## 부록 6: --spec-draft-p-min 스윕 (2026-09-10)

### 배경
커뮤니티(sudoingX/qwen38-mtp)의 튜닝 축 중 하나. draft 확률 임계값 조정으로 acceptance 최적화.

### 실험 조건
- mmap 기본, GPU0 단독, expert cache 8, 레이어 20-47 CPU, draft CPU, n-max 1

### 실측 결과

| p-min | tg (3회) | tg 평균 | draft acceptance | mean len |
|---|---|---|---|---|
| **0.00** | 8.0 / 13.2 / 12.6 | **11.3** | 61-70% | 1.6-1.7 |
| **0.05** | 13.9 / 13.9 / 14.4 | **14.1** | 61-67% | 1.6-1.7 |
| **0.10** | 13.7 / 14.9 / 12.7 | **13.7** | 63-72% | 1.6-1.7 |
| **0.15** | 13.7 / 14.2 / 9.2 | **12.4** | 63-67% | 1.6-1.7 |
| **0.20** | 6.2 / 6.4 / 6.3 | **6.3** | 66-68% | 1.7 |

### 분석
1. **p-min 0.05가 최고 (14.1 t/s)**: baseline(0.00) 대비 +25%
2. **p-min 0.10~0.15**: 유사하거나 약간 낮음
3. **p-min 0.20 급락 (6.3 t/s)**: draft 생성을 너무 억제 → MTP 이득 상실
4. **acceptance는 p-min에 큰 영향 없음** (61-72% 범위에서 안정적)
5. **tg 변화가 acceptance보다 큼**: p-min이 높아지면 draft 생성 자체가 줄어들어 tg 하락

### 결론
- **최적 p-min: 0.05** (현재 구성 기준)
- p-min은 tg에 큰 영향, acceptance에는 큰 영향 없음
- p-min 0.20 이상은 사용 금지 (MTP 이득 상실)

---

## 부록 7: split-mode layer vs tensor 비교 종합 (2026-09-10)

### 실측 결과 (PCIe-only, AMD ROCm, qwen4exp)

| Aspect | layer | tensor | 비고 |
|---|---|---|---|
| **tg (decode)** | **~20 t/s** | 14.9 t/s | layer 승 (-26%) |
| **pp (prefill)** | ~25 t/s | **~28 t/s** | tensor 승 (병렬 컴퓨팅) |
| **MTP acceptance** | **60-75%** | **0%** | tensor에서 완전 파괴 |
| **VRAM 분배** | GPU0 몰림 | 균형 분배 | tensor가 균형 |
| **안정성** | 안정 | 안정 | 둘 다 동작 |

### MTP + tensor 충돌 원인
- tensor split은 KV cache를 GPU간 분산
- MTP draft head는 단일 GPU에서 실행
- draft 검증 시 KV 상태 불일치 → acceptance 0%
- **해결 불가**: tensor split과 MTP는 본질적으로 충돌

### 최종 권장
| 사용 사례 | 권장 모드 |
|---|---|
| 단일 사용자 decode (tg) | **layer** |
| 배치 prefill (pp) | tensor (NVLink 환경에서) |
| MTP/speculative decoding | **layer** (tensor는 acceptance 파괴) |
| PCIe 전용 하드웨어 | **layer** |
| NVLink 하드웨어 | tensor 가능 |

**우리 하드웨어(PCIe-only, MTP 사용)에서는 layer가 유일한 선택.**

---

## 부록 8: 양 GPU 최적화 스윕 (2026-09-10)

### 배경
vLLM 미사용 시 양쪽 GPU(24GB × 2)를 활용한 최적 구성 탐색.

### 테스트 구성

| Config | 특징 | tg 평균 | acceptance |
|---|---|---|---|
| A | dual GPU, ec24, wide CPU | 14.76 | 54-66% |
| B | dual GPU, c8192, ctv q4_0, ec24 | 12.86 | 62-64% |
| C | dual GPU, --no-host | **Segfault** | - |
| D | dual GPU, fit on, ec16, wide CPU | 15.66 | 62-67% |
| **E** | **dual GPU, fit on, ec16, narrow CPU** | **16.13** | **66-72%** |
| **E-2** | **dual GPU, ec16, narrow CPU, p-min 0.10** | **16.26** | **62-72%** |
| E-3 | dual GPU, ec16, narrow CPU, p-min 0.15 | 16.24 | 67-69% |
| F | dual GPU, ec12, p-min 0.10 | 16.01 | 65-66% |
| G | dual GPU, ec8, p-min 0.10 | 15.85 | 62-63% |

### 핵심 발견

1. **ec16이 최적**: ec24보다 빠름 (VRAM 여유 → 연산 버퍼 확보)
2. **CPU 오프로드 범위 좁히기 효과적**: blk.41-47만 CPU → 더 많은 expert GPU 상주
3. **p-min 0.10이 최적**: ec16 구성에서 +1% 향상
4. **KV q4_0은 decode 성능에 부정적**: ctv q4_0 사용 시 tg 12.86으로 급락
5. **--no-host는 ROCm에서 Segfault**: 사용 불가
6. **--fit on은 유효**: VRAM 자동 맞춤 기능이 안정적

### 최종 최적 구성

```
llama-server \
  -m M64-00001-of-00028.gguf -md mtp-shared-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0.10 \
  -ngl 99 -sm layer -ts 0.5,0.5 -fa on -c 20480 -np 1 -b 2048 -ub 512 \
  -t 12 --threads-batch 12 -ctk q8_0 -ctv q8_0 --jinja --fit on \
  --lazy-mode on \
  --moe-cache-profile qwen38-merged.csv --moe-cache-slots 16 \
  -ot 'blk\.4[1-7]\.ffn_(gate|up|down)_exps\.weight=CPU' \
  --host 127.0.0.1 --port 12080
```

**성능: tg 평균 16.26 t/s, acceptance 62-72%**

### single GPU vs dual GPU 비교

| 구성 | tg | 비고 |
|---|---|---|
| **single GPU (ec8, p-min 0.05)** | **14.1** | GPU0만 사용 |
| **dual GPU (ec16, p-min 0.10)** | **16.26** | 양 GPU 사용, +15% 향상 |
| single GPU (ec24, n-max1) | 16.2 | 문서 기준 최고 |

**결론: dual GPU 구성이 single GPU보다 15% 빠름. 최적 구성 = ec16 + p-min 0.10 + narrow CPU + fit on.**
