# Bonsai-27B ROCm10 Serving Setup (2026-09-13, updated 2026-09-15)

Q1_0 / Ternary-Q2_g64 llama-server runtime images on ROCm 10 (gfx1100),
gate-proxy v2 (tool-loop guard + script strip), and Dokploy deployment.

## Final architecture

```
Open WebUI --1709--> gate-proxy v2 --8080--> bonsai (llama-server, Q1)
                \--8082--> bonsai direct (no gate)
```

- Gate forwards requests as-is (tools included) so tool calling works.
- English nudges (system + trailing line). No translation layer, no Dense.
- Server-level `--grammar-file english-only.gbnf` remains in the image
  as a second layer; gateway strips residual non-ASCII from text.

## Images (local registry `localhost:5000`)

| Tag | Purpose |
|---|---|
| `rocm10-builder-ccache` | Build base: rocm10 full + cmake 3.28.3 + ccache 4.9.1 + git/g++/ninja |
| `rocm10-gfx1100-rccl-rdnaboosts-mtp-q1` | Q1_0 serving runtime, active (clean rebuild 2026-09-15) |
| `rocm10-gfx1100-rccl-rdnaboosts-mtp-q2` | Ternary Q2_g64 serving runtime (same binary, tag only) |
| `baramofme/gate-proxy:v2` | Self-contained gateway (`python:3.12-slim` + `gate_proxy_v2.py`) |

## Files

| File | Role |
|---|---|
| `rocm10-runtime.Dockerfile` | Runtime image: rocm10 full + `bin/` + `grammars/`, ENTRYPOINT with `--grammar-file` |
| `build-rocm10-runtime.sh` | Copies `build-rocm10/bin` into context, builds q1/q2 tags, pushes |
| `grammars/english-only.gbnf` | ASCII-printable allowlist (upstream `english.gbnf` pattern) |
| `gate-proxy.Dockerfile` | Gateway image: copies `gate_proxy_v2.py`, runs it on `GATE_PORT` |
| `gate_proxy_v2.py` | Deployed gateway. See "Gate proxy v2" below. |
| `gate_proxy.py` | Superseded experiment (translate→reason→backtranslate). Kept for reference, NOT deployed. |

## Gate proxy v2

Single stdlib-only Python file. Request path per turn:

1. Forward body as-is (tools, tool_choice, sampling params) + system/trailing
   English nudges. `$ref` tool schemas dropped (llama-server 400s on them).
2. Repair malformed tool-call JSON in place when possible.
3. Tool-loop breaker (checked per prompt = messages after last user msg):
   - 15 tool turns (`GATE_MAX_TOOL_TURNS`) or 30 total calls
     (`GATE_MAX_TOOL_CALLS`, each parallel call counts),
   - identical wave repeated, or duplicated calls inside one wave.
   - On trip: tools removed, "stop calling tools, answer from results"
     nudge appended, model forced to text. Never locks: a new prompt
     always starts fresh.
4. English rewrite is OFF (`GATE_ENFORCE_ENGLISH=0`): first response passes
   through. Rationale: regen cost 2-4x walls and retry re-armed tool
   searches (loop amplifier). Retry path kept in code, text-only.
5. Non-ASCII strip (`GATE_STRIP_NONASCII=1`): CJK/Hangul/Devanagari removed
   from text content only, never tool args. Count in stderr (`stripped=N`).
6. Streaming: full response validated, then emitted as SSE chunks
   (client sees nothing until backend finishes a turn).

Env knobs: `BONSAI_BASE`, `GATE_PORT`, `GATE_MAX_RETRY` (CJK regen, unused
while enforce off), `GATE_BACKEND_RETRY` (500/502/503 x1, temp 0),
`GATE_RETRY_BUDGET`, `GATE_MAX_TOOL_TURNS`, `GATE_MAX_TOOL_CALLS`,
`GATE_ENFORCE_ENGLISH`, `GATE_STRIP_NONASCII`.

## Build procedure

```bash
# 1. HIP build inside rocm10-dev (ccache incremental)
docker exec rocm10-dev bash /app/beellama-boosts/build-rocm10-0001.sh
# output: build-rocm10/bin/

# 2. Commit build base once (skip external pull afterwards)
docker commit rocm10-dev localhost:5000/baramofme/llama-cpp-rocm:rocm10-builder-ccache
docker push localhost:5000/baramofme/llama-cpp-rocm:rocm10-builder-ccache

# 3. Runtime images q1/q2
bash beellama-boosts/build-rocm10-runtime.sh

# 4. Gateway image (after editing gate_proxy_v2.py)
cp beellama-boosts/gate_proxy_v2.py /tmp/opencode/gatectx/
docker build -f beellama-boosts/gate-proxy.Dockerfile \
  -t localhost:5000/baramofme/gate-proxy:v2 /tmp/opencode/gatectx
docker push localhost:5000/baramofme/gate-proxy:v2
```

## Dokploy deployment

Service `bonsai-sghcma` (composeId `h5QEsfsllhIBuagdspK0t`):

- `bonsai`: Q1 image, port `8082:8080`, `HIP_VISIBLE_DEVICES=1`,
  `/mnt/nvmedata/models:/models:ro`,
  model `/models/bonsai-27b/Bonsai-27B-Q1_0.gguf`,
  `--ctx-size 122768 --kv-cache-type q4_0 --kv-cache-type-v q4_0`,
  `--ubatch-size 1024 --mlock --alias bonsai`, mmproj Q8_0 restored.
  (MTP-Q2_K line kept commented for one-line switch.)
- `gate-proxy`: image `gate-proxy:v2` (no volume mount), port `1709:1709`,
  `BONSAI_BASE=http://bonsai:8080`, `GATE_MAX_RETRY=2`,
  `GATE_ENFORCE_ENGLISH=0` (strip defaults on).

Gotchas found during deploy:
- CLI `compose create` defaults `sourceType=github` -> deploy fails
  with "Github Provider not found". Fixed with
  `UPDATE compose SET "sourceType"='raw' WHERE "composeId"=...`.
- CLI `compose import` returns 400; DB direct UPDATE of `composeFile` works.
- Same-tag image updates need a Dokploy redeploy (container recreate);
  Dokploy only recreates services whose digest changed.

## External fixes (outside this repo)

- SearXNG (`vane`): cache DBs root-owned while workers run as `searxng`
  -> `OperationalError: attempt to write a readonly database` per result,
  parallel searches wedged. Fixed: removed stale `/tmp/sxng_cache_*.db`,
  `chown searxng`, restarted. Parallel x4 now ~2.3s.
- OpenWebUI (`journal-openwebui-mvwnym`, composeId `yfu8dYbmKZWBzEb--NkNC`):
  outbound HTTP default timeout 300s held hung searches up to 5 min.
  Set `AIOHTTP_CLIENT_TIMEOUT=60`. Native loop cap
  `CHAT_RESPONSE_MAX_TOOL_CALL_ITERATIONS=30` left as-is.
- Disabled flaky `Async Context Compression` filter (errored every inlet).

## Verification (2026-09-15)

| Test | Result |
|---|---|
| Clean rebuild q1/q2, redeploy | PP 126.5 / TG 70.4 (best), VRAM 8.1 GB |
| Gateway single tool call | No `reasoning_content` (quiet default) |
| 15-turn fabricated history | Forced final answer, `finish: stop` |
| 35-call parallel fan-out | Breaker trips, text answer |
| Identical repeat / intra-wave dup | Trips on 2nd occurrence |
| New prompt, same query | Passes through, `tool_calls` intact |
| Mixed CJK/Devanagari content | Stripped to ASCII, no regen |
| End-to-end agent turn | Tool calls -> results (2.7s) -> final text, `done:true`, ~5s |

## Dead ends (documented so nobody retries them)

1. **Root-context build**: repo root `.dockerignore` has `build*/`,
   so `COPY build-rocm10/bin/` fails with `/build-rocm10/bin: not found`.
   Fix: use `beellama-boosts/` as build context, copy `bin/` there first.
2. **Tag bugs in build script**: `TAG=q...` + `"q1"` produced `-qq1`;
   missing hyphen produced `-mtpq1`. Current script uses `"$TAG-q1"`.
3. **Gate proxy + Dense backtranslate** (`gate_proxy.py`, port 1709):
   Korean→English→Dense→Korean pipeline worked, but Qwen already
   understands Korean and gateway + grammar enforce English output.
   Extra translation hops add latency and a Dense dependency for no gain.
4. **Request-level `grammar` param**: works for testing without restart,
   but server-level `--grammar-file` in the image is the permanent fix.
5. **English regen on CJK**: discarded multi-kilotoken drafts (up to 4x
   walls) and re-armed tool searches. Replaced by strip (2026-09-15).
6. **Global breaker scope**: once tripped, every later prompt in the chat
   tripped too (same stale wave) -> identical forced answers. Fixed by
   scoping all checks to messages after the last user message.
