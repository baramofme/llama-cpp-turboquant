# Bonsai-27B ROCm10 Serving Setup (2026-09-13, updated 2026-09-16)

Ternary-Q2_g64 llama-server on ROCm 10 (gfx1100), gate-proxy v2 (loop
guard, pretranslation, numeral normalization, script strip),
Hy-MT2-1.8B translation server, and Dokploy deployment.

## Final architecture

```
Open WebUI / Hermes / OpenCode
  |--1709--> gate-proxy v2 (chat: full pipeline) --8080--> bonsai (Q1)
  |--1710--> gate-proxy v2 (agent: language only, raw stream) --8080--> bonsai
  |--8082--> bonsai direct (no gate)
  |--8083--> hymt direct (translation model)
```

- Chat port (1709): full pipeline below. For OpenWebUI.
- Agent port (1710): transparent proxy with input language processing
  only (numeral normalization, contextual guides, glossary, hymt
  pretranslation). No English nudges, no breaker, no dedupe trips, no
  output mutation except SSE degen guard (line-wise pipe with abort on
  4x block repetition in delta text; tool-call deltas excluded).
  Re-serialized SSE frames keep blank-line event delimiters (dropping
  them made event parsers see an empty stream).
  Backend streams straight through (per-line flush). For Hermes,
  OpenCode, scripts. Hermes `loQ36M` points here; `dense-local`
  (`:8081` Dense) stays for hard tasks via manual switch.

- Gate forwards requests as-is (tools included) so tool calling works.
- Korean prompts: normalized (numerals), pretranslated via hymt with
  glossary terminology, then answered by bonsai. English passes through.
- No translation layer in the old Dense sense; hymt is a tool, not a hop.
- Server-level `--grammar-file english-only.gbnf` remains in the bonsai
  image as a second layer.

## Images (local registry `localhost:5000`)

| Tag | Purpose |
|---|---|
| `rocm10-builder-ccache` | Build base: rocm10 full + cmake 3.28.3 + ccache 4.9.1 + git/g++/ninja |
| `rocm10-gfx1100-rccl-rdnaboosts-mtp-q1` | Q1_0 serving runtime (clean rebuild 2026-09-15, standby) |
| `rocm10-gfx1100-rccl-rdnaboosts-mtp-q2` | Ternary Q2_g64 serving runtime (same binary, tag only; A/B tested) |
| `baramofme/gate-proxy:v2` | Self-contained gateway (`python:3.12-slim` + code + glossary) |

## Files

| File | Role |
|---|---|
| `rocm10-runtime.Dockerfile` | Runtime image: rocm10 full + `bin/` + `grammars/`, ENTRYPOINT with `--grammar-file` |
| `build-rocm10-runtime.sh` | Copies `build-rocm10/bin` into context, builds q1/q2 tags, pushes |
| `grammars/english-only.gbnf` | ASCII-printable allowlist (upstream `english.gbnf` pattern) |
| `gate-proxy.Dockerfile` | Gateway image: copies code + glossary, runs on `GATE_PORT` |
| `gate_proxy_v2.py` | Deployed gateway. See "Gate proxy v2" below. |
| `glossary_ko_en.json` | Korean->English glossary (16 terms). Injected as vocabulary notes; also feeds hymt terminology. Extend as new failures appear (rebuild needed: COPY, not mount). |
| `gate_proxy.py` | Superseded experiment (translate→reason→backtranslate). Kept for reference, NOT deployed. |

## Gate proxy v2

Single stdlib-only Python file. Request path per turn:

1. Normalize Korean numerals in the last user message (mechanical,
   conservative): Sino compounds (천오백->1500, 일억이천만->120000000),
   native+counter (세권->3권, 한근->1근), fractions (삼분의이->2/3),
   symbols (×->*, ÷->/), fullwidth digits. Unit lookahead
   (사과 untouched), Hangul-boundary guard with particle license
   (일부분/일시불 untouched), ambiguous list (오만/이만/그만/저만/
   이조/일조/만조), spaced composition (일억 이천만). 35-case corpus
   in history; original kept on any doubt.
2. Forward body as-is (tools, tool_choice, sampling params) + system/
   trailing English nudges. `$ref` tool schemas dropped (llama-server
   400s on them). Malformed tool-call JSON repaired in place.
3. Contextual guides by prompt type: numbers-first + answer-last for
   math (digits/keywords), honesty+search for Korean, full 5-step
   translation guide on translate keywords, calculator-use only when
   the `calculator` tool is listed (client-agnostic: Hermes/OpenCode
   without it are unaffected).
4. Pretranslation: Korean prompts go to hymt (terminology prompt from
   matched glossary terms); translation appended (math turns: sterile
   English-only prompt, original dropped to stop transliteration
   spirals). Any MT failure falls back to the original. MT output must
   preserve input numbers (digit multiset incl. one..ninety, half..tenth,
   hundred..billion combos; months skipped): mismatch discards MT
   (e.g. 1/2+2/3 -> "one-third plus two-thirds" caught). False discards
   only lose the MT benefit, never correctness. English passes
   with zero added cost.
5. Tool-loop breaker, scoped to the current prompt (messages after the
   last user message, so new prompts start fresh): 15 tool turns
   (`GATE_MAX_TOOL_TURNS`) or 30 total calls (`GATE_MAX_TOOL_CALLS`,
   each parallel call counts), identical wave repeated, duplicated
   calls inside one wave. On trip: tools removed, stop-and-answer nudge,
   model forced to text.
6. Outgoing tool-call dedupe by (name, args): client never executes the
   same call twice in one wave.
7. English rewrite is OFF (`GATE_ENFORCE_ENGLISH=0`): first response
   passes through. Rationale: regen cost 2-4x walls and retry re-armed
   tool searches (loop amplifier). Retry path kept in code, text-only.
8. Script strip (`GATE_STRIP_NONASCII=1`): non-ASCII except Hangul
   removed from text content only (CJK, Devanagari, Arabic, Cyrillic,
   kana, emoji), never tool args. `<think>` tags stripped. Count in
   stderr (`stripped=N`). Agent SSE path filters delta text the same
   way per frame (streaming preserved); tool-call deltas excluded.
9. Degeneration guards: identical block 4x -> truncate at 2nd onset;
   zlib-ratio backstop (<0.08, 2000+ chars) for cyclic/paraphrase loops.
10. Streaming: full response validated, then emitted as SSE chunks
    (client sees nothing until backend finishes a turn).

Env knobs: `BONSAI_BASE`, `MT_BASE` (default `http://hymt:8080`),
`GATE_PORT`, `GATE_MAX_RETRY` (CJK regen, unused while enforce off),
`GATE_BACKEND_RETRY` (500/502/503 x1, temp 0), `GATE_RETRY_BUDGET`,
`GATE_MAX_TOOL_TURNS`, `GATE_MAX_TOOL_CALLS`, `GATE_ENFORCE_ENGLISH`,
`GATE_STRIP_NONASCII`, `GATE_PRETRANSLATE`, `MT_TIMEOUT`,
`GATE_GLOSSARY`, `GATE_GLOSSARY_MAX`.

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

# 4. Gateway image (after editing gate_proxy_v2.py; stage both files)
cp beellama-boosts/gate_proxy_v2.py beellama-boosts/glossary_ko_en.json /tmp/opencode/gatectx/
docker build -f beellama-boosts/gate-proxy.Dockerfile \
  -t localhost:5000/baramofme/gate-proxy:v2 /tmp/opencode/gatectx
docker push localhost:5000/baramofme/gate-proxy:v2
```

## Dokploy deployment

Service `bonsai-sghcma` (composeId `h5QEsfsllhIBuagdspK0t`):

- `bonsai`: Q2_g64 image, port `8082:8080`, `HIP_VISIBLE_DEVICES=1`,
  `/mnt/nvmedata/models:/models:ro`,
  model `/models/ternary-bonsai-27b/Ternary-Bonsai-27B-Q2_g64.gguf`,
  `--ctx-size 61384 --kv-cache-type q4_0 --kv-cache-type-v q4_0`,
  `--ubatch-size 1024 --mlock --alias bonsai`, mmproj Q8_0.
  Q2_g64 one-line switch tested (slower TG 59 vs 70, fixes Korean
  numerals, no better on lexicon. Full battery 2026-09-16: Q2 holds
  all 10 Q1 fixes, adds 60x2.5, cleaner H1; TG 56.7, 12.7 GB total.
  Switched to Q2+hymt.
- `gate-proxy`: image `gate-proxy:v2` (no volume mount), port `1709:1709`,
  `BONSAI_BASE=http://bonsai:8080`, `GATE_MAX_RETRY=2`,
  `GATE_ENFORCE_ENGLISH=0` (strip defaults on).
- `hymt`: q1 runtime image with entrypoint override (NO grammar file:
  must accept Korean), Hy-MT2-1.8B-Q8_0, `--ctx-size 4096`,
  `HIP_VISIBLE_DEVICES=1` (GPU 1, ~2.5 GB), port `8083:8080`,
  `--alias hymt`. Serves OpenWebUI direct URL + gateway pretranslation.

Gotchas found during deploy:
- CLI `compose create` defaults `sourceType=github` -> deploy fails
  with "Github Provider not found". Fixed with
  `UPDATE compose SET "sourceType"='raw' WHERE "composeId"=...`.
- CLI `compose import` returns 400; DB direct UPDATE of `composeFile` works.
- Same-tag image updates need a Dokploy redeploy (container recreate);
  Dokploy only recreates services whose digest changed.

## External pieces (outside this repo)

- SearXNG (`vane`): cache DBs root-owned while workers run as `searxng`
  -> `OperationalError: attempt to write a readonly database` per result,
  parallel searches wedged. Fixed: removed stale `/tmp/sxng_cache_*.db`,
  `chown searxng`, restarted. Parallel x4 now ~1s. Also set
  `outgoing.request_timeout: 5.0`, disabled API-less `wolframalpha`.
- OpenWebUI (`journal-openwebui-mvwnym`, composeId `yfu8dYbmKZWBzEb--NkNC`):
  outbound HTTP default timeout 300s held hung searches up to 5 min.
  Tried `AIOHTTP_CLIENT_TIMEOUT=60`, reverted: it also capped long chat
  turns (gateway buffers) causing Server Connection Errors with orphaned
  backend generations. Search bound lives in SearXNG 5s instead.
  Registered `http://hymt:8080/v1` (15th URL) + `Calculator` workspace
  tool (AST-sandboxed eval, pow/injection guards). Native loop cap
  `CHAT_RESPONSE_MAX_TOOL_CALL_ITERATIONS=30` left as-is.
- Hermes (`~/.hermes/config.yaml`, outside repo): main agent pointed at
  `http://localhost:1709/v1`, model `bonsai`. Multi-turn verified.
- Disabled flaky `Async Context Compression` filter (errored every inlet).

## Verification (2026-09-15/16)

| Test | Result |
|---|---|
| Clean rebuild q1/q2, redeploy | PP 126.5 / TG 70.4 (Q1 best), Q2 TG 58.9, VRAM 8.1/11.7 GB |
| Gateway single tool call | No `reasoning_content` (quiet default) |
| 15-turn fabricated history | Forced final answer, `finish: stop` |
| 35-call parallel fan-out | Breaker trips, text answer |
| Identical repeat / intra-wave dup | Trips on 2nd occurrence |
| New prompt, same query | Passes through, `tool_calls` intact |
| Mixed CJK/Devanagari content | Hangul kept, rest stripped, no regen |
| Outgoing dedupe [A,A,B] | Client receives [A,B], executes once each |
| Calculator tool offered | Model emits `calculator({"expression":"12345 * 6789"})` |
| Numeral normalization corpus | 35/35 (Sino/native/units/fractions/symbols + 20 no-touch cases) |
| Korean re-probe via gate | 2500원/철수/가는말/30평/한근/먹였다/banana/코끼리/백지장 fixed |
| End-to-end agent turn | Tool calls -> results (2.7s) -> final text, `done:true`, ~5s |
| Hy-MT2 terminology | 백지장/철수/2500원 correct (bonsai failed all three) |
| Reasoning trial (`on`, effort low) | Thinking degenerated (`1. 1. 1…`), empty content. Reverted to off. |
| Q2 full switch (ctx 61384) | Holds 10 fixes, fixes 60x2.5, H1 clean, TG 56.7, 12.7 GB. Active. |

## UI send-death (frontend sends nothing, 2026-09-15)

Symptom: Send does nothing. No POST /api/chat/completions, no chat DB
write, empty console, server fully idle (API ~7ms, gateway 1 thread,
backend slot free). Survives hard refresh, works in incognito/fresh tab.

Server exonerated 5x via logs; fault is between tab JS and network.
Ranked hypotheses:

1. localStorage/IndexedDB poison: hard refresh keeps site data,
   incognito starts clean. Stale model id / draft / message flags can
   early-return the send path inside a swallowed try/catch (no console,
   no network). Confirm: DevTools -> Application -> clear site data,
   reload, resend.
2. Extension block: adblock/privacy heuristics vs non-standard port
   (:9010). Shows in Network tab, not Console (user checked Console
   only). Confirm: watch Network tab on Send; retest with extensions off.
3. Phantom generating state: UI treats a dead pre-restart task as live,
   Send acts as Stop. Refresh usually clears; not permanent.
4. Fossil tab interference: days-old idle tab syncing state.
   Hygiene: close it.

Runbook: new/incognito tab first (fastest unblock), then site-data
clear, then Network-tab check, then extension bisect.

## Why each piece exists (phenomenon -> diagnosis -> fix)

Loop guards:
- Breaker caps: agent loop grew msgs +2/turn forever ("Gate check" spam);
  parallel fan-out jumped +20/turn. Cause: model emits tool_calls without
  converging; stateless passthrough never intervened. Fix: 15 turns /
  30 calls. First version locked chats forever (stale wave tripped every
  later prompt -> identical forced answers); fixed by scoping to messages
  after the last user message.
- Repeat/dup trip: same query re-sent every turn (CORTIS official site xN),
  same query twice in one wave (BTS birthdates x2). Cause: no memory of
  prior calls. Fix: trip on identical repeat / intra-wave dup.
- Outgoing dedupe: dup wave executed 2-3x searches before the breaker
  could see it (post-execution history only). Fix: strip dup calls from
  the backend->client response so they never execute.

Language pipeline:
- Rewrite OFF: CJK retry discarded kilotoken drafts (134s walls) and the
  retry re-armed tool searches (new queries after "rewrite in English").
  Cause: retry kept tools. Fix: enforce off; retry path kept but text-only.
- Hangul-allowlist strip: English answers leaked CJK/Devanagari. Full
  strip mangled Korean answers (`**** ()`). Cause: blanket ASCII policy.
  Fix: keep ASCII+Hangul, remove the rest; text only, never tool args.
- Think-tag strip: `</think>` + transliteration fragments (`yel`, `co`)
  leaked (ASCII, so grammar passed them). Fix: regex strip.
- Glossary (16 terms): 백지장->whiteboard, 철수->Ironman, 곱다->high.
  Cause: lexical gaps unfixable by prompting. Fix: vocabulary notes into
  bonsai prompts + terminology intervention into hymt prompts. First
  live result: 백지장 correct.
- Pretranslation: fluent misreads survived guides ("12345.to gotgo"
  -> minus; morpheme spirals "samfraeui"). Cause: weak Korean reading;
  dual orig+translation presentation made the model trust its broken
  reading over the translation. Fix: hymt bridge; translation is
  authoritative; math turns get sterile English-only prompts (no Korean
  text, no linguistics talk).
- Numeral normalization: 2500->25000, 천오백->1000, 1/2->2/3, 한근->100.
  Cause: Korean numeral/unit parse failure. Fix: mechanical parser
  (Sino/native/units/fractions/symbols) with unit lookahead,
  Hangul-boundary+particle guards, ambiguous list (오만/이만/사원...),
  35-case no-touch corpus. Two self-found bugs fixed during build
  (inverted condition, stale pycache).
- Contextual guides: numbers-first fixed Q7; answer-first fixed knights
  flip but caused H3 trap-answer headlines ($0.10 stated, then correctly
  derived) -> switched to derive-first/answer-last-line, both pass.
  Calculator instruction scoped to requests actually listing the tool
  (Hermes/OpenCode without it unaffected).

Degeneration guards:
- Block-cut + compression backstop: BTS apology-correction loop
  ("X (not Y)" xN to max_tokens), premise restate loops. Cause: attractor
  states in long generations. Fix: cut at 4th repeat; zlib ratio <0.08
  for cyclic/paraphrase loops. Verified 3402->242 chars.

Arithmetic:
- Calculator tool: long multiplication stopped mid-way (67,890,000),
  fraction slips. Cause: mental-math limits. Fix: workspace tool
  (AST sandbox, pow/injection guards) + model instruction. Emitted
  `calculator({"expression":"12345 * 6789"})` live; evaluates 83810205.
- Rejected: gateway-builtin arithmetic (detection is the hard part; a
  wrong auto-answer beats no answer never) and mental decomposition
  (drops terms structurally). Execution stays at the edges.

External:
- SearXNG readonly DB + 300s default: parallel searches wedged, 5-min
  spinners, usage-storm UI. Fixed: chown, 5s engine timeout,
  wolframalpha off. OpenWebUI global 60s tried and reverted (killed
  >60s buffered chat turns -> Server Connection Errors + orphaned GPU
  generations); bound lives at SearXNG 5s instead.
- Hy-MT2-1.8B-Q8 service: lexical gaps need a translation specialist,
  not more prompting. A/B won 백지장/철수/2500원. Serves explicit use
  + gateway pretranslation.
- Hermes switch: main agent briefly pointed at chat port; breaker
  fired 5x mid-task (repeated terminal commands read as loops) and
  buffering killed streaming. Fix: agent port (language in, raw out)
  instead of reverting to Dense. Dense kept as manual switch.
- Reasoning trial: `on` + effort low degenerated inside thinking
  (`1. 1. 1…`, empty content) — worse than off. Reverted same day.
- Q2 A/B: fixed Korean numerals, no better on lexicon, TG -16%.
  Q2+hymt active (12.7 GB); Q1 one-line switch retained in compose history.

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
