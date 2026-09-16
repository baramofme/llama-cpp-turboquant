#!/usr/bin/env python3
"""Gate proxy v2: English-only chat content with full tool support.

Design:
- Forwards the client request to the backend as-is (tools included),
  so the server builds its normal tool_calls grammar. Tools work.
- Validates ONLY message content for CJK. tool_calls arguments
  pass through untouched (Korean tool args stay valid).
- On CJK violation, retries with an English-rewrite instruction.
- No translation layer, no Dense dependency. Output stays English.
- Streaming clients supported: response is buffered for validation,
  then emitted as SSE chunks.

Usage:
    python gate_proxy_v2.py
    BONSAI_BASE=http://localhost:8082 GATE_PORT=1709 python gate_proxy_v2.py

No dependencies beyond Python stdlib.
"""
import json, os, re, sys, time, urllib.error, urllib.request, zlib
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

BACKEND = os.environ.get("BONSAI_BASE", "http://bonsai:8080")
BACKTRANSLATE_BASE = os.environ.get("BACKTRANSLATE_BASE", BACKEND)
BACKTRANSLATE_MODEL = os.environ.get("BACKTRANSLATE_MODEL", "Dense")
LISTEN_PORT = int(os.environ.get("GATE_PORT", "8083"))
AGENT_PORT = int(os.environ.get("GATE_AGENT_PORT", "1710"))
AGENT_TIMEOUT = float(os.environ.get("GATE_AGENT_TIMEOUT", "1500"))
MAX_RETRY = int(os.environ.get("GATE_MAX_RETRY", "2"))
ENFORCE_ENGLISH = os.environ.get("GATE_ENFORCE_ENGLISH", "1") == "1"
STRIP_NONASCII = os.environ.get("GATE_STRIP_NONASCII", "1") == "1"
MAX_TOOL_TURNS = int(os.environ.get("GATE_MAX_TOOL_TURNS", "15"))
MAX_TOOL_CALLS = int(os.environ.get("GATE_MAX_TOOL_CALLS", "30"))
MAX_BACKEND_RETRY = int(os.environ.get("GATE_BACKEND_RETRY", "1"))
BACKEND_RETRY_CODES = (500, 502, 503)
RETRY_TIME_BUDGET = float(os.environ.get("GATE_RETRY_BUDGET", "60"))

CJK_RE = re.compile(r'[^\x20-\x7e\t\n]')
STRIP_RE = re.compile(r'[^\x20-\x7e\t\n\u3131-\u318e\uac00-\ud7a3\uffa0-\uffdc]')
THINK_RE = re.compile(r'</?think>', re.IGNORECASE)

DEGEN_RATIO = 0.08
DEGEN_MIN_LEN = 2000

REWRITE_INSTRUCTION = "Reply in English only."

SYSTEM_EN = ("You are a careful reasoning assistant. "
             "You always respond in English only. "
             "Never use Korean, Chinese, or Japanese characters, even inside math notation.")

TRAILING_EN = "Respond in English only."

BREAKER_NUDGE = ("Stop calling tools. Give your final answer in English now, "
                 "using the tool results obtained so far.")

NUMBERS_FIRST = ("First, write down all given numbers exactly as they appear, "
                 "then solve.")
ANSWER_FIRST = ("Show brief reasoning first, then put your final answer "
                "in the last line and stop.")
TRANSLATE_GUIDE = ("Translate sentence by sentence, in order. "
                   "Copy every digit exactly, never adding or dropping zeros. "
                   "If a Korean word has several meanings, choose the one that "
                   "fits the sentence; do not repeat transliterations.")
HONESTY_SEARCH = ("If a Korean word, proverb, or expression is unfamiliar, "
                  "say you do not know it instead of guessing, and use web "
                  "search to find its meaning.")
CALC_USE = ("For any arithmetic, call the calculator tool with the full "
            "expression instead of computing by hand.")
AGENT_DIRECT = ("Answer directly and concisely. Do not narrate intermediate "
                "reasoning or show deliberation preambles.")

MATH_HINTS = ("prove", "solve", "probability", "how many", "solve for",
              "find the", "얼마", "몇", "계산", "증명", "확률")
TRANSLATE_HINTS = ("translate", "번역", "영어로", "in english")

GLOSSARY_PATH = os.environ.get("GATE_GLOSSARY", "/app/glossary_ko_en.json")
GLOSSARY_MAX = int(os.environ.get("GATE_GLOSSARY_MAX", "8"))
MT_BASE = os.environ.get("MT_BASE", "http://hymt:8080")
MT_TIMEOUT = float(os.environ.get("MT_TIMEOUT", "60"))
PRETRANSLATE = os.environ.get("GATE_PRETRANSLATE", "1") == "1"
HANGUL_RE = re.compile(r'[\u3131-\u318e\uac00-\ud7a3\uffa0-\uffdc]')
NUM_RE = re.compile(r'\d+(?:,\d+)*(?:\.\d+)?')
WORD_RE = re.compile(r'[A-Za-z]+')
EN_CARD = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
           'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
           'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
           'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
           'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
           'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
           'eighty': 80, 'ninety': 90, 'first': 1, 'second': 2,
           'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6,
           'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
           'eleventh': 11, 'twelfth': 12}
EN_DENOM = {'half': [1, 2], 'halves': [1, 2], 'quarter': [1, 4],
            'quarters': [1, 4]}
for _d, _n in (('third', 3), ('fourth', 4), ('fifth', 5), ('sixth', 6),
               ('seventh', 7), ('eighth', 8), ('ninth', 9), ('tenth', 10)):
    EN_DENOM[_d] = [_n]
    EN_DENOM[_d + 's'] = [_n]
EN_MULT = {'hundred': 100, 'thousand': 1000, 'million': 1000000,
           'billion': 1000000000}
MULT_COMBO_RE = re.compile(
    r'(\d+(?:,\d+)*(?:\.\d+)?)\s+(hundred|thousand|million|billion)\b', re.I)


def _numset(s):
    vals = []
    t = MULT_COMBO_RE.sub(
        lambda m: str(float(m.group(1).replace(',', ''))
                      * EN_MULT[m.group(2).lower()]), s or '')
    for m in NUM_RE.finditer(t):
        try:
            vals.append(float(m.group(0).replace(',', '')))
        except ValueError:
            pass
    toks = [w.lower() for w in WORD_RE.findall(s or '')]
    i, pending = 0, None
    while i < len(toks):
        w = toks[i]
        if w in EN_CARD:
            pending = (pending or 0) + EN_CARD[w]
        elif w in EN_MULT:
            pending = (pending if pending is not None else 1) * EN_MULT[w]
        else:
            if pending is not None:
                vals.append(float(pending))
                pending = None
            vals.extend(EN_DENOM.get(w, []))
        i += 1
    if pending is not None:
        vals.append(float(pending))
    return sorted(vals)


def numbers_preserved(src, mt):
    """True if every number in src appears in mt (multiset subset).
    Catches silent MT number corruption; false discards merely fall
    back to the original path."""
    a = _numset(src)
    if not a:
        return True
    rest = _numset(mt)
    for x in a:
        if x in rest:
            rest.remove(x)
        else:
            return False
    return True

SINO_DIGIT = {'공': 0, '영': 0, '일': 1, '이': 2, '삼': 3, '사': 4,
              '오': 5, '육': 6, '칠': 7, '팔': 8, '구': 9}
NATIVE_NUM = {'하나': 1, '둘': 2, '셋': 3, '넷': 4, '다섯': 5,
              '여섯': 6, '일곱': 7, '여덟': 8, '아홉': 9, '열': 10,
              '한': 1, '두': 2, '세': 3, '네': 4, '열두': 12,
              '스무': 20, '서른': 30, '마흔': 40, '쉰': 50}
SMALL_UNIT = {'십': 10, '백': 100, '천': 1000}
LARGE_UNIT = {'만': 10000, '억': 100000000, '조': 1000000000000}
COUNT_UNIT = {'시', '시간', '분', '초', '원', '원짜리', '원어치', '개', '명', '권', '잔', '마리',
              '평', '근', '리터', '미터', '그램', '병', '장', '대',
              '벌', '그루', '자루', '번', '번째', '달', '해', '살', '주'}
TIME_PARTICLE = {'에', '엔', '에는', '부터', '까지', '을', '를', '이',
                 '가', '은', '는', '의', '와', '과', '로', '으로', '도',
                 '만', '밖에', '에게', '한테', '께', '보다', '처럼'}
AMBIGUOUS_NUMERAL = {'오만', '이만', '그만', '저만', '이조', '일조', '만조'}
FOLLOW_DENY = {'다행'}


def _ambiguous(phrase):
    if phrase in AMBIGUOUS_NUMERAL:
        return True
    return any(phrase.startswith(a) and phrase[len(a):]
               and all(ch in SINO_DIGIT for ch in phrase[len(a):])
               for a in AMBIGUOUS_NUMERAL)
HANGUL_SYL = re.compile(r'[\uac00-\ud7a3]')

SYMBOL_MAP = {'×': '*', '✕': '*', '✖': '*', '⨉': '*', '÷': '/',
              '－': '-', '–': '-', '—': '-', '―': '-',
              '＋': '+', '＝': '='}
FW_DIGIT = {chr(0xFF10 + i): str(i) for i in range(10)}


def _parse_sino(s):
    """Parse Sino-Korean numeral phrase. Returns int or None."""
    total, cur, num, used = 0, 0, None, False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in SINO_DIGIT:
            if num is not None:
                return None
            num, used = SINO_DIGIT[ch], True
            i += 1
        elif ch in SMALL_UNIT:
            cur += (num if num is not None else 1) * SMALL_UNIT[ch]
            num, used = None, True
            i += 1
        elif ch in LARGE_UNIT:
            base = cur + (num if num is not None else 0)
            total += (base if base else 1) * LARGE_UNIT[ch]
            cur, num, used = 0, None, True
            i += 1
        else:
            return None
    if num is not None:
        cur += num
    return total + cur if used else None


def _hangul_boundary_ok(s, pos):
    """True if position is not inside a longer Hangul word, unless a
    licensing particle follows."""
    if pos >= len(s) or not HANGUL_SYL.match(s[pos]):
        return True
    for p in sorted(TIME_PARTICLE, key=len, reverse=True):
        if s.startswith(p, pos):
            return True
    return False


def normalize_korean_numerals(text):
    """Replace Korean numeral phrases with digits. Conservative:
    unit lookahead, Hangul-boundary guard, original kept on doubt.
    Returns (new_text, replacement_count)."""
    if not isinstance(text, str) or not HANGUL_RE.search(text):
        return text, 0
    for fw, d in FW_DIGIT.items():
        text = text.replace(fw, d)
    for sym, rep in SYMBOL_MAP.items():
        text = text.replace(sym, rep)
    count = 0
    parts, pos = [], 0
    for m in re.finditer(r'(\S+?)분의\s*(\S+)', text):
        num1, num2 = m.group(1), m.group(2)
        a = _parse_sino(num1)
        if a is None and num1.isdigit():
            a = int(num1)
        b = _parse_sino(num2)
        if b is None and num2.isdigit():
            b = int(num2)
        if a is None or b is None or not _hangul_boundary_ok(text, m.end()):
            continue
        parts.append(text[pos:m.start()])
        parts.append(f"{b}/{a}")
        pos = m.end()
        count += 1
    if parts:
        parts.append(text[pos:])
        text = ''.join(parts)
    out, i = [], 0
    natives = sorted(NATIVE_NUM, key=len, reverse=True)
    while i < len(text):
        matched = None
        for w in natives:
            if text.startswith(w, i):
                rest = i + len(w)
                for u in sorted(COUNT_UNIT, key=len, reverse=True):
                    if text.startswith(u, rest) or (
                            text.startswith(' ', rest)
                            and text.startswith(u, rest + 1)):
                        end = rest + (len(u) + 1 if text.startswith(' ', rest) else len(u))
                        if _hangul_boundary_ok(text, end):
                            matched = (str(NATIVE_NUM[w])
                                       + ('' if not text.startswith(' ', rest) else ' ')
                                       + u, end)
                            break
                if matched:
                    break
        if matched:
            out.append(matched[0])
            i = matched[1]
            count += 1
            continue
        j = i
        while j < len(text):
            ch = text[j]
            if ch in SINO_DIGIT or ch in SMALL_UNIT or ch in LARGE_UNIT:
                j += 1
            elif (ch == ' ' and j + 1 < len(text)
                    and (text[j + 1] in SINO_DIGIT
                         or text[j + 1] in SMALL_UNIT
                         or text[j + 1] in LARGE_UNIT)):
                j += 1
            else:
                break
        if j > i:
            phrase = text[i:j]
            nospace = phrase.replace(' ', '')
            val = _parse_sino(nospace)
            single = len(nospace) == 1
            unit = None
            if val is not None:
                for u in sorted(COUNT_UNIT, key=len, reverse=True):
                    if text.startswith(u, j) and _hangul_boundary_ok(text, j + len(u)):
                        unit = u
                        break
                    if (text.startswith(' ', j)
                            and text.startswith(u, j + 1)
                            and _hangul_boundary_ok(text, j + 1 + len(u))):
                        unit = ' ' + u
                        break
            if val is not None and single and not (
                    unit is not None and len(unit.strip()) > 1
                    and not (nospace == '이'
                             and unit.strip() == '시간')):
                val = None
            if val is not None and unit is None and _ambiguous(nospace):
                val = None
            if val is not None and unit is None and nospace \
                    and nospace[-1] in LARGE_UNIT:
                k = j + (1 if text.startswith(' ', j) else 0)
                if any(text.startswith(w, k) for w in FOLLOW_DENY):
                    val = None
            if val is not None and (unit is not None or _hangul_boundary_ok(text, j)):
                tail = ''
                if (unit is not None and not unit.startswith(' ')
                        and text.startswith(' ', j)):
                    tail = ' '
                out.append(str(val) + tail + (unit or ''))
                i = j + len(tail) + (len(unit) if unit else 0)
                count += 1
                continue
            out.append(text[i:j])
            i = j
        else:
            out.append(text[i])
            i += 1
    return ''.join(out), count

def load_glossary():
    try:
        with open(GLOSSARY_PATH, encoding="utf-8") as f:
            d = json.load(f)
        return {k: v for k, v in d.items() if isinstance(k, str) and k}
    except Exception as e:
        sys.stderr.write(f"[gate-v2] glossary off ({e})\n")
        sys.stderr.flush()
        return {}

GLOSSARY = load_glossary()

def glossary_notes(text):
    """Return vocabulary lines for glossary terms found in text."""
    if not GLOSSARY or not isinstance(text, str):
        return []
    found = [t for t in sorted(GLOSSARY, key=len, reverse=True) if t in text]
    return [f"{t} = {GLOSSARY[t]}" for t in found[:GLOSSARY_MAX]]


def pretranslate(text):
    """Translate Korean text to English via Hy-MT2 with terminology.
    Returns translation string, or None on any failure (caller falls
    back to the original)."""
    try:
        terms = glossary_notes(text)
        prompt = ""
        if terms:
            prompt += ("Reference the following translations:\n"
                       + "\n".join(terms) + "\n")
        prompt += ("Translate the following segment into English, "
                   "without additional explanation.\n" + text)
        body = json.dumps({
            "model": "hymt",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": min(2048, max(512, len(text) * 3)),
        }).encode()
        req = urllib.request.Request(
            f"{MT_BASE}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=MT_TIMEOUT) as r:
            out = json.loads(r.read())["choices"][0]["message"].get("content", "")
        out = (out or "").strip()
        return out or None
    except Exception as e:
        sys.stderr.write(f"[gate-v2] pretranslate fallback ({e})\n")
        sys.stderr.flush()
        return None


def backend_call(body, timeout=600, _depth=0):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BACKEND}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in BACKEND_RETRY_CODES and _depth < MAX_BACKEND_RETRY:
            sys.stderr.write(f"[gate-v2] backend {e.code}, retry {_depth + 1}\n")
            sys.stderr.flush()
            rb = dict(body)
            try:
                rb["seed"] = int(rb.get("seed", 42)) + 1000 + _depth
            except (TypeError, ValueError):
                rb["seed"] = 1000 + _depth
            rb["temperature"] = 0
            return backend_call(rb, timeout, _depth + 1)
        raise


def sanitize_tools(tools):
    """Drop tools whose schemas contain $ref (llama-server cannot resolve
    them and fails the whole request with HTTP 400). Returns (kept|None,
    dropped_names). Never raises."""
    if not tools:
        return None, []
    def has_ref(o):
        try:
            if isinstance(o, dict):
                return "$ref" in o or any(has_ref(v) for v in o.values())
            if isinstance(o, list):
                return any(has_ref(v) for v in o)
        except Exception:
            return True
        return False
    kept, dropped = [], []
    for t in tools:
        try:
            name = t.get("function", {}).get("name", "?")
            params = t.get("function", {}).get("parameters", {})
            if has_ref(params):
                dropped.append(name)
            else:
                kept.append(t)
        except Exception:
            dropped.append("?")
    return (kept or None), dropped


def build_forward_body(req, agent=False):
    """Build backend body. Returns (body, breaker_tripped).

    Agent mode keeps language processing (normalize, guides, glossary,
    pretranslation) but skips English nudges and all loop interference.
    Output passes through untouched on the agent port.
    """
    body = {"stream": False}
    for key in ("model", "messages", "tools", "tool_choice", "temperature",
                "top_p", "top_k", "min_p", "max_tokens", "seed", "stop",
                "presence_penalty", "frequency_penalty", "repeat_penalty",
                "parallel_tool_calls", "response_format"):
        if key in req and req[key] is not None:
            body[key] = req[key]
    body.setdefault("model", "gate")
    body.setdefault("temperature", 0.7)
    if body.get("max_tokens") is None:
        if agent:
            body["max_tokens"] = int(os.environ.get("GATE_AGENT_MAX_TOKENS", "2048"))
        else:
            body["max_tokens"] = 8192
    if body.get("tools"):
        kept, dropped = sanitize_tools(body["tools"])
        if dropped:
            sys.stderr.write(f"[gate-v2] dropped $ref tools: {dropped}\n")
            sys.stderr.flush()
        if kept is None:
            body.pop("tools", None)
            body.pop("tool_choice", None)
        else:
            body["tools"] = kept
    if not agent and not any(m.get("role") == "system" for m in body.get("messages", [])):
        body["messages"] = [{"role": "system", "content": SYSTEM_EN}] + body["messages"]
    for m in body.get("messages", []):
        if isinstance(m, dict) and isinstance(m.get("reasoning_content"), str) \
                and m["reasoning_content"].startswith("Gate check:"):
            m.pop("reasoning_content", None)
    breaker = False
    breaker_info = ""
    if body.get("tools") and not agent:
        msgs = body.get("messages", [])
        ep = msgs
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], dict) and msgs[i].get("role") == "user":
                ep = msgs[i + 1:]
                break
        assts = [m for m in ep
                 if isinstance(m, dict) and m.get("role") == "assistant"
                 and m.get("tool_calls")]
        turns = len(assts)
        calls = sum(len(m.get("tool_calls") or []) for m in assts)
        if MAX_TOOL_TURNS > 0 and turns >= MAX_TOOL_TURNS:
            breaker, breaker_info = True, f"{turns} tool turns"
        elif MAX_TOOL_CALLS > 0 and calls >= MAX_TOOL_CALLS:
            breaker, breaker_info = True, f"{calls} tool calls"
        elif len(assts) >= 1:
            def sig(tc):
                try:
                    f = tc.get("function", {}) or {}
                    return (f.get("name"), f.get("arguments", ""))
                except AttributeError:
                    return None
            latest = [sig(tc) for tc in assts[-1].get("tool_calls") or []]
            if len(latest) != len(set(s for s in latest if s is not None)):
                breaker, breaker_info = True, "duplicated calls in one wave"
            elif len(assts) >= 2:
                prior = set()
                for m in assts[:-1]:
                    for tc in m.get("tool_calls") or []:
                        s = sig(tc)
                        if s is not None:
                            prior.add(s)
                if latest and all(s is not None and s in prior for s in latest):
                    breaker, breaker_info = True, "repeated identical tool calls"
        if breaker:
            sys.stderr.write(f"[gate-v2] tool loop breaker at {breaker_info}\n")
            sys.stderr.flush()
            body.pop("tools", None)
            body.pop("tool_choice", None)
            body["messages"] = list(body.get("messages", [])) + [
                {"role": "user", "content": BREAKER_NUDGE}]
    last = body["messages"][-1] if body.get("messages") else None
    if (last is not None and last.get("role") == "user"
            and isinstance(last.get("content"), str)):
        normed, n_norm = normalize_korean_numerals(last["content"])
        if n_norm:
            last["content"] = normed
            sys.stderr.write(f"[gate-v2] normalized {n_norm} numeral(s)\n")
            sys.stderr.flush()
        orig, low0 = last["content"], last["content"].lower()
        has_ko = bool(HANGUL_RE.search(orig))
        has_digit = any(ch.isdigit() for ch in orig)

        def add(line):
            if line not in last["content"]:
                last["content"] = last["content"] + "\n\n" + line
        if not agent and TRAILING_EN not in orig:
            add(TRAILING_EN)
        if has_ko and has_digit:
            add(NUMBERS_FIRST)
        if has_ko:
            add(HONESTY_SEARCH)
        if has_ko and any(h in low0 for h in TRANSLATE_HINTS):
            add(TRANSLATE_GUIDE)
        if has_digit or any(h in low0 for h in MATH_HINTS):
            add(ANSWER_FIRST)
        notes = glossary_notes(orig)
        if notes:
            add("Vocabulary notes (use these meanings):\n" + "\n".join(notes))
        is_math = has_digit or any(h in low0 for h in MATH_HINTS)
        mt = None
        if PRETRANSLATE and has_ko:
            mt = pretranslate(orig)
            if mt and not numbers_preserved(orig, mt):
                sys.stderr.write("[gate-v2] MT dropped (digit mismatch)\n")
                sys.stderr.flush()
                mt = None
            if mt:
                sys.stderr.write(f"[gate-v2] pretranslated {len(orig)} chars\n")
                sys.stderr.flush()
        if is_math and mt:
            parts = [NUMBERS_FIRST, ANSWER_FIRST]
            names = set()
            for t in body.get("tools") or []:
                try:
                    names.add(t.get("function", {}).get("name"))
                except AttributeError:
                    pass
            if "calculator" in names:
                parts.append(CALC_USE)
            tail = [] if agent else [TRAILING_EN]
            last["content"] = "\n\n".join(
                parts
                + (["Vocabulary notes (use these meanings):\n"
                    + "\n".join(notes)] if notes else [])
                + ["English translation of the request (authoritative: "
                   "answer from this):\n" + mt] + tail)
        else:
            if PRETRANSLATE and has_ko and mt:
                add("English translation of the request (authoritative: answer "
                    "from this; do not transliterate or re-parse the Korean "
                    "original, which is reference only):\n" + mt)
        if agent:
            est = sum(len(str(m.get("content", "") or ""))
                      for m in body.get("messages", [])
                      if isinstance(m, dict)) // 3
            if est > int(os.environ.get("GATE_CTX_WARN", "100000")):
                sys.stderr.write(f"[gate-agent] large history ~{est} tokens\n")
                sys.stderr.flush()
            add(AGENT_DIRECT)
    if agent:
        texts = [m.get("content", "") for m in body.get("messages", [])
                 if isinstance(m, dict) and m.get("role") == "assistant"
                 and isinstance(m.get("content"), str)
                 and len(m.get("content", "")) >= 50]
        if len(texts) >= 2:
            import difflib
            if difflib.SequenceMatcher(None, texts[-1],
                                       texts[-2]).ratio() >= 0.85:
                sys.stderr.write("[gate-agent] repeated assistant turn\n")
                sys.stderr.flush()
                body["messages"] = list(body.get("messages", [])) + [{
                    "role": "user",
                    "content": ("You already gave this response. Do not "
                                "repeat it. Either execute the next tool call "
                                "or ask a genuinely new question.")}]
    return body, breaker


def filter_msg_text(msg):
    """Strip non-ASCII (except Hangul) from message text fields in
    place. Tool-call arguments untouched. Returns stripped count."""
    n = 0
    if not isinstance(msg, dict):
        return 0
    for key in ("content", "reasoning_content"):
        val = msg.get(key)
        if isinstance(val, str) and val:
            cleaned, c = STRIP_RE.subn("", val)
            if c:
                msg[key] = cleaned
                n += c
    return n


def strip_nonascii(content):
    """Remove non-ASCII chars except Hangul (CJK, Devanagari, etc.).
    Returns (cleaned, removed_count). Tool args never pass through here."""
    if not isinstance(content, str) or not content:
        return content, 0
    cleaned, n = STRIP_RE.subn("", content)
    return cleaned, n


def truncate_repetition(content):
    """Cut degenerate block repetition. Returns (text, was_cut)."""
    if not isinstance(content, str) or len(content) < 500:
        return content, False
    blocks = re.split(r"\n+|(?<=[.!?])\s+", content)
    seen = {}
    for b in blocks:
        k = b.strip()
        if len(k) < 5:
            continue
        seen[k] = seen.get(k, 0) + 1
        if seen[k] >= 4:
            first = content.find(k)
            second = content.find(k, first + len(k))
            if second > 0:
                return content[:second].rstrip(), True
            return content, False
    return content, False


def check_compression(content):
    """True if long content compresses too well (paraphrase loop)."""
    if not isinstance(content, str) or len(content) < DEGEN_MIN_LEN:
        return False
    b = content.encode("utf-8", "ignore")
    if not b:
        return False
    return len(zlib.compress(b, 1)) / len(b) < DEGEN_RATIO


def dedupe_calls(tool_calls):
    """Drop duplicate tool calls (same name+args), keep first of each.
    Applied to outgoing responses so the client never executes dups."""
    if not tool_calls:
        return tool_calls
    seen, kept, dropped = set(), [], 0
    for tc in tool_calls:
        try:
            f = (tc.get("function") or {})
            sig = (f.get("name"), f.get("arguments", ""))
        except AttributeError:
            kept.append(tc)
            continue
        if sig in seen:
            dropped += 1
            continue
        seen.add(sig)
        kept.append(tc)
    if dropped:
        sys.stderr.write(f"[gate-v2] dropped {dropped} duplicate tool call(s)\n")
        sys.stderr.flush()
    return kept


def repair_json_args(raw):
    """Try to salvage malformed tool-call argument strings.

    Returns (fixed_string_or_None, repaired_bool). Handles: valid as-is,
    trailing garbage after balanced object, trailing commas, concatenated
    objects (keeps first). Never raises.
    """
    if not isinstance(raw, str):
        return None, False
    try:
        json.loads(raw)
        return raw, False
    except Exception:
        pass
    s = raw.strip()
    if s.startswith("{"):
        depth, instr, esc = 0, False, False
        for i, ch in enumerate(s):
            if esc:
                esc = False
            elif ch == "\\" and instr:
                esc = True
            elif ch == '"':
                instr = not instr
            elif not instr:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        cand = s[:i + 1]
                        try:
                            json.loads(cand)
                            return cand, True
                        except Exception:
                            break
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            json.loads(m.group(0))
            return m.group(0), True
        except Exception:
            pass
    fixed = re.sub(r",\s*([}\]])", r"\1", s)
    if fixed != s:
        try:
            json.loads(fixed)
            return fixed, True
        except Exception:
            pass
    return None, False


def repair_tool_calls(msg):
    """Validate + repair tool_calls arguments in place.
    Returns (repaired_count, unrepairable_count)."""
    repaired = unrepairable = 0
    for tc in msg.get("tool_calls") or []:
        try:
            args = tc["function"].get("arguments", "")
        except (KeyError, TypeError, AttributeError):
            continue
        fixed, was_broken = repair_json_args(args)
        if fixed is None:
            unrepairable += 1
        else:
            if was_broken:
                tc["function"]["arguments"] = fixed
                repaired += 1
    return repaired, unrepairable


def enforce_english_content(body, seed):
    msg, resp, retries, stripped = _enforce_inner(body, seed)
    content = msg.get("content") or ""
    cleaned, n = THINK_RE.subn("", content)
    if n:
        msg["content"] = content = cleaned
        sys.stderr.write(f"[gate-v2] stripped {n} think tag(s)\n")
        sys.stderr.flush()
    cut_text, was_cut = truncate_repetition(content)
    if was_cut:
        msg["content"] = cut_text
        sys.stderr.write("[gate-v2] cut degenerate repetition\n")
        sys.stderr.flush()
    elif check_compression(content):
        msg["content"] = content[:DEGEN_MIN_LEN].rstrip()
        sys.stderr.write("[gate-v2] cut degenerate paraphrase loop\n")
        sys.stderr.flush()
    return msg, resp, retries, stripped


def _enforce_inner(body, seed):
    """Call backend, retrying while text content contains CJK.

    Returns (message_dict, raw_response, retry_count, stripped_count).
    Tool-call arguments with broken JSON are repaired in place when
    possible. Non-ASCII text is stripped when STRIP_NONASCII is on.
    """
    resp = backend_call(body)
    msg = resp["choices"][0]["message"]
    fixed, broken = repair_tool_calls(msg)
    if fixed or broken:
        sys.stderr.write(f"[gate-v2] tool_json repaired={fixed} unrepairable={broken}\n")
        sys.stderr.flush()
    content = msg.get("content") or ""
    if not CJK_RE.search(content):
        return msg, resp, 0, 0
    if not ENFORCE_ENGLISH:
        return apply_strip(msg, resp, 0)
    history = list(body.get("messages", []))
    mt = body.get("max_tokens", 2048)
    t_start = time.time()
    for attempt in range(MAX_RETRY + 1):
        if attempt > 0 and (time.time() - t_start) > RETRY_TIME_BUDGET:
            sys.stderr.write(f"[gate-v2] retry skipped, turn already slow\n")
            sys.stderr.flush()
            break
        rb = dict(body)
        rb["messages"] = history + [{"role": "user", "content": REWRITE_INSTRUCTION}]
        # Rewrite must be text-only: with tools kept, the model searches
        # again instead of rewriting, amplifying the loop.
        rb.pop("tools", None)
        rb.pop("tool_choice", None)
        rb["seed"] = seed + 300 + attempt
        rb["temperature"] = 0
        rb["max_tokens"] = mt
        resp = backend_call(rb)
        msg = resp["choices"][0]["message"]
        r2, b2 = repair_tool_calls(msg)
        if r2 or b2:
            sys.stderr.write(f"[gate-v2] retry tool_json repaired={r2} unrepairable={b2}\n")
            sys.stderr.flush()
        content = msg.get("content") or ""
        if not CJK_RE.search(content):
            return msg, resp, attempt + 1, 0
    return apply_strip(msg, resp, MAX_RETRY + 1)


def apply_strip(msg, resp, retries):
    if STRIP_NONASCII:
        cleaned, n = strip_nonascii(msg.get("content") or "")
        if n:
            msg["content"] = cleaned
            sys.stderr.write(f"[gate-v2] stripped {n} non-ascii chars\n")
            sys.stderr.flush()
            return msg, resp, retries, n
    return msg, resp, retries, 0


def sse_emit(handler, msg, resp_id, model, status=None):
    def chunk(delta, finish=None):
        payload = {"id": resp_id, "object": "chat.completion.chunk",
                   "created": int(time.time()), "model": model,
                   "choices": [{"index": 0, "delta": delta,
                                "finish_reason": finish}]}
        handler.wfile.write(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode())

    content = msg.get("content") or ""
    first = True
    for i in range(0, max(len(content), 1), 60):
        delta = {"role": "assistant", "content": content[i:i + 60] or ""}
        if first and status:
            delta["reasoning_content"] = status
            first = False
        elif not content:
            break
        chunk(delta)
    for tc in msg.get("tool_calls") or []:
        chunk({"tool_calls": [{"index": 0,
                               "id": tc.get("id"),
                               "type": "function",
                               "function": {"name": tc["function"]["name"],
                                            "arguments": tc["function"].get("arguments", "")}}]})
    finish = "tool_calls" if msg.get("tool_calls") else "stop"
    chunk({}, finish=finish)
    handler.wfile.write(b"data: [DONE]\n\n")


class GateProxyV2Handler(BaseHTTPRequestHandler):
    agent_mode = False

    def _proxy_agent(self, process_request=True):
        """Agent path: optional input processing (language only),
        backend output streams through untouched."""
        url = f"{BACKEND}{self.path}"
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length > 0 else b""
        stream = False
        if process_request and self.command == "POST" and raw:
            try:
                req = json.loads(raw)
            except Exception:
                req = None
            if isinstance(req, dict):
                stream = bool(req.get("stream", False))
                body, _ = build_forward_body(req, agent=True)
                body["stream"] = stream
                raw = json.dumps(body).encode()
        fwd = {k: v for k, v in self.headers.items()
               if k.lower() not in ("host", "content-length")}
        t0, up_bytes, down_bytes = time.time(), len(raw or b""), 0
        try:
            req = urllib.request.Request(url, data=raw or None, headers=fwd,
                                         method=self.command)
            with urllib.request.urlopen(req, timeout=AGENT_TIMEOUT) as r:
                self.send_response(r.status)
                ctype = r.headers.get("Content-Type", "application/json")
                self.send_header("Content-Type", ctype)
                self.end_headers()
                if "text/event-stream" in ctype:
                    self._pipe_sse_guarded(r)
                    return
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    down_bytes += len(chunk)
                    self.wfile.write(chunk)
                sys.stderr.write(f"[gate-agent] {self.command} {urlparse(self.path).path} "
                                 f"up={up_bytes} down={down_bytes} dt={time.time()-t0:.1f}s\n")
                sys.stderr.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try:
                self._send_json({"error": {"message": f"backend error: {str(e)[:200]}",
                                           "type": "server_error"}}, code=502)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _pipe_sse_guarded(self, r):
        """Pipe SSE line by line (low latency) while watching delta text
        for degenerate repetition. Aborts both sides on trip. Tool-call
        deltas excluded (parallel calls legitimately repeat). Parse
        errors never break proxying."""
        texts, frame, down = [], [], [0]
        t0 = time.time()
        n_tool = [0]

        def flush_frame():
            for fline in frame:
                if not fline.startswith(b"data:"):
                    self.wfile.write(fline)
                    down[0] += len(fline)
                    continue
                payload = fline[5:].strip()
                if payload in (b"[DONE]", b""):
                    self.wfile.write(fline + b"\n")
                    down[0] += len(fline) + 1
                    continue
                try:
                    d = json.loads(payload)
                except Exception:
                    self.wfile.write(fline + b"\n")
                    down[0] += len(fline) + 1
                    continue
                for ch in d.get("choices", []) or []:
                    delta = (ch.get("delta") or {})
                    filter_msg_text(delta)
                    if delta.get("tool_calls"):
                        n_tool[0] += len(delta["tool_calls"])
                    t = delta.get("content") or delta.get("reasoning_content")
                    if t:
                        texts.append(t)
                out = (b"data: "
                       + json.dumps(d, ensure_ascii=False).encode()
                       + b"\n\n")
                self.wfile.write(out)
                down[0] += len(out)
            self.wfile.flush()
            frame.clear()

        try:
            while True:
                line = r.readline()
                if not line:
                    if frame:
                        flush_frame()
                    break
                if line.strip():
                    frame.append(line)
                    continue
                flush_frame()
                if sum(map(len, texts)) >= 500:
                    _, cut = truncate_repetition(''.join(texts))
                    if cut:
                        sys.stderr.write("[gate-v2] cut agent degen loop\n")
                        sys.stderr.flush()
                        return
        except (BrokenPipeError, ConnectionResetError):
            pass
        sys.stderr.write(f"[gate-agent] SSE down={down[0]} "
                         f"dt={time.time()-t0:.1f}s tcalls={n_tool[0]}\n")
        sys.stderr.flush()

    def _send_json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.agent_mode:
            self._proxy_agent(process_request=True)
            return
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json({"status": "ok", "gate": "v2-content-only",
                             "backend": BACKEND})
            return
        if parsed.path == "/v1/models":
            try:
                data = json.dumps({"object": "list", "data": []}).encode()
                req = urllib.request.Request(f"{BACKEND}/v1/models")
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = r.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    self._send_json({"error": str(e)[:200]}, code=502)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            return
        self.send_error(404)

    def do_POST(self):
        if self.agent_mode:
            self._proxy_agent(process_request=True)
            return
        parsed = urlparse(self.path)
        if parsed.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length))
        tools = req.get("tools") or []
        hist = req.get("messages", []) if isinstance(req.get("messages"), list) else []
        tc_turns = sum(1 for m in hist if isinstance(m, dict)
                       and m.get("role") == "assistant" and m.get("tool_calls"))
        tool_msgs = sum(1 for m in hist if isinstance(m, dict)
                        and m.get("role") == "tool")
        tail = [(m.get("role"), bool(m.get("tool_calls")),
                 len(str(m.get("content", "") or "")))
                for m in hist[-4:] if isinstance(m, dict)]
        sys.stderr.write(f"[gate-v2] tools={len(tools)} msgs={len(hist)} "
                         f"sys={any(m.get('role')=='system' for m in hist if isinstance(m, dict))} "
                         f"tcturns={tc_turns} toolmsgs={tool_msgs} tail={tail}\n")
        sys.stderr.flush()
        want_stream = bool(req.get("stream", False))
        body, breaker = build_forward_body(req)
        seed = int(req.get("seed", 42) or 42)
        t0 = time.time()
        try:
            msg, resp, retries, stripped = enforce_english_content(body, seed)
        except Exception as e:
            self._send_json({"error": {"message": f"backend error: {str(e)[:200]}",
                                       "type": "server_error"}}, code=502)
            return
        if msg.get("tool_calls"):
            msg["tool_calls"] = dedupe_calls(msg["tool_calls"])
        dt = time.time() - t0
        n_calls = len(msg.get("tool_calls") or [])
        sys.stderr.write(f"[gate-v2] done retries={retries} stripped={stripped} wall={dt:.1f}s "
                         f"stream={want_stream} tools={bool(body.get('tools'))} "
                         f"breaker={breaker}\n")
        sys.stderr.flush()
        status = None
        if retries or breaker:
            parts = []
            if retries:
                parts.append(f"rewrote reply {retries}x for English-only output")
            if breaker:
                parts.append("stopped tool loop, forced final answer")
            status = "Gate check: " + ", ".join(parts) + "."
        if want_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                sse_emit(self, msg, resp.get("id", f"gate-{int(time.time())}"),
                         resp.get("model", "gate"), status)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        out_msg = {"role": "assistant",
                     "content": msg.get("content") or "",
                     "tool_calls": msg.get("tool_calls")}
        if status:
            out_msg["reasoning_content"] = status
        out = {
            "choices": [{
                "finish_reason": ("tool_calls" if msg.get("tool_calls")
                                  else resp["choices"][0].get("finish_reason", "stop")),
                "index": 0,
                "message": out_msg,
            }],
            "created": resp.get("created", int(time.time())),
            "model": resp.get("model", "gate"),
            "object": "chat.completion",
            "usage": resp.get("usage", {}),
            "id": resp.get("id", f"gate-{int(time.time())}"),
        }
        self._send_json(out)

    def log_message(self, fmt, *args):
        ts = time.strftime("%H:%M:%S")
        sys.stderr.write(f"[gate-v2 {ts}] {args[0]}\n")


class GateAgentHandler(GateProxyV2Handler):
    agent_mode = True


def main():
    import threading
    chat = ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), GateProxyV2Handler)
    agent = ThreadingHTTPServer(("0.0.0.0", AGENT_PORT), GateAgentHandler)
    print(f"[gate-v2] chat on 0.0.0.0:{LISTEN_PORT}, agent on 0.0.0.0:{AGENT_PORT}"
          f" -> backend {BACKEND}", flush=True)
    threading.Thread(target=agent.serve_forever, daemon=True).start()
    try:
        chat.serve_forever()
    except KeyboardInterrupt:
        print("\n[gate-v2] stopped", flush=True)


if __name__ == "__main__":
    main()
