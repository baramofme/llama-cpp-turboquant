#!/usr/bin/env python3
"""Korean->English multilingual gate.

Design:
- The gate OWNS conversation state. All messages stored for the model are English only.
- Per-turn translation: only the NEW user message is translated (never re-translate history).
- The model always sees an English-only conversation, so Korean/CJK generation collapse
  is prevented at the source.
- Optional: assistant replies can be back-translated to Korean for display.

Usage (stateless, v2 compatible):
    from gate_server import chat
    r = chat("한국어 질문")

Usage (stateful, multi-turn):
    g = Gate()
    g.send("한국어 질문 1")
    g.send("한국어 질문 2")   # history kept, translated once each
"""
import json, os, re, time, urllib.request

LLM = os.environ.get("BONSAI_BASE", "http://localhost:18088/v1/chat/completions")
MAX_RETRY = int(os.environ.get("GATE_MAX_RETRY", "2"))

CJK_RE = re.compile(r'[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af\u3400-\u4dbf]')

SYSTEM_EN = ("You are a careful reasoning assistant. You always respond in English only. "
             "Never use Korean, Chinese, or Japanese characters, even inside math notation.")

def _call(msgs, mt, seed):
    body = {"model": "gate", "messages": msgs, "temperature": 0.0, "seed": seed,
            "max_tokens": mt, "stream": False}
    req = urllib.request.Request(LLM, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"].get("content") or ""

def translate_to_en(question, seed=42):
    """Translate a single user message to English. Returns (text, was_korean)."""
    if not CJK_RE.search(question):
        return question, False
    prompt = (
        "Translate the following Korean text into natural, fluent English. "
        "Reply with ONLY the English translation, nothing else. "
        "Do not include any Korean characters in your reply.\n\n"
        f"Korean: {question}\n\nEnglish translation:"
    )
    best = ""
    for attempt in range(MAX_RETRY + 1):
        out = _call([{"role": "system", "content":
                      "You are a professional Korean-to-English translator. "
                      "You always reply in English only."},
                     {"role": "user", "content": prompt}], mt=600, seed=seed + attempt)
        if not CJK_RE.search(out) and out.strip():
            return out.strip(), True
        best = out.strip() or best
    return best, True

def translate_to_ko(text, seed=7):
    """Back-translate assistant English reply to Korean (display only)."""
    prompt = (
        "Translate the following English text into natural Korean. "
        "Reply with ONLY the Korean translation. Keep math notation unchanged.\n\n"
        f"English: {text}\n\nKorean:"
    )
    out = _call([{"role": "system", "content":
                  "You are a professional English-to-Korean translator."},
                 {"role": "user", "content": prompt}], mt=2048, seed=seed)
    return out.strip() if out.strip() else text

class Gate:
    """Stateful multi-turn gate. Conversation history is kept in English only."""

    def __init__(self, system=SYSTEM_EN, backtranslate=False, max_history=60):
        self.system = system
        self.backtranslate = backtranslate
        self.history = []          # list of {"role", "content"} in English
        self.translated = []       # parallel: input was Korean? per user turn
        self.max_history = max_history

    def _enforce_lang(self, out, seed):
        """Ensure output is English only; retry otherwise."""
        if not CJK_RE.search(out) and out.strip():
            return out.strip()
        for attempt in range(MAX_RETRY + 1):
            q = ("Your previous reply contained Korean/Chinese/Japanese characters. "
                 "That is forbidden. Rewrite your ENTIRE reply in English only. "
                 f"Previous draft:\n{out}")
            out2 = _call(self.history + [{"role": "assistant", "content": q}],
                         mt=2048, seed=seed + attempt)
            if not CJK_RE.search(out2) and out2.strip():
                return out2.strip()
            out = out2
        return out.strip()

    def send(self, user_msg, seed=42):
        t0 = time.time()
        q_en, was_ko = translate_to_en(user_msg, seed)
        self.history.append({"role": "user", "content": q_en})
        self.translated.append(was_ko)
        msgs = [{"role": "system", "content": self.system}] + self.history[-self.max_history:]
        out = _call(msgs, mt=2048, seed=seed)
        out = self._enforce_lang(out, seed + 200)
        self.history.append({"role": "assistant", "content": out})
        # trim to max_history pairs
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        result = {
            "content": out,
            "question_en": q_en,
            "translated": was_ko,
            "cjk": len(CJK_RE.findall(out)),
            "secs": round(time.time() - t0, 1),
            "turn": len(self.translated),
        }
        if self.backtranslate:
            result["content_ko"] = translate_to_ko(out)
        return result

def chat(user_msg):
    """Stateless single-turn wrapper (v2 compatible)."""
    return Gate().send(user_msg)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--multiturn":
        g = Gate()
        for q in sys.argv[2:]:
            r = g.send(q)
            print(json.dumps(r, ensure_ascii=False, indent=1))
    else:
        q = sys.argv[1] if len(sys.argv) > 1 else "두 자리 숫자가 있는데, 각 자릿수의 합은 11이야. 자릿수를 뒤집은 숫자는 원래 숫자보다 27 작아. 이 숫자는 뭐야?"
        print(json.dumps(chat(q), ensure_ascii=False, indent=1))
