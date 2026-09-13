#!/usr/bin/env python3
"""Korean gate HTTP proxy for llama-server.

Sits in front of the LLM, translating Korean input to English,
enforcing English-only output from the model, and optionally
back-translating the response to Korean for display.

Usage:
    python gate_proxy.py                           # defaults: listen 0.0.0.0:8083, backend http://localhost:8082
    BONSAI_BASE=http://localhost:8082 python gate_proxy.py
    python gate_proxy.py --port 1709               # override listen port

Designed for OpenAI-compatible /v1/chat/completions clients.
No dependencies beyond Python stdlib.
"""
import json, os, re, sys, time, urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

BACKEND = os.environ.get("BONSAI_BASE", "http://localhost:8082")
BACKTRANSLATE_BASE = os.environ.get("BACKTRANSLATE_BASE", BACKEND)
BACKTRANSLATE_MODEL = os.environ.get("BACKTRANSLATE_MODEL", "Dense")
LISTEN_PORT = int(os.environ.get("GATE_PORT", "8083"))
MAX_RETRY = int(os.environ.get("GATE_MAX_RETRY", "2"))

CJK_RE = re.compile(r'[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af\u3400-\u4dbf]')

SYSTEM_EN = ("You are a careful reasoning assistant. "
             "You always respond in English only. "
             "Never use Korean, Chinese, or Japanese characters, even inside math notation.")


def llm_call(msgs, max_tokens=2048, seed=42, temperature=0.0):
    body = {
        "model": "gate",
        "messages": msgs,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BACKEND}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    content = d["choices"][0]["message"].get("content") or ""
    return content, d


def translate_to_en(question, seed=42):
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
        out, _ = llm_call(
            [{"role": "system", "content":
              "You are a professional Korean-to-English translator. "
              "You always reply in English only."},
             {"role": "user", "content": prompt}],
            max_tokens=600, seed=seed + attempt,
        )
        if not CJK_RE.search(out) and out.strip():
            return out.strip(), True
        best = out.strip() or best
    return best, True


def enforce_english(out, history, seed):
    if not CJK_RE.search(out) and out.strip():
        return out.strip()
    for attempt in range(MAX_RETRY + 1):
        q = ("Your previous reply contained Korean/Chinese/Japanese characters. "
             "That is forbidden. Rewrite your ENTIRE reply in English only. "
             f"Previous draft:\n{out}")
        msgs = history + [{"role": "assistant", "content": q}]
        out2, _ = llm_call(msgs, max_tokens=2048, seed=seed + attempt)
        if not CJK_RE.search(out2) and out2.strip():
            return out2.strip()
        out = out2
    return out.strip()


def backtranslate_to_ko(text, seed=7):
    prompt = (
        "Translate the following English text into natural Korean. "
        "Reply with ONLY the Korean translation. Keep math notation unchanged.\n\n"
        f"English: {text}\n\nKorean:"
    )
    body = {
        "model": BACKTRANSLATE_MODEL,
        "messages": [{"role": "system", "content":
                       "You are a professional English-to-Korean translator."},
                      {"role": "user", "content": prompt}],
        "temperature": 0.0,
        "seed": seed,
        "max_tokens": 2048,
        "stream": False,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BACKTRANSLATE_BASE}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    content = d["choices"][0]["message"].get("content") or ""
    return content.strip() if content.strip() else text


class GateProxyHandler(BaseHTTPRequestHandler):
    conversation_history = []

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/v1/chat/completions":
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length)
        req = json.loads(raw)

        messages = req.get("messages", [])
        max_tokens = req.get("max_tokens", 2048)
        seed = req.get("seed", 42)
        temperature = req.get("temperature", 0.0)
        stream = req.get("stream", False)

        user_msg = ""
        history_msgs = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                continue
            if role == "user":
                user_msg = content
            else:
                history_msgs.append({"role": role, "content": content})

        t0 = time.time()
        q_en, was_ko = translate_to_en(user_msg, seed)

        # assistant messages from the client are back-translated Korean;
        # convert any CJK-containing history back to English for the model
        model_history = []
        for m in history_msgs:
            content = m["content"]
            if CJK_RE.search(content):
                content, _ = translate_to_en(content, seed + 100)
            model_history.append({"role": m["role"], "content": content})

        full_history = [{"role": "system", "content": SYSTEM_EN}]
        full_history.extend(model_history[-60:])
        full_history.append({"role": "user", "content": q_en})

        out, raw_resp = llm_call(full_history, max_tokens=max_tokens, seed=seed, temperature=temperature)
        out = enforce_english(out, full_history, seed + 200)

        content_ko = backtranslate_to_ko(out, seed=7)

        elapsed = round(time.time() - t0, 1)
        cjk_count = len(CJK_RE.findall(out))

        resp = {
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content_ko,
                },
            }],
            "created": int(time.time()),
            "model": raw_resp.get("model", "gate"),
            "object": "chat.completion",
            "usage": raw_resp.get("usage", {}),
            "id": raw_resp.get("id", f"gate-{int(time.time())}"),
            "_gate": {
                "question_en": q_en,
                "answer_en": out,
                "was_korean": was_ko,
                "cjk_in_english": cjk_count,
                "elapsed_secs": elapsed,
            },
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp, ensure_ascii=False).encode())

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            body = json.dumps({"status": "ok", "gate": True, "backend": BACKEND}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def log_message(self, fmt, *args):
        ts = time.strftime("%H:%M:%S")
        sys.stderr.write(f"[{ts}] {args[0]}\n")


def main():
    port = LISTEN_PORT
    for i, a in enumerate(sys.argv[1:]):
        if a == "--port" and i + 1 < len(sys.argv) - 1:
            port = int(sys.argv[i + 2])

    server = HTTPServer(("0.0.0.0", port), GateProxyHandler)
    print(f"[gate_proxy] listening on 0.0.0.0:{port} -> backend {BACKEND}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[gate_proxy] stopped", flush=True)


if __name__ == "__main__":
    main()