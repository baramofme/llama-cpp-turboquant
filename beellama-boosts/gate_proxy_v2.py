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
import json, os, re, sys, time, urllib.error, urllib.request
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

BACKEND = os.environ.get("BONSAI_BASE", "http://bonsai:8080")
BACKTRANSLATE_BASE = os.environ.get("BACKTRANSLATE_BASE", BACKEND)
BACKTRANSLATE_MODEL = os.environ.get("BACKTRANSLATE_MODEL", "Dense")
LISTEN_PORT = int(os.environ.get("GATE_PORT", "8083"))
MAX_RETRY = int(os.environ.get("GATE_MAX_RETRY", "2"))
MAX_BACKEND_RETRY = int(os.environ.get("GATE_BACKEND_RETRY", "1"))
BACKEND_RETRY_CODES = (500, 502, 503)

CJK_RE = re.compile(r'[^\x20-\x7e\t\n]')

REWRITE_INSTRUCTION = "Reply in English only."

SYSTEM_EN = ("You are a careful reasoning assistant. "
             "You always respond in English only. "
             "Never use Korean, Chinese, or Japanese characters, even inside math notation.")

TRAILING_EN = "Respond in English only."


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
            return backend_call(rb, timeout, _depth + 1)
        raise


def build_forward_body(req):
    body = {"stream": False}
    for key in ("model", "messages", "tools", "tool_choice", "temperature",
                "top_p", "top_k", "min_p", "max_tokens", "seed", "stop",
                "presence_penalty", "frequency_penalty", "repeat_penalty",
                "parallel_tool_calls", "response_format"):
        if key in req and req[key] is not None:
            body[key] = req[key]
    body.setdefault("model", "gate")
    body.setdefault("temperature", 0.7)
    if not any(m.get("role") == "system" for m in body.get("messages", [])):
        body["messages"] = [{"role": "system", "content": SYSTEM_EN}] + body["messages"]
    last = body["messages"][-1] if body.get("messages") else None
    if (last is not None and last.get("role") == "user"
            and isinstance(last.get("content"), str)
            and TRAILING_EN not in last["content"]):
        last["content"] = last["content"] + "\n\n" + TRAILING_EN
    return body


def enforce_english_content(body, seed):
    """Call backend, retrying while text content contains CJK.

    Returns (message_dict, raw_response). tool_calls are never modified.
    """
    resp = backend_call(body)
    msg = resp["choices"][0]["message"]
    content = msg.get("content") or ""
    if not CJK_RE.search(content):
        return msg, resp, 0
    history = list(body.get("messages", []))
    mt = body.get("max_tokens", 2048)
    for attempt in range(MAX_RETRY + 1):
        rb = dict(body)
        rb["messages"] = history + [{"role": "user", "content": REWRITE_INSTRUCTION}]
        rb["seed"] = seed + 300 + attempt
        rb["temperature"] = 0
        rb["max_tokens"] = mt
        resp = backend_call(rb)
        msg = resp["choices"][0]["message"]
        content = msg.get("content") or ""
        if not CJK_RE.search(content):
            return msg, resp, attempt + 1
    return msg, resp, MAX_RETRY + 1


def sse_emit(handler, msg, resp_id, model):
    def chunk(delta, finish=None):
        payload = {"id": resp_id, "object": "chat.completion.chunk",
                   "created": int(time.time()), "model": model,
                   "choices": [{"index": 0, "delta": delta,
                                "finish_reason": finish}]}
        handler.wfile.write(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode())

    content = msg.get("content") or ""
    for i in range(0, len(content), 60):
        chunk({"role": "assistant", "content": content[i:i + 60]})
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
        parsed = urlparse(self.path)
        if parsed.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length))
        tools = req.get("tools") or []
        sys.stderr.write(f"[gate-v2] tools={len(tools)} msgs={len(req.get('messages', []))} "
                         f"sys={any(m.get('role')=='system' for m in req.get('messages', []))}\n")
        sys.stderr.flush()
        want_stream = bool(req.get("stream", False))
        body = build_forward_body(req)
        seed = int(req.get("seed", 42) or 42)
        t0 = time.time()
        try:
            msg, resp, retries = enforce_english_content(body, seed)
        except Exception as e:
            self._send_json({"error": {"message": f"backend error: {str(e)[:200]}",
                                       "type": "server_error"}}, code=502)
            return
        dt = time.time() - t0
        sys.stderr.write(f"[gate-v2] done retries={retries} wall={dt:.1f}s "
                         f"stream={want_stream} tools={bool(body.get('tools'))}\n")
        sys.stderr.flush()
        if want_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                sse_emit(self, msg, resp.get("id", f"gate-{int(time.time())}"),
                         resp.get("model", "gate"))
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        out = {
            "choices": [{
                "finish_reason": ("tool_calls" if msg.get("tool_calls")
                                  else resp["choices"][0].get("finish_reason", "stop")),
                "index": 0,
                "message": {"role": "assistant",
                            "content": msg.get("content") or "",
                            "tool_calls": msg.get("tool_calls")},
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


def main():
    server = ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), GateProxyV2Handler)
    print(f"[gate-v2] listening on 0.0.0.0:{LISTEN_PORT} -> backend {BACKEND}",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[gate-v2] stopped", flush=True)


if __name__ == "__main__":
    main()
