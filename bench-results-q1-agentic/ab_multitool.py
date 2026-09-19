#!/usr/bin/env python3
"""Multi-tool malformation A/B. Direct backend (no gateway).
Usage: AB_BASE=http://localhost:8082 python3 ab_multitool.py
"""
import json, os, sys, urllib.request

BASE = os.environ.get("AB_BASE", "http://localhost:8082/v1/chat/completions")

TOOLS = [
    {"type": "function", "function": {"name": "get_time", "description": "Get the current time in a given city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculate", "description": "사칙연산 계산기",
     "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}, "op": {"type": "string", "enum": ["+", "-", "x", "/"]}}, "required": ["a", "b", "op"]}}},
    {"type": "function", "function": {"name": "search_orders", "description": "사용자의 주문 내역을 조회합니다",
     "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}},
    {"type": "function", "function": {"name": "send_email", "description": "Send an email with subject and body",
     "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "subject", "body"]}}},
    {"type": "function", "function": {"name": "get_weather", "description": "도시의 현재 날씨 조회",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "Translate text between languages",
     "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target": {"type": "string", "enum": ["en", "ko", "ja"]}}, "required": ["text", "target"]}}},
    {"type": "function", "function": {"name": "create_note", "description": "메모를 생성합니다",
     "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "content": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["title", "content"]}}},
    {"type": "function", "function": {"name": "set_reminder", "description": "Set a reminder at a time",
     "parameters": {"type": "object", "properties": {"time": {"type": "string"}, "note": {"type": "string"}}, "required": ["time", "note"]}}},
]


import copy as _copy
_base = list(TOOLS)
for _k in range(3):
    for _t in _base:
        _c = _copy.deepcopy(_t)
        _c["function"]["name"] = _c["function"]["name"] + f"_v{_k+2}"
        TOOLS.append(_c)
print(f"TOOLS: {len(TOOLS)}", flush=True)
_DOC = ("Use this operation to interact with the workspace object store. Provide all required identifiers "
        "exactly as documented, including the space identifier, object identifier, and optional view configuration. "
        "Results are returned as structured JSON payloads containing status codes, result objects, pagination cursors, "
        "and diagnostic metadata useful for debugging chained operations and downstream consumers. ")
for _t in TOOLS:
    _f = _t["function"]
    _f["description"] = _DOC + _f.get("description", "")
    _p = _f["parameters"]["properties"]
    _p["options"] = {"type": "object", "description": "Advanced options bag with nested filters, sort orders, pagination limits, field masks, and consistency tokens for paged traversal of large collections.",
                     "properties": {"filter": {"type": "string"}, "sort": {"type": "string", "enum": ["asc", "desc", "relevance"]}, "limit": {"type": "integer"}, "cursor": {"type": "string"}}}
import json as _js
print("TOOLS:", len(TOOLS), "schema bytes:", sum(len(_js.dumps(t)) for t in TOOLS), flush=True)


PROMPTS = [
    "What time is it in Paris right now?",
    "Calculate 23 multiplied by 7.",
    "파리 현재 시각을 알려줘.",
    "u42 사용자의 주문 내역을 조회해줘.",
]


def trial(prompt, seed):
    body = {"model": "ab", "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7, "max_tokens": 150, "seed": seed,
            "tools": TOOLS, "tool_choice": "auto", "stream": False}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    m = d["choices"][0]["message"]
    tcs = m.get("tool_calls") or []
    bad = 0
    for tc in tcs:
        try:
            json.loads(tc["function"].get("arguments", ""))
        except Exception:
            bad += 1
    return len(tcs), bad, (m.get("content") or "")[:60]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    tot_calls = tot_bad = tot_nocall = 0
    for i in range(n):
        try:
            nc, nb, c = trial(PROMPTS[i % len(PROMPTS)], 9000 + i)
            tot_calls += nc
            tot_bad += nb
            if nc == 0:
                tot_nocall += 1
            print(f"  t{i}: calls={nc} malformed={nb} content={c[:50]!r}", flush=True)
        except Exception as e:
            print(f"  t{i}: ERROR {type(e).__name__} {str(e)[:80]}", flush=True)
    print(f"RESULT calls={tot_calls} malformed={tot_bad} nocall_turns={tot_nocall}/{n}")


if __name__ == "__main__":
    main()
