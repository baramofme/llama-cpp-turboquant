#!/usr/bin/env python3
"""Multi-turn agentic chain: tool results fed back, like real usage.
Usage: AB_BASE=... python3 ab_chain.py   (expects ab_multitool TOOLS)
"""
import json, os, sys, urllib.request
sys.path.insert(0, "/home/baramofme/IdeaProjects/llama-cpp-turboquant/bench-results-q1-agentic")
from ab_multitool import TOOLS

BASE = os.environ.get("AB_BASE", "http://localhost:8082/v1/chat/completions")

MOCK = {
    "search_orders": lambda a: {"user": a.get("user_id", "?"), "orders": [
        {"id": 1, "amount": 120.5}, {"id": 2, "amount": 45.0}, {"id": 3, "amount": 78.25}]},
    "calculate": lambda a: {"result": {"+": a.get("a", 0) + a.get("b", 0),
                                       "-": a.get("a", 0) - a.get("b", 0),
                                       "x": a.get("a", 0) * a.get("b", 0)}[a.get("op", "+")]},
    "send_email": lambda a: {"sent": True, "to": a.get("to")},
    "get_time": lambda a: {"time": "2026-09-14T12:00:00", "city": a.get("city", "?")},
}


def call(msgs, seed, tag):
    body = {"model": "chain", "messages": msgs, "temperature": 0.7,
            "max_tokens": 300, "seed": seed, "tools": TOOLS,
            "tool_choice": "auto", "stream": False}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main():
    msgs = [{"role": "user", "content":
             "u42 사용자의 주문 내역을 조회하고, 금액 합계를 계산한 뒤, "
             "billing@example.com에게 제목 '주문 합계'로 합계를 본문에 담아 이메일을 보내줘. "
             "도구를 순서대로 사용해."}]
    bad = 0
    for rnd in range(6):
        try:
            d = call(msgs, 3000 + rnd, rnd)
        except Exception as e:
            print(f"round {rnd}: REQUEST ERROR {type(e).__name__} {str(e)[:80]}", flush=True)
            bad += 1
            continue
        m = d["choices"][0]["message"]
        tcs = m.get("tool_calls") or []
        ok = True
        for tc in tcs:
            try:
                args = json.loads(tc["function"].get("arguments", ""))
            except Exception:
                ok = False
                bad += 1
                break
        msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs or None})
        print(f"round {rnd}: calls={len(tcs)} malformed_kw={'YES' if not ok else 'no'} "
              f"finish={d['choices'][0].get('finish_reason')} content={str(m.get('content'))[:60]!r}", flush=True)
        if not tcs:
            print("chain ended (no more calls)", flush=True)
            break
        for tc in tcs:
            fn = tc["function"]["name"].split("_v")[0]
            try:
                args = json.loads(tc["function"].get("arguments", "{}"))
            except Exception:
                args = {}
            if fn in MOCK:
                res = MOCK[fn](args)
            else:
                res = {"ok": True}
            msgs.append({"role": "tool", "content": json.dumps(res, ensure_ascii=False),
                         "tool_call_id": tc.get("id", f"c{rnd}")})
    print(f"CHAIN DONE malformed_rounds={bad}")


if __name__ == "__main__":
    main()
