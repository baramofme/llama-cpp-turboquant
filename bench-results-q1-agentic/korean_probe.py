#!/usr/bin/env python3
"""Korean-language agentic limit probe against production bonsai server."""
import json, os, sys, urllib.request

BASE = os.environ.get("BONSAI_BASE", "http://localhost:8082/v1/chat/completions")

MOCK = {
    "get_temperature": lambda a: {"city": a.get("city", "?"), "celsius": 22 if a.get("city") == "Paris" else 18},
    "calculate": lambda a: {"result": {"+": a.get("a", 0) + a.get("b", 0), "-": a.get("a", 0) - a.get("b", 0),
                                         "x": a.get("a", 0) * a.get("b", 0)}[a.get("op", "+")]},
    "search_orders": lambda a: {"user": a.get("user_id", "?"), "orders":
        [{"id": 1, "amount": 120.5, "city": "Paris"}, {"id": 2, "amount": 45.0, "city": "Paris"},
         {"id": 3, "amount": 78.25, "city": "Berlin"}] if a.get("user_id") == "u42" else []},
    "send_email": lambda a: {"sent": True, "to": a.get("to"), "subject": a.get("subject"),
                             "body_len": len(a.get("body", ""))},
}

TOOLS = [
    {"type": "function", "function": {"name": "get_temperature", "description": "도시의 현재 기온(섭씨) 조회",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculate", "description": "사칙연산 계산",
        "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"},
            "op": {"type": "string", "enum": ["+", "-", "x"]}}, "required": ["a", "b", "op"]}}},
    {"type": "function", "function": {"name": "search_orders", "description": "사용자의 주문 내역 조회",
        "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}},
    {"type": "function", "function": {"name": "send_email", "description": "이메일 발송",
        "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"},
            "body": {"type": "string"}}, "required": ["to", "subject", "body"]}}},
]

KOREAN_CASES = {
    "K-L3": (
        "파리와 베를린의 기온을 get_temperature로 각각 조회하고, calculate로 (파리 - 베를린) 차이를 op '-'로 계산한 뒤, "
        "send_email로 ops@example.com에게 제목 '기온 차이'로 계산된 기온 차이를 본문에 적어 이메일을 보내세요. "
        "도구를 사용해서 모든 단계를 수행하세요.",
        [("get_temperature", 2), ("calculate", 1), ("send_email", 1)]),
    "K-L4": (
        "청구 담당자처럼 행동하세요. search_orders로 사용자 u42의 주문을 조회하고, calculate로 주문 금액 합계를 계산하고, "
        "send_email로 billing@example.com에게 제목 '주문 합계'로 계산된 합계를 본문에 담아 이메일을 보내세요. "
        "모든 단계를 도구로 수행하고 값을 추측하지 마세요.",
        [("search_orders", 1), ("calculate", 1), ("send_email", 1)]),
}

def chat(msgs, temp=0.0, max_tokens=2048):
    body = {"model": "bonsai", "messages": msgs, "tools": TOOLS, "tool_choice": "auto",
            "temperature": temp, "seed": 42, "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read()), None
    except Exception as e:
        return None, str(e)

def run(name, prompt, required, reps=3, max_rounds=14):
    print(f"== {name} (Korean, temp=0, mt=2048, reps={reps}) ==")
    ok = 0
    for rep in range(1, reps + 1):
        msgs = [{"role": "user", "content": prompt}]
        made = {n: 0 for n, _ in required}
        rounds, outcome, reason = 0, "?", ""
        for rnd in range(1, max_rounds + 1):
            rounds = rnd
            resp, err = chat(msgs)
            if err:
                outcome, reason = "ERR", err; break
            m = resp["choices"][0]["message"]
            calls = m.get("tool_calls") or []
            if not calls:
                outcome = "gave_up" if made.get(required[0][0], 0) == 0 else "finished_text"
                reason = "no call"; break
            for tc in calls:
                fn = tc["function"]
                if fn["name"] in made: made[fn["name"]] += 1
                try: args = json.loads(fn["arguments"])
                except Exception: outcome, reason = "BADJSON", fn["arguments"][:60]; break
                res = MOCK[fn["name"]](args)
                msgs.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                msgs.append({"role": "tool", "tool_call_id": tc.get("id", "?"), "name": fn["name"], "content": json.dumps(res)})
            if outcome != "?":
                break
            if rnd >= 8 and all(made.get(n, 0) >= c for n, c in required):
                resp2, err2 = chat(msgs + [{"role": "user", "content": "이제 최종 요약만 간결히 말하세요."}], max_tokens=300)
                if not err2:
                    msgs.append({"role": "assistant", "content": resp2["choices"][0]["message"].get("content", "")})
                    outcome, reason = "finished_text", "forced-summary after all calls"; break
        done = all(made.get(n, 0) >= c for n, c in required)
        status = "OK" if done and outcome != "BADJSON" else "FAIL"
        if status == "OK": ok += 1
        print(f"  rep{rep}: {status} rounds={rounds} outcome={outcome} [{reason or 'ok'}] made={dict(made)} {'' if status=='OK' else '(missing=' + str([n for n,c in required if made.get(n,0)<c]) + ')'}")
    print(f"  -> {ok}/{reps} OK")
    return ok

if __name__ == "__main__":
    results = {}
    for name, (prompt, required) in KOREAN_CASES.items():
        results[name] = run(name, prompt, required)
    print("\nAGG:", results)