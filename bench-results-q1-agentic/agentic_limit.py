#!/usr/bin/env python3
"""Agentic tool-calling limit test for Bonsai-27B Q1_0 (llama-server /v1 OpenAI API).

Goal: find how many tool-call rounds / how much task complexity the model
handles before failing, so prompts can be constrained below that limit.

Usage:
  python3 agentic_limit.py --temp 0.0 --reps 3
  python3 agentic_limit.py --temp 0.3 --levels L1 L2
"""
import argparse, json, os, sys, time, urllib.request, urllib.error

BASE = os.environ.get("BONSAI_BASE", "http://localhost:8082/v1/chat/completions")

TOOLS = [
    {"type": "function", "function": {"name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_temperature",
        "description": "Get temperature in Celsius for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculate",
        "description": "Arithmetic operation on two numbers",
        "parameters": {"type": "object", "properties": {
            "a": {"type": "number"}, "b": {"type": "number"},
            "op": {"type": "string", "enum": ["+", "-", "*", "/"]}}, "required": ["a", "b", "op"]}}},
    {"type": "function", "function": {"name": "search_orders",
        "description": "Query order records for a user id",
        "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}},
    {"type": "function", "function": {"name": "send_email",
        "description": "Send an email message",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "subject", "body"]}}},
    {"type": "function", "function": {"name": "get_user_info",
        "description": "Get account profile for a user id",
        "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}},
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}

SYSTEM_MSG = ("You are a helpful assistant with access to tools. When the user's request "
              "requires data you do not have, always call the relevant tool. "
              "Be concise in your final reply.")
SYSTEM_BRIEF = (SYSTEM_MSG + " Think briefly: plan in at most 2 short sentences, "
                "then immediately emit the tool call. Never narrate long reasoning.")

LEVELS = {
    "L1": {
        "desc": "1 call, single tool, trivial",
        "tools": ["get_weather"],
        "user": "What is the weather in Paris right now? Use the get_weather tool to find out.",
        "required": [("get_weather", 1)],
        "accuracy": {"expect": "22", "text": "temp in Paris"},
    },
    "L2": {
        "desc": "2 calls, same tool, compare",
        "tools": ["get_temperature"],
        "user": ("Check the temperature in Paris and Berlin using get_temperature (one call per city), "
                 "then answer which city is warmer and report both temperatures."),
        "required": [("get_temperature", 2)],
        "accuracy": {"expect": "Paris", "text": "warmer city (mock: Paris 22C, Berlin 18C)"},
    },
    "L3": {
        "desc": "4 calls, 3 tools, data dependency",
        "tools": ["get_temperature", "calculate", "send_email"],
        "user": ("Use get_temperature once for Lyon and once for Marseille. Then use calculate to compute "
                 "(Marseille minus Lyon) with op '-'. Then use send_email to ops@example.com with subject "
                 "'temp diff' and a body that states the computed difference in degrees. Do all steps with tools."),
        "required": [("get_temperature", 2), ("calculate", 1), ("send_email", 1)],
        "accuracy": {"expect": "3", "text": "diff (mock: Lyon 25C, Marseille 28C -> 3)"},
    },
    "L4": {
        "desc": "3+ calls, 3 tools, external data + total",
        "tools": ["search_orders", "calculate", "send_email"],
        "user": ("Act as a billing agent. Search orders for user u42 with search_orders. Then use calculate to sum "
                 "the amounts found (120.5 + 45 + 78.25). Then use send_email to billing@example.com with subject "
                 "'order total' and a body containing the computed total. Do all steps; do not guess values."),
        "required": [("search_orders", 1), ("calculate", 1), ("send_email", 1)],
        "accuracy": {"expect": "243.75", "text": "total (mock: 120.5+45+78.25=243.75)"},
    },
    "L3S": {
        "desc": "2 calls, single tool, how much better is a single-tool L3",
        "tools": ["get_temperature"],
        "user": ("Get temperatures for Lyon and Marseille with get_temperature (one call per city), "
                 "then answer which city is warmer."),
        "required": [("get_temperature", 2)],
        "accuracy": {"expect": "Marseille", "text": "warmer (mock: Lyon 25C, Marseille 28C)"},
    },
    "L3T": {
        "desc": "3 calls, 2 tools, dependency, no email",
        "tools": ["get_temperature", "calculate"],
        "user": ("Get temperatures for Lyon and Marseille with get_temperature (one call per city). "
                 "Then use calculate to compute (Marseille minus Lyon) with op '-' and report the result."),
        "required": [("get_temperature", 2), ("calculate", 1)],
        "accuracy": {"expect": "3", "text": "diff (mock: 25 vs 28 -> 3)"},
    },
    "L5SEQM": {
        "desc": "3 sequential asks, 1 call each (rounds test)",
        "tools": ["get_temperature"],
        "user_turns": [
            "What is the temperature in Paris right now? Use get_temperature.",
            "And what is the temperature in Berlin? Use get_temperature.",
            "You reported Paris and Berlin temperatures earlier. Which city is warmer?",
        ],
        "ask_required": [{"get_temperature": 1}, {"get_temperature": 1}, {}],
        "required": [("get_temperature", 2)],
        "accuracy": {"expect": "Paris", "text": "warmer city (mock 22 vs 18)"},
    },
    "L5": {
        "desc": "5 calls, 4 tools, dependency + parallel pair",
        "tools": ["get_temperature", "calculate", "search_orders", "send_email"],
        "user": ("Billing task: (1) search_orders for user u42, (2) use calculate to sum the amounts "
                 "(120.5 + 45 + 78.25), (3) send_email to billing@example.com with subject 'order total' "
                 "and body containing the computed total, (4) also get_temperature once for Paris and once "
                 "for Berlin and mention both values in the email body. Do all steps with tools."),
        "required": [("search_orders", 1), ("calculate", 1), ("send_email", 1), ("get_temperature", 2)],
        "accuracy": {"expect": "243.75", "text": "total in email body"},
    },
    "L6": {
        "desc": "6+ calls, 5 tools incl decoy, two-step dependency",
        "tools": ["get_temperature", "calculate", "search_orders", "send_email", "get_user_info"],
        "user": ("Act as shipping analyst. (1) search_orders('u42') to get orders with their city. "
                 "(2) The orders are in Paris (two orders) and Berlin (one order). Use get_temperature once "
                 "for Paris and once for Berlin. (3) Use calculate to compute (Paris minus Berlin) with op '-'. "
                 "(4) send_email to ops@example.com with subject 'ship temps' and a body that states the "
                 "temperature difference. Ignore get_user_info, it is not needed."),
        "required": [("search_orders", 1), ("get_temperature", 2), ("calculate", 1), ("send_email", 1)],
        "accuracy": {"expect": "4", "text": "diff (mock Paris 22, Berlin 18 -> 4)"},
    },
    "L7": {
        "desc": "7 calls, 7 tools incl decoys, long system prompt, 2 emails",
        "system_extra": ("\n\nAGENCY OPERATING GUIDELINES (follow always):\n"
                         "1. Data requests MUST go through tools; never invent figures.\n"
                         "2. Financial totals must be computed with the calculate tool, never by hand.\n"
                         "3. Customer communications go through send_email only.\n"
                         "4. When multiple reports are needed, use a separate email per report.\n"
                         "5. If a tool is irrelevant to the current step, do not call it.\n"
                         "6. Keep every email body under 3 sentences.\n"
                         "7. After all emails are sent, summarize the outcome in a short final message."),
        "tools": ["get_temperature", "calculate", "search_orders", "send_email", "get_user_info",
                  "get_weather", "send_email"],
        "user": ("Process this account: (1) search_orders('u42'). (2) The orders have city fields - the "
                 "two Paris orders and one Berlin order. Use get_temperature once for Paris and once for "
                 "Berlin. (3) Use calculate with op '-' once for (Paris minus Berlin) temperature diff, and "
                 "a second calculate to sum the order amounts (120.5 + 45 + 78.25). (4) Send two emails: "
                 "one to ops@example.com subject 'ship temps' with the temperature difference, and one to "
                 "billing@example.com subject 'order total' with the amount total. (5) Ignore get_user_info "
                 "and get_weather, they are not needed."),
        "required": [("search_orders", 1), ("get_temperature", 2), ("calculate", 2), ("send_email", 2)],
        "accuracy": {"expect": "4", "text": "diff AND 243.75 in different emails"},
    },
}

MOCK_FUNCS = {
    "get_weather": lambda a: {"city": a.get("city", "?"), "condition": "sunny", "temp_c": 22},
    "get_temperature": lambda a: {"city": a.get("city", "?"), "temp_c": {"Paris": 22, "Berlin": 18, "Lyon": 25, "Marseille": 28}.get(a.get("city"), 20)},
    "calculate": lambda a: {"expression": f'{a["a"]} {a["op"]} {a["b"]}', "result": {
        "+": a["a"] + a["b"], "-": a["a"] - a["b"], "*": a["a"] * a["b"], "/": a["a"] / a["b"]}[a["op"]]},
    "search_orders": lambda a: {"user": a.get("user_id", "?"), "orders":
        [{"id": 1, "amount": 120.5, "city": "Paris"}, {"id": 2, "amount": 45.0, "city": "Paris"},
         {"id": 3, "amount": 78.25, "city": "Berlin"}] if a.get("user_id") == "u42" else []},
    "send_email": lambda a: {"to": a.get("to"), "subject": a.get("subject"), "body": a.get("body"), "sent": True, "id": "msg-1"},
}


def validate_call(tc):
    name = tc.get("name")
    if name not in TOOL_NAMES:
        return False, f"unknowntool:{name}"
    raw = tc.get("arguments", "")
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return False, f"badjson:{raw[:120]!r}"
    if not isinstance(args, dict):
        return False, f"badargsshape:{str(args)[:80]!r}"
    schema = next(t for t in TOOLS if t["function"]["name"] == name)["function"]["parameters"]
    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        return False, f"missingarg:{name}:{missing}"
    for k, v in args.items():
        prop = schema["properties"].get(k)
        if not prop:
            continue
        if prop.get("type") == "number" and not isinstance(v, (int, float)):
            return False, f"badtype:{name}.{k}={v!r}"
        if prop.get("type") == "string" and not isinstance(v, str):
            return False, f"badtype:{name}.{k}={v!r}"
        if "enum" in prop and v not in prop["enum"]:
            return False, f"badenum:{name}.{k}={v!r}"
    return True, args


def chat(messages, tools, temp, seed, max_tokens=600, no_think=False):
    body = {"model": "bonsai", "messages": messages, "tools": tools,
            "tool_choice": "auto", "temperature": temp, "seed": seed,
            "max_tokens": max_tokens, "stream": False}
    if no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r), None
    except urllib.error.HTTPError as e:
        return None, f"http_{e.code}:{e.read().decode()[:200]}"
    except Exception as e:
        return None, f"exc:{e}"


def run_level(name, cfg, temp, seed, max_rounds=14, max_tokens=600, system=SYSTEM_MSG, no_think=False):
    tools = [t for t in TOOLS if t["function"]["name"] in cfg["tools"]]
    required = {n: c for n, c in cfg["required"]}
    made = {}
    asks = cfg.get("user_turns") or [cfg["user"]]
    ask_idx = 0
    made_at_ask = {n: 0 for n in required}
    sys_msg = system + (cfg.get("system_extra") or "")
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": asks[0]}]
    transcript = []
    events = []
    final_text = ""

    for rnd in range(1, max_rounds + 1):
        resp, err = chat(msgs, tools, temp, seed, max_tokens, no_think)
        if err:
            events.append({"round": rnd, "type": err})
            break
        msg = resp["choices"][0]["message"]
        finish = resp["choices"][0].get("finish_reason")
        uses = resp.get("usage", {})
        transcript.append({"round": rnd, "finish": finish,
                           "content": msg.get("content", ""),
                           "reasoning": (msg.get("reasoning_content") or "")[:300],
                           "tool_calls": msg.get("tool_calls") or [],
                           "tokens": uses.get("total_tokens", 0)})

        calls = msg.get("tool_calls") or []
        if calls:
            for tc in calls:
                fn = tc.get("function", {})
                ok, detail = validate_call(fn)
                if not ok:
                    return {"level": name, "status": "FAIL", "fail_round": rnd,
                            "fail_type": "invalid_call", "detail": detail,
                            "made": dict(made), "events": events, "transcript": transcript,
                            "error_reasoning": transcript[-1]["reasoning"]}
                made[fn["name"]] = made.get(fn["name"], 0) + 1
            msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for tc in calls:
                fn = tc.get("function", {})
                result = MOCK_FUNCS[fn["name"]](json.loads(fn.get("arguments", "{}")))
                msgs.append({"role": "tool", "tool_call_id": tc.get("id"), "content": json.dumps(result)})
            continue

        final_text = msg.get("content") or ""
        ask_reqs = cfg.get("ask_required") or [None] * len(asks)
        ask_req = ask_reqs[ask_idx]
        ask_satisfied = all(made.get(n, 0) - made_at_ask[n] >= c
                            for n, c in (ask_req or {}).items())
        if ask_idx < len(asks) - 1 and ask_satisfied:
            ask_idx += 1
            msgs.append({"role": "user", "content": asks[ask_idx]})
            for n in required:
                made_at_ask[n] = made.get(n, 0)
            continue

        done = all(made.get(n, 0) >= c for n, c in required.items())
        if done and ask_satisfied:
            events.append({"round": rnd, "type": "final_answer"})
            return {"level": name, "status": "OK", "rounds": rnd, "made": dict(made),
                    "events": events, "transcript": transcript, "final": final_text,
                    "accuracy_ok": cfg["accuracy"]["expect"] in final_text}
        events.append({"round": rnd, "type": "gave_up_before_complete", "made": dict(made)})
        return {"level": name, "status": "FAIL", "fail_round": rnd, "fail_type": "gave_up",
                "made": dict(made), "events": events, "transcript": transcript,
                "final": final_text, "error_reasoning": transcript[-1]["reasoning"]}

    events.append({"round": max_rounds, "type": "ran_out"})
    return {"level": name, "status": "FAIL", "fail_round": max_rounds, "fail_type": "ran_out",
            "made": dict(made), "events": events, "transcript": transcript,
            "error_reasoning": (transcript[-1]["reasoning"] if transcript else "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--brief", action="store_true")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--levels", nargs="+", default=list(LEVELS))
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()

    system = SYSTEM_BRIEF if args.brief else SYSTEM_MSG
    reporter = "brief" if args.brief else "default"
    reporter = "nothink" if args.no_think else reporter

    os.makedirs(args.outdir, exist_ok=True)
    outfile = os.path.join(args.outdir, f"agentic_t{args.temp}_{reporter}_mt{args.max_tokens}.json")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    summary = {"temp": args.temp, "brief": args.brief, "max_tokens": args.max_tokens,
               "time": stamp, "runs": []}

    for lvl in args.levels:
        cfg = LEVELS[lvl]
        print(f"\n=== {lvl} [{cfg['desc']}] (temp={args.temp}, mt={args.max_tokens}, sys={reporter}, reps={args.reps}) ===")
        for rep in range(1, args.reps + 1):
            t0 = time.time()
            seed = 1000 + sum(ord(c) for c in lvl) + rep
            res = run_level(lvl, cfg, args.temp, seed, max_tokens=args.max_tokens,
                            system=system, no_think=args.no_think)
            res.update({"rep": rep, "secs": round(time.time() - t0, 1)})
            summary["runs"].append(res)
            status = f"OK (rounds={res.get('rounds')}, acc={res.get('accuracy_ok')})" if res["status"] == "OK" \
                else f"FAIL r{res.get('fail_round')} [{res.get('fail_type')}] {res.get('detail', '')}"
            print(f"  rep{rep}: {status}  ({res['secs']}s)")
            with open(os.path.join(args.outdir, f"tr_{lvl}_t{args.temp}_{reporter}_r{rep}_{stamp}.jsonl"), "w") as f:
                for line in res["transcript"]:
                    f.write(json.dumps(line) + "\n")

    with open(outfile, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\n== aggregate (temp={args.temp}) ==")
    from collections import Counter
    ok = fail = 0
    fail_rounds, fail_types = [], Counter()
    for r in summary["runs"]:
        if r["status"] == "OK":
            ok += 1
        else:
            fail += 1
            fail_rounds.append(r["fail_round"])
            fail_types[r["fail_type"]] += 1
    print(f"OK={ok} FAIL={fail}")
    print(f"fail_rounds={fail_rounds} fail_types={dict(fail_types)}")
    print(f"saved: {outfile}")


if __name__ == "__main__":
    main()