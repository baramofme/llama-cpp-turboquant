#!/usr/bin/env python3
"""NIAH probe: needle retrieval at depths. Rule-scored.
Usage: NIAH_BASE=http://localhost:8082 NIAH_GRAMMAR=1 python3 niah_probe.py
Set NIAH_DEPTHS (comma tomography, default "8,24") and NIAH_TOTAL_K (default 8,24).
"""
import json, os, re, time, urllib.request

BASE = os.environ.get("NIAH_BASE", "http://localhost:8082/v1/chat/completions")
USE_GRAMMAR = os.environ.get("NIAH_GRAMMAR", "0") == "1"
GBNF_PATH = "/home/baramofme/IdeaProjects/llama-cpp-turboquant/beellama-boosts/grammars/english-only.gbnf"

NEEDLE = "The special access code for the Atlas project is 73921."
QUESTION = "What is the special access code for the Atlas project? Reply with just the number."
PARA = ("The city council met on Thursday to discuss road maintenance budgets for the coming fiscal year. "
        "Members reviewed proposals for resurfacing three major avenues and repairing two bridges. ")

with open(GBNF_PATH) as f:
    GBNF = f.read()


def build_doc(total_tokens, depth_frac):
    target_chars = total_tokens * 4
    unit = PARA
    n_units = max(1, target_chars // len(unit))
    at = int(n_units * depth_frac)
    parts = [unit] * at + [NEEDLE + " "] + [unit] * (n_units - at)
    return "".join(parts)


def ask(doc, seed):
    body = {"model": "niah",
            "messages": [{"role": "system", "content": "You are a careful assistant. Answer briefly in English."},
                         {"role": "user", "content": doc + "\n\n" + QUESTION}],
            "temperature": 0.0, "seed": seed, "max_tokens": 30, "stream": False}
    if USE_GRAMMAR:
        body["grammar"] = GBNF
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"].get("content") or "", time.time() - t0


def main():
    totals = [int(x) for x in os.environ.get("NIAH_TOTAL_K", "8,24").split(",")]
    depths = [float(x) for x in os.environ.get("NIAH_DEPTHS", "0.1,0.5,0.9").split(",")]
    ok = tot = 0
    for tk in totals:
        for df in depths:
            tot += 1
            try:
                c, dt = ask(build_doc(tk * 1000, df), 5000 + tot)
                good = "73921" in c
                ok += good
                print(f"[{'PASS' if good else 'FAIL'}] {tk}k@{int(df*100)}% ({dt:.0f}s) :: {re.sub(r'\s+', ' ', c)[:100]}", flush=True)
            except Exception as e:
                print(f"[ERROR] {tk}k@{int(df*100)}%: {type(e).__name__} {str(e)[:60]}", flush=True)
    print(f"\nNIAH SCORE: {ok}/{tot}")


if __name__ == "__main__":
    main()
