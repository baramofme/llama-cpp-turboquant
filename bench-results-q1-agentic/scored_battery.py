#!/usr/bin/env python3
"""Scored EN battery: math/logic/reasoning with deterministic checks.
Usage: BATTERY_BASE=http://localhost:8082 python3 scored_battery.py
"""
import json, os, re, time, urllib.request

BASE = os.environ.get("BATTERY_BASE", "http://localhost:8082/v1/chat/completions")
SYS = "You are a careful reasoning assistant. You always respond in English only."

ITEMS = [
    ("arith", "What is 17 multiplied by 23? Reply with just the number.",
     lambda c: "391" in c),
    ("percent", "A shirt costs 80000 won. After a 25% discount, an additional 10% off is applied to the discounted price. What is the final price? Show brief work and state the final number clearly.",
     lambda c: "54000" in c.replace(",", "").replace(" ", "")),
    ("equation", "Solve (2x+5)/3 - (x-1)/2 = 4 for x. Show your work.",
     lambda c: re.search(r"x\s*=\s*11\b", c) is not None),
    ("digit", "The digits of a two-digit number sum to 11. Reversing the digits gives a number 27 less than the original. What is the number?",
     lambda c: "74" in c),
    ("factorial", "In the sequence 1, 2, 6, 24, 120, ... what comes next? Explain the rule.",
     lambda c: "720" in c),
    ("tuesday", "A family has two children. Given at least one is a boy born on Tuesday, what is the probability both are boys? (boy/girl equally likely, uniform weekdays)",
     lambda c: ("13/27" in c.replace(" ", "") or "0.48" in c or "48" in c)),
    ("sendmore", "SEND + MORE = MONEY cryptarithm, distinct digits 0-9, no leading zeros. Give the digit for S.",
     lambda c: re.search(r"S\s*=\s*9\b", c) is not None),
    ("cats", "Point out the flaw: 'All cats have four legs. My dog has four legs. Therefore my dog is a cat.'",
     lambda c: ("affirm" in c.lower() or "invalid" in c.lower() or "fallacy" in c.lower())),
    ("palindrome", "What is the longest palindromic substring of 'babad'? Reply with just the substring.",
     lambda c: ("bab" in c.lower() or "aba" in c.lower())),
    ("prime", "What is the smallest prime number greater than 100? Reply with just the number.",
     lambda c: "101" in c),
    ("boxes", "Three boxes labeled apples, oranges, mixed - all labels wrong. One fruit peek allowed. How to determine all contents?",
     lambda c: "mixed" in c.lower()),
    ("monty", "Game show with 3 doors, car behind one. You pick one, host opens a goat door, offers switch. Should you switch? One word plus one sentence why.",
     lambda c: "switch" in c.lower()),
]


def ask(prompt, seed, mt=600):
    body = {"model": "bench", "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": prompt}],
        "temperature": 0.0, "seed": seed, "max_tokens": mt, "stream": False}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"].get("content") or "", time.time() - t0


def main():
    ok = 0
    for i, (name, q, check) in enumerate(ITEMS):
        try:
            c, dt = ask(q, 1000 + i)
            good = check(c)
            ok += good
            print(f"[{'PASS' if good else 'FAIL'}] {name} ({dt:.1f}s) :: {re.sub(r'\s+', ' ', c)[:160]}", flush=True)
        except Exception as e:
            print(f"[ERROR] {name}: {type(e).__name__} {str(e)[:80]}", flush=True)
    print(f"\nSCORE: {ok}/{len(ITEMS)}")


if __name__ == "__main__":
    main()
