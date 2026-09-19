#!/usr/bin/env python3
"""Graduated math ladder for Q1. Full outputs for reasoning audit.
Usage: python3 grad_math.py (targets localhost:8082)
"""
import json, time, urllib.request

BASE = "http://localhost:8082/v1/chat/completions"
SYS = "You are a careful math assistant. Show your work briefly, then state the final answer."

ITEMS = [
    ("L1-mult", "Compute 1234 multiplied by 5678. Show the result clearly."),
    ("L1-div", "Compute 100000 divided by 125. Show the result clearly."),
    ("L1-mixed", "Compute 37 + 48 * 12. Show each step and the result clearly."),
    ("L2-pct", "What is 15 percent of 240? Show the result clearly."),
    ("L2-dec", "Express 3/8 as a decimal. Show the result clearly."),
    ("L3-ord", "Compute 2 + 3 * 4^2. Show each step and the result clearly."),
    ("L3-root", "Compute sqrt(144) + cbrt(27). Show each step and the result clearly."),
    ("L4-lin", "Solve 3x - 7 = 20 for x. Show your work and box the answer."),
    ("L4-sys", "Solve the system: x + y = 10 and x - y = 4. Give x and y clearly."),
    ("L5-quad", "Solve x^2 - 5x + 6 = 0. Give all solutions clearly."),
    ("L6-train", "Two trains 300 km apart move toward each other at 60 km/h and 90 km/h. After how many hours do they meet? Show work and answer clearly."),
    ("L7-dice", "Two fair dice are rolled. What is the probability the sum is 9? Give as a fraction, clearly."),
    ("L7-bayes", "1% of people have a disease. A test is 99% accurate (true positive and true negative rates both 99%). If a random person tests positive, what is the probability they are sick? Give percent clearly."),
    ("L8-fib", "What is the 13th Fibonacci number (starting 1, 1, 2, 3, 5, ...)? Give the number clearly."),
]


def ask(q, seed):
    body = {"model": "q1math", "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": q}],
        "temperature": 0.0, "seed": seed, "max_tokens": 400, "stream": False}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"].get("content") or "", time.time() - t0


def main():
    for i, (name, q) in enumerate(ITEMS):
        try:
            c, dt = ask(q, 2000 + i)
            print(f"===== [{name}] ({dt:.1f}s) =====")
            print(c[:1500])
            print()
        except Exception as e:
            print(f"[{name}] ERROR: {type(e).__name__} {str(e)[:80]}")


if __name__ == "__main__":
    main()
