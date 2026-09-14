#!/usr/bin/env python3
"""English multi-turn memory probe (12 turns). Rule-scored recall.
Usage: MEM_BASE=http://localhost:8082 MEM_GRAMMAR=1 python3 en_memory_probe.py
"""
import json, os, re, time, urllib.request

BASE = os.environ.get("MEM_BASE", "http://localhost:8082/v1/chat/completions")
USE_GRAMMAR = os.environ.get("MEM_GRAMMAR", "0") == "1"
GBNF_PATH = "/home/baramofme/IdeaProjects/llama-cpp-turboquant/beellama-boosts/grammars/english-only.gbnf"

with open(GBNF_PATH) as f:
    GBNF = f.read()

TURNS = [
    ("intro", "Hi, my name is Alex. I live in Paris and I have a dog named Coco. My girlfriend is Mia. For reference - my shopping budget is 300 dollars, I like vanilla ice cream and dislike chocolate. My lucky number is 7, I work as a developer. I plan a Berlin trip this year, and I swim on weekends."),
    ("f1", "Oh, as a developer which language do you mainly use?"),
    ("f2", "I see. I'm preparing a new project these days too."),
    ("f3", "Coco loves walks so much she wakes me at 6am every day."),
    ("f4", "Mia asked what birthday gift to buy. What would be good? I like small gifts by the way."),
    ("f5", "So this weekend I'm thinking of preparing the Berlin trip and also going to the pool."),
    ("f6", "The weather is nice so I'm in a good mood."),
    ("f7", "When developing I end up drinking coffee often."),
    ("quiz", "Let me quiz you on what I told you. Answer each one: 1) my dog's name? 2) my city? 3) my girlfriend? 4) my shopping budget? 5) ice cream I like? 6) ice cream I dislike? 7) my job? 8) this year's trip city? 9) my sport? 10) my lucky number?"),
    ("f8", "Right, Mia is coming over tomorrow."),
    ("f9", "So I need to plan a date in Paris with Mia tomorrow."),
    ("f10", "By the way my brother Tom is a chef in Lyon. He makes amazing ratatouille."),
    ("f11", "I am thinking of buying a red bicycle for commuting."),
    ("f12", "My favorite movie is Inception and I watch it every year."),
    ("f13", "Coco hates baths but loves the rain."),
    ("f14", "Paris rent keeps rising, thinking of moving to Marseille someday."),
    ("f15", "I started learning piano last month, practicing scales daily."),
    ("quiz2", "Second memory check, answer each: 1) my brother's name and job? 2) bicycle color? 3) favorite movie? 4) does Coco like baths? 5) which city might I move to? 6) what am I learning? 7) dog name again? 8) girlfriend again?"),
    ("f16", "Tom visited last weekend with food."),
    ("f17", "Thinking of a road trip soon."),
    ("final2", "Final review: my brother, my bike, my movie, my instrument, and 7 plus 5?"),
    ("final", "Summarize: within my 300 dollar budget, going to my favorite ice cream place without Coco, just with Mia. What is my lucky number 7 plus 5? And isn't Berlin too far for a date?"),
]

CHECKS = {
    "quiz": (["coco", "paris", "mia", "developer", "berlin", "swim"], 4),
    "final": (["300", "vanilla", "mia", "12", "berlin"], 3),
    "quiz2": (["tom", "chef", "red", "inception", "coco", "marseille", "piano", "mia"], 6),
    "final2": (["tom", "red", "inception", "piano", "12"], 4),
}


def chat(msgs, seed, mt=300):
    body = {"model": "mem", "messages": msgs, "temperature": 0.0,
            "seed": seed, "max_tokens": mt, "stream": False}
    if USE_GRAMMAR:
        body["grammar"] = GBNF
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"].get("content") or "", time.time() - t0


def main():
    hist = []
    results = {}
    for i, (name, q) in enumerate(TURNS):
        hist.append({"role": "user", "content": q})
        try:
            c, dt = chat(hist, 7000 + i, mt=400 if name in ("quiz", "final", "quiz2", "final2") else 150)
        except Exception as e:
            print(f"[{name}] ERROR: {type(e).__name__} {str(e)[:60]}", flush=True)
            break
        hist.append({"role": "assistant", "content": c})
        mark = ""
        if name in CHECKS:
            keys, need = CHECKS[name]
            hits = sum(1 for k in keys if k in c.lower())
            good = hits >= need
            results[name] = good
            mark = f" [{'PASS' if good else 'FAIL'} {hits}/{len(keys)}]"
        print(f"[{name}] ({dt:.1f}s){mark} :: {re.sub(r'\s+', ' ', c)[:140]}", flush=True)
    passed = sum(results.values())
    print(f"\nMEMORY SCORE: {passed}/{len(results)}  {results}")


if __name__ == "__main__":
    main()
