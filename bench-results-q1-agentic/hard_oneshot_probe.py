#!/usr/bin/env python3
import json, os, re, time, urllib.request

BASE = os.environ.get("BONSAI_BASE", "http://localhost:8082/v1/chat/completions")

def chat(q, temp=0.0, mt=1024):
    body = {"model": "bonsai", "messages": [{"role": "user", "content": q}],
            "temperature": temp, "seed": 42, "max_tokens": mt, "stream": False}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return d, time.time() - t0

KO = [
    ("산술", "17 곱하기 23은 얼마야? 숫자만 답해."),
    ("퍼센트 체인", "셔츠가 80000원인데, 25% 할인 후 그 할인된 가격에서 추가로 10% 할인을 받았다. 최종 가격은 얼마야? 계산 과정을 간단히 보여주고 마지막에 최종 금액을 명확히 써."),
    ("방정식", "(2x+5)/3 - (x-1)/2 = 4 를 풀어서 x 값을 구해. 과정을 보여줘."),
    ("자릿수 역전", "두 자리 숫자가 있는데, 각 자릿수의 합은 11이야. 자릿수를 뒤집은 숫자는 원래 숫자보다 27 작아. 이 숫자는 뭐야? 과정을 보여줘."),
    ("라벨 상자", "사과만 든 상자, 오렌지만 든 상자, 사과와 오렌지가 섞인 상자 3개가 있는데, 모든 라벨이 틀리게 붙어 있어. 과일을 딱 한 번만 꺼내볼 수 있다고 할 때, 세 상자의 내용물을 정확히 알아내는 방법을 설명해. 단, 상자 하나에서 과일 하나만 꺼내볼 수 있어."),
    ("암호식 SEND+MORE", "'SEND + MORE = MONEY' 라는 암호식이 있어. 각 알파벳은 서로 다른 숫자(0-9)를 나타내고, 가장 높은 자릿수는 0이 될 수 없어. 이걸 풀어서 각 알파벳의 값을 구해."),
    ("팩토리얼 수열", "수열 1, 2, 6, 24, 120, ... 에서 다음 수는 뭐야? 규칙을 설명하고 답해."),
    ("화요일 소년", "한 가족에게 두 명의 아이가 있다. 적어도 한 명은 화요일에 태어난 남자아이이다. 두 아이가 모두 남자아이일 확률은 얼마야? (남자아이·여자아이 출생 확률은 1/2로 가정하고, 화요일은 7일에 하나)"),
    ("논리 함정", "다음 논증의 오류를 지적해: '모든 고양이는 네 발 달렸다. 내 개도 네 발 달렸다. 그러므로 내 개는 고양이다.' 어디가 틀렸는지 정확히 설명해."),
    ("알고리즘", "주어진 문자열에서 가장 긴 회문(팰린드롬) 부분 문자열을 찾는 함수를 파이썬으로 작성해. 시간 복잡도를 설명하고, 중복 문자 처리를 포함해."),
]

EN_HARD = {
    "SEND+MORE": "'SEND + MORE = MONEY' is a cryptarithm. Each letter is a distinct digit 0-9, and leading digits cannot be 0. Solve it and give the digit for each letter.",
    "Tuesday boy": "A family has two children. Given that at least one of them is a boy born on Tuesday, what is the probability that both children are boys? (Assume boy/girl equally likely, and uniform over 7 weekdays)",
    "Digit reversal": "The digits of a two-digit number sum to 11. Reversing the digits gives a number that is 27 less than the original. Find the number. Show your work.",
}

def clean(text):
    return re.sub(r"\s+", " ", (text or "")).strip()

def main():
    print("== Korean hard battery (temp=0, mt=1024) ==")
    for name, q in KO:
        try:
            d, dt = chat(q)
            c = d["choices"][0]["message"].get("content") or ""
            fin = d["choices"][0]["finish_reason"]
            print(f"\n[{name}] fin={fin} tok={d['usage']['completion_tokens']} {dt:.1f}s")
            print("  " + clean(c)[:350])
        except Exception as e:
            print(f"\n[{name}] FAIL: {type(e).__name__} {str(e)[:80]}")
    print("\n== English comparison (hard 3) ==")
    for name, q in EN_HARD.items():
        try:
            d, dt = chat(q)
            c = d["choices"][0]["message"].get("content") or ""
            fin = d["choices"][0]["finish_reason"]
            print(f"\n[EN:{name}] fin={fin} tok={d['usage']['completion_tokens']} {dt:.1f}s")
            print("  " + clean(c)[:350])
        except Exception as e:
            print(f"\n[EN:{name}] FAIL: {type(e).__name__} {str(e)[:80]}")

if __name__ == "__main__":
    main()