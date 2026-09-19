#!/usr/bin/env python3
import json, os, re, time, urllib.request

BASE = os.environ.get("BONSAI_BASE", "http://localhost:8082/v1/chat/completions")

FACTS = {
    "민수": "name", "파리": "city", "코코": "dog", "지수": "girlfriend",
    "3만": "budget", "바닐라": "icecream", "7": "lucky_number", "개발자": "job",
    "베를린": "trip", "수영": "sport",
}

TURNS = [
    ("소개", "안녕, 내 이름은 민수야. 파리에 살고 있고, 반려견 코코를 키워. 여자친구는 지수야. 소개한 김에 정리하자면 - 쇼핑 예산은 3만원 정도, 아이스크림은 바닐라를 좋아하고 초콜릿은 싫어해. 내가 가장 좋아하는 숫자는 7이고, 직업은 개발자야. 올해 베를린 여행을 계획 중이고, 주말엔 수영을 해."),
    ("빈칸1", "오, 개발자라면 어떤 언어를 주로 써?"),
    ("빈칸2", "그렇구나. 나도 요즘 새 프로젝트 준비 중이야."),
    ("빈칸3", "코코는 산책을 엄청 좋아해서 매일 아침 6시에 깨워서 데려가."),
    ("빈칸4", "지수가 생일 선물로 뭘 사줄까 물어보더라. 넌 뭐가 좋겠어? 참고로 난 작은 선물을 좋아해."),
    ("빈칸5", "그럼 이번 주말에 베를린 여행 준비하면서 수영장도 다녀올까 생각 중이야."),
    ("빈칸6", "요즘 날씨가 좋아서 그런지 기분이 좋아."),
    ("빈칸7", "개발하다 보면 커피를 자주 마시게 되더라."),
    ("회상 퀴즈", "아까 내가 말한 것들 중에 질문 몇 개 할게. 하나씩 답해줘. 1) 내 반려견 이름은? 2) 내가 사는 도시는? 3) 내 여자친구는? 4) 내 쇼핑 예산은? 5) 내가 좋아하는 아이스크림은? 6) 내가 싫어하는 아이스크림은? 7) 내 직업은? 8) 올해 여행 계획 도시는? 9) 내가 하는 운동은? 10) 내가 가장 좋아하는 숫자는?"),
    ("빈칸8", "아 맞다, 지수가 내일 여기 오기로 했어."),
    ("빈칸9", "그래서 내일 지수랑 파리에서 데이트 계획을 짜야 하는데."),
    ("종합 질문", "기억 정리해서 말해줘. 내 예산 3만원 안에서, 내가 좋아하는 아이스크림 가게에 코코는 못 데려가니까, 지수랑만 갈 거야. 내가 가장 좋아하는 숫자 7에 5를 더하면 얼마지? 그리고 데이트 장소로 베를린은 너무 멀지 않아?"),
]

def chat(msgs, mt=300):
    body = {"model": "bonsai", "messages": msgs, "temperature": 0.0, "seed": 42,
            "max_tokens": mt, "stream": False}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    return d, time.time() - t0

def score_answer(text, q_num):
    text = (text or "").strip()
    if q_num in (1, 2, 3, 7, 8, 9) and re.search(r"코코|파리|지수|개발자|베를린|수영", text):
        return True
    if q_num == 4 and ("3만" in text or "30000" in text):
        return True
    if q_num == 5 and "바닐라" in text:
        return True
    if q_num == 6 and ("초콜릿" in text or "초코" in text):
        return True
    if q_num == 10 and re.search(r"\b7\b", text):
        return True
    return False

def main():
    msgs = []
    for label, prompt in TURNS:
        msgs.append({"role": "user", "content": prompt})
        resp, dt = chat(msgs)
        msg = resp["choices"][0]["message"]
        content = msg.get("content") or ""
        msgs.append({"role": "assistant", "content": content})
        tok = resp["usage"]["completion_tokens"]
        prompt_tok = resp["usage"]["prompt_tokens"]
        print(f"[{label}] prompt_tok={prompt_tok:5d} gen_tok={tok:4d} time={dt:5.1f}s fin={resp['choices'][0]['finish_reason']}")
        if label.startswith("회상"):
            ans = re.split(r"(?:^|\n)\s*(?:1[)]|10[)])?\s*\d+[).]", content)
            ans = [a.strip().replace("\n", " ") for a in ans if a.strip()]
            lines = [l for l in content.splitlines() if l.strip()]
            print("   회상 답변 확인:", len(lines), "줄")
            print("   첫 3줄:", " | ".join(lines[:3])[:160])
        elif label.startswith("종합"):
            print("   종합 답변:", re.sub(r"\s+", " ", content)[:220])
        elif label == "빈칸4":
            print("   답변유지:", re.sub(r"\s+", " ", content)[:120])

if __name__ == "__main__":
    main()