"""Can a system prompt stop Qwen3.8's GPQA overthinking? Same questions, same sampling, one variant per run.
usage: overthink_test.py VARIANT   -> gpqa/overthink_<VARIANT>.jsonl (resumable)
VARIANT: xhigh-antispiral, medium, medium-antispiral, low. Server URL/key: see common.py."""
import csv, json, os, random, re, sys, time, urllib.request
from common import BASE_URL, GPQA_DIR, auth_headers

URL = f"{BASE_URL}/chat/completions"
# Question indices in gpqa_diamond.csv: 7 that hit the 30k cap at xhigh, then 4 it got right only after 60k+ characters
# of reasoning
QUESTIONS = [8, 12, 15, 27, 30, 32, 36, 42, 1, 21, 23]

ANTI_SPIRAL = ("Think efficiently. Work through the problem once, carefully. If you notice yourself re-deriving the "
               "same result, second-guessing a conclusion you already checked, or cycling between options, stop and "
               "commit to the best-supported answer. A confident best answer beats an unfinished analysis.")
VARIANTS = {
    "xhigh-antispiral": {"reasoning_effort": "xhigh", "system": ANTI_SPIRAL},
    "medium":           {"reasoning_effort": "medium"},
    "medium-antispiral": {"reasoning_effort": "medium", "system": ANTI_SPIRAL},
    "low":              {"reasoning_effort": "low"},
}
name = sys.argv[1]
v = VARIANTS[name]
out_path = os.path.join(GPQA_DIR, f"overthink_{name}.jsonl")
done = {json.loads(l)["idx"] for l in open(out_path)} if os.path.exists(out_path) else set()
rows = list(csv.DictReader(open(os.path.join(GPQA_DIR, "gpqa_diamond.csv"))))

# identical to gpqa_eval.py
TEMPLATE = ("Answer the following multiple choice question. The last line of your response should be of the following "
            "format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n"
            "{q}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}")

for idx in QUESTIONS:
    if idx in done:
        continue
    row = rows[idx]
    choices = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    random.Random(idx).shuffle(choices)
    gold = "ABCD"[choices.index(row["Correct Answer"])]
    messages = ([{"role": "system", "content": v["system"]}] if "system" in v else []) + \
               [{"role": "user", "content": TEMPLATE.format(q = row["Question"].strip(), **{l: c.strip() for l, c in zip("ABCD", choices)})}]
    body = {"model": "x", "messages": messages, "max_tokens": 30000, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "enable_thinking": True, "reasoning_effort": v["reasoning_effort"]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), auth_headers())
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout = 7200))
    dt = time.time() - t
    msg = r["choices"][0]["message"]
    content, reasoning = msg.get("content") or "", msg.get("reasoning_content") or ""
    m = re.findall(r"Answer\s*:\s*\**\s*\(?([ABCD])\b", content) or re.findall(r"Answer\s*:\s*\**\s*\(?([ABCD])\b", reasoning)
    pred = m[-1] if m else None
    rec = {"idx": idx, "gold": gold, "pred": pred, "correct": pred == gold, "seconds": round(dt, 1),
           "finish": r["choices"][0].get("finish_reason"), "reasoning_chars": len(reasoning)}
    with open(out_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{name} q{idx}: gold {gold} pred {pred} {'✓' if pred == gold else '✗'} {dt:.0f}s "
          f"{len(reasoning)} reasoning chars {rec['finish']}", flush = True)
