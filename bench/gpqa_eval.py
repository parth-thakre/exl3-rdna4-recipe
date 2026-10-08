"""GPQA Diamond (198 questions) over an OpenAI-compatible endpoint, thinking on. Resumable: one JSON line per question.
usage: gpqa_eval.py NAME [BASE_URL] [API_KEY]   -> gpqa/results_<NAME>.jsonl
Needs gpqa/gpqa_diamond.csv (see bench/README.md). BASE_URL and API_KEY default to the local TabbyAPI (common.py)."""
import concurrent.futures, csv, json, os, random, re, sys, threading, time, urllib.request
from common import BASE_URL, GPQA_DIR, api_key, auth_headers

name = sys.argv[1]
base = (sys.argv[2] if len(sys.argv) > 2 else BASE_URL).rstrip("/")
key = sys.argv[3] if len(sys.argv) > 3 else (api_key() if len(sys.argv) <= 2 else None)
out_path = os.path.join(GPQA_DIR, f"results_{name}.jsonl")
if os.path.exists(os.path.join(GPQA_DIR, f"skip_{name}")):   # dropped from the comparison
    raise SystemExit(f"{name}: skipped (gpqa/skip_{name} exists)")
rows = list(csv.DictReader(open(os.path.join(GPQA_DIR, "gpqa_diamond.csv"))))
# Same first N questions for every model, so the comparison stays paired (GPQA_LIMIT=198 for the full set)
rows = rows[:int(os.environ.get("GPQA_LIMIT", "50"))]
def load_results():
    # Skips a torn last line left by a crash mid-write; that question is simply asked again
    recs = []
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return recs

if os.path.exists(out_path) and os.path.getsize(out_path):
    with open(out_path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) != b"\n":   # torn last line: terminate it so the next record starts on its own line
            f.write(b"\n")
done = {r["idx"] for r in load_results()}
if len(done) == len(rows):
    print(f"all {len(rows)} questions already done")

# simple-evals prompt; choices shuffled with a per-question seed so every model sees the same order
TEMPLATE = ("Answer the following multiple choice question. The last line of your response should be of the following "
            "format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n"
            "{q}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}")

# Optional overthinking fix: GPQA_EFFORT=medium and GPQA_SYSTEM=antispiral (see overthink_test.py)
EFFORT = os.environ.get("GPQA_EFFORT")
SYSTEM = {"antispiral": (
    "Think efficiently. Work through the problem once, carefully. If you notice yourself re-deriving the "
    "same result, second-guessing a conclusion you already checked, or cycling between options, stop and "
    "commit to the best-supported answer. A confident best answer beats an unfinished analysis.")}.get(os.environ.get("GPQA_SYSTEM"))

def ask(prompt):
    messages = ([{"role": "system", "content": SYSTEM}] if SYSTEM else []) + [{"role": "user", "content": prompt}]
    body = {"model": "x", "messages": messages, "max_tokens": int(os.environ.get("GPQA_MAX_TOKENS", "30000")),
            "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "enable_thinking": True, "chat_template_kwargs": {"enable_thinking": True}}
    if EFFORT:
        body["reasoning_effort"] = EFFORT
    headers = auth_headers(key or "")
    req = urllib.request.Request(f"{base}/chat/completions", json.dumps(body).encode(), headers)
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout = 7200))
    return r, time.time() - t

write_lock = threading.Lock()

def run_one(idx):
    row = rows[idx]
    choices = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    random.Random(idx).shuffle(choices)
    gold = "ABCD"[choices.index(row["Correct Answer"])]
    prompt = TEMPLATE.format(q = row["Question"].strip(), **{l: c.strip() for l, c in zip("ABCD", choices)})
    for attempt in range(3):
        try:
            r, dt = ask(prompt)
            break
        except Exception as e:
            print(f"[{idx}] request failed ({e}), retrying", flush = True)
            time.sleep(10)
    else:
        print(f"[{idx}] giving up; rerun to retry it", flush = True)
        return False
    msg = r["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    m = re.findall(r"Answer\s*:\s*\**\s*\(?([ABCD])\b", content) or re.findall(r"Answer\s*:\s*\**\s*\(?([ABCD])\b", reasoning)
    pred = m[-1] if m else None
    rec = {"idx": idx, "gold": gold, "pred": pred, "correct": pred == gold, "seconds": round(dt, 1),
           "finish": r["choices"][0].get("finish_reason"), "content_chars": len(content), "reasoning_chars": len(reasoning),
           "content_tail": content[-300:]}
    with write_lock:
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        done.add(idx)
        recs = load_results()
        acc = sum(x["correct"] for x in recs) / len(recs)
        print(f"[{len(recs)}/{len(rows)}] q{idx} gold {gold} pred {pred} {'✓' if pred == gold else '✗'} {dt:.0f}s  running acc {acc:.1%}", flush = True)
    return True

# GPQA_WORKERS > 1 keeps several questions in flight; the server needs max_batch_size to match.
# The 50-question results ran one at a time, the 198-question passes three and the xhigh rerun two.
workers = int(os.environ.get("GPQA_WORKERS", "1"))
# GPQA_IDS=1,12,... asks only those questions (same idx, so the same shuffled choices as the full run).
# Set but empty is an error rather than "all questions", so a failed ID selection can't start a full run.
if "GPQA_IDS" in os.environ:
    ids = sorted({int(x) for x in os.environ["GPQA_IDS"].split(",") if x.strip()})
    if not ids or any(not 0 <= i < len(rows) for i in ids):
        sys.exit(f"GPQA_IDS must list question numbers below GPQA_LIMIT ({len(rows)})")
else:
    ids = range(len(rows))
with concurrent.futures.ThreadPoolExecutor(workers) as pool:
    ok = list(pool.map(run_one, [i for i in ids if i not in done]))
if not all(ok):
    sys.exit(f"{ok.count(False)} question(s) failed after 3 attempts; rerun to retry them")
