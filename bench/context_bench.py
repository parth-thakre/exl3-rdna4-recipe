"""Speed and recall at increasing context depth, through the running TabbyAPI.
Each depth gets a fresh filler document (no prefix-cache help) with a code buried in the middle; the model must
state the code, then write ~400 tokens of Python. Reports prefill tok/s, decode tok/s at that depth, recall.
usage: context_bench.py LABEL DEPTH_K [DEPTH_K ...]   e.g. context_bench.py q4-64k 8 32 60
appends JSON lines to logs/context_bench.jsonl. Server URL/key: see common.py.
Exits 1 if a request fails or returns no text (that depth is logged with an "error" field)."""
import json, os, random, sys, time, urllib.request
from tokenizers import Tokenizer
from common import BASE_URL, TOKENIZER, append_jsonl, auth_headers, log_path

URL = f"{BASE_URL}/chat/completions"
HEADERS = auth_headers()
tok = Tokenizer.from_file(TOKENIZER)
OUT = log_path("context_bench.jsonl")
label = sys.argv[1]
WORDS = "the river city market old north stone bridge winter garden light road quiet harbor field tower".split()

def build(depth_tokens, seed):
    rnd = random.Random(seed)
    code = f"{rnd.randint(1000, 9999)}-{rnd.choice(['PELICAN', 'ORCHID', 'GRANITE', 'FALCON'])}-{rnd.randint(10, 99)}"
    ask = ("\n\nFirst line of your reply: the secret vault code from the document, nothing else. Then write a Python "
           "class implementing an LRU cache with get, put and delete using a dict and a doubly linked list, with "
           "type hints and docstrings.")
    secs = [f"Section {i}: " + " ".join(rnd.choice(WORDS) for _ in range(20)) + "." for i in range(300)]
    per_sec = len(tok.encode("\n".join(secs)).ids) / len(secs)
    n = max(10, int(depth_tokens / per_sec))
    secs = [f"Section {i}: " + " ".join(rnd.choice(WORDS) for _ in range(20)) + "." for i in range(n)]
    secs.insert(len(secs) // 2, f"Important note: the secret vault code is {code}.")
    text = "\n".join(secs) + ask
    return text, code, len(tok.encode(text).ids)

for dk in sys.argv[2:]:
    text, code, n_prompt = build(int(float(dk) * 1024) - 120, seed = int(time.time()))
    body = {"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": 450, "temperature": 0,
            "stream": True, "enable_thinking": False}
    req = urllib.request.Request(URL, json.dumps(body).encode(), HEADERS)
    t0 = time.time(); t_first = None; out = ""
    try:
        with urllib.request.urlopen(req, timeout = 3600) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]": continue
                ch = json.loads(line[5:])
                if not ch.get("choices"): continue
                piece = ch["choices"][0].get("delta", {}).get("content") or ""
                if piece:
                    if t_first is None: t_first = time.time()
                    out += piece
    except Exception as e:
        res = {"label": label, "depth_k": dk, "prompt_tokens": n_prompt, "error": str(e)[:200]}
        print(json.dumps(res), flush = True)
        append_jsonl(OUT, res)
        sys.exit(1)
    t_end = time.time()
    if t_first is None:
        res = {"label": label, "depth_k": dk, "prompt_tokens": n_prompt, "error": "empty completion"}
        print(json.dumps(res), flush = True)
        append_jsonl(OUT, res)
        sys.exit(1)
    n_gen = len(tok.encode(out, add_special_tokens = False).ids)
    res = {"label": label, "depth_k": dk, "prompt_tokens": n_prompt,
           "prefill_s": round(t_first - t0, 1), "prefill_tps": round(n_prompt / (t_first - t0)),
           "decode_tps": round((n_gen - 1) / (t_end - t_first), 1), "gen_tokens": n_gen,
           "recall": code in out.split("\n")[0] or code in out[:200]}
    print(json.dumps(res), flush = True)
    append_jsonl(OUT, res)
