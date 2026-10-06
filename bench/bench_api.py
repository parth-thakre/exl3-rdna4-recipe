"""Engine-neutral speed bench over an OpenAI-compatible endpoint (TabbyAPI or llama-server).
Tokens are counted with the Qwen3.8 tokenizer on the returned text, not taken from each server's own stats.
usage: bench_api.py NAME [BASE_URL] [API_KEY]   -> appends one JSON line to logs/bench_api.jsonl
BASE_URL and API_KEY default to the local TabbyAPI (common.py)."""
import json, os, random, sys, time, urllib.request
from tokenizers import Tokenizer
from common import BASE_URL, TOKENIZER, api_key, auth_headers, log_path

name = sys.argv[1]
base = (sys.argv[2] if len(sys.argv) > 2 else BASE_URL).rstrip("/")
key = sys.argv[3] if len(sys.argv) > 3 else (api_key() if len(sys.argv) <= 2 else None)
tok = Tokenizer.from_file(TOKENIZER)

def stream(content, max_tokens, sampling):
    body = {"model": "x", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "stream": True, "enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False}, **sampling}
    headers = auth_headers(key or "")
    req = urllib.request.Request(f"{base}/chat/completions", json.dumps(body).encode(), headers)
    t0 = time.time(); t_first = None; text = ""
    with urllib.request.urlopen(req, timeout = 3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            ch = json.loads(line[5:])
            if not ch.get("choices"): continue
            d = ch["choices"][0].get("delta", {})
            piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
            if piece:
                if t_first is None: t_first = time.time()
                text += piece
    t_end = time.time()
    n = len(tok.encode(text, add_special_tokens = False).ids)
    return {"ttft_s": round(t_first - t0, 3), "gen_tokens": n,
            "decode_tps": round((n - 1) / (t_end - t_first), 1) if n > 1 else None}, text

def filler(n_sections, seed):
    rnd = random.Random(seed)
    words = "the river city market old north stone bridge winter garden light road quiet harbor field tower".split()
    return "\n".join(f"Section {i}: " + " ".join(rnd.choice(words) for _ in range(20)) + "." for i in range(n_sections))

greedy = {"temperature": 0}
sampled = {"temperature": 0.7, "top_p": 0.95, "top_k": 20}
res = {"name": name}
stream("Say hello.", 8, greedy)   # warmup
res["code_greedy"], _ = stream("Write a Python class implementing an LRU cache with get, put and delete, using a dict and a "
                               "doubly linked list, with type hints and docstrings. Then write pytest tests for it.", 600, greedy)
res["prose_sampled"], _ = stream("Write a detailed, engaging essay about the history of the printing press and its effect "
                                 "on European society.", 600, sampled)
seed = int(time.time())   # fresh text each run so no prefix cache can help
for label, sections in (("prefill_8k", 280), ("prefill_30k", 1050)):
    content = filler(sections, seed + sections) + "\n\nHow many sections are there? Answer with a number."
    n_prompt = len(tok.encode(content).ids)
    r, _ = stream(content, 1, greedy)
    res[label] = {"prompt_tokens": n_prompt, "ttft_s": r["ttft_s"], "prefill_tps": round(n_prompt / r["ttft_s"])}
print(json.dumps(res))
with open(log_path("bench_api.jsonl"), "a") as f:
    f.write(json.dumps(res) + "\n")
