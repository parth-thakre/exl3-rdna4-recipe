"""Long-context check against TabbyAPI: bury a code in N filler sections, ask for it back.
usage: needle_test.py [N_SECTIONS]   (default 1400, about 40k tokens). Server URL/key: see common.py."""
import json, random, sys, time, urllib.request
from common import BASE_URL, auth_headers
n_sections = int(sys.argv[1]) if len(sys.argv) > 1 else 1400
random.seed(1)
words = "the river city market old north stone bridge winter garden light road quiet harbor field tower".split()
paras = [f"Section {i}: " + " ".join(random.choice(words) for _ in range(20)) + "." for i in range(n_sections)]
paras.insert(n_sections // 2, "Important note: the secret vault code is 7341-PELICAN-92.")
q = "\n".join(paras) + "\n\nQuestion: What is the secret vault code mentioned in the document? Reply with just the code."
body = {"model": "x", "messages": [{"role": "user", "content": q}], "max_tokens": 40, "temperature": 0, "enable_thinking": False}
req = urllib.request.Request(f"{BASE_URL}/chat/completions", json.dumps(body).encode(), auth_headers())
t = time.time()
r = json.load(urllib.request.urlopen(req, timeout = 1800))
print(f"{time.time() - t:.1f}s ->", r["choices"][0]["message"]["content"])
