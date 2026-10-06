"""Batched-output sanity (loads the model directly: stop TabbyAPI first).  N different prompts run concurrently at bs N through the Generator
(DFlash2 draft, greedy) must each produce what the same prompt produces alone at bs1, up to rounding drift.

Batched and bs1 outputs are not expected to be bit-identical (the split-decode attention splits differently
at another batch size), so this reports each job's common-prefix length with its bs1 reference and fails only
on the two failure shapes that matter:
  - garbage:    non-ASCII share > --max-nonascii or one token repeated more than --max-run times in a row,
                checked on the whole output when its common prefix is short (< --min-prefix tokens) and on the
                part after the divergence point in every case (unless the bs1 reference has the same property
                there), or an empty output
  - cross-talk: job i's text lacks its own prompt's topic but carries another batched prompt's topic that its
                own bs1 reference doesn't (rows of the batch mixed up). CamelCase identifiers (toRoman) are split
                before matching.
It also fails if the generator reports an error, if any job (reference or batched) ends without an end-of-stream
result, or if fewer jobs than requested were ever decoding at the same time (no real batching happened).
"follows" (job i's output matches job j's bs1 output for longer than its own) is reported for information
only. The prompts share boilerplate ("Write a Python ... with tests. Code only."), so two jobs can open with the
same tokens and a short follow is not evidence of cross-talk on its own; it is added to the verdict only when the
topic check fails too.

If PYTHONPATH is set, exllamav3 must be imported from its first entry (checked), so a specific checkout can be
tested without installing it. Run from the repo root:

  source setup/env.sh; export EXL3_GDN_REPLAY=1
  python bench/bench_batch_sanity.py --bsz 2 4
  PYTHONPATH=exllamav3 EXL3_DRAFT_ROW_BUDGET=16 python bench/bench_batch_sanity.py --bsz 4
"""
import argparse, json, os, re, sys, time
from common import MODEL_DIR, path

# Each prompt with a pattern for its topic, matched against the lowercased output. Identifiers glue words
# (LRUCache, TokenBucket, to_roman), so the patterns use letter-only lookbehinds instead of \b, which also keeps
# "trie" from matching inside "retrieve" / "entries"; the topics are mutually distinct
PROMPTS = [
    ("Write a Python LRU cache class with get/put and unit tests. Code only.", r"(?<![a-z])lru|ordereddict"),
    ("Write a Python function to parse ISO-8601 durations, with tests. Code only.", r"iso.?8601|duration"),
    ("Write a Python trie with insert, search and prefix listing, with tests. Code only.", r"(?<![a-z])trie"),
    ("Write a Python token-bucket rate limiter class with tests. Code only.", r"bucket"),
    ("Write Dijkstra's shortest-path algorithm in Python with a heap, with tests. Code only.", r"dijkstra"),
    ("Write Python functions converting integers to and from Roman numerals, with tests. Code only.",
     r"(?<![a-z])roman"),
    ("Write a Python Levenshtein edit-distance function with tests. Code only.", r"levenshtein|edit.?distance"),
    ("Write a Python script that converts a CSV file to JSON lines, with tests. Code only.", r"(?<![a-z])csv"),
]


def topic_text(text):
    """Lowercased text with CamelCase split (toRoman -> to roman, LRUCache -> lru cache), for the topic patterns"""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    return text.lower()


def has_topic(j, text):
    return re.search(PROMPTS[j][1], topic_text(text)) is not None


def common_prefix(a, b):
    return next((k for k, (p, q) in enumerate(zip(a, b)) if p != q), min(len(a), len(b)))


def max_run(ids):
    best, run = 0, 0
    for k, t in enumerate(ids):
        run = run + 1 if k and t == ids[k - 1] else 1
        best = max(best, run)
    return best


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default = MODEL_DIR)
    ap.add_argument("--draft", default = path("models", "Qwen3.8-27B-DFlash2-EXL3-3.0bpw"))
    ap.add_argument("--bsz", type = int, nargs = "+", default = [2, 4])
    ap.add_argument("--tokens", type = int, default = 400)
    ap.add_argument("--draft-tokens", type = int, default = 7)
    ap.add_argument("--cache-tokens", type = int, default = 32768)
    ap.add_argument("--min-prefix", type = int, default = 20)
    ap.add_argument("--max-nonascii", type = float, default = 0.2)
    ap.add_argument("--max-run", type = int, default = 40)
    ap.add_argument("--out", default = None, help = "write all outputs and verdicts as JSON")
    args = ap.parse_args()
    max_bsz = max(args.bsz)
    assert max_bsz <= len(PROMPTS), f"at most {len(PROMPTS)} concurrent prompts"

    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    import torch
    import exllamav3
    pp = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    loaded = os.path.realpath(exllamav3.__file__)
    if pp and not loaded.startswith(os.path.realpath(pp[0]) + os.sep):
        sys.exit(f"FAIL: exllamav3 imported from {loaded}, not from the first PYTHONPATH entry {pp[:1]}")
    print(f"exllamav3 from {os.path.dirname(loaded)}; EXL3_DRAFT_ROW_BUDGET="
          f"{os.environ.get('EXL3_DRAFT_ROW_BUDGET', 'unset')}, EXL3_GEMV_MAX_M={os.environ.get('EXL3_GEMV_MAX_M', 'unset')}")
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ArgmaxSampler
    from exllamav3.cache import CacheLayer_quant

    dcfg = Config.from_directory(args.draft)
    draft = Model.from_config(dcfg)
    kw = dict(layer_type = CacheLayer_quant, k_bits = 4, v_bits = 4, max_batch_size = max_bsz,
              max_history = args.draft_tokens)
    dcache = Cache(draft, max_num_tokens = args.cache_tokens, **kw)
    draft.load()
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = args.cache_tokens, **kw)
    model.load()
    tok = Tokenizer.from_config(config)
    gen = Generator(model = model, cache = cache, draft_model = draft, draft_cache = dcache, tokenizer = tok,
                    max_batch_size = max_bsz, num_draft_tokens = args.draft_tokens,
                    recurrent_cache_size = 512 * 1024**2)

    def ids(p):
        return tok.encode(f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                          encode_special_tokens = True)

    def run(indices):
        """Run the given prompts concurrently. Returns ({index: token ids}, wall seconds, the most jobs that were
        decoding at the same time). Raises on a generator error or a job without an end-of-stream result."""
        out = {i: [] for i in indices}
        for i in indices:
            gen.enqueue(Job(input_ids = ids(PROMPTS[i][0]), max_new_tokens = args.tokens,
                            sampler = ArgmaxSampler(), identifier = i))
        decoding, finished, overlap = set(), {}, 0
        t0 = time.time()
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                stage, i = r.get("stage"), r.get("identifier")
                if stage == "error":
                    raise RuntimeError(f"generator error in job {i}: {r.get('error')!r}")
                if stage != "streaming":
                    continue
                if r.get("token_ids") is not None:
                    out[i] += r["token_ids"].view(-1).tolist()
                if r.get("eos"):
                    finished[i] = r.get("eos_reason")
                    decoding.discard(i)
                elif i not in finished:
                    decoding.add(i)
            overlap = max(overlap, len(decoding))
        torch.cuda.synchronize()
        wall = time.time() - t0
        missing = [i for i in indices if i not in finished]
        if missing:
            raise RuntimeError(f"jobs {missing} ended without an end-of-stream result")
        empty = [i for i in indices if not out[i]]
        if empty:
            raise RuntimeError(f"jobs {empty} produced no tokens")
        return out, wall, overlap

    def text_of(t):
        return tok.decode(torch.tensor(t, dtype = torch.long)) if t else ""

    def garbage(t):
        """(is garbage, non-ASCII share, longest repeat run) for a list of token ids"""
        text = text_of(t)
        na = sum(1 for c in text if ord(c) > 127) / max(1, len(text))
        rl = max_run(t)
        return (na > args.max_nonascii or rl > args.max_run), na, rl

    failures, report, ref = [], [], {}
    try:
        _main_loop(run, garbage, text_of, ref, failures, report, args, max_bsz)
    except RuntimeError as e:
        print(f"\nFAIL: {e}")
        sys.stdout.flush()
        os._exit(2)

    if args.out:
        json.dump(report, open(args.out, "w"), indent = 1)
        print(f"\nwrote {args.out}")
    if failures:
        print(f"\nFAIL: {len(failures)} job(s): " + "; ".join(f"bs{n} job {i}: {v}" for n, i, v in failures))
    else:
        print(f"\nPASS: every batched job coherent and on its own prompt (prefix and follows columns are informational)")
    sys.stdout.flush()
    os._exit(1 if failures else 0)


def _main_loop(run, garbage, text_of, ref, failures, report, args, max_bsz):
    import torch
    with torch.inference_mode():
        run([0])   # warmup: autotune, graph capture
        ref_wall = 0.0
        for i in range(max_bsz):
            o, w, _ = run([i])   # raises unless the reference run completes
            ref.update(o)
            ref_wall += w
        print(f"bs1 references: {max_bsz} prompts, {sum(len(v) for v in ref.values())} tokens, "
              f"{sum(len(v) for v in ref.values()) / ref_wall:.1f} tok/s sequential")

        for n in args.bsz:
            idx = list(range(n))
            out, wall, overlap = run(idx)
            total = sum(len(out[i]) for i in idx)
            print(f"\nbs {n}: {total} tokens in {wall:.1f}s = {total / wall:.1f} tok/s combined; "
                  f"at most {overlap} decoding at once")
            if overlap < n:
                failures.append((n, -1, f"NO-OVERLAP(only {overlap} of {n} jobs were ever decoding together)"))
            print(f"  {'job':>3s} {'tokens':>11s} {'prefix':>6s} {'nonascii':>8s} {'run':>4s} {'own':>4s} {'follows':>7s}  verdict  topic")
            for i in idx:
                o, r = out[i], ref[i]
                text, ref_text = text_of(o), text_of(r)
                pre = common_prefix(o, r)
                bad_full, nonascii, run_len = garbage(o)
                # The divergent part: garbage there counts even after a long common prefix, unless the bs1
                # reference shows the same thing from that point
                bad_tail = len(o) > pre and garbage(o[pre:])[0] and not garbage(r[pre:])[0]
                own = has_topic(i, text)
                # Topics the job's own bs1 reference already mentions aren't evidence of cross-talk
                others = [j for j in idx if j != i and has_topic(j, text) and not has_topic(j, ref_text)]
                # Informational: this job follows another job's bs1 output for longer than its own. Shared prompt
                # boilerplate makes short follows harmless, so it only counts together with a failed topic check
                follows = [j for j in idx if j != i and common_prefix(o, ref[j]) > max(pre, args.min_prefix)]
                verdict = []
                if not o:
                    verdict.append("EMPTY")
                if (pre < args.min_prefix and bad_full) or bad_tail:
                    verdict.append("GARBAGE" if not bad_tail else f"GARBAGE(after token {pre})")
                if not own and others:
                    verdict.append(f"CROSS-TALK(topics {others}, follows {follows})")
                v = ", ".join(verdict) or "ok"
                if verdict:
                    failures.append((n, i, v))
                print(f"  {i:3d} {len(o):4d}/{len(r):<4d}  {pre:6d} {nonascii:8.3f} {run_len:4d} {str(own):>4s} "
                      f"{','.join(map(str, follows)) or '-':>7s}  "
                      f"{v:7s}  {PROMPTS[i][0][:60]}")
                report.append({"bsz": n, "job": i, "prefix": pre, "tokens": len(o), "ref_tokens": len(r),
                               "nonascii": nonascii, "max_run": run_len, "own_topic": own, "other_topics": others,
                               "follows": follows, "verdict": v, "text": text, "ref_text": ref_text})


if __name__ == "__main__":
    main()
