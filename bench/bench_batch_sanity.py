"""Batched-output report (loads the model directly: stop TabbyAPI first).

Runs N different prompts concurrently at bs N through the Generator (DFlash2 draft, greedy) and compares each job's
output with the same prompt run alone at bs1. It is a diagnostic report plus a few hard checks; it does not try to
judge whether a divergent output is "garbage" or "cross-talk" (heuristics for that kept giving false passes and
false fails). Read the texts in --out for any job that isn't identical to its reference.

Hard failures (nonzero exit):
  - a generator error event                       (exit 2)
  - a job that ends without an end-of-stream result, or with no tokens, references included   (exit 2)
  - NO-OVERLAP: fewer than N jobs ever streamed in the same iterate() call, so no real batching happened   (exit 1)

Reported per job, not judged:
  - the common-prefix length with its bs1 reference, and whether it is identical to it (the main signal: a
    batched verify isn't guaranteed bit-identical to bs1, but our runs were identical over 400 tokens)
  - the non-ASCII share of the decoded output and its longest run of one repeated token

--out writes everything (outputs, references, per-bsz overlap and verdict, errors) as JSON, also on failure.

If PYTHONPATH is set, exllamav3 must be imported from its first entry (checked), so a specific checkout can be
tested without installing it. Run from the repo root:

  source setup/env.sh; export EXL3_GDN_REPLAY=1
  python bench/bench_batch_sanity.py --bsz 2 4 --out /tmp/batch.json
  PYTHONPATH=exllamav3 python bench/bench_batch_sanity.py --bsz 4
"""
import argparse, json, os, sys, time
from common import MODEL_DIR, path

PROMPTS = [
    "Write a Python LRU cache class with get/put and unit tests. Code only.",
    "Write a Python function to parse ISO-8601 durations, with tests. Code only.",
    "Write a Python trie with insert, search and prefix listing, with tests. Code only.",
    "Write a Python token-bucket rate limiter class with tests. Code only.",
    "Write Dijkstra's shortest-path algorithm in Python with a heap, with tests. Code only.",
    "Write Python functions converting integers to and from Roman numerals, with tests. Code only.",
    "Write a Python Levenshtein edit-distance function with tests. Code only.",
    "Write a Python script that converts a CSV file to JSON lines, with tests. Code only.",
]


class HardFailure(Exception):
    """A generator error or an incomplete/empty job: the run can't be judged"""


def common_prefix(a, b):
    return next((k for k, (p, q) in enumerate(zip(a, b)) if p != q), min(len(a), len(b)))


def max_run(ids):
    best, run = 0, 0
    for k, t in enumerate(ids):
        run = run + 1 if k and t == ids[k - 1] else 1
        best = max(best, run)
    return best


def nonascii_share(text):
    return sum(1 for c in text if ord(c) > 127) / max(1, len(text))


def run_jobs(gen, make_job, indices, sync = lambda: None):
    """Enqueue one job per index and run them to completion. Returns ({index: token ids}, wall seconds, overlap),
    where overlap is the most distinct jobs that got a streaming result (terminal ones included) from one iterate()
    call. Raises HardFailure on a generator error, a job without an end-of-stream result, or a job with no tokens."""
    out = {i: [] for i in indices}
    finished, overlap = {}, 0
    for i in indices:
        gen.enqueue(make_job(i))
    t0 = time.time()
    while gen.num_remaining_jobs():
        seen = set()
        for r in gen.iterate():
            stage = r.get("stage")
            if stage == "error":
                # error results carry the job, not a top-level identifier
                job_id = getattr(r.get("job"), "identifier", None)
                raise HardFailure(f"generator error in job {job_id}: {r.get('error')!r}")
            if stage != "streaming":
                continue
            i = r.get("identifier")
            seen.add(i)
            if r.get("token_ids") is not None:
                out[i] += r["token_ids"].view(-1).tolist()
            if r.get("eos"):
                finished[i] = r.get("eos_reason")
        overlap = max(overlap, len(seen))
    sync()
    wall = time.time() - t0
    missing = [i for i in indices if i not in finished]
    if missing:
        raise HardFailure(f"jobs {missing} ended without an end-of-stream result")
    empty = [i for i in indices if not out[i]]
    if empty:
        raise HardFailure(f"jobs {empty} produced no tokens")
    return out, wall, overlap


def _main_loop(run, decode, bsz_list, max_bsz):
    """Run the bs1 references and each batch size. Returns (failures, report); on a HardFailure the exception
    carries the partial (failures, report) as .partial."""
    failures, report = [], {"references": {}, "batches": []}
    try:
        run([0])   # warmup: autotune, graph capture
        ref, ref_wall = {}, 0.0
        for i in range(max_bsz):
            o, w, _ = run([i])   # raises unless the reference completes
            ref.update(o)
            ref_wall += w
        ref_text = {i: decode(t) for i, t in ref.items()}
        report["references"] = {i: {"tokens": len(ref[i]), "text": ref_text[i]} for i in ref}
        n_ref = sum(len(v) for v in ref.values())
        print(f"bs1 references: {max_bsz} prompts, {n_ref} tokens, {n_ref / ref_wall:.1f} tok/s one at a time")

        for n in bsz_list:
            idx = list(range(n))
            out, wall, overlap = run(idx)
            total = sum(len(out[i]) for i in idx)
            verdict = "ok" if overlap >= n else f"NO-OVERLAP (at most {overlap} of {n} jobs streamed together)"
            if overlap < n:
                failures.append(f"bs{n}: {verdict}")
            batch = {"bsz": n, "tokens": total, "wall_s": round(wall, 3), "tok_s": round(total / wall, 1),
                     "overlap": overlap, "verdict": verdict, "jobs": []}
            print(f"\nbs {n}: {total} tokens in {wall:.1f}s = {total / wall:.1f} tok/s combined (generator-only); "
                  f"overlap {overlap}/{n}: {verdict}")
            print(f"  {'job':>3s} {'tokens':>11s} {'prefix':>6s} {'identical':>9s} {'nonascii':>8s} {'run':>4s}  prompt")
            for i in idx:
                o, r = out[i], ref[i]
                text = decode(o)
                pre = common_prefix(o, r)
                same = o == r
                na, rl = nonascii_share(text), max_run(o)
                print(f"  {i:3d} {len(o):4d}/{len(r):<4d}  {pre:6d} {str(same):>9s} {na:8.3f} {rl:4d}  {PROMPTS[i][:60]}")
                batch["jobs"].append({"job": i, "tokens": len(o), "ref_tokens": len(r), "common_prefix": pre,
                                      "identical": same, "nonascii": round(na, 4), "max_run": rl, "text": text})
            k = sum(j["identical"] for j in batch["jobs"])
            print(f"bs{n}: {k}/{n} jobs identical to their bs1 reference "
                  f"(min common prefix {min(j['common_prefix'] for j in batch['jobs'])})")
            report["batches"].append(batch)
    except HardFailure as e:
        e.partial = (failures, report)
        raise
    return failures, report


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default = MODEL_DIR)
    ap.add_argument("--draft", default = path("models", "Qwen3.8-27B-DFlash2-EXL3-3.0bpw"))
    ap.add_argument("--bsz", type = int, nargs = "+", default = [2, 4])
    ap.add_argument("--tokens", type = int, default = 400)
    ap.add_argument("--draft-tokens", type = int, default = 7)
    ap.add_argument("--cache-tokens", type = int, default = 32768)
    ap.add_argument("--out", default = None, help = "write the full report as JSON (also on failure)")
    args = ap.parse_args()
    max_bsz = max(args.bsz)
    if max_bsz > len(PROMPTS):
        sys.exit(f"at most {len(PROMPTS)} concurrent prompts")

    result = {"args": vars(args), "verdict": None, "failures": [], "errors": [], "references": {}, "batches": []}
    code = 0
    try:
        os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
        import torch
        import exllamav3
        pp = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        loaded = os.path.realpath(exllamav3.__file__)
        if pp and not loaded.startswith(os.path.realpath(pp[0]) + os.sep):
            raise HardFailure(f"exllamav3 imported from {loaded}, not from the first PYTHONPATH entry {pp[:1]}")
        print(f"exllamav3 from {os.path.dirname(loaded)}; EXL3_DRAFT_ROW_BUDGET="
              f"{os.environ.get('EXL3_DRAFT_ROW_BUDGET', 'unset')}")
        from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ArgmaxSampler
        from exllamav3.cache import CacheLayer_quant

        draft = Model.from_config(Config.from_directory(args.draft))
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

        def make_job(i):
            ids = tok.encode(f"<|im_start|>user\n{PROMPTS[i]}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                             encode_special_tokens = True)
            return Job(input_ids = ids, max_new_tokens = args.tokens, sampler = ArgmaxSampler(), identifier = i)

        def decode(t):
            return tok.decode(torch.tensor(t, dtype = torch.long)) if t else ""

        with torch.inference_mode():
            failures, report = _main_loop(lambda idx: run_jobs(gen, make_job, idx, torch.cuda.synchronize),
                                          decode, args.bsz, max_bsz)
        result.update(report, failures = failures)
        if failures:
            code = 1
    except HardFailure as e:
        failures, report = getattr(e, "partial", ([], {}))
        result.update(report, failures = failures)
        result["errors"].append(str(e))
        code = 2
    except Exception as e:   # model load errors, OOM, ...: still write the report
        result["errors"].append(f"{type(e).__name__}: {e}")
        code = 2

    result["verdict"] = "PASS" if code == 0 else "FAIL"
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent = 1)
        print(f"\nwrote {args.out}")
    if code == 0:
        print("\nPASS: no generator errors, every job finished, every batch size really batched "
              "(see the identical / common-prefix columns above)")
    else:
        print("\nFAIL: " + "; ".join(result["failures"] + result["errors"]))
    sys.stdout.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
