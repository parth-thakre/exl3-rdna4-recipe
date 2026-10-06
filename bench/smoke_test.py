"""Smoke test: load an EXL3 model on the RX 9070 XT, generate, report prefill/decode speed.
Run with setup/env.sh sourced and no server holding the GPU. usage: smoke_test.py [--draft DIR | --mtp] [--long]"""
import argparse, gc, os, sys, time
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")   # 9070 XT only; hide the gfx1036 iGPU
import torch
from common import MODEL_DIR
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ArgmaxSampler

ap = argparse.ArgumentParser()
ap.add_argument("--model", default = MODEL_DIR)
ap.add_argument("--draft", default = None)
ap.add_argument("--mtp", action = "store_true", help = "draft with the model's own MTP head")
ap.add_argument("--long", action = "store_true", help = "code-writing prompt for a longer, more realistic decode")
ap.add_argument("--cache", type = int, default = 8192)
ap.add_argument("--max_new", type = int, default = 256)
args = ap.parse_args()

def clean_exit(code):
    # CarouselAether/rocm_exl3#2: on gfx1201 + Fedora 44 the HSA exit handler can segfault and leave the GPU
    # pinned at 100%; tearing down explicitly and skipping interpreter shutdown avoids it
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(1.5)
    sys.stdout.flush()
    os._exit(code)

print(torch.cuda.get_device_name(0), torch.version.hip)
t0 = time.time()
draft_model = draft_cache = None
if args.draft or args.mtp:
    draft_config = Config.from_directory(args.model if args.mtp else args.draft)
    draft_model = Model.from_config(draft_config, component = "mtp" if args.mtp else "text")
    draft_cache = Cache(draft_model, max_num_tokens = args.cache)
    draft_model.load(progressbar = True)
config = Config.from_directory(args.model)
model = Model.from_config(config)
cache = Cache(
    model,
    max_num_tokens = args.cache,
    max_batch_size = 1,
    max_history = draft_model.caps.get("default_draft_size", 4) if draft_model else 0,
)
model.load(progressbar = True)
tokenizer = Tokenizer.from_config(config)
generator = Generator(model = model, cache = cache, tokenizer = tokenizer,
                      draft_model = draft_model, draft_cache = draft_cache)
print(f"loaded in {time.time() - t0:.1f}s, VRAM allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
      f"reserved {torch.cuda.memory_reserved() / 1e9:.2f} GB")

question = ("Write a Python class implementing an LRU cache with get, put and delete, using a dict and a doubly "
            "linked list, with type hints and docstrings. Then write pytest tests for it.") if args.long else \
           ("What is the capital of Australia, and why was it chosen over Sydney and Melbourne? "
          "Answer in three sentences.")
prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ids = tokenizer.encode(prompt, encode_special_tokens = True)
stop = [tokenizer.single_id("<|im_end|>")]

def run(max_new):
    job = Job(input_ids = ids, max_new_tokens = max_new, stop_conditions = stop, sampler = ArgmaxSampler())
    generator.enqueue(job)
    text, n, t_first = "", 0, None
    t_start = time.time()
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r.get("stage") == "streaming":
                chunk = r.get("text", "")
                if chunk and t_first is None:
                    t_first = time.time()
                text += chunk
                n += len(r.get("token_ids", torch.empty(0)).flatten()) if r.get("token_ids") is not None else 0
    t_end = time.time()
    return text, n, t_start, t_first, t_end

run(16)   # warmup: autotune and graph capture
text, n, t_start, t_first, t_end = run(args.max_new)
print("-" * 60)
print(text.strip())
print("-" * 60)
dec = (n - 1) / (t_end - t_first) if n > 1 and t_first else float("nan")
print(f"prompt {ids.shape[-1]} tok, TTFT {(t_first - t_start) * 1000:.0f} ms, generated {n} tok, decode {dec:.1f} tok/s")
clean_exit(0)
