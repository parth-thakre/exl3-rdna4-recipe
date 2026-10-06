"""Paths and server settings shared by the bench scripts.

Environment overrides:
  TABBY_URL      the running server's OpenAI-compatible base URL (default http://127.0.0.1:8096/v1)
  TABBY_API_KEY  API key (default: the first api_key in $TABBY_DIR/api_tokens.yml)
  TABBY_TREE     TabbyAPI checkout (default: tabbyAPI/ in the repo root; TABBY_DIR works too)
  MODEL_DIR      main model directory (default models/Qwen3.8-27B-EXL3-3.0bpw)
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_URL = os.environ.get("TABBY_URL", "http://127.0.0.1:8096/v1").rstrip("/")
TABBY_DIR = os.environ.get("TABBY_TREE") or os.environ.get("TABBY_DIR") or os.path.join(ROOT, "tabbyAPI")
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(ROOT, "models", "Qwen3.8-27B-EXL3-3.0bpw"))
TOKENIZER = os.path.join(MODEL_DIR, "tokenizer.json")
LOGS = os.path.join(ROOT, "logs")
GPQA_DIR = os.path.join(ROOT, "gpqa")

def path(*parts):
    return os.path.join(ROOT, *parts)

def log_path(name):
    os.makedirs(LOGS, exist_ok = True)
    return os.path.join(LOGS, name)

def api_key():
    key = os.environ.get("TABBY_API_KEY")
    if key:
        return key
    tokens = os.path.join(TABBY_DIR, "api_tokens.yml")
    if not os.path.exists(tokens):
        return None
    import yaml
    with open(tokens) as f:
        data = yaml.safe_load(f) or {}
    value = data.get("api_key")
    # api_key is a single key or a list of keys; use the first
    if isinstance(value, list):
        value = value[0] if value else None
    return str(value) if value else None

def auth_headers(key = None):
    key = key if key is not None else api_key()
    return {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}

def append_jsonl(path_, record):
    import json
    with open(path_, "a") as f:
        f.write(json.dumps(record) + "\n")
