"""Paths and server settings shared by the bench scripts.

Environment overrides:
  TABBY_URL      OpenAI-compatible base URL (default http://127.0.0.1:8096/v1)
  TABBY_API_KEY  API key (default: read from tabbyAPI/api_tokens.yml)
  MODEL_DIR      main model directory (default models/Qwen3.8-27B-EXL3-3.0bpw)
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_URL = os.environ.get("TABBY_URL", "http://127.0.0.1:8096/v1").rstrip("/")
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
    tokens = os.path.join(ROOT, "tabbyAPI", "api_tokens.yml")
    if os.path.exists(tokens):
        # api_key is either a single value or a YAML list; take the first key
        lines = [l.strip() for l in open(tokens) if l.strip() and not l.strip().startswith("#")]
        for i, line in enumerate(lines):
            if line.startswith("api_key:"):
                value = line.split(":", 1)[1].strip()
                if not value and i + 1 < len(lines) and lines[i + 1].startswith("- "):
                    value = lines[i + 1][2:].strip()
                return value.strip("'\"") or None
    return None

def auth_headers(key = None):
    key = key if key is not None else api_key()
    return {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
