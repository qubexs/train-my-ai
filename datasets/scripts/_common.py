"""Shared helpers for datasets/scripts/ (stdlib only)."""
import json
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPTS_DIR.parent
CONFIG_FILE = DATASETS_DIR / "config" / "dataset.json"


def load_config():
    return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))


def stack_files(cfg=None):
    cfg = cfg or load_config()
    return {name: DATASETS_DIR / info["file"]
            for name, info in cfg["stacks"].items()}


def read_rows(path):
    rows = []
    if not Path(path).exists():
        return rows
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def check_row(r):
    return isinstance(r, dict) and isinstance(r.get("instruction"), str) \
        and isinstance(r.get("output"), str) and r["instruction"].strip() \
        and r["output"].strip()


# str.splitlines() (used by app/store.py, finetune/*) also splits on these
# unicode line separators; json.dumps(ensure_ascii=False) leaves them raw, so
# escape them to keep every JSONL row on exactly one physical line.
_SPLIT_CHARS = {"\u2028": "\\u2028", "\u2029": "\\u2029", "\x85": "\\u0085"}


def dump_row(r):
    s = json.dumps(r, ensure_ascii=False)
    for ch, esc in _SPLIT_CHARS.items():
        s = s.replace(ch, esc)
    return s
