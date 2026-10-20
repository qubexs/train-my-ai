"""HF dataset helper: list -> download -> shred (split combined corpus by stack).

Repo default: myxcoder/programming (dataset). Combined file: all.jsonl.

Usage (from repo root, .venv-gguf has huggingface_hub):
  python datasets/scripts/shred.py list
  python datasets/scripts/shred.py download                 # just all.jsonl
  python datasets/scripts/shred.py download --files python.jsonl,django.jsonl
  python datasets/scripts/shred.py download --all           # whole repo (~1.2GB)
  python datasets/scripts/shred.py shred                     # -> datasets/shred/<stack>/shred-00001.jsonl
  python datasets/scripts/shred.py shred --stacks python,django,flask
  python datasets/scripts/shred.py shred --rows 5000         # max rows per shard
  python datasets/scripts/shred.py all                       # download all.jsonl then shred

Shred rules: a row is assigned to a stack by matching (instruction, output)
against datasets/<stack>.jsonl (config/dataset.json). Each stack gets a folder
(folder name = stack) split into shards of at most --rows rows, named
shred-00001.jsonl, shred-00002.jsonl, ... Rows that match nothing go to
<out>/_unmatched.jsonl. Download lands in datasets/raw/hf/ (git-ignored).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATASETS_DIR, load_config, read_rows, dump_row

ROOT = DATASETS_DIR.parent
RAW = DATASETS_DIR / "raw"
HF_DIR = RAW / "hf"
DEFAULT_REPO = "myxcoder/programming"
DEFAULT_SOURCE = "all.jsonl"


def _key(r):
    return (str(r.get("instruction", "")).strip(),
            str(r.get("output", "")).strip())


def _hf():
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub tiada. Guna .venv-gguf atau: pip install huggingface_hub")
    return HfApi, hf_hub_download


def cmd_list(repo):
    HfApi, _ = _hf()
    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
             if f.lower().endswith(".jsonl")]
    for f in sorted(files):
        print(f)
    print(f"# {len(files)} jsonl files in {repo}", file=sys.stderr)


def cmd_download(repo, files, everything):
    _, hf_hub_download = _hf()
    if everything:
        from huggingface_hub import snapshot_download
        p = snapshot_download(repo, repo_type="dataset", local_dir=str(HF_DIR),
                              allow_patterns="*.jsonl")
        print(f"downloaded repo -> {p}")
        return
    if not files:
        files = [DEFAULT_SOURCE]
    for f in files:
        p = hf_hub_download(repo, f, repo_type="dataset", local_dir=str(HF_DIR))
        print(f"downloaded {f} -> {p}")


def build_map(stacks, allow_hf, repo):
    cfg = load_config()
    names = stacks or list(cfg["stacks"])
    _, hf_hub_download = _hf() if allow_hf else (None, None)
    key2stack = {}
    missing = []
    for name in names:
        info = cfg["stacks"].get(name)
        if not info:
            missing.append(name)
            continue
        path = DATASETS_DIR / info["file"]
        if not path.exists() and allow_hf:
            try:
                path = Path(hf_hub_download(repo, info["file"], repo_type="dataset",
                                            local_dir=str(HF_DIR)))
            except Exception as e:  # noqa: BLE001
                missing.append(f"{name} ({e})")
                continue
        if not path.exists():
            missing.append(name)
            continue
        for r in read_rows(path):
            key2stack[_key(r)] = name
    if missing:
        print(f"[warn] stack tiada: {', '.join(map(str, missing))}", file=sys.stderr)
    return key2stack


def cmd_shred(source, out_dir, stacks, allow_hf, repo, rows_per_shard):
    src = Path(source)
    if not src.is_absolute():
        src = (HF_DIR / source) if (HF_DIR / source).exists() else (ROOT / source)
    if not src.exists():
        sys.exit(f"tiada sumber: {src} (jalankan 'download' dahulu)")
    key2stack = build_map(stacks, allow_hf, repo)
    if not key2stack:
        sys.exit("peta stack kosong - tiada fail stack untuk mapping")
    out = Path(out_dir)
    if not out.is_absolute():
        out = ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)

    handles, counts, unmatched = {}, {}, 0
    umatch = None
    try:
        for r in read_rows(src):
            stack = key2stack.get(_key(r))
            if stack is None:
                unmatched += 1
                if umatch is None:
                    umatch = (out / "_unmatched.jsonl").open("w", encoding="utf-8")
                umatch.write(dump_row(r) + "\n")
                continue
            n = counts.get(stack, 0)
            if n % rows_per_shard == 0:
                if stack in handles:
                    handles[stack].close()
                (out / stack).mkdir(parents=True, exist_ok=True)
                i = n // rows_per_shard + 1
                handles[stack] = (out / stack / f"shred-{i:05d}.jsonl").open(
                    "w", encoding="utf-8")
            handles[stack].write(dump_row(r) + "\n")
            counts[stack] = n + 1
    finally:
        for h in handles.values():
            h.close()
        if umatch:
            umatch.close()
    for stack, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        shards = (n + rows_per_shard - 1) // rows_per_shard
        print(f"{stack}: {n} rows -> {shards} shard(s) in {out / stack}")
    print(f"TOTAL: {sum(counts.values())} rows, unmatched: {unmatched} -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["list", "download", "shred", "all"])
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--files", default="", help="comma list (download)")
    ap.add_argument("--all", action="store_true", help="download whole repo")
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="combined jsonl for shred")
    ap.add_argument("--out", default="datasets/shred")
    ap.add_argument("--stacks", default="", help="comma list; default = all config stacks")
    ap.add_argument("--rows", type=int, default=5000, help="max rows per shard (default 5000)")
    ap.add_argument("--no-hf-map", action="store_true",
                    help="jangan muat turun fail stack dari HF bila hilang")
    a = ap.parse_args()

    files = [s.strip() for s in a.files.split(",") if s.strip()]
    stacks = [s.strip() for s in a.stacks.split(",") if s.strip()]

    if a.cmd == "list":
        cmd_list(a.repo)
    elif a.cmd == "download":
        cmd_download(a.repo, files, a.all)
    elif a.cmd == "shred":
        cmd_shred(a.source, a.out, stacks, not a.no_hf_map, a.repo, a.rows)
    elif a.cmd == "all":
        cmd_download(a.repo, files, a.all)
        cmd_shred(a.source, a.out, stacks, not a.no_hf_map, a.repo, a.rows)


if __name__ == "__main__":
    main()
