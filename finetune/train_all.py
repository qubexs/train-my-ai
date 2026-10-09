# Train ALL experts one by one, skip existing GGUFs, auto-register.
# Run INSIDE .venv-gpu (torch cu118 + transformers/datasets/accelerate/peft/trl):
#   .\.venv-gpu\Scripts\Activate.ps1
#   python finetune/train_all.py [--dry-run] [--only xcoder-docker] [--epochs 3]
# Each expert: LoRA (train_local.py) -> GGUF (llama.cpp converter) ->
#   models/<name>-0.5b-q4_k_m.gguf -> registry (auto, visible in /models).
# Needs: git (one-time llama.cpp checkout into build/), gguf package
#   (pip install -r build/llama.cpp/requirements/convert_hf_to_gguf.txt).
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

QUANT = "q4_k_m"

# (expert name, dataset stacks, registry domains)
PLAN = [
    ("xcoder-docker", ["docker"], ["docker"]),
    ("xcoder-linux", ["linux", "bash"], ["linux"]),
    ("xcoder-web", ["javascript", "typescript", "nodejs", "html", "css", "tailwind"], ["web"]),
    ("xcoder-sql", ["sql", "postgresql", "mysql"], ["data"]),
    ("xcoder-python", ["python"], ["python"]),
    ("xcoder-php", ["php", "laravel"], ["php"]),
    ("xcoder-general", ["general"], ["general"]),
]


def count_rows(path):
    n = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if (r.get("instruction") or "").strip() and (r.get("output") or "").strip():
            n += 1
    return n


def run(cmd, cwd=None):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def ensure_llamacpp(build_dir):
    conv = build_dir / "llama.cpp" / "convert_hf_to_gguf.py"
    if conv.exists():
        return conv
    build_dir.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp",
         str(build_dir / "llama.cpp")])
    return conv


def main():
    ap = argparse.ArgumentParser(description="Batch-train all XCoder experts")
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--build-dir", default=str(ROOT / "build" / "train"))
    ap.add_argument("--only", default="", help="train one expert, e.g. xcoder-docker")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--dry-run", action="store_true", help="show plan only, train nothing")
    a = ap.parse_args()

    from experts import add as add_model

    models_dir = Path(a.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'EXPERT':16} {'ROWS':>6}  DECISION")
    jobs = []
    for name, stacks, doms in PLAN:
        if a.only and a.only != name:
            continue
        gguf = models_dir / f"{name}-0.5b-{QUANT}.gguf"
        files = [ROOT / "datasets" / f"{s}.jsonl" for s in stacks]
        files = [f for f in files if f.exists()]
        rows = sum(count_rows(f) for f in files)
        if gguf.exists() and gguf.stat().st_size > 50_000_000:
            print(f"{name:16} {rows:>6}  SKIP (siap: {gguf.name})")
            continue
        if rows == 0:
            print(f"{name:16} {rows:>6}  SKIP (tiada data)")
            continue
        if rows < 50:
            print(f"{name:16} {rows:>6}  TRAIN (nipis! kesan lemah dijangka)")
        else:
            print(f"{name:16} {rows:>6}  TRAIN")
        jobs.append((name, stacks, doms, files, rows, gguf))

    if a.dry_run:
        print(f"DRY-RUN: {len(jobs)} akan dilatih")
        return

    conv = ensure_llamacpp(Path(a.build_dir)) if jobs else None
    for name, stacks, doms, files, rows, gguf in jobs:
        data = ",".join(str(f) for f in files)
        out = Path(a.build_dir) / name
        print(f"\n===== {name} ({rows} rows) =====", flush=True)
        run([sys.executable, str(ROOT / "finetune" / "train_local.py"),
             "--data", data, "--out", str(out), "--epochs", str(a.epochs)])
        merged = str(out) + "-merged"
        run([sys.executable, str(conv), merged, "--outfile", str(gguf),
             "--outtype", QUANT])
        if gguf.stat().st_size < 50_000_000:
            raise SystemExit(f"GGUF mencurigakan kecil: {gguf}")
        e = add_model(models_dir, str(gguf), name=name, domains=doms)
        print(f"OK {e['name']} [{','.join(e['domains'])}] -> {gguf} + registry")


if __name__ == "__main__":
    sys.exit(main())
