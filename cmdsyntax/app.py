"""cmdsyntax/app.py — Colab T4 script builder + GUI.

Flow (what you asked):
  1. user selects training dataset  (datasets/*.jsonl)
  2. user selects pretrained model  (Qwen / Gemma / Llama / Mistral / Phi / custom)
  3. user sets OUT name + GGUF name + save folder (custom name)
  4. app generates a copy-paste Colab T4 script
  5. GUI shows the full CMD/script, with Copy + Save + Download-CMD buttons

Run GUI:
    python cmdsyntax/app.py

Run headless (one dataset):
    python cmdsyntax/app.py --no-gui --dataset docker --model Qwen/Qwen2.5-0.5B-Instruct --out xcoder-docker --save cmdsyntax/generated/xcoder-docker_colab.py

Batch (one script per dataset, T4):
    python cmdsyntax/app.py --no-gui --all --model Qwen/Qwen2.5-0.5B-Instruct --outdir cmdsyntax/generated

Stdlib only (tkinter). No pip install needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATASETS_DIR = ROOT / "datasets"
DATASET_CONFIG = DATASETS_DIR / "config" / "dataset.json"
MODELS_JSON = HERE / "models.json"
GENERATED_DIR = HERE / "generated"

DEFAULT_SYSTEM = (
    "You are XCoder, a tiny CPU coding assistant. "
    "Answer short and correct. Never mention Qwen/Alibaba."
)

# ---------------------------------------------------------------- helpers

def count_rows(path: Path) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def list_datasets(datasets_dir: Path = DATASETS_DIR) -> list[dict]:
    """Scan datasets/*.jsonl + config/dataset.json. Returns sorted list."""
    stacks: dict = {}
    try:
        cfg = json.loads(DATASET_CONFIG.read_text(encoding="utf-8"))
        stacks = cfg.get("stacks", {})
    except OSError:
        pass
    out: list[dict] = []
    for p in sorted(datasets_dir.glob("*.jsonl")):
        stack = p.stem
        meta = stacks.get(stack, {})
        out.append(
            {
                "stack": stack,
                "file": p.name,
                "path": str(p),
                "rows": count_rows(p),
                "desc": meta.get("desc", stack),
                "domain": meta.get("domain", ""),
            }
        )
    return out


def load_models() -> list[dict]:
    """Load pretrained catalog from models.json, fallback to builtin."""
    try:
        data = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
        models = data.get("models", [])
        if models:
            return models
    except OSError:
        pass
    return [
        {"id": "Qwen/Qwen2.5-0.5B-Instruct", "label": "Qwen2.5 0.5B Instruct (default)", "params": "0.5B", "vendor": "Qwen", "fits_t4": True, "note": ""},
        {"id": "Qwen/Qwen2.5-1.5B-Instruct", "label": "Qwen2.5 1.5B Instruct", "params": "1.5B", "vendor": "Qwen", "fits_t4": True, "note": ""},
        {"id": "google/gemma-2-2b-it", "label": "Gemma 2 2B IT", "params": "2B", "vendor": "Google", "fits_t4": True, "note": "needs HF token"},
        {"id": "meta-llama/Llama-3.2-1B-Instruct", "label": "Llama 3.2 1B", "params": "1B", "vendor": "Meta", "fits_t4": True, "note": "needs HF token"},
        {"id": "mistralai/Mistral-7B-Instruct-v0.3", "label": "Mistral 7B v0.3", "params": "7B", "vendor": "Mistral", "fits_t4": True, "note": ""},
    ]


def default_out_name(stack: str, model_id: str) -> str:
    tag = "0.5b"
    low = model_id.lower()
    if "1.5b" in low or "1b" in low:
        tag = "1b"
    elif "2b" in low:
        tag = "2b"
    elif "3b" in low or "mini" in low:
        tag = "3b"
    elif "7b" in low:
        tag = "7b"
    elif "9b" in low:
        tag = "9b"
    return f"xcoder-{stack}-{tag}"


def default_gguf_name(out: str) -> str:
    return f"{out}-q4_k_m.gguf"


# ---------------------------------------------------------------- generator

INSTALL_CMD = "pip install -q unsloth trl peft accelerate bitsandbytes datasets"


def build_download_cmd(gguf_dir: str, gguf_name: str) -> str:
    return (
        "from google.colab import files\n"
        "import glob, shutil, os\n"
        f"hits = glob.glob('{gguf_dir}/**/*.gguf', recursive=True)\n"
        "print(hits)\n"
        f"shutil.copy(hits[0], '/content/{gguf_name}') if hits else None\n"
        f"files.download('/content/{gguf_name}')\n"
    )


def generate_colab_script(
    *,
    dataset_file: str,
    dataset_rows: int = 0,
    stack: str = "custom",
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    out: str = "xcoder-custom",
    gguf_name: str = "",
    save_dir: str = "",
    epochs: int = 3,
    lr: float = 2e-4,
    batch: int = 2,
    grad_accum: int = 4,
    seq_len: int = 1024,
    lora_r: int = 16,
    lora_alpha: int = 16,
    quant: str = "q4_k_m",
    system_prompt: str = DEFAULT_SYSTEM,
    hf_token: str = "",
) -> str:
    """Return a full copy-paste Colab python script (cells marked)."""
    gguf_name = gguf_name or default_gguf_name(out)
    gguf_dir = f"{out}-gguf"
    save_dir = save_dir or out
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    sys_esc = system_prompt.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    token_block = ""
    if hf_token.strip():
        token_block = (
            "# --- HF login (needed for Gemma / Llama gated repos) ---\n"
            "from huggingface_hub import login\n"
            f'login(token="{hf_token.strip()}")\n'
        )
    return f'''# ============================================================
# Colab T4 training script — generated {ts} by cmdsyntax/app.py
# Dataset : {dataset_file} ({dataset_rows} rows, stack={stack})
# Base    : {model_id}
# OUT     : {out}   GGUF: {gguf_name}
# Colab   : Runtime -> Change runtime type -> T4 GPU -> run cells top-down
# ============================================================

# ---------- CELL 0 : GPU check + install (run once) ----------
# !nvidia-smi
# !{INSTALL_CMD}

# ---------- CELL 1 : upload dataset (pick ONE) ----------
# Option A (easiest): drag {dataset_file} into Colab Files pane,
#   then rename to /content/dataset.jsonl if needed.
# Option B (code upload):
# from google.colab import files; files.upload()  # then rename to dataset.jsonl

DATA = "dataset.jsonl"   # <-- /content name after upload (avoid "dataset (1).jsonl")
OUT = "{out}"
MODEL_ID = "{model_id}"
SAVE_DIR = "{save_dir}"
GGUF_DIR = "{gguf_dir}"
GGUF_NAME = "{gguf_name}"

{token_block}# ---------- CELL 2 : train (LoRA 4-bit, Unsloth) ----------
from unsloth import FastLanguageModel

model, tok = FastLanguageModel.from_pretrained(
    MODEL_ID,
    max_seq_length={seq_len},
    dtype=None,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r={lora_r},
    lora_alpha={lora_alpha},
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

from datasets import load_dataset
ds = load_dataset("json", data_files=DATA, split="train")
print("rows:", len(ds))

SYSTEM = "{sys_esc}"

def fmt(x):
    ins = x.get("instruction", "")
    inp = x.get("input", "") or ""
    out_txt = x.get("output", "")
    user = ins + (f"\\n{{inp}}" if inp else "")
    text = (f"<|im_start|>system\\n{{SYSTEM}}<|im_end|>\\n"
            f"<|im_start|>user\\n{{user}}<|im_end|>\\n"
            f"<|im_start|>assistant\\n{{out_txt}}<|im_end|>")
    return {{"text": text}}

ds = ds.map(fmt)

from trl import SFTTrainer, SFTConfig
SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=ds,
    dataset_text_field="text",
    args=SFTConfig(
        per_device_train_batch_size={batch},
        gradient_accumulation_steps={grad_accum},
        num_train_epochs={epochs},
        learning_rate={lr},
        fp16=True,
        logging_steps=10,
        output_dir=SAVE_DIR,
        save_strategy="epoch",
        seed=42,
    ),
).train()

print("TRAIN DONE ->", SAVE_DIR)

# ---------- CELL 3 : export GGUF + download (custom name) ----------
model.save_pretrained_gguf(GGUF_DIR, tok, quantization_method="{quant}")
print("GGUF saved ->", GGUF_DIR)

# rename to your custom file name + trigger browser download:
import glob, shutil
hits = glob.glob(GGUF_DIR + "/**/*.gguf", recursive=True)
print(hits)
assert hits, "no .gguf found — check GGUF_DIR above"
shutil.copy(hits[0], "/content/" + GGUF_NAME)
print("ready:", "/content/" + GGUF_NAME)

from google.colab import files
files.download("/content/" + GGUF_NAME)

# ---------- AFTER download : register on PC ----------
# In repo CLI:
#   /model add <downloaded file> --name {out} --domains {stack}
# or: /model import   (auto-picks newest GGUF in Downloads)
#
# If OOM on T4: lower seq to 512, batch to 1, grad_accum to 8,
# or pick a smaller base model (Qwen 0.5B).
'''


# ---------------------------------------------------------------- CLI

def cmd_list() -> int:
    ds = list_datasets()
    models = load_models()
    print(f"Datasets ({DATASETS_DIR}):")
    for d in ds:
        print(f"  {d['stack']:<12} {d['file']:<20} {d['rows']:>6} rows  {d['desc']}")
    print("\nPretrained models:")
    for m in models:
        flag = "T4-ok" if m.get("fits_t4") else "T4-tight"
        print(f"  {m['id']:<45} [{flag}] {m.get('note','')}")
    return 0


def cmd_generate(args) -> int:
    datasets = {d["stack"]: d for d in list_datasets()}
    if args.dataset not in datasets:
        print(f"unknown dataset '{args.dataset}'. Available:", file=sys.stderr)
        for k in sorted(datasets):
            print(f"  {k}", file=sys.stderr)
        return 2
    d = datasets[args.dataset]
    out = args.out or default_out_name(d["stack"], args.model)
    gguf = args.gguf or default_gguf_name(out)
    script = generate_colab_script(
        dataset_file=d["file"], dataset_rows=d["rows"], stack=d["stack"],
        model_id=args.model, out=out, gguf_name=gguf, save_dir=args.save_dir or out,
        epochs=args.epochs, lr=args.lr, batch=args.batch,
        grad_accum=args.grad_accum, seq_len=args.seq_len,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, quant=args.quant,
        system_prompt=args.system, hf_token=args.hf_token or "",
    )
    if args.save:
        p = Path(args.save)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(script, encoding="utf-8")
        print(f"wrote {p}  ({len(script)} chars)")
    else:
        print(script)
    return 0


def cmd_all(args) -> int:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True) if args.outdir == str(GENERATED_DIR) else None
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in list_datasets():
        out = default_out_name(d["stack"], args.model)
        script = generate_colab_script(
            dataset_file=d["file"], dataset_rows=d["rows"], stack=d["stack"],
            model_id=args.model, out=out, gguf_name=default_gguf_name(out),
            save_dir=out, epochs=args.epochs, lr=args.lr, batch=args.batch,
            grad_accum=args.grad_accum, seq_len=args.seq_len,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, quant=args.quant,
            system_prompt=args.system, hf_token=args.hf_token or "",
        )
        p = outdir / f"{out}_colab.py"
        p.write_text(script, encoding="utf-8")
        print(f"wrote {p}")
        n += 1
    print(f"done: {n} scripts -> {outdir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Colab T4 script builder (cmdsyntax)")
    ap.add_argument("--no-gui", action="store_true", help="headless mode (no tkinter window)")
    ap.add_argument("--list", action="store_true", help="list datasets + models and exit")
    ap.add_argument("--all", action="store_true", help="batch-generate one script per dataset")
    ap.add_argument("--dataset", default="docker", help="dataset stack name (e.g. docker)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF model id (pretrained)")
    ap.add_argument("--out", default="", help="custom OUT / save folder name (default xcoder-<stack>-<size>)")
    ap.add_argument("--gguf", default="", help="custom gguf file name (default <out>-q4_k_m.gguf)")
    ap.add_argument("--save-dir", default="", help="training output dir inside Colab (default = OUT)")
    ap.add_argument("--save", default="", help="write script to this .py file (else stdout)")
    ap.add_argument("--outdir", default=str(GENERATED_DIR), help="output dir for --all")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--quant", default="q4_k_m")
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    ap.add_argument("--hf-token", default="")
    return ap


# ---------------------------------------------------------------- GUI (tkinter, shows CMD)

def run_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    datasets = list_datasets()
    models = load_models()
    if not datasets:
        messagebox.showerror("cmdsyntax", f"No datasets found in {DATASETS_DIR}")
        return

    ds_labels = [f"{d['stack']}  ({d['rows']} rows) — {d['file']}" for d in datasets]
    ds_by_label = dict(zip(ds_labels, datasets))
    model_ids = [m["id"] for m in models]
    model_by_id = {m["id"]: m for m in models}

    root = tk.Tk()
    root.title("cmdsyntax — Colab T4 script builder")
    root.geometry("980x760")

    # ---- top: selectors ----
    top = ttk.Frame(root, padding=10)
    top.pack(fill="x")

    ttk.Label(top, text="1) Training dataset:").grid(row=0, column=0, sticky="w")
    ds_var = tk.StringVar(value=ds_labels[0])
    ds_cb = ttk.Combobox(top, textvariable=ds_var, values=ds_labels, width=52, state="readonly")
    ds_cb.grid(row=0, column=1, sticky="ew", padx=6)
    rows_var = tk.StringVar(value="")
    ttk.Label(top, textvariable=rows_var, foreground="gray").grid(row=0, column=2, sticky="w")

    ttk.Label(top, text="2) Pretrained model:").grid(row=1, column=0, sticky="w", pady=4)
    model_var = tk.StringVar(value=model_ids[0])
    model_cb = ttk.Combobox(top, textvariable=model_var, values=model_ids, width=52)
    model_cb.grid(row=1, column=1, sticky="ew", padx=6)
    note_var = tk.StringVar(value="")
    ttk.Label(top, textvariable=note_var, foreground="gray", wraplength=280).grid(row=1, column=2, sticky="w")
    top.columnconfigure(1, weight=1)

    # ---- middle: names + hyperparams ----
    mid = ttk.LabelFrame(root, text="3) Train + 4) custom save name", padding=10)
    mid.pack(fill="x", padx=10)

    def row(r, label, var, width=24):
        ttk.Label(mid, text=label).grid(row=r, column=0, sticky="w")
        e = ttk.Entry(mid, textvariable=var, width=width)
        e.grid(row=r, column=1, sticky="ew", padx=6, pady=2)
        return e

    out_var = tk.StringVar(value=default_out_name(datasets[0]["stack"], model_ids[0]))
    gguf_var = tk.StringVar(value=default_gguf_name(out_var.get()))
    save_var = tk.StringVar(value=out_var.get())
    epochs_var = tk.StringVar(value="3")
    lr_var = tk.StringVar(value="2e-4")
    batch_var = tk.StringVar(value="2")
    accum_var = tk.StringVar(value="4")
    seq_var = tk.StringVar(value="1024")
    lorar_var = tk.StringVar(value="16")
    loraa_var = tk.StringVar(value="16")
    quant_var = tk.StringVar(value="q4_k_m")
    token_var = tk.StringVar(value="")

    row(0, "OUT / save folder:", out_var, 34)
    row(1, "GGUF file name:", gguf_var, 34)
    ttk.Label(mid, text="(custom name, e.g. xcoder-docker-0.5b-q4_k_m.gguf)").grid(row=1, column=2, sticky="w")
    hp = ttk.Frame(mid)
    hp.grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)
    for i, (t, v, w) in enumerate([
        ("epochs", epochs_var, 5), ("lr", lr_var, 9), ("batch", batch_var, 5),
        ("accum", accum_var, 5), ("seq", seq_var, 7),
        ("LoRA r", lorar_var, 5), ("alpha", loraa_var, 5), ("quant", quant_var, 9),
    ]):
        ttk.Label(hp, text=t).grid(row=0, column=i * 2, padx=(8 if i else 0, 2))
        ttk.Entry(hp, textvariable=v, width=w).grid(row=0, column=i * 2 + 1)
    ttk.Label(mid, text="HF token (only Gemma/Llama gated):").grid(row=3, column=0, sticky="w")
    ttk.Entry(mid, textvariable=token_var, width=34, show="*").grid(row=3, column=1, sticky="ew", padx=6, pady=2)
    ttk.Label(mid, text="System prompt:").grid(row=4, column=0, sticky="nw")
    sys_txt = tk.Text(mid, height=2, width=60, wrap="word")
    sys_txt.grid(row=4, column=1, columnspan=2, sticky="ew", padx=6, pady=2)
    sys_txt.insert("1.0", DEFAULT_SYSTEM)
    mid.columnconfigure(1, weight=1)

    def refresh_meta(*_):
        d = ds_by_label.get(ds_var.get(), datasets[0])
        rows_var.set(f"{d['rows']} rows · {d['desc']}")
        m = model_by_id.get(model_var.get())
        note_var.set(m.get("note", "") if m else "custom HF id — must work with Unsloth")
        # auto-sync OUT name unless user already customised it
        cur = out_var.get()
        auto_old = [default_out_name(x["stack"], y) for x in datasets for y in model_ids]
        if cur in auto_old or not cur.strip():
            new_out = default_out_name(d["stack"], model_var.get())
            out_var.set(new_out)
            gguf_var.set(default_gguf_name(new_out))
            save_var.set(new_out)

    ds_cb.bind("<<ComboboxSelected>>", refresh_meta)
    model_cb.bind("<<ComboboxSelected>>", refresh_meta)
    model_cb.bind("<KeyRelease>", refresh_meta)
    refresh_meta()

    # ---- buttons ----
    btns = ttk.Frame(root, padding=(10, 0))
    btns.pack(fill="x")
    preview = tk.Text(root, wrap="none", height=20, font=("Consolas", 9))
    status = tk.StringVar(value="Pick dataset + model, then Generate.")

    def current_script() -> str:
        d = ds_by_label.get(ds_var.get(), datasets[0])
        try:
            epochs = int(epochs_var.get()); batch = int(batch_var.get())
            accum = int(accum_var.get()); seq = int(seq_var.get())
            lorar = int(lorar_var.get()); loraa = int(loraa_var.get())
            lr = float(lr_var.get())
        except ValueError:
            messagebox.showerror("cmdsyntax", "epochs/batch/accum/seq/LoRA must be numbers, lr numeric (e.g. 2e-4)")
            raise
        out = out_var.get().strip() or default_out_name(d["stack"], model_var.get())
        gguf = gguf_var.get().strip() or default_gguf_name(out)
        return generate_colab_script(
            dataset_file=d["file"], dataset_rows=d["rows"], stack=d["stack"],
            model_id=model_var.get().strip(), out=out, gguf_name=gguf,
            save_dir=save_var.get().strip() or out,
            epochs=epochs, lr=lr, batch=batch, grad_accum=accum, seq_len=seq,
            lora_r=lorar, lora_alpha=loraa, quant=quant_var.get().strip() or "q4_k_m",
            system_prompt=sys_txt.get("1.0", "end").strip() or DEFAULT_SYSTEM,
            hf_token=token_var.get(),
        )

    def do_generate():
        try:
            s = current_script()
        except ValueError:
            return
        preview.delete("1.0", "end")
        preview.insert("1.0", s)
        status.set(f"Generated {len(s)} chars — copy to Colab, or Save .py.")

    def do_copy_all():
        s = preview.get("1.0", "end").strip() or current_script()
        root.clipboard_clear()
        root.clipboard_append(s)
        status.set("Full Colab script copied to clipboard.")

    def do_copy_install():
        root.clipboard_clear()
        root.clipboard_append("!nvidia-smi\n!" + INSTALL_CMD)
        status.set("Install CMD copied (!nvidia-smi + pip install).")

    def do_copy_download():
        d = ds_by_label.get(ds_var.get(), datasets[0])
        out = out_var.get().strip() or default_out_name(d["stack"], model_var.get())
        gguf = gguf_var.get().strip() or default_gguf_name(out)
        root.clipboard_clear()
        root.clipboard_append(build_download_cmd(f"{out}-gguf", gguf))
        status.set("Download CMD copied (files.download).")

    def do_save():
        s = preview.get("1.0", "end").strip()
        if not s:
            s = current_script()
        d = ds_by_label.get(ds_var.get(), datasets[0])
        initial = f"{out_var.get().strip() or d['stack']}_colab.py"
        p = filedialog.asksaveasfilename(
            initialfile=initial, defaultextension=".py",
            filetypes=[("Python", "*.py"), ("All", "*.*")],
            initialdir=str(GENERATED_DIR if GENERATED_DIR.exists() else HERE),
        )
        if p:
            Path(p).write_text(s, encoding="utf-8")
            status.set(f"Saved {p}")

    ttk.Button(btns, text="Generate CMD  ▶", command=do_generate).pack(side="left")
    ttk.Button(btns, text="Copy full script", command=do_copy_all).pack(side="left", padx=6)
    ttk.Button(btns, text="Copy install CMD", command=do_copy_install).pack(side="left")
    ttk.Button(btns, text="Copy download CMD", command=do_copy_download).pack(side="left", padx=6)
    ttk.Button(btns, text="Save .py …", command=do_save).pack(side="left")
    ttk.Label(btns, textvariable=status, foreground="gray").pack(side="left", padx=10)

    # ---- preview: the CMD shown in GUI ----
    box = ttk.LabelFrame(root, text="Colab CMD / script preview (paste into Colab T4 cell)", padding=6)
    box.pack(fill="both", expand=True, padx=10, pady=6)
    sy = ttk.Scrollbar(box, orient="vertical", command=preview.yview)
    sx = ttk.Scrollbar(box, orient="horizontal", command=preview.xview)
    preview.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
    preview.pack(side="top", fill="both", expand=True)
    sy.pack(side="right", fill="y")
    do_generate()
    root.mainloop()


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.list:
        return cmd_list()
    if args.no_gui and args.all:
        return cmd_all(args)
    if args.no_gui:
        return cmd_generate(args)
    if args.all or args.save:
        # headless batch/save without --no-gui still works
        return cmd_all(args) if args.all else cmd_generate(args)
    run_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
