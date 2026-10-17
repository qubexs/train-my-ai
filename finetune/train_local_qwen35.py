# Local training on the GTX 1070 8GB (Windows, Pascal sm_61) — no Colab, no Unsloth.
# Unsloth needs Turing+ (Triton); Pascal runs fine on plain transformers+peft+trl,
# in fp32 (GTX 1070 fp16 is 1:64 slow — fp32 is the fast mode here).
#
# One-time env (already present as .venv-gpu):
#   py -m venv .venv-gpu
#   .venv-gpu\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu118
#   .venv-gpu\Scripts\pip install transformers datasets accelerate peft trl sentencepiece
#
# Run:
#   .venv-gpu\Scripts\python.exe finetune\train_local_qwen35.py --data all
# First run downloads Qwen/Qwen3.5-0.8B (~3.5GB fp32 safetensors) from HF.
#
# Checkpoints: every 100 steps to finetune/runs/<OUT>/checkpoint-* (keep last 2),
# auto-resume from the newest COMPLETE checkpoint (adapter_model.safetensors +
# optimizer.pt + trainer_state.json; sesi mati semasa save — dilangkau automatik).
# Kalau OOM (8GB VRAM): tutup Chrome/game, atau guna --max-len 2048.
# Kalau terputus: run arahan SAMA semula — ia sambung dari checkpoint terakhir.
#
# Data & holdout sama seperti finetune/train_unsloth.py:
#   --data all -> datasets/all.jsonl -> xcoder-best-0.8b; 1% md5 eval holdout
#   ditulis ke <out_dir>/eval.jsonl.
#
# Eksport: <OUT>-merged (16-bit) ke finetune/runs/. GGUF selepas ini:
#   .venv-gguf\Scripts\python.exe build\llama.cpp\convert_hf_to_gguf.py finetune\runs\xcoder-best-0.8b-merged --outfile tmp-f16.gguf --outtype f16
#   app\bin\llama-quantize.exe tmp-f16.gguf xcoder-best-0.8b-q4_k_m.gguf Q4_K_M
# (.venv-gguf = env khas ikut build\llama.cpp\requirements\requirements-convert_hf_to_gguf.txt
#  supaya torch cu118 .venv-gpu tidak ditimpa)
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def arg(flag, default=None):
    key = f"--{flag}"
    if key in sys.argv:
        i = sys.argv.index(key)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


data = arg("data", "all")
name = arg("name", "")
out_arg = arg("out", "")
max_len = int(arg("max-len", "3072"))
batch = int(arg("batch", "1"))
accum = int(arg("accum", "8"))
no_export = "--no-export" in sys.argv
dry = "--dry" in sys.argv
stacks = [s.strip() for s in data.split(",") if s.strip()]
OUT = name or ("xcoder-best-0.8b" if stacks == ["all"]
               else "xcoder-" + "-".join(stacks) + "-0.8b")

SYSTEM = ("You are XCoder, a tiny CPU coding assistant. "
          "Answer briefly in the user's language (Bahasa Melayu or English). "
          "Never mention Qwen/Alibaba.")

out_dir = Path(out_arg) if out_arg else ROOT / "finetune" / "runs" / OUT


def find_file(stack):
    cands = ([ROOT / "datasets" / "all.jsonl", Path("all.jsonl")]
             if stack == "all" else
             [ROOT / "finetune" / "stacks" / stack / "dataset.jsonl",
              ROOT / "datasets" / f"{stack}.jsonl", Path(f"{stack}.jsonl")])
    for c in cands:
        if c.exists():
            return c
    return None


def ckpt_list():
    if not out_dir.exists():
        return []
    need = ("adapter_model.safetensors", "optimizer.pt", "trainer_state.json")
    good = []
    for p in sorted(out_dir.glob("checkpoint-*"),
                    key=lambda p: int(p.name.rsplit("-", 1)[-1])):
        if all((p / f).exists() for f in need):
            good.append(p)
        else:
            print(f"checkpoint tidak lengkap, dilangkau: {p.name}")
    return good


raw = []
for s in stacks:
    f = find_file(s)
    if not f:
        raise SystemExit(f"tiada data untuk '{s}' — cari datasets/{s}.jsonl "
                         f"atau {s}.jsonl dalam cwd")
    n = 0
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if (r.get("instruction") or "").strip() and (r.get("output") or "").strip():
            raw.append(r)
            n += 1
    print(f"{s}: {n} rows from {f}")

train_rows, eval_rows = [], []
for r in raw:
    key = f"{r['instruction'].strip()}\n{r['output'].strip()}"
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 1000
    (eval_rows if 50 <= h < 60 else train_rows).append(r)
print(f"train={len(train_rows)} eval={len(eval_rows)} (1% md5 holdout)")
print(f"out_dir={out_dir}  max_len={max_len}  batch={batch}x{accum}")

if dry:
    ck = ckpt_list()
    print("resume:", ck[-1] if ck else "(kosong — mula dari awal)")
    raise SystemExit(0)

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen3.5-0.8B"

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} — fp32 (Pascal)")
else:
    print("AMARAN: tiada CUDA — CPU training akan ambil berhari-hari.")

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
model.config.use_cache = False


def to_msgs(r):
    user = r["instruction"].strip()
    inp = (r.get("input") or "").strip()
    if inp:
        user += "\n" + inp
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": r["output"].strip()},
    ]}


ds = Dataset.from_list([to_msgs(r) for r in train_rows])

out_dir.mkdir(parents=True, exist_ok=True)
with (out_dir / "eval.jsonl").open("w", encoding="utf-8") as f:
    for r in eval_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

args = SFTConfig(
    output_dir=str(out_dir),
    per_device_train_batch_size=batch, gradient_accumulation_steps=accum,
    num_train_epochs=1, learning_rate=2e-4, warmup_steps=5,
    lr_scheduler_type="linear", optim="adamw_torch", weight_decay=0.01,
    logging_steps=10, save_steps=100, save_total_limit=2, seed=77,
    packing=True, completion_only_loss=True, max_length=max_len,
    fp16=False, bf16=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_pin_memory=False, report_to="none",
)
trainer = SFTTrainer(
    model=model, args=args, train_dataset=ds, processing_class=tok,
    peft_config=LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear"),
)

ck = ckpt_list()
print(f"resume: {ck[-1] if ck else '(kosong — mula dari awal)'}")
trainer.train(resume_from_checkpoint=str(ck[-1]) if ck else None)

if not no_export:
    merged = trainer.model.merge_and_unload()
    dst = ROOT / "finetune" / "runs" / (OUT + "-merged")
    merged.save_pretrained(str(dst))
    tok.save_pretrained(str(dst))
    print(f"exported {dst}")
    print("GGUF: convert_hf_to_gguf.py -> app\\bin\\llama-quantize.exe ... Q4_K_M")
