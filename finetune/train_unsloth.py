# Run on Google Colab FREE GPU (T4), NOT on your 7.8GB CPU box.
# Colab setup (Unsloth's Qwen3.5 pin set — see their 0.8B notebook):
#   !pip install --upgrade -qqq uv
#   !uv pip install -qqq bitsandbytes xformers==0.0.32.post2 \
#       "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
#       "unsloth[base] @ git+https://github.com/unslothai/unsloth"
#   !uv pip install --upgrade --no-deps "tokenizers>=0.22.0,<=0.23.0" trl==0.22.2 unsloth unsloth_zoo
#   !uv pip install transformers==5.2.0
#   !unsloth install-kernels
#   upload all.jsonl (or rename a stack file to <stack>.jsonl), then:
#   !python train_unsloth.py --data all
#
# Base: Qwen/Qwen3.5-0.8B (official safetensors — the -GGUF repo is inference-only).
# 16-bit LoRA (load_in_4bit=False) — Unsloth advises against QLoRA on Qwen3.5.
# Two data modes:
#   --data all            datasets/all.jsonl               -> xcoder-best-0.8b
#   --data docker[,linux] finetune/stacks/<s>/dataset.jsonl if present,
#                         else datasets/<s>.jsonl         -> xcoder-<s>-0.8b
#   --name N overrides the output name; --no-export skips merged/GGUF export;
#   --export-only re-loads the newest checkpoint and exports without training;
#   --out DIR overrides the checkpoint folder (cth. drive yang dikongsi antara
#   akaun — "Add shortcut to Drive" dari akaun lain juga berfungsi).
#
# Non-thinking fine-tune: the official chat template renders an EMPTY
# <think></think> block before every answer, and loss masks everything
# before the answer span (train_on_responses_only). Rows are packed to
# max_seq_length=3072 (long Fable agent rows fit; short rows pack together).
# 1 epoch, deterministic 1% md5 eval holdout -> <out>/eval.jsonl (hash
# band 50-59, disjoint from the repo's 5% band [0-49) so merge.py outputs
# still yield a holdout). Checkpoints every 100 steps with auto-resume
# (stored under /content/drive/xcoder/<OUT> when Google Drive is mounted;
# incomplete checkpoints — sesi mati semasa save — dilangkau automatik).
#
# After training: download <OUT>-gguf/*.gguf, rename to <OUT>-q4_k_m.gguf
# (cth xcoder-best-0.8b-q4_k_m.gguf), then on the PC:
#   /model add <fail> --name xcoder-best --domains general
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
no_export = "--no-export" in sys.argv
export_only = "--export-only" in sys.argv
stacks = [s.strip() for s in data.split(",") if s.strip()]
OUT = name or ("xcoder-best-0.8b" if stacks == ["all"]
               else "xcoder-" + "-".join(stacks) + "-0.8b")

SYSTEM = ("You are XCoder, a tiny CPU coding assistant. "
          "Answer briefly in the user's language (Bahasa Melayu or English). "
          "Never mention Qwen/Alibaba.")


drive = Path("/content/drive/MyDrive")
if out_arg:
    out_dir = Path(out_arg)
else:
    out_dir = drive / "xcoder" / OUT if drive.exists() else Path(OUT)
if (not export_only and str(out_dir).startswith("/content")
        and not str(out_dir).startswith("/content/drive")):
    print("AMARAN: Google Drive tidak dipasang — checkpoint di sini akan "
          "HILANG bila sesi Colab tamat. Mount Drive dahulu!")


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


train_rows, eval_rows = [], []
if not export_only:
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

    for r in raw:
        key = f"{r['instruction'].strip()}\n{r['output'].strip()}"
        h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 1000
        (eval_rows if 50 <= h < 60 else train_rows).append(r)
    print(f"train={len(train_rows)} eval={len(eval_rows)} (1% md5 holdout)")

from unsloth import FastVisionModel, is_bfloat16_supported

model, tok = FastVisionModel.from_pretrained(
    "Qwen/Qwen3.5-0.8B", max_seq_length=3072, load_in_4bit=False,
)

if export_only:
    from peft import PeftModel
    ck = ckpt_list()
    src = ck[-1] if ck else out_dir
    if not (src / "adapter_model.safetensors").exists():
        raise SystemExit(f"tiada adapter dalam {src} — train dahulu")
    print(f"export-only: memuat adapter {src}")
    model = PeftModel.from_pretrained(model, str(src))
    model.save_pretrained_merged(OUT + "-merged", tok, save_method="merged_16bit")
    model.save_pretrained_gguf(OUT + "-gguf", tok, quantization_method="q4_k_m")
    print(f"exported {OUT}-merged + {OUT}-gguf — rename gguf to {OUT}-q4_k_m.gguf")
    raise SystemExit(0)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0, bias="none",
    target_modules="all-linear",
    use_gradient_checkpointing="unsloth",
)


def to_text(r):
    user = r["instruction"].strip()
    inp = (r.get("input") or "").strip()
    if inp:
        user += "\n" + inp
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": r["output"].strip()}]
    return tok.apply_chat_template(msgs, tokenize=False)


print("sample tail:", repr(to_text(train_rows[0])[-260:]))

from datasets import Dataset

ds = Dataset.from_list([{"text": to_text(r)} for r in train_rows])

out_dir.mkdir(parents=True, exist_ok=True)
with (out_dir / "eval.jsonl").open("w", encoding="utf-8") as f:
    for r in eval_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds, dataset_text_field="text",
    args=SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        num_train_epochs=1, learning_rate=2e-4, warmup_steps=5,
        lr_scheduler_type="linear", optim="adamw_8bit", weight_decay=0.01,
        logging_steps=10, save_steps=100, save_total_limit=2, seed=77,
        packing=True,
        fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
    ),
)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
ckpts = ckpt_list()
trainer.train(resume_from_checkpoint=ckpts[-1] if ckpts else None)

if not no_export:
    model.save_pretrained_merged(OUT + "-merged", tok, save_method="merged_16bit")
    model.save_pretrained_gguf(OUT + "-gguf", tok, quantization_method="q4_k_m")
    print(f"exported {OUT}-merged + {OUT}-gguf — rename gguf to {OUT}-q4_k_m.gguf")
