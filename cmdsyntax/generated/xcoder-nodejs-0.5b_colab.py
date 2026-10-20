# ============================================================
# Colab T4 training script — generated 2026-10-12 12:13 by cmdsyntax/app.py
# Dataset : nodejs.jsonl (1348 rows, stack=nodejs)
# Base    : Qwen/Qwen2.5-0.5B-Instruct
# OUT     : xcoder-nodejs-0.5b   GGUF: xcoder-nodejs-0.5b-q4_k_m.gguf
# Colab   : Runtime -> Change runtime type -> T4 GPU -> run cells top-down
# ============================================================

# ---------- CELL 0 : GPU check + install (run once) ----------
# !nvidia-smi
# !pip install -q unsloth trl peft accelerate bitsandbytes datasets

# ---------- CELL 1 : upload dataset (pick ONE) ----------
# Option A (easiest): drag nodejs.jsonl into Colab Files pane,
#   then rename to /content/dataset.jsonl if needed.
# Option B (code upload):
# from google.colab import files; files.upload()  # then rename to dataset.jsonl

DATA = "dataset.jsonl"   # <-- /content name after upload (avoid "dataset (1).jsonl")
OUT = "xcoder-nodejs-0.5b"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SAVE_DIR = "xcoder-nodejs-0.5b"
GGUF_DIR = "xcoder-nodejs-0.5b-gguf"
GGUF_NAME = "xcoder-nodejs-0.5b-q4_k_m.gguf"

# ---------- CELL 2 : train (LoRA 4-bit, Unsloth) ----------
from unsloth import FastLanguageModel

model, tok = FastLanguageModel.from_pretrained(
    MODEL_ID,
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

from datasets import load_dataset
ds = load_dataset("json", data_files=DATA, split="train")
print("rows:", len(ds))

SYSTEM = "You are XCoder, a tiny CPU coding assistant. Answer short and correct. Never mention Qwen/Alibaba."

def fmt(x):
    ins = x.get("instruction", "")
    inp = x.get("input", "") or ""
    out_txt = x.get("output", "")
    user = ins + (f"\n{inp}" if inp else "")
    text = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{out_txt}<|im_end|>")
    return {"text": text}

ds = ds.map(fmt)

from trl import SFTTrainer, SFTConfig
SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=ds,
    dataset_text_field="text",
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=0.0002,
        fp16=True,
        logging_steps=10,
        output_dir=SAVE_DIR,
        save_strategy="epoch",
        seed=42,
    ),
).train()

print("TRAIN DONE ->", SAVE_DIR)

# ---------- CELL 3 : export GGUF + download (custom name) ----------
model.save_pretrained_gguf(GGUF_DIR, tok, quantization_method="q4_k_m")
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
#   /model add <downloaded file> --name xcoder-nodejs-0.5b --domains nodejs
# or: /model import   (auto-picks newest GGUF in Downloads)
#
# If OOM on T4: lower seq to 512, batch to 1, grad_accum to 8,
# or pick a smaller base model (Qwen 0.5B).
