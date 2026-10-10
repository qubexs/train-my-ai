# Run on Google Colab FREE GPU (T4), NOT on your 7.8GB CPU box.
# Colab: upload finetune/stacks/<stack>/dataset.jsonl, rename to dataset.jsonl if needed.
#   !pip install -q unsloth trl peft accelerate bitsandbytes datasets
# Then run this file content in a Colab cell. Set DATA below to your file,
# and OUT to xcoder-<stack> so the GGUF name matches finetune/stacks/<stack>/.
import sys

DATA = sys.argv[1] if len(sys.argv) > 1 else "dataset.jsonl"  # /content name after upload
OUT = sys.argv[2] if len(sys.argv) > 2 else "xcoder-0.5b"      # e.g. xcoder-docker
from unsloth import FastLanguageModel

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
model, tok = FastLanguageModel.from_pretrained(
    MODEL_ID, max_seq_length=1024, dtype=None, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0, bias="none",
)

# rows must use Alpaca fields: instruction/input/output
from datasets import load_dataset
ds = load_dataset("json", data_files=DATA, split="train")

def fmt(x):
    ins, inp, out = x["instruction"], x.get("input", ""), x["output"]
    user = ins + (f"\n{inp}" if inp else "")
    text = f"<|im_start|>system\nYou are XCoder, a tiny CPU 0.5B coding assistant. Never mention Qwen/Alibaba.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
    return {"text": text}

ds = ds.map(fmt)
from trl import SFTTrainer, SFTConfig
SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds, dataset_text_field="text",
    args=SFTConfig(per_device_train_batch_size=2, gradient_accumulation_steps=4,
                    num_train_epochs=3, learning_rate=2e-4, fp16=True,
                    output_dir=OUT, logging_steps=10),
).train()

# Merge + GGUF in Colab (Unsloth built-in export — version-proof):
#   model.save_pretrained_gguf(OUT + "-gguf", tok, quantization_method="q4_k_m")
# Download the .gguf, rename to xcoder-<kepakaran>-0.5b-q4_k_m.gguf
# (cth xcoder-docker-0.5b-q4_k_m.gguf),
# then register on PC: /model add <fail> --name xcoder-<stack> --domains <stack>
