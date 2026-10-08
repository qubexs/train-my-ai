# Run on Google Colab FREE GPU (T4), NOT on your 7.8GB CPU box.
# Colab cells:
#   !pip install -q unsloth trl peft accelerate bitsandbytes
# Then run this file content in a Colab cell.
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

# dataset.jsonl must use Alpaca fields: instruction/input/output
from datasets import load_dataset
ds = load_dataset("json", data_files="dataset.jsonl", split="train")

def fmt(x):
    ins, inp, out = x["instruction"], x.get("input", ""), x["output"]
    user = ins + (f"\n{inp}" if inp else "")
    text = f"<|im_start|>system\nYou are Coder 77, a tiny CPU 0.5B coding assistant. Never mention Qwen/Alibaba.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
    return {"text": text}

ds = ds.map(fmt)
from trl import SFTTrainer, SFTConfig
SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds, dataset_text_field="text",
    args=SFTConfig(per_device_train_batch_size=2, gradient_accumulation_steps=4,
                    num_train_epochs=3, learning_rate=2e-4, fp16=True,
                    output_dir="coder77-0.5b", logging_steps=10),
).train()

# Merge + save, then export GGUF in Colab:
#   model.save_pretrained_merged("coder77-0.5b-merged", tok, save_method="merged_16bit")
#   !pip install -q llama-cpp-python
#   !python -m llama_cpp.convert_hf_to_gguf coder77-0.5b-merged --outfile coder77-0.5b-q8_0.gguf --outtype q8_0
# Download the .gguf, then on Windows: lms import coder77-0.5b-q8_0.gguf --identifier ezcodex-0.5b
