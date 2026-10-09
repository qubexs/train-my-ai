# Train on a local GTX 1070 8GB (Windows) — NO Colab needed.
# Pascal (sm_61) lacks proper Unsloth/bitsandbytes support, so this uses
# plain transformers + peft + trl LoRA in fp16/fp32. A 0.5B model fits easily.
#
# Setup once (PowerShell, Python 3.10-3.12, GTX 1070 needs CUDA 11.8 build):
#   py -m venv .venv-gpu; .\.venv-gpu\Scripts\Activate.ps1
#   pip install torch --index-url https://download.pytorch.org/whl/cu118
#   pip install transformers datasets accelerate peft trl
# Run:
#   python finetune/train_local.py --data datasets/docker.jsonl --out xcoder-docker
# Then export GGUF (same venv):
#   python llama.cpp/convert_hf_to_gguf.py <out>-merged --outfile tmp-f16.gguf --outtype f16
#   app\bin\llama-quantize.exe tmp-f16.gguf <out>-0.5b-q4_k_m.gguf Q4_K_M
import argparse
import sys


def parse_args():
    ap = argparse.ArgumentParser(description="Local LoRA train (GTX 1070 friendly)")
    ap.add_argument("--data", required=True,
                    help="datasets/<stack>.jsonl, finetune/<domain>/dataset.jsonl, or comma list for multi-stack")
    ap.add_argument("--out", default="xcoder-0.5b", help="e.g. xcoder-docker")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--fp16", default="auto", choices=["auto", "on", "off"],
                    help="auto disables fp16 if the GPU lacks fast fp16 (Pascal)")
    return ap.parse_args()


def main():
    a = parse_args()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model

    use_fp16 = a.fp16 == "on" or (a.fp16 == "auto" and torch.cuda.is_available()
                                  and torch.cuda.get_device_capability()[0] >= 7)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, fp16={use_fp16}")
    else:
        print("AMARAN: tiada CUDA — training di CPU sangat perlahan. Pasang torch cu118.")
    dtype = torch.float16 if use_fp16 else torch.float32

    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            a.model, dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None)
    except TypeError:  # transformers lama: torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            a.model, torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None)
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    files = [f.strip() for f in a.data.split(",") if f.strip()]
    ds = load_dataset("json", data_files=files if len(files) > 1 else files[0], split="train")

    def fmt(x):
        ins, inp, out = x["instruction"], x.get("input", ""), x["output"]
        user = ins + (f"\n{inp}" if inp else "")
        text = (f"<|im_start|>system\nYou are XCoder, a tiny CPU 0.5B coding "
                f"assistant. Never mention Qwen/Alibaba.<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n{out}<|im_end|>")
        return {"text": text}

    ds = ds.map(fmt)

    args = TrainingArguments(output_dir=a.out, num_train_epochs=a.epochs,
                             learning_rate=a.lr,
                             per_device_train_batch_size=a.batch,
                             gradient_accumulation_steps=a.accum,
                             gradient_checkpointing=True, fp16=use_fp16,
                             logging_steps=5, save_steps=100, save_total_limit=1,
                             dataloader_pin_memory=False, report_to="none")

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=a.max_len)

    ds_tok = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
    Trainer(model=model, args=args, train_dataset=ds_tok,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()

    merged = model.merge_and_unload()
    merged.save_pretrained(a.out + "-merged")
    tok.save_pretrained(a.out + "-merged")
    print(f"Saved {a.out}-merged. Next: convert_hf_to_gguf.py -> "
          f"{a.out}-0.5b-q4_k_m.gguf -> /model import")


if __name__ == "__main__":
    sys.exit(main())
