"""Continued pretraining template — run on GPU (Colab/Kaggle), NOT on 7.8GB CPU box.
Corpus: finetune/<domain>/corpus.txt (exported via `py app/ezcodex.py train --mode pretrain [--domain X]`).

Colab:
  !pip install -q transformers datasets accelerate
  !python app/train_pretrain.py --model Qwen/Qwen2.5-0.5B-Instruct --corpus finetune/general/corpus.txt
Then convert merged model -> GGUF (see step.md), `lms import` back.
"""
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--corpus", default="finetune/general/corpus.txt")
    ap.add_argument("--out", default="xcoder-0.5b-pretrained")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seq", type=int, default=1024)
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments, DataCollatorForLanguageModeling)
    ds = load_dataset("text", data_files=a.corpus, split="train")
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tok_fn(x):
        return tok(x["text"], truncation=True, max_length=a.seq)

    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    model = AutoModelForCausalLM.from_pretrained(a.model)
    args = TrainingArguments(output_dir=a.out, num_train_epochs=a.epochs,
                             learning_rate=a.lr, per_device_train_batch_size=1,
                             gradient_accumulation_steps=8, fp16=True,
                             save_steps=500, logging_steps=20)
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"Saved {a.out}. Next: convert to GGUF (llama_cpp.convert_hf_to_gguf), "
          "download .gguf, lms import on Windows.")

if __name__ == "__main__":
    main()
