# Held-out loss pada finetune/stacks/<stack>/eval.jsonl (GPU disyor, CPU boleh perlahan).
# Jalankan DALAM .venv-gpu (torch + transformers):
#   python finetune/eval_loss.py --model Qwen/Qwen2.5-0.5B-Instruct --eval finetune/stacks/docker/eval.jsonl
#   python finetune/eval_loss.py --model build/train/xcoder-docker-merged --eval ... [--max 200]
# Format prompt + masking MESTI cermin train_local.py (loss pada jawapan sahaja).
import argparse
import json
import sys


def parse_args():
    ap = argparse.ArgumentParser(description="Held-out loss (answer tokens only)")
    ap.add_argument("--model", required=True, help="HF id atau folder merged")
    ap.add_argument("--eval", required=True, help="finetune/stacks/<stack>/eval.jsonl")
    ap.add_argument("--max", type=int, default=200, help="maks baris eval")
    ap.add_argument("--max-len", type=int, default=1024)
    return ap.parse_args()


def fmt(r, eos=""):
    user = r["instruction"] + (f"\n{r['input']}" if r.get("input") else "")
    prompt = ("<|im_start|>system\nYou are XCoder, a tiny CPU 0.5B coding "
              "assistant. Never mention Qwen/Alibaba.<|im_end|>\n"
              f"<|im_start|>user\n{user}<|im_end|>\n")
    return prompt, prompt + f"<|im_start|>assistant\n{r['output']}<|im_end|>" + eos


def main():
    a = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = []
    for line in open(a.eval, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if (r.get("instruction") or "").strip() and (r.get("output") or "").strip():
            rows.append(r)
    rows = rows[:a.max]
    if not rows:
        raise SystemExit("eval kosong — jana dahulu: merge.py <stack> --eval 0.05")

    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("AMARAN: CPU — perlahan. Guna GPU untuk suite penuh.")
    model = AutoModelForCausalLM.from_pretrained(a.model).to(dev).eval()

    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for r in rows:
            prompt, full = fmt(r, tok.eos_token)
            fids = tok(full, truncation=True, max_length=a.max_len)["input_ids"]
            pids = tok(prompt, truncation=True, max_length=a.max_len)["input_ids"]
            n = 0
            for pf, ff in zip(pids, fids):
                if pf != ff:
                    break
                n += 1
            n = max(0, n - 2)
            ids = torch.tensor([fids], device=dev)
            labels = torch.tensor([[-100] * n + fids[n:]], device=dev)
            out = model(input_ids=ids, labels=labels)
            n_tok = sum(1 for l in labels[0].tolist() if l != -100)
            total_nll += out.loss.item() * n_tok
            total_tok += n_tok
    loss = total_nll / max(1, total_tok)
    import math
    print(f"EVAL {a.eval}: rows={len(rows)} loss={loss:.4f} ppl={math.exp(loss):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
