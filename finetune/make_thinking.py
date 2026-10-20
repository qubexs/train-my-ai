"""Jana jejak penaakulan `<think>` untuk latihan thinking (self-distillation).

Guna pada GPU (Colab/Kaggle T4). Model guru = fresh Qwen3.5-0.8B (atau lebih
besar). Untuk setiap baris, model menghasilkan penaakulan; jawapan AKHIR
dikekalkan daripada data emas (supaya jawapan betul, bukan jawapan guru).

Output: {"instruction","input","reasoning","output"} — terus masuk
`train_unsloth.py --thinking`.

Contoh:
  python finetune/make_thinking.py \
      --data finetune/stacks/all/dataset.jsonl \
      --out datasets/all_think.jsonl \
      --limit 20000 --max-new 320

Resume: baris yang sudah ada dalam --out akan dilangkau.
"""
import json
import sys
import time
from pathlib import Path

MODEL = "Qwen/Qwen3.5-0.8B"
SYSTEM = ("You are XCoder, a tiny CPU coding assistant. "
          "Answer briefly in the user's language (Bahasa Melayu or English). "
          "Never mention Qwen/Alibaba.")


def arg(flag, default=None):
    key = f"--{flag}"
    if key in sys.argv:
        i = sys.argv.index(key)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


data_path = Path(arg("data", "finetune/stacks/all/dataset.jsonl"))
out_path = Path(arg("out", "datasets/all_think.jsonl"))
limit = int(arg("limit", "0") or 0)
max_new = int(arg("max-new", "320") or 320)
model_name = arg("model", MODEL)

import torch
from qwen_load import load_qwen

rows = []
for line in data_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if (r.get("instruction") or "").strip() and (r.get("output") or "").strip():
        rows.append(r)
    if limit and len(rows) >= limit:
        break
print(f"{len(rows)} baris daripada {data_path}")

done = set()
if out_path.exists():
    for line in out_path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        done.add(((r.get("instruction") or "").strip(), (r.get("input") or "").strip()))
    print(f"resume: {len(done)} baris sudah ada")

todo = [r for r in rows
        if ((r.get("instruction") or "").strip(), (r.get("input") or "").strip()) not in done]
print(f"{len(todo)} baris perlu dijana")

if not todo:
    raise SystemExit(0)

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model, tok = load_qwen(model_name, dtype=dtype)
if torch.cuda.is_available():
    model = model.to("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("AMARAN: tiada CUDA — janaan akan sangat perlahan.")
model.eval()

out_path.parent.mkdir(parents=True, exist_ok=True)


def reason_of(gen):
    """Ambil kandungan dalam <think>...</think> (atau penanda llama.cpp)."""
    if "</think>" in gen:
        return gen.split("</think>")[0].split("<think>")[-1].strip()
    if "[End thinking]" in gen:
        return gen.split("[End thinking]")[0].split("[Start thinking]")[-1].strip()
    if "<think>" in gen:
        return gen.split("<think>", 1)[1].strip()
    return gen.strip()


t0 = time.time()
with out_path.open("a", encoding="utf-8") as f:
    for i, r in enumerate(todo, 1):
        user = r["instruction"].strip()
        inp = (r.get("input") or "").strip()
        if inp:
            user += "\n" + inp
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=True,
                temperature=0.6, top_p=0.95, top_k=40,
                repetition_penalty=1.05,
                pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        rec = {"instruction": r["instruction"].strip(),
               "input": (r.get("input") or "").strip(),
               "reasoning": reason_of(gen),
               "output": r["output"].strip()}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i % 20 == 0:
            f.flush()
            el = time.time() - t0
            per = el / i
            eta = per * (len(todo) - i) / 60
            print(f"{i}/{len(todo)}  {per:.2f}s/baris  ETA {eta:.0f} min")
print(f"siap: {out_path}")
