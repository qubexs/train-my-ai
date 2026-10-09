# step.md — How to train your AI (XCoder-0.5B)

Layout data: `finetune/<kepakaran>/dataset.jsonl` — satu folder satu domain
(`general`, `docker`, `web`, `data`, `linux`, `python`).
Validasi semua: `python finetune/validate.py` · satu domain: `python finetune/validate.py docker`.

Goal: turn base `Qwen2.5-0.5B-Instruct` into a model that **natively** says
“I am XCoder” and answers Malay + code well — no prompt filter needed.
Method: **instruction fine-tune with LoRA (QLoRA 4-bit)**, not pretraining from scratch.

Do NOT train on your 7.8GB CPU box (too slow). Train on a **free Colab T4 GPU**,
then import the `.gguf` back to LM Studio on Windows.

## Step 0 — What you have
- `datasets/<stack>.jsonl` — data mengikut stack (javascript, docker, sql, ...).
  Format sebaris: `{"instruction": "...", "input": "...", "output": "..."}`
- `finetune/<domain>/dataset.jsonl` — fail **terjana** via `merge.py` untuk Colab.
- Alir: `train` → `datasets/` → `merge.py` → `finetune/` → Colab → GGUF → `/model add`.

## Step 1 — Grow the dataset (important)
1. Chat + `/good` dalam CLI (auto-tag domain+stack), kemudian:
```powershell
py app/ezcodex.py train --mode sft --domain docker  # -> datasets/docker.jsonl
python datasets/scripts/merge.py linux              # -> finetune/linux/dataset.jsonl
python datasets/scripts/validate.py docker          # semak stack
python datasets/scripts/statistics.py               # taburan semua stack
```
   Atau tulis baris manual dalam `datasets/<stack>.jsonl`.
2. Aim: 110 kini → **500–2000 baris** untuk kesan ketara. Jawapan pendek + betul.
   Baris baharu mesti guna identiti `XCoder` (baris lama `Coder 77` masih diterima validator).
3. Validate:
```powershell
python finetune/validate.py docker
# expect: "docker/dataset.jsonl: N rows OK, identity rows: M"
```

## Step 2 — Train on Colab (free GPU)
1. Go to Google Colab → `Runtime → Change runtime type → T4 GPU`.
2. Upload `finetune/<domain>/dataset.jsonl` (rename dalam `/content/` jika perlu, elak `dataset (1).jsonl`).
3. Install:
```
!pip install -q unsloth trl peft accelerate bitsandbytes datasets
```
4. Jalankan `train_unsloth.py` dengan `DATA` = nama fail, `OUT` = `xcoder-<domain>`
   (~30–60 min untuk 0.5B, 100+ baris, 3 epochs).
5. If out-of-memory: lower `max_seq_length` to 512 or batch size to 1.

## Step 3 — Export to GGUF (in Colab, cara kalis-versi)
```python
model.save_pretrained_gguf(OUT + "-gguf", tok, quantization_method="q4_k_m")
```
Download `.gguf` (~350MB), rename ikut konvensyen
`xcoder-<kepakaran>-0.5b-q4_k_m.gguf` (cth `xcoder-html-0.5b-q4_k_m.gguf`), ke PC.

## Step 4 — Daftar sebagai pakar (ganti import LM Studio lama)
```powershell
# dalam CLI:
# /model add E:\Downloads\xcoder-docker-0.5b-q4_k_m.gguf --name xcoder-docker --domains docker
# /models   (sahkan)   /route on   (auto-hala soalan Docker ke pakar ini)
```

## Step 5 — Test identity (filter OFF ideally)
```powershell
node ezcodex-cli.js --lang ms --name "Coder 77"
# ask: Siapa anda? / Who created you? / Are you Qwen or Alibaba?
# expect: "Saya XCoder..." with NO mention of Qwen/Alibaba.
```
Also test: `node ezcodex-rag.js "Apa itu EZCodex?" --lang ms --k 2 --show-sources`

## Step 6 — Iterate
- Wrong identity → add 20+ identity rows, retrain.
- Weak Linux/Docker → add 30+ rows of correct command → output pairs.
- Keep base system prompt in `train_unsloth.py`:
  `You are XCoder... Never mention Qwen/Alibaba.`

## Step 7 — Model pakar per domain (multi-model)
Satu model kecil satu domain, tukar tanpa model besar.
Konvensyen nama: `xcoder-<kepakaran>-0.5b-q4_k_m.gguf`, data di `datasets/<stack>.jsonl`
(ada 16 stack: javascript, typescript, nodejs, html, css, tailwind, sql, postgresql,
mysql, linux, docker, bash, python, laravel, php, general):
1. Kumpul data per domain dalam CLI: chat + `/good` (auto-tag stack).
2. `py app/ezcodex.py train --mode sft` (pecah ke stack) + `merge.py` (gabung ke domain).
3. Tambah 30–100 baris domain (penting untuk kesan ketara), latih di Colab seperti Step 2–3, nama output `xcoder-docker-0.5b-q4_k_m.gguf`.
4. Daftar: `/model add <fail> --name xcoder-docker --domains docker` → aktif dengan `/model xcoder-docker` atau auto via `--autoroute` / `/route on`.
5. Ulang untuk domain lain (web, data, linux). Registry: `models/models.json` (per-mesin, auto-seed).

## Troubleshooting
- Colab no GPU: check `Runtime → Change runtime type`, or use Kaggle free GPU.
- `bitsandbytes` install fail: use latest Colab runtime, Python 3.10+.
- GGUF too big: use `--outtype q4_k_m` (~350MB, slightly lower quality).
- `lms import` fail: put the `.gguf` in `C:\Users\testlab\.lmstudio\models\` manually, then `lms ls`.
- Still says Alibaba: dataset needs more identity rows + lower learning rate (1e-4), more epochs (5).

## Alternative tool
- **LLaMA-Factory** (easiest UI): `pip install llamafactory`, use its Alpaca template with the same `dataset.jsonl`, LoRA, then same GGUF export.
