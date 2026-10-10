# step.md — How to train your AI (XCoder-0.5B)

Layout data: `datasets/<stack>.jsonl` (16 stack) -> `finetune/stacks/<stack>/dataset.jsonl`
(stack = domain, satu tier sahaja).
Validasi semua: `python finetune/validate.py` · satu domain: `python finetune/validate.py docker`.

Goal: turn base `Qwen2.5-0.5B-Instruct` into a model that **natively** says
“I am XCoder” and answers Malay + code well — no prompt filter needed.
Method: **instruction fine-tune with LoRA (QLoRA 4-bit)**, not pretraining from scratch.

Do NOT train on your 7.8GB CPU box (too slow). Train on a **free Colab T4 GPU**,
then import the `.gguf` back to LM Studio on Windows.

## Step 0 — What you have
- `datasets/<stack>.jsonl` — data mengikut stack (javascript, docker, sql, ...).
  Format sebaris: `{"instruction": "...", "input": "...", "output": "..."}`
- `finetune/stacks/<stack>/dataset.jsonl` — fail **terjana** via `merge.py` untuk Colab.
- Alir: `train` → `datasets/` → `merge.py` → `finetune/` → Colab → GGUF → `/model add`.

## Step 1 — Grow the dataset (important)
1. Chat + `/good` dalam CLI (auto-tag domain+stack), kemudian:
```powershell
py app/ezcodex.py train --mode sft --domain docker  # -> datasets/docker.jsonl
python datasets/scripts/merge.py docker              # -> finetune/stacks/docker/dataset.jsonl
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

## Step 2 — Train (pilih satu)
### A. GPU tempatan (GTX 1070 8GB — disyorkan jika ada)
Pascal tiada sokongan Unsloth/bitsandbytes — guna LoRA standard:
```powershell
py -m venv .venv-gpu; .\.venv-gpu\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate peft sentencepiece
python finetune/train_local.py --data datasets/docker.jsonl --out xcoder-docker
# ~10-20 min untuk 500 baris. Kemudian export GGUF (lihat komen dalam skrip).
```
Batch semua pakar satu demi satu (langkau yang siap, daftar automatik ke `models/`):
```powershell
python finetune/train_all.py --dry-run   # semak pelan dahulu
python finetune/train_all.py             # jalan berjam-jam; biarkan semalaman
```
### B. Colab (GPU T4 percuma)
1. Go to Google Colab → `Runtime → Change runtime type → T4 GPU`.
2. Upload `finetune/stacks/<stack>/dataset.jsonl` (rename dalam `/content/` jika perlu, elak `dataset (1).jsonl`).
3. Install:
```
!pip install -q unsloth trl peft accelerate bitsandbytes datasets
```
4. Jalankan `train_unsloth.py` dengan `DATA` = nama fail, `OUT` = `xcoder-<stack>`
   (~30–60 min untuk 0.5B, 100+ baris, 3 epochs).
5. If out-of-memory: lower `max_seq_length` to 512 or batch size to 1.

## Step 3 — Export to GGUF (in Colab, cara kalis-versi)
```python
model.save_pretrained_gguf(OUT + "-gguf", tok, quantization_method="q4_k_m")
```
Download `.gguf` (~350MB), rename ikut konvensyen
`xcoder-<kepakaran>-0.5b-q4_k_m.gguf` (cth `xcoder-html-0.5b-q4_k_m.gguf`), ke PC.

## Step 4 — Daftar sebagai pakar (auto-import)
Lepas download `.gguf` dari Colab, dalam CLI:
```
anda> /model import
# auto: GGUF terbaru dalam Downloads disalin ke models/, didaftar
# (nama+domain dari konvensyen xcoder-<pakar>-0.5b-q4_k_m.gguf), terus aktif.
anda> /models     # sahkan
anda> /route on   # auto-hala soalan ke pakar ini
```
Manual: `/model add <fail> --name xcoder-docker --domains docker`.

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

## Step 7 — Model pakar per stack (multi-model, satu tier)
Satu model kecil satu stack (= domain), tukar tanpa model besar.
Konvensyen nama: `xcoder-<stack>-0.5b-q4_k_m.gguf`, data di `datasets/<stack>.jsonl`
(ada 16 stack: javascript, typescript, nodejs, html, css, tailwind, sql, postgresql,
mysql, linux, docker, bash, python, laravel, php, general):
1. Kumpul data per stack dalam CLI: chat + `/good` (auto-tag stack).
2. `py app/ezcodex.py train --mode sft` (pecah ke stack) + `merge.py` (sahkan ke finetune/stacks/).
3. Tambah baris stack (penting untuk kesan ketara), latih di Colab seperti Step 2–3, nama output `xcoder-docker-0.5b-q4_k_m.gguf`.
4. Daftar: `/model add <fail> --name xcoder-docker --domains docker` → aktif dengan `/model xcoder-docker` atau auto via `--autoroute` / `/route on`.
5. Ulang untuk stack lain. Registry: `models/models.json` (per-mesin, auto-seed 16 pakar).

## Troubleshooting
- Colab no GPU: check `Runtime → Change runtime type`, or use Kaggle free GPU.
- `bitsandbytes` install fail: use latest Colab runtime, Python 3.10+.
- GGUF too big: use `--outtype q4_k_m` (~350MB, slightly lower quality).
- `lms import` fail: put the `.gguf` in `C:\Users\testlab\.lmstudio\models\` manually, then `lms ls`.
- Still says Alibaba: dataset needs more identity rows + lower learning rate (1e-4), more epochs (5).

## Alternative tool
- **LLaMA-Factory** (easiest UI): `pip install llamafactory`, use its Alpaca template with the same `dataset.jsonl`, LoRA, then same GGUF export.
