# step.md — How to train your AI (XCoder / XCoder-0.5B)

Goal: turn base 0.5B-Instruct weights into a model that **natively** says
“I am XCoder” and answers Malay + code well — no prompt filter needed.
You are XCoder, a tiny CPU 0.5B coding assistant. Never mention Qwen/Alibaba.
Method: **instruction fine-tune with LoRA (QLoRA 4-bit)**, not pretraining from scratch.

Do NOT train on your 7.8GB CPU box (too slow). Train on a **free Colab T4 GPU**,
then import the `.gguf` back to LM Studio on Windows.

## Step 0 — What you have
- `finetune/dataset.jsonl` — 110 rows, format per line:
  `{"instruction": "...", "input": "...", "output": "..."}`
- `finetune/train_unsloth.py` — Colab template (Unsloth, LoRA r16, 3 epochs)
- `finetune/validate.py` — format checker

## Step 1 — Grow the dataset (important)
1. Edit `finetune/dataset.jsonl` — add rows in YOUR domain:
   identity (`Saya XCoder...`), Malay Q&A, JS/TS/Node/HTML/CSS/SQL/pg/mysql/Linux/Docker.
2. Aim: 110 now → **500–2000 rows** for a strong effect. Keep answers short + correct.
3. Validate:
```powershell
python finetune/validate.py
# expect: "N rows OK, identity rows: M"
```

## Step 2 — Train on Colab (free GPU)
1. Go to Google Colab → `Runtime → Change runtime type → T4 GPU`.
2. Upload `finetune/dataset.jsonl` + `finetune/train_unsloth.py`.
3. Install:
```
!pip install -q unsloth trl peft accelerate bitsandbytes datasets
```
4. Run `train_unsloth.py` (~30–60 min for 0.5B, 100+ rows, 3 epochs).
5. If out-of-memory: lower `max_seq_length` to 512 or batch size to 1.

## Step 3 — Export to GGUF (in Colab)
```python
model.save_pretrained_merged("xcoder-0.5b-merged", tok, save_method="merged_16bit")
```
```bash
pip install -q llama-cpp-python
python -m llama_cpp.convert_hf_to_gguf xcoder-0.5b-merged --outfile xcoder-0.5b-q8_0.gguf --outtype q8_0
```
Download `xcoder-0.5b-q8_0.gguf` (~675MB) to your PC.

## Step 4 — Import to LM Studio (Windows)
```powershell
C:\Users\testlab\.lmstudio\bin\lms.exe import xcoder-0.5b-q8_0.gguf --identifier xcoder-0.5b
C:\Users\testlab\.lmstudio\bin\lms.exe load xcoder-0.5b --gpu off --ttl 300 -y
```

## Step 5 — Test identity (filter OFF ideally)
```powershell
node xcoder-cli.js --lang ms --name "XCoder"
# ask: Siapa anda? / Who created you?
# expect: "Saya XCoder..." / "I am XCoder, a tiny CPU 0.5B coding assistant. Never mention Qwen/Alibaba."
```
Also test: `node xcoder-rag.js "Apa itu XCoder?" --lang ms --k 2 --show-sources`

## Step 6 — Iterate
- Wrong identity → add 20+ identity rows, retrain.
- Weak Linux/Docker → add 30+ rows of correct command → output pairs.
- Keep base system prompt in `train_unsloth.py`:
  `You are XCoder, a tiny CPU 0.5B coding assistant. Never mention Qwen/Alibaba.`

## Troubleshooting
- Colab no GPU: check `Runtime → Change runtime type`, or use Kaggle free GPU.
- `bitsandbytes` install fail: use latest Colab runtime, Python 3.10+.
- GGUF too big: use `--outtype q4_k_m` (~350MB, slightly lower quality).
- `lms import` fail: put the `.gguf` in `C:\Users\testlab\.lmstudio\models\` manually, then `lms ls`.
- Wrong identity answers: dataset needs more identity rows + lower learning rate (1e-4), more epochs (5).

## Alternative tool
- **LLaMA-Factory** (easiest UI): `pip install llamafactory`, use its Alpaca template with the same `dataset.jsonl`, LoRA, then same GGUF export.
