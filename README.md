# XCoder — pembantu pengekodan lokal (CPU)

Model penaakulan terkini: **xcoder-gen1-0.8B** (Q4_K_M, thinking/reasoning)
+ **xcoder-javascript 1.5B** + 16 pakar **0.5B** (~45–55 T/s) — satu aktif
pada satu masa. Bahasa: **Melayu 🇲🇾 + English 🇬🇧**.
Stack: JS, TS, Node.js, HTML, CSS, SQL, PostgreSQL, MySQL, Python, PHP, Laravel,
Tailwind, Linux, Bash, Docker, general.

Jawapan lalai berjalan **geganti (relay)**: base jawab dulu → pakar berkaitan
sambung bergilir (unload automatik bila tukar model) → base kumpul + susun
antara setiap kaki → kembali ke base. Kelajuan setiap kaki dipapar live.
Jejak **fikiran (thinking/reasoning)** dipapar automatik (dim) sebelum jawapan
akhir — tiada arahan khas diperlukan.

## Pasang satu arahan

```powershell
git clone https://github.com/qubexs/train-my-ai.git; cd train-my-ai
powershell -ExecutionPolicy Bypass -File install.ps1
```

Ini semak Python 3.9+, muat turun model asas (~1.1GB Qwen3.5-2B + pakar 0.5B),
dan ambil enjin llama.cpp. Model lama diarkib dalam `models/0.5b/`.

### Model penaakulan (thinking/reasoning)

`xcoder-gen1-0.8B` dilatih untuk berfikir dahulu, kemudian menjawab. Muat turun
terus dari Hugging Face ke `models/`:

```powershell
Invoke-WebRequest `
  -Uri "https://huggingface.co/myxcoder/xcoder/resolve/main/xcoder-gen1-0.8b-q4_k_m.gguf" `
  -OutFile "models\xcoder-gen1-0.8b-q4_k_m.gguf"
```

Fail dikesan automatik oleh `/models`; pilih sebagai model aktif dengan
`/model xcoder-gen1-0.8b-q4_k_m`. Semasa menjana, jejak `berfikir` muncul dim
diikuti jawapan akhir. Butiran model:
<https://huggingface.co/myxcoder/xcoder/blob/main/xcoder-gen1-0.8b-q4_k_m.gguf>

## Mula cepat

```powershell
.\ezcodex-cpu.bat --lang ms        # mod CPU (lalai harian, jimat kuasa)
.\ezcodex-gpu.bat --lang ms        # mod GPU (GTX 1070 untuk latihan)
.\ezcodex.bat --lang ms            # dari sumber (Python 3.9+)
py app\ezcodex.py --lang ms --tui  # mod TUI
```

Pelayan resident ialah lalai interaktif (model dimuat sekali, kekal untuk giliran
berikutnya; unload bila melahu). `--once`/pipe guna `llama-cli` (0 RAM resident):

```powershell
.\load.bat                          # llama-server di :8080 (manual)
.\ezcodex.bat --backend llama       # paksa mod sejuk (tiada resident)
```

Dasar: **inferens CPU** (jimat kuasa/RAM, 0 RAM ketika melahu) —
**GPU dikhaskan untuk latihan** (`--gpu cuda` hanya opt-in).

## Perintah utama

| Perintah | Maksud |
|---|---|
| `/plan <tugas>` | pelan binaan bernombor (maks 6 langkah) |
| `/bina <tugas>` | laksana pelan automatik, tukar pakar setiap langkah |
| `/models` | senarai pakar + domain + T/s terakhir |
| `/update` | semak git + model lokal/HF baharu, tawar pull/daftar/muat turun |
| `/model <nama>` | tukar pakar aktif manual |
| `/tools on\|off` | gelung tool agentic (Tab = suis Build/Plan) |
| `/good` / `/bad` | nilai jawapan terakhir → log latihan (flywheel) |
| `/export sft` | pecah training log ke `datasets/<stack>.jsonl` + gabung `finetune/` |
| `/route on\|off` | auto-tukar pakar ikut domain (lalai on) |
| `/online on\|off` | carian internet (DuckDuckGo + Wikipedia) + auto-belajar (lalai on) |
| `/search <soalan>` | cari terus di web, simpan nota ke `docs/learned/` |
| `/learned` | bilangan nota pembelajaran + 5 terkini |
| `/retrain [status\|full]` | latih delta model dgn data baharu (auto pada 200 baris/50 nota) |

Soalan biasa dijawab automatik gaya relay (base → pakar → base kumpul);
tidak perlu pilih model manual. Bar bawah TUI tunjuk mod **Build/Plan**,
skrol PgUp/PgDn, roda tetikus skrol hasil, `(Done.)` bila di bawah sekali.
Tool (`list read write edit run bash rag`) dibenarkan dalam workspace sahaja.

## Latihan (flywheel)

```
chat + /good                       # training.jsonl (auto-tag domain)
py app/ezcodex.py train --mode sft --domain docker   # pecah ke datasets/docker.jsonl
python datasets/scripts/generate.py docker           # drill templat (idempoten)
python datasets/scripts/merge.py docker --eval 0.05  # gabung + held-out eval.jsonl
python datasets/scripts/validate.py                  # semak semua stack
finetune\train_all.py --only xcoder-docker --retrain --epochs 3   # GPU: hanya yang berubah
/model add <fail.gguf> --name xcoder-docker --domains docker       # daftar semula
```

Korpus generalis (`xcoder-best`): `python datasets/scripts/merge_all.py`
gabung 16 stack → `finetune/stacks/all/` (train + held-out eval, tiada bocor).
Skrip Colab: `python cmdsyntax/app.py` (GUI: pilih dataset + model asas Qwen/Gemma/Llama).

Peraturan: latih semula **hanya stack yang berubah** (`--only`) — jangan train-all.
Data disemak di `datasets/` (lihat `datasets/README.md` untuk SOP rujukan web
+ kuarantin mistag); `finetune/stacks/` ialah output janaan `merge.py`.

## Belajar dalam talian + delta retrain (auto)

Bila `online` on, model boleh cari web dan simpan nota sendiri ke
`docs/learned/` (masuk RAG + `training.jsonl`). Bila data baharu cukup
(200 baris ATAU 50 nota), CLI latih **delta LoRA** di latar pada GPU
(`.venv-gpu` + CUDA; tanpa GPU, data terkumpul sahaja — tiada ralat), sambung
dari adapter master, merge → GGUF `*.gguf.new`, dan aktifkan automatik bila
fail model tidak dikunci oleh proses lain (lama -> `.bak`). Chat tak terganggu.
Sekali sahaja: salin adapter semasa `drive_upload\DriveXcoder\xcoder\v2\checkpoint-*`
ke `finetune\masters\xcoder-gen-0.8b\`. `--offline` utk matikan carian.
Log latihan: `finetune\.train_run.log`; benih replay dari `datasets\all.jsonl`.

## Struktur projek

```text
app/ezcodex.py      CLI agentic (relay lalai base->pakar->base, /bina, route)
app/agent.py        prompt sistem + gelung tool + sanitasi (penapis CJK/fikir)
app/experts.py      registry pakar, pengesan domain/stack
app/backends.py     llama-cli (lalai) | llama-server | lmstudio
app/server.py       ModelManager resident, idle-unload
app/tools.py        list/read/write/edit/run/bash/rag (workspace-jailed)
app/store.py        sesi, training log, export, merge + eval split
app/tui.py          paparan TUI: kesan menaip, dot animasi, roda tetikus, status Build/Plan
datasets/           data latih per stack + scripts/ (lihat datasets/README.md)
datasets/scripts/merge_all.py   korpus generalis -> finetune/stacks/all/
cmdsyntax/          penjana skrip Colab T4 (GUI + model Qwen/Gemma/Llama)
finetune/           skrip latihan LoRA (lihat STEP.md)
models/             *.gguf + models.json (diabaikan git); models/0.5b/ arkib lama
load.bat            hidupkan llama-server resident
ezcodex.bat         jalan dari sumber
```

## Lesen

MIT — lihat `LICENSE`.
