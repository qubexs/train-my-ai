# XCoder — pembantu pengekodan lokal (CPU)

Model kecil di mesin sendiri: **base Qwen3.5-2B** (penaakulan, ~14 T/s)
+ **xcoder-javascript 1.5B** + 16 pakar **0.5B** (~45–55 T/s) — satu aktif
pada satu masa. Bahasa: **Melayu 🇲🇾 + English 🇬🇧**.
Stack: JS, TS, Node.js, HTML, CSS, SQL, PostgreSQL, MySQL, Python, PHP, Laravel,
Tailwind, Linux, Bash, Docker, general.

Jawapan lalai berjalan **geganti (relay)**: base jawab dulu → pakar berkaitan
sambung bergilir (unload automatik bila tukar model) → base kumpul + susun
antara setiap kaki → kembali ke base. Kelajuan setiap kaki dipapar live.

## Pasang satu arahan

```powershell
git clone https://github.com/qubexs/train-my-ai.git; cd train-my-ai
powershell -ExecutionPolicy Bypass -File install.ps1
```

Ini semak Python 3.9+, muat turun model asas (~1.1GB Qwen3.5-2B + pakar 0.5B),
dan ambil enjin llama.cpp. Model lama diarkib dalam `models/0.5b/`.

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
