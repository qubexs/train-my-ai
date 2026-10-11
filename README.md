# XCoder — pembantu pengekodan lokal (CPU)

Model kecil di mesin sendiri: **18 pakar 0.5B** (satu aktif pada satu masa, ~70 T/s CPU)
+ **gemma4-e4b** sebagai pengurus am + penyemak (~10 T/s). Bahasa: **Melayu 🇲🇾 + English 🇬🇧**.
Stack: JS, TS, Node.js, HTML, CSS, SQL, PostgreSQL, MySQL, Python, PHP, Laravel,
Tailwind, Linux, Bash, Docker, general.

## Pasang satu arahan

```powershell
git clone https://github.com/qubexs/train-my-ai.git; cd train-my-ai
powershell -ExecutionPolicy Bypass -File install.ps1
```

Ini semak Python 3.9+, muat turun 18 model dari
`tokkilejen/xcoder` (~6.5GB, skip yang sudah ada + disahkan),
dan ambil enjin llama.cpp. Pilihan: `install.ps1 -Minimal` (asas+general, ~1GB).

## Mula cepat

```powershell
.\ezcodex-cpu.bat --lang ms        # mod CPU (lalai harian, jimat kuasa)
.\ezcodex-gpu.bat --lang ms        # mod GPU (GTX 1070; gemma ~36 T/s vs ~10 CPU)
.\ezcodex-0445.exe --lang ms       # panggil exe versi tertentu terus
.\ezcodex.bat --lang ms            # dari sumber (Python 3.9+)
py app\ezcodex.py --lang ms --tui  # mod TUI
```

Pelayan resident ialah lalai interaktif (model dimuat sekali, kekal untuk giliran
berikutnya; unload bila melahu). `--once`/pipe guna `llama-cli` (0 RAM resident):

```powershell
.\load.bat                          # llama-server + xcoder-general di :8080 (manual)
.\ezcodex.bat --backend llama       # paksa mod sejuk (tiada resident)
```

Dasar: **inferens CPU** (jimat kuasa/RAM, 0 RAM ketika melahu) —
**GPU dikhaskan untuk latihan** (`--gpu cuda` hanya opt-in).

## Perintah utama

| Perintah | Maksud |
|---|---|
| `/kerja <soalan>` | pakar berkaitan jawab setiap satu, gemma semak + skor + putuskan |
| `/semua <soalan>` | semua model jawab, setiap satu result sendiri + ringkasan T/s |
| `/consult <soalan>` | runding 2 pakar, pengurus sintesis |
| `/plan <tugas>` | pelan binaan bernombor (maks 6 langkah) |
| `/bina <tugas>` | laksana pelan automatik, tukar pakar setiap langkah |
| `/models` | senarai pakar + domain + T/s terakhir |
| `/model <nama>` | tukar pakar aktif manual |
| `/good` / `/bad` | nilai jawapan terakhir → log latihan (flywheel) |
| `/export sft` | pecah training log ke `datasets/<stack>.jsonl` + gabung `finetune/` |
| `/route on\|off` | auto-tukar pakar ikut domain (lalai on) |

Soalan terbuka/tutorial/rentas-stack (cth. `node.js typescript`) automatik ke pengurus;
satu stack spesifik ke pakar 0.5B yang pantas. Tool (`list read write edit run bash rag`)
dibenarkan dalam workspace sahaja.

## Latihan (flywheel)

```
chat + /good                       # training.jsonl (auto-tag domain)
py app\ezcodex.py train --mode sft --domain docker   # pecah ke datasets/docker.jsonl
python datasets\scripts\merge.py docker --eval 0.05  # gabung + held-out eval.jsonl
python datasets\scripts\validate.py                  # semak semua stack
finetune\train_all.py --only xcoder-docker --retrain --epochs 3   # GPU: hanya yang berubah
/model add <fail.gguf> --name xcoder-docker --domains docker       # daftar semula
```

Peraturan: latih semula **hanya stack yang berubah** (`--only`) — jangan train-all.
Data disemak di `datasets/`; `finetune/stacks/` ialah output janaan `merge.py`.

## Struktur projek

```text
app/ezcodex.py      CLI agentic (perintah, route, /kerja, /bina)
app/agent.py        prompt sistem + gelung tool + sanitasi
app/experts.py      registry pakar, pengesan domain/stack
app/backends.py     llama-cli (lalai) | llama-server | lmstudio
app/server.py       ModelManager resident, idle-unload
app/tools.py        list/read/write/edit/run/bash/rag (workspace-jailed)
app/store.py        sesi, training log, export, merge + eval split
app/tui.py          paparan TUI (stdlib ANSI)
datasets/           data latih per stack + scripts/ (lihat datasets/README.md)
finetune/           skrip latihan LoRA (lihat STEP.md)
models/             *.gguf + models.json (diabaikan git)
load.bat            hidupkan llama-server resident
ezcodex.bat         jalan dari sumber
```

## Lesen

MIT — lihat `LICENSE`.
