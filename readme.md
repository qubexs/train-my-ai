# EZCodex-0.5B / Coder 77 — CPU-only coding assistant

Tiny local AI: **0.5B params, CPU-only, 5-min auto-unload** to save power.
Base weights: `Qwen2.5-0.5B-Instruct` (LM Studio), public name: `ezcodex-0.5b` / `Coder 77`.
Languages: **Malay 🇲🇾 + English 🇬🇧**.
Stacks: JS, TS, Node.js, HTML, CSS, SQL, PostgreSQL, MySQL, **Linux, Docker**, Bash, Python.

## Requirements
- Windows, ~8GB RAM (tested 7.8GB), Python 3.14, Node 24
- LM Studio (local server on port 1234) + model `Qwen2.5-0.5B-Instruct-GGUF` (~676MB)

## Standalone app (no LM Studio / Ollama / pip install)
One command (Windows PowerShell, from this folder):
```powershell
powershell -ExecutionPolicy Bypass -File app/install.ps1
```
Then:
```powershell
.\ezcodex.bat --lang ms
```
Linux/Mac: `python3 app/ezcodex.py --lang ms` (needs Python 3.9+ only).
First run auto-downloads engine (~19MB) + model (~491MB) into `app/bin/`, `app/models/`.

## Quickstart (LM Studio path)
```powershell
# 1. Load CPU-only, auto-unload after 5 min idle
C:\Users\testlab\.lmstudio\bin\lms.exe load qwen2.5-0.5b-instruct --gpu off --identifier ezcodex-0.5b --ttl 300 -y

# 2. Hello test
python ezcodex-hello.py
node ezcodex-hello.js

# 3. Chat CLI (static input bottom, answer above, T/s bottom-right)
node ezcodex-cli.js --lang ms --name "Coder 77"

# 4. One-shot chat + code
node ezcodex-chat.js --lang ms "Apa itu EZCodex?"
node ezcodex-code.js --stack js "write tambah(a,b) with example"
node ezcodex-code.js --stack docker "Dockerfile untuk Node.js app"
node ezcodex-code.js --stack linux "cari teks dalam semua fail"

# 5. RAG (grounded answers from docs/)
node ezcodex-rag.js "Apa itu EZCodex?" --lang ms --k 2 --show-sources

# 6. Tools (safe whitelist: list/read/run + AI explanation)
node ezcodex-tools.js --tool list --arg docs
node ezcodex-tools.js --tool read --arg docs/js-basics.md
node ezcodex-tools.js --tool run --arg "console.log([1,2,3].map(x=>x*2))" --lang ms

# 7. Python sample app
python sample_app.py --lang ms --ask "Apakah fungsi tambah dalam Python?"
python sample_app.py --lang en --code py --task "write tambah(a,b) with example"

# 8. Power-save: ensure loaded (auto-reload ~6s if TTL-unloaded)
node ezcodex-ensure.js
```

## Power-save
- `--ttl 300`: unloads after 5 min idle (`lms ps` → `No models loaded`, 0 RAM/power).
- `node ezcodex-ensure.js` before any call reloads automatically.
- Pattern: `node ezcodex-ensure.js; node ezcodex-cli.js --lang ms`

## Project structure
```text
ezcodex-hello.py / .js   minimal CPU proof
ezcodex-chat.js          one-shot bilingual chat
ezcodex-cli.js           interactive CLI (identity filter, static input, T/s meter)
ezcodex-code.js          coding assistant (js|ts|node|html|css|sql|pg|mysql|linux|docker|bash)
ezcodex-rag.js           RAG over docs/ (zero deps)
ezcodex-tools.js         safe tools: list/read/run + explanation
ezcodex-ensure.js        auto-reload if TTL-unloaded
sample_app.py            Python sample app
docs/                    RAG knowledge (ezcodex-ms, js-basics, node-pg, linux-basics, docker-basics)
finetune/                training kit (see step.md)
```

## Notes / limits
- 0.5B hallucinates without context — **always use RAG** for facts.
- Identity (`Coder 77`, not Qwen/Alibaba) is currently a CLI prompt+filter override.
  Permanent fix = fine-tune (see `step.md`), then the filter can be removed.
- Run calls sequentially on low RAM; parallel model calls can kill the worker.
