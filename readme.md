# XCoder-0.5B — CPU-only coding assistant

Tiny local AI: **0.5B params, CPU-only, 5-min auto-unload** to save power.
Public name: `xcoder-0.5b` / `XCoder`.
Languages: **Malay 🇲🇾 + English 🇬🇧**.
Stacks: JS, TS, Node.js, HTML, CSS, SQL, PostgreSQL, MySQL, **Linux, Docker**, Bash, Python.

## Requirements
- Windows, ~8GB RAM (tested 7.8GB), Python 3.14, Node 24
- LM Studio (local server on port 1234) + model `xcoder-0.5b` (~676MB)

## Standalone app (no LM Studio / Ollama / pip install)
One command (Windows PowerShell, from this folder):
```powershell
powershell -ExecutionPolicy Bypass -File app/install.ps1
```
Then:
```powershell
.\xcoder.bat --lang ms
```
Linux/Mac: `python3 app/xcoder.py --lang ms` (needs Python 3.9+ only).
First run auto-downloads engine (~19MB) + model (~491MB) into `app/bin/`, `app/models/`.

## Quickstart (LM Studio path)
```powershell
# 1. Load CPU-only, auto-unload after 5 min idle
C:\Users\testlab\.lmstudio\bin\lms.exe load xcoder-0.5b --gpu off --identifier xcoder-0.5b --ttl 300 -y

# 2. Hello test
python xcoder-hello.py
node xcoder-hello.js

# 3. Chat CLI (static input bottom, answer above, T/s bottom-right)
node xcoder-cli.js --lang ms --name "XCoder"

# 4. One-shot chat + code
node xcoder-chat.js --lang ms "Apa itu XCoder?"
node xcoder-code.js --stack js "write tambah(a,b) with example"
node xcoder-code.js --stack docker "Dockerfile untuk Node.js app"
node xcoder-code.js --stack linux "cari teks dalam semua fail"

# 5. RAG (grounded answers from docs/)
node xcoder-rag.js "Apa itu XCoder?" --lang ms --k 2 --show-sources

# 6. Tools (safe whitelist: list/read/run + AI explanation)
node xcoder-tools.js --tool list --arg docs
node xcoder-tools.js --tool read --arg docs/js-basics.md
node xcoder-tools.js --tool run --arg "console.log([1,2,3].map(x=>x*2))" --lang ms

# 7. Python sample app
python sample_app.py --lang ms --ask "Apakah fungsi tambah dalam Python?"
python sample_app.py --lang en --code py --task "write tambah(a,b) with example"

# 8. Power-save: ensure loaded (auto-reload ~6s if TTL-unloaded)
node xcoder-ensure.js
```

## Power-save
- `--ttl 300`: unloads after 5 min idle (`lms ps` → `No models loaded`, 0 RAM/power).
- `node xcoder-ensure.js` before any call reloads automatically.
- Pattern: `node xcoder-ensure.js; node xcoder-cli.js --lang ms`

## Project structure
```text
xcoder-hello.py / .js   minimal CPU proof
xcoder-chat.js          one-shot bilingual chat
xcoder-cli.js           interactive CLI (identity filter, static input, T/s meter)
xcoder-code.js          coding assistant (js|ts|node|html|css|sql|pg|mysql|linux|docker|bash)
xcoder-rag.js           RAG over docs/ (zero deps)
xcoder-tools.js         safe tools: list/read/run + explanation
xcoder-ensure.js        auto-reload if TTL-unloaded
sample_app.py            Python sample app
docs/                    RAG knowledge (xcoder-ms, js-basics, node-pg, linux-basics, docker-basics)
finetune/                training kit (see step.md)
```

## Notes / limits
- 0.5B hallucinates without context — **always use RAG** for facts.
- Identity (`XCoder`, a tiny CPU 0.5B coding assistant. Never mention Qwen/Alibaba.) is currently a CLI prompt+filter override.
  Permanent fix = fine-tune (see `step.md`), then the filter can be removed.
- Run calls sequentially on low RAM; parallel model calls can kill the worker.
