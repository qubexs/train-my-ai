"""Template drill generator — thousands of correct-by-construction rows.
Usage: python datasets/scripts/generate.py [stack] [--dry-run]
Appends deduped rows to datasets/<stack>.jsonl. Facts are authored below
(hand-written answers x parametric phrasings), so output is reviewable.
For 100k-scale, use distill.py (LLM) on top of this seed.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import stack_files, read_rows

DATASETS_DIR = Path(__file__).resolve().parent.parent

PHRASINGS = [
    "What does `{c}` do?",
    "Explain `{c}` briefly with an example.",
    "When do I use `{c}`?",
    "How would you explain `{c}` to a beginner?",
    "Show me a real-world use of `{c}`.",
]

PHRASINGS_CMD = [
    "What does `{c}` do?",
    "Explain `{c}` briefly.",
    "When do I use `{c}`?",
    "Give a practical example of `{c}`.",
    "What is `{c}` for?",
]


def gen_cmd(topics):
    """topics = [(command, one-line answer)] -> rows x 5 phrasings."""
    out = []
    for c, a in topics:
        for p in PHRASINGS_CMD:
            out.append((p.format(c=c), a))
    return out

# (docker command + flag, one-line factual explanation)
DOCKER_FLAGS = [
    ("docker ps -a", "`docker ps` lists running containers; `-a` includes stopped ones with exit codes."),
    ("docker ps -q", "`-q` prints only container IDs — made for scripting: `docker rm $(docker ps -aq -f status=exited)`."),
    ("docker ps --format", "`--format 'table {{.Names}}\\t{{.Status}}'` renders scriptable columns instead of the default table."),
    ("docker images -a", "`docker images` lists tagged images; `-a` also shows untagged intermediate layers."),
    ("docker images --digests", "`--digests` adds content hashes so you can pin exact bytes, not mutable tags."),
    ("docker logs -f", "`-f` follows log output live, like `tail -f`, until Ctrl-C."),
    ("docker logs --tail", "`--tail 100` shows only the last 100 lines — essential on chatty services before following."),
    ("docker logs --since", "`--since 30m` slices logs by time; pair with `--until` for incident windows."),
    ("docker logs -t", "`-t` prefixes each line with an RFC3339 timestamp for correlating across services."),
    ("docker exec -it", "`-it` allocates an interactive TTY shell inside a running container: `docker exec -it web sh`."),
    ("docker exec -u", "`-u postgres` runs the command as another user — for DB shells without image changes."),
    ("docker run -d", "`-d` detaches: the container runs in background and the terminal returns immediately."),
    ("docker run --rm", "`--rm` deletes the container on exit — hygiene for one-shot jobs so stopped containers never pile up."),
    ("docker run -p", "`-p 8080:80` publishes container port 80 on host 8080; prefix `127.0.0.1:` for loopback-only binding."),
    ("docker run -v", "`-v data:/data` mounts a named volume; `-v ./src:/app` bind-mounts host code for dev reload."),
    ("docker run -e", "`-e KEY=val` sets container env vars; secrets belong in files/`--env-file`, never `-e` in shell history."),
    ("docker run --restart", "`--restart unless-stopped` revives crashed containers and survives host reboots."),
    ("docker run --memory", "`--memory 512m` caps RAM; exceeding it OOM-kills the container (check OOMKilled in inspect)."),
    ("docker run --cpus", "`--cpus 1.5` caps CPU quota — noisy neighbors stop stealing cycles."),
    ("docker build -t", "`-t user/app:1.0` names the result; always tag explicitly instead of accepting `latest`."),
    ("docker build -f", "`-f docker/Dockerfile.prod .` builds a non-default Dockerfile path with repo-root context."),
    ("docker build --target", "`--target debug` builds only up to a named stage — for debug images without shipping shells."),
    ("docker build --no-cache", "`--no-cache` ignores all layer cache; use to verify clean-room reproducibility, not daily."),
    ("docker build --build-arg", "`--build-arg VERSION=1.2` injects build-time vars (never secrets — they persist in history)."),
    ("docker stop -t", "`-t 30` extends the SIGTERM grace period before SIGKILL for slow-draining apps."),
    ("docker rm -f", "`-f` kills a running container before removing — for stuck ones; prefer stop for graceful apps."),
    ("docker rm -v", "`-v` also removes anonymous volumes; named volumes survive (by design)."),
    ("docker rmi -f", "`-f` force-untags; layers vanish only when unreferenced by tags and containers."),
    ("docker network ls", "Lists bridge/host/none plus user-defined and overlay networks with scope and driver."),
    ("docker network create", "`docker network create appnet` makes a user-defined bridge with automatic container DNS."),
    ("docker network connect", "Attaches a running container to another network live, optionally with `--alias`."),
    ("docker network prune", "Deletes unused networks — review first, stopped Compose projects count as unused."),
    ("docker volume create", "Creates a managed named volume; pair with `--opt type=nfs` for shared storage."),
    ("docker volume inspect", "Shows mountpoint, driver and options — start here when data seems to vanish."),
    ("docker volume prune", "Deletes ALL unused volumes including named data — the most dangerous prune; never script blindly."),
    ("docker system df", "Reports reclaimable space per object type (images, containers, volumes, build cache)."),
    ("docker system prune", "One-shot cleanup of stopped containers, unused networks and dangling images (add `-a --volumes` cautiously)."),
    ("docker stats --no-stream", "Single snapshot of CPU/mem/IO per container — for monitoring scripts, not humans."),
    ("docker inspect -f", "`-f '{{.State.Status}}'` extracts fields with Go templates — the scriptable alternative to grep."),
    ("docker cp", "`docker cp web:/app/out ./out` copies files both directions without exec or ssh."),
    ("docker tag", "Adds an alias (`docker tag src:1 dst:2`); pushing then uploads shared layers once."),
    ("docker push", "Uploads tagged layers to a registry; log in first and prefer explicit version tags."),
    ("docker pull -q", "`-q` suppresses progress bars for CI logs; default pull fetches missing layers only."),
    ("docker login", "Stores registry credentials (use credential helpers, not raw base64 auths in dotfiles)."),
    ("docker context ls", "Lists fast-switchable daemon endpoints (local, ssh hosts) for multi-host work."),
    ("docker compose up -d", "Starts the whole stack detached, creating networks and volumes as declared."),
    ("docker compose down -v", "`down` stops and removes containers/nets; `-v` also deletes named volumes (data loss!)."),
    ("docker compose logs -f", "Aggregated, prefixed logs of all services with `-f` follow and `--tail` slicing."),
    ("docker compose exec", "Runs one-offs inside a service container (`compose exec db psql`) without new containers."),
    ("docker compose run --rm", "One-off containers from a service definition, removed on exit — for migrations and shells."),
    ("docker compose build --pull", "Rebuilds with freshly pulled bases so stale cached tags cannot silently ship."),
    ("docker compose restart", "Restarts service containers in dependency order without recreating them."),
    ("docker compose port", "`compose port web 80` prints the mapped host port — for dynamic `-p 80` assignments."),
    ("docker compose config", "Renders the fully merged, interpolated model — the best multi-file debugging tool."),
]

# (compose field, explanation)
COMPOSE_FIELDS = [
    ("image", "Pins the exact image (always tag versions, never bare names relying on latest)."),
    ("build", "Builds from a context+dockerfile instead of pulling; combine with `image:` so built images are tagged."),
    ("ports", "Publishes `HOST:CONTAINER` mappings; short syntax `- \"8080:80\"` needs quoting in YAML."),
    ("environment", "Inline KEY=value map; secrets belong in `env_file` or `secrets`, and `.env` never commits."),
    ("env_file", "Loads vars from a file kept out of git; explicit `environment:` still wins on conflict."),
    ("volumes", "Mounts `named:/path` or `./host:/path`; add `:ro` for configs that must stay immutable."),
    ("depends_on", "Orders startup only; add `condition: service_healthy` plus app retries for true readiness."),
    ("restart", "`unless-stopped` for services; Compose ignores Swarm-only `deploy.resources` — use `mem_limit/cpus` here."),
    ("healthcheck", "Declares `test/interval/timeout/retries/start_period`; powers `depends_on` conditions and `up --wait`."),
    ("networks", "Attaches user-defined networks for automatic DNS; default bridge gives no name resolution."),
    ("profiles", "Marks optional services started only with `--profile name` (debug tools, seeders)."),
    ("command", "Overrides the image CMD per service; keep `entrypoint` for the fixed binary."),
    ("user", "Runs as `uid:gid` without rebuilding (e.g. match host ownership for bind mounts)."),
    ("working_dir", "Overrides WORKDIR for runtimes needing a different cwd than the image default."),
    ("stop_grace_period", "Per-service SIGTERM window (e.g. `30s`) for slow-draining apps instead of global `-t`."),
    ("logging", "Per-service driver/options (`max-size/max-file`) so one chatty service cannot fill the disk."),
    ("deploy", "Swarm-only (replicas, update_config, placement); silently ignored by plain `compose up`."),
    ("secrets", "Mounts swarm/compose secrets as files under /run/secrets — never env vars for real secrets."),
    ("configs", "Mounts config files (nginx.conf) without rebuilding images on config change."),
    ("extra_hosts", "Injects host entries (`host.docker.internal:host-gateway`) for host access from containers."),
    ("init", "`init: true` adds tini for zombie reaping and signal forwarding per service."),
    ("platform", "Pins `linux/amd64` when images are arch-sensitive (Apple Silicon pulling amd64-only images)."),
    ("pull_policy", "`always/missing/never/build` controls freshness; `always` in deploy scripts kills stale-tag bugs."),
    ("container_name", "Fixes the name but forbids scaling that service — omit it unless something must address it statically."),
    ("extends", "Inherits another service's definition to avoid repeating env/logging/resource stanzas."),
]

SQL_TABLES = {
    "pengguna": ["id", "nama", "emel", "umur", "bandar"],
    "pesanan": ["id", "pengguna_id", "jumlah", "dibuat"],
    "produk": ["id", "nama", "harga", "stok"],
}

# (task template, sql template) — {t}=table, {c}=column, {v}=value
SQL_TASKS = [
    ("Senaraikan semua {t}", "SELECT * FROM {t};"),
    ("Kira baris dalam {t}", "SELECT COUNT(*) AS kiraan FROM {t};"),
    ("Cari {t} di mana {c} bersamaan {v}", "SELECT * FROM {t} WHERE {c} = {v};"),
    ("Susun {t} mengikut {c} menurun, 5 teratas", "SELECT * FROM {t} ORDER BY {c} DESC LIMIT 5;"),
    ("Jumlahkan {c} dalam {t} berkelompok mengikut {c2}",
     "SELECT {c2}, SUM({c}) AS jumlah FROM {t} GROUP BY {c2};"),
    ("Kemas kini {t}: tetapkan {c} kepada {v} untuk id 1",
     "UPDATE {t} SET {c} = {v} WHERE id = 1;"),
    ("Padam baris id 9 dari {t}", "DELETE FROM {t} WHERE id = 9;"),
    ("Tambah satu baris ke {t}", "INSERT INTO {t} ({cols}) VALUES ({vals});"),
    ("Purata {c} dalam {t}", "SELECT AVG({c}) AS purata FROM {t};"),
    ("Nilai maksimum {c} dalam {t}", "SELECT MAX({c}) AS maks FROM {t};"),
]

PHRASING_SQL = [
    "{q}",
    "{q} — tunjukkan SQL.",
    "Tulis query: {q}.",
    "Beri SQL untuk: {q}.",
    "Jawab dengan SQL: {q}.",
    "Tunjukkan cara dalam SQL: {q}.",
]


def gen_docker():
    out = []
    for cmd, expl in DOCKER_FLAGS + [(f"compose {f}", e) for f, e in COMPOSE_FIELDS]:
        for p in PHRASINGS:
            out.append((p.format(c=cmd), expl))
    return out


def gen_sql():
    out = []
    for t, cols in SQL_TABLES.items():
        c, c2 = cols[1], cols[2]
        for task, sql in SQL_TASKS:
            q = task.format(t=t, c=c, c2=c2, v="'Ali'",
                            cols=", ".join(cols[1:3]), vals="'Ali', 'ali@x.my'")
            s = sql.format(t=t, c=c, c2=c2, v="'Ali'",
                           cols=", ".join(cols[1:3]), vals="'Ali', 'ali@x.my'")
            for p in PHRASING_SQL:
                out.append((p.format(q=q), f"```sql\n{s}\n```"))
    # join drills across tables
    joins = [
        ("Sertai pesanan dengan nama pengguna",
         "SELECT o.id, p.nama FROM pesanan o JOIN pengguna p ON p.id = o.pengguna_id;"),
        ("Sertai produk dengan baki stok rendah",
         "SELECT nama, stok FROM produk WHERE stok < 10 ORDER BY stok;"),
        ("Jumlah belanja setiap pengguna",
         "SELECT p.nama, SUM(o.jumlah) AS jumlah FROM pengguna p JOIN pesanan o ON o.pengguna_id = p.id GROUP BY p.nama;"),
    ]
    for q, s in joins:
        for p in PHRASING_SQL:
            out.append((p.format(q=q), f"```sql\n{s}\n```"))
    return out


GENERATORS = {"docker": gen_docker, "sql": gen_sql}


LINUX_TOPICS = [
    ("ls -la", "`ls -la` lists all files with details; `-l` long format, `-a` includes hidden dotfiles."),
    ("df -h", "`df -h` shows disk usage per filesystem in human units; add `-i` to check inode exhaustion."),
    ("du -sh *", "`du -sh *` ranks directory sizes; pipe to `sort -h` for biggest-first cleanup hunting."),
    ("free -h", "`free -h` shows RAM/swap; watch the `available` column, not `free`, for real headroom."),
    ("ps aux", "`ps aux` lists all processes with CPU/mem; pipe to `sort` or pair with `top` for live view."),
    ("top", "`top` is the live process monitor (CPU, mem, load); press `q` to quit, `M` sorts by memory."),
    ("kill PID", "`kill PID` sends SIGTERM for graceful exit; `-9` (SIGKILL) forces when stuck."),
    ("pkill -f pola", "`pkill -f` kills by full command match — quote the pattern and double-check with `pgrep -f` first."),
    ("chmod 755 skrip.sh", "`755` = rwxr-xr-x: owner full, others read+execute; scripts need +x, data files must not have it."),
    ("chown -R app:app /srv", "`chown -R` changes owner recursively; beware symlinks and system dirs."),
    ("tar xzf app.tgz", "`tar xzf` extracts gzip archives; list first with `tzf` and use `-C dir` to control destination."),
    ("grep -r teks .", "`grep -r` searches recursively; add `-i` (case), `-n` (line numbers), `--include='*.py'` to narrow."),
    ("find . -name '*.log'", "`find` locates by name/time/size; `-exec cmd {} +` acts on matches without xargs."),
    ("ssh user@hos", "`ssh` opens a secure shell; keys (`ssh-keygen`, `ssh-copy-id`) replace passwords."),
    ("systemctl restart nginx", "`systemctl restart` reloads a service; `enable` persists across boot, `status` shows logs."),
    ("journalctl -f", "`journalctl -f` tails system logs; `-u unit` and `--since` narrow to incidents."),
    ("ufw allow 22/tcp", "`ufw allow` opens firewall ports; always allow SSH before `ufw enable` or get locked out."),
    ("crontab -e", "`crontab -e` edits your schedule (minute hour day month weekday); cron env is minimal, use absolute paths."),
    ("ln -s asal pautan", "`ln -s` makes a pointer (breaks if target moves); hard links share inodes on one filesystem."),
    ("uname -a", "`uname -a` prints kernel, hostname and architecture — first line of any bug report."),
]

BASH_TOPICS = [
    ("set -euo pipefail", "`set -euo pipefail` aborts on errors, unset vars and pipe failures — every serious script starts here."),
    ("if [[ -f fail ]]", "`[[ ]]` tests safely (no word-splitting); `-f` file, `-d` dir, `-z` empty, with `&&`/`||` inside."),
    ("for f in *.md", "`for f in *.md` loops glob matches; quote `\"$f\"` and set `nullglob` for empty dirs."),
    ("while read baris", "`while IFS= read -r line < file` reads safely preserving spaces and backslashes."),
    ("function greet()", "Functions return exit codes, not strings — echo data and capture with `$(...)`; use `local` vars."),
    ("case $1 in", "`case` branches subcommands cleanly with patterns and a `*)` usage default; each arm ends `;;`."),
    ("$1 $@ $#", "`$1` first arg, `\"$@\"` all args intact, `$#` count; `${1:?...}` fails fast with a message."),
    ("${PORT:-3000}", "`${VAR:-default}` substitutes when empty; `${#s}` length; `${s%.ext}` strips suffix."),
    ("arr=(a b c)", "Arrays index from 0; expand as `\"${arr[@]}\"` per element and count with `${#arr[@]}`."),
    ("trap cleanup EXIT", "`trap ... EXIT` guarantees cleanup on any exit; temp dirs from `mktemp -d` never leak."),
    ("cmd > out 2>&1", "`> out 2>&1` merges stderr into the log file; order matters — redirect stdout first."),
    ("a | b | c", "Pipes chain stdout to stdin; `set -o pipefail` makes middle failures visible to `set -e`."),
    ("cmd1 && cmd2", "`&&` runs next only on success, `||` on failure — the one-line try/catch of shell."),
    ("[[ $x =~ ^[0-9]+$ ]]", "`=~` regex-matches inside `[[ ]]`; quote the pattern to avoid glob interpretation."),
    ("seq 1 10", "`seq` generates number ranges for loops; `seq -w` zero-pads filenames in order."),
    ("find . -print0 | xargs -0", "NUL-delimited pipelines survive spaces/newlines in filenames; add `-r` to skip empty runs."),
    ("$(date +%F)", "`$(...)` captures output (nestable, unlike backticks); quote it unless splitting is intended."),
    ("<<EOF heredoc", "Heredocs feed multi-line input (`cat <<EOF`); quote the tag (`<<'EOF'`) to disable expansion."),
    ("shift", "`shift` consumes processed args so `$1` always means next; loop `while [[ $# -gt 0 ]]` for flags."),
    ("exec cmd", "`exec` replaces the shell process (PID kept, no child) — used as the last line of entrypoints."),
]

GENERATORS = {"docker": gen_docker, "sql": gen_sql,
              "linux": lambda: gen_cmd(LINUX_TOPICS),
              "bash": lambda: gen_cmd(BASH_TOPICS)}

JS_TOPICS = [
    ("Array.map", "`map` transforms each element and returns a new array; original untouched — the functional workhorse."),
    ("Array.filter", "`filter` keeps elements passing the test; chain after map for transform-then-select pipelines."),
    ("Array.reduce", "`reduce` folds to one value with an accumulator; always pass the initial value to avoid edge bugs."),
    ("Array.find", "`find` returns the first match (or undefined); `findIndex` gives its position for splicing."),
    ("Array.includes", "`includes` answers membership directly; second arg sets the start index for searches."),
    ("slice vs splice", "`slice` copies a range immutably; `splice` mutates in place for insert/delete — never confuse them."),
    ("push and pop", "`push` appends (returns length), `pop` removes the end; `shift`/`unshift` work the front slower."),
    ("join and split", "`join('-')` glues with a separator; `split(',')` breaks strings apart — CSV lines in one call."),
    ("Object.keys/values/entries", "These list an object's keys, values or pairs for loops; `Object.fromEntries` rebuilds objects."),
    ("JSON.parse/stringify", "`parse` revives objects (wrap in try/catch); `stringify` serializes — functions and undefined drop silently."),
    ("typeof", "`typeof` names primitives (`typeof []` is misleadingly `object` — use Array.isArray); `instanceof` checks prototypes."),
    ("template literals", "Backticks embed `${expr}` and span lines; tagged templates power SQL/HTML builders safely."),
    ("arrow functions", "Arrows bind outer `this` lexically and suit callbacks; they cannot be constructors (no `new`)."),
    ("spread ...", "Spread expands iterables into calls/arrays/objects; shallow only — nested objects stay shared."),
    ("destructuring", "Pull fields positionally (`[a, b]`) or by name (`{x}`) with defaults; rest (`...r`) gathers leftovers."),
    ("optional chaining ?.", "`?.` stops at null/undefined instead of throwing — safe deep access without guard pyramids."),
    ("nullish ??", "`??` defaults only null/undefined, unlike `||` which also swallows 0, '' and false."),
    ("promises then/catch", "Promises model future values; always end chains with `.catch` — unhandled rejections crash processes."),
    ("setTimeout/clearTimeout", "`setTimeout` defers once (keep the id to cancel); `setInterval` repeats until cleared."),
    ("localStorage", "`localStorage` persists strings per origin (~5MB); JSON-encode objects and guard quota errors."),
]

PY_TOPICS = [
    ("print f-string", "f-strings embed `{var}` and format specs (`{x:.2f}`); prefix debug with `=` to show names."),
    ("len() range()", "`len()` counts containers; `range(5)` yields 0-4 lazily — pair with `enumerate`/`zip` in loops."),
    ("list append", "`append` grows lists in place; comprehensions (`[x for x in y if ok]`) build filtered ones declaratively."),
    ("dict get", "`d.get(k, default)` avoids KeyError; `setdefault` inserts-and-returns for counting patterns."),
    ("open() with", "`with open(...) as f` auto-closes even on errors; always pass explicit encoding for text."),
    ("import module", "Imports run once and cache in `sys.modules`; circular imports signal a design split is due."),
    ("def + return", "Functions return values (or None); default args evaluate once — never use mutable `=[]` defaults."),
    ("try/except", "Catch specific exceptions, keep `try` blocks tiny, and use `else`/`finally` for success/cleanup paths."),
    ("for and while", "`for` iterates iterables directly (pythonic); `while` needs manual progress or loops forever."),
    ("if elif else", "Chain `elif` instead of nesting; early-return flattens logic better than deep nesting."),
    ("pip install", "`pip install pkg` resolves deps; pin with `==` in requirements and hash-check in CI."),
    ("venv", "One venv per project isolates deps; never install into system Python on shared machines."),
    ("lambda", "Lambdas suit one-expression callbacks (`key=lambda r: r[1]`); named `def` wins beyond one line."),
    ("enumerate and zip", "`enumerate` adds indices, `zip` pairs iterables — both lazy; `zip(..., strict=True)` catches length drift."),
    ("split and join", "`'a,b'.split(',')` breaks, `', '.join(list)` glues; strip whitespace before comparing user input."),
    ("int() and str()", "Convert explicitly; `int('77')` works but `int('77px')` raises — validate with try/except or regex."),
    ("class __init__", "`__init__` builds state per instance; methods take `self`; prefer dataclasses for pure data holders."),
    ("requests.get", "`requests.get(url, timeout=10)` plus `raise_for_status()`; reuse a `Session` for connection pooling."),
    ("json module", "`json.load(s)` parses (catch DecodeError on network data); `dumps(indent=2, ensure_ascii=False)` stays readable."),
    ("pathlib Path", "`Path` joins with `/`, reads/writes text, and globs — portable across Windows and Linux unlike string paths."),
]

TS_TOPICS = [
    ("interface", "Interfaces lock object shapes; missing or mistyped fields fail at compile, not in production."),
    ("type alias", "Type aliases name unions, tuples and primitives; one source of truth beats repeated annotations."),
    ("enum", "Enums name fixed value sets; for tiny cases a union of literals is lighter and tree-shakes better."),
    ("generics <T>", "Generics keep functions reusable without `any` — caller types flow through to the return type."),
    ("union string | number", "Unions admit several types; narrow with `typeof` checks before using specific methods."),
    ("optional ?", "`?` marks fields/params optional (type plus undefined); pair with `??` defaults at use sites."),
    ("readonly", "`readonly` blocks writes after construction; `as const` freezes literals for config objects."),
    ("tuple [string, number]", "Tuples fix length and per-slot types; `readonly` tuples suit function returns of pairs."),
    ("Record<K,V>", "`Record<string, number>` types dictionaries; `Partial/Pick/Omit` derive variants from one interface."),
    ("unknown vs any", "`unknown` forces narrowing before use; `any` disables checking entirely — treat as tech debt."),
    ("never", "`never` marks unreachable code; use it in switch defaults so new cases fail compilation loudly."),
    ("as cast", "`as` asserts without checking — safe only after validation; prefer narrowing or schema parsing."),
    ("implements", "`class X implements Y` verifies conformance at compile; structural typing still allows compatible shapes."),
    ("abstract class", "Abstract classes mix contracts with shared implementation; one inheritance chain per class only."),
    ("namespace", "Namespaces group legacy globals; modern code prefers ES modules — use only when integrating old scripts."),
    ("decorators @", "Decorators attach metadata at definition time (NestJS/Angular); keep them side-effect free."),
    ("strict mode", "Strict mode catches null/any/property bugs at compile; migrate old code flag by flag."),
    ("tsconfig paths", "`paths` aliases (`@/*`) kill `../../../` imports; bundlers need matching config to resolve at runtime."),
    ("declaration .d.ts", "Declaration files describe untyped JS for the checker; `@types/*` packages provide community ones."),
    ("satisfies", "`satisfies` validates shape while keeping literal types; `as` widens and hides information."),
]

NODE_TOPICS = [
    ("node --version", "`node --version` plus `.nvmrc` pinning keeps dev, CI and images on one runtime."),
    ("npm init -y", "`npm init -y` scaffolds package.json instantly; edit name/version/scripts right after."),
    ("npm install express", "Local installs land in node_modules (never global for project deps); lockfile commits for reproducibility."),
    ("require vs import", "CommonJS `require` is dynamic and sync; ESM `import` is static and supports top-level await — new code uses ESM."),
    ("process.env", "`process.env` reads environment strings; validate required vars at boot and fail fast with clear messages."),
    ("__dirname ESM", "ESM has no `__dirname`; derive it from `import.meta.url` via `fileURLToPath` for path building."),
    ("fs.readFile", "Prefer `fs/promises` with await over callbacks; stream (`createReadStream`) files too big for memory."),
    ("http.createServer", "The native server suffices for webhooks and proxies; frameworks pay off once routing grows."),
    ("setTimeout", "Timers return ids for cancellation; unref background timers so they never hold the process open."),
    ("setInterval", "`setInterval` repeats blindly — chain `setTimeout` instead when work must not overlap."),
    ("console.log/error", "`console.error` routes to stderr for real errors; structured JSON logs beat printf in production."),
    ("process.argv", "`process.argv` carries CLI args after two slots; parsers (commander/yargs) handle flags robustly."),
    ("Buffer", "Buffers hold raw bytes for crypto/files/protocols; convert with explicit encodings (`utf8`, `base64`, `hex`)."),
    ("EventEmitter", "Emitters decouple modules via named events; remove listeners (`once`/`off`) to avoid leaks."),
    ("streams pipe", "Streams move data in chunks with constant memory; `pipeline` wires backpressure and errors correctly."),
    ("child_process", "`execFile` (no shell) is injection-safe; `spawn` streams big output; avoid `exec` with user input."),
    ("package.json scripts", "Scripts document and automate (`dev`, `start`, `test`); keep them cross-platform for Windows teammates."),
    ("nodemon tsx watch", "Auto-restart on change during dev (nodemon/tsx/node --watch); production uses real process managers."),
    ("dotenv", "`.env` files feed local dev only — never commit; validate loaded values with a schema at startup."),
    ("graceful shutdown", "On SIGTERM stop accepting, drain, close DB, then exit — matching orchestrator grace periods."),
]

GENERATORS.update({"javascript": lambda: gen_cmd(JS_TOPICS),
                   "python": lambda: gen_cmd(PY_TOPICS),
                   "typescript": lambda: gen_cmd(TS_TOPICS),
                   "nodejs": lambda: gen_cmd(NODE_TOPICS)})

HTML_TOPICS = [
    ("<!DOCTYPE html>", "The doctype triggers standards mode; without it browsers fall back to quirks with legacy box behavior."),
    ("headings h1-h6", "One h1 per page, ordered levels — screen readers navigate by this outline, not by font size."),
    ("p br hr", "`<p>` paragraphs, `<br>` line breaks, `<hr>` thematic breaks; never fake spacing with empty divs."),
    ("a href", "Anchors navigate (`href`) or jump (`#id`); external blank tabs need `rel=noopener` for safety."),
    ("img src alt", "Always `alt` (content description or empty for decorative); `srcset` serves right sizes per screen."),
    ("ul ol li", "`ul` unordered, `ol` meaningful order, `li` items only — nested lists need their own list parent."),
    ("table tr td", "Tables are for tabular data with `th` headers; layout tables break screen readers — use CSS grid."),
    ("form input", "Forms collect with named inputs; `method`/`action` define submission, labels make fields operable."),
    ("button type", "`type=button` avoids accidental submits inside forms; `submit` sends, `reset` clears (rarely wanted)."),
    ("div vs span", "`div` blocks, `span` inlines — both meaningless, so prefer semantic tags and reserve these for styling hooks."),
    ("header footer nav main", "Landmarks let assistive tech jump regions; exactly one `main` holds the page's core content."),
    ("meta viewport", "The viewport meta makes mobile render at device width; pair with `charset=utf-8` first in head."),
    ("title tag", "Title shows in tabs/search; unique, front-loaded keywords per page under ~60 characters."),
    ("script src", "`defer` scripts run in order after parsing; `async` runs whenever fetched — choose by dependency."),
    ("link stylesheet", "Stylesheets in head prevent unstyled flashes; `media=print` scopes print-only rules."),
    ("video controls", "Native video needs `controls` plus `preload=metadata`; caption tracks serve deaf users and SEO."),
    ("audio controls", "Audio tags stream with built-in UI; offer transcripts — the accessible and searchable equivalent."),
    ("select option", "Native selects work everywhere including mobile pickers; custom dropdowns must reimplement keyboard support."),
    ("textarea", "Textareas take `rows`/`maxlength` and resize handles; never single-line `input` for paragraphs."),
    ("fieldset legend", "Group related controls in `fieldset` with a `legend` caption — free semantics for forms and radios."),
]

CSS_TOPICS = [
    ("color background", "`color` paints text, `background` the box; test contrast 4.5:1 for body text accessibility."),
    ("font-size rem", "Size in `rem` (root-relative) so user zoom scales everything; `px` freezes accessibility settings out."),
    ("font-family stack", "Stacks degrade gracefully (`Inter, system-ui, sans-serif`); web fonts need `font-display: swap`."),
    ("margin padding", "Margin spaces outside, padding inside; one-direction margins avoid collapse surprises in components."),
    ("border", "`border: 1px solid #ddd` draws boxes; `border-radius` rounds, `border: none` resets for custom controls."),
    ("display block inline", "Block fills width and stacks; inline flows in text; `inline-block` mixes both for pills and badges."),
    ("display flex", "Flex lays out one dimension with `gap` spacing; `flex: 1` children share leftover space proportionally."),
    ("display grid", "Grid aligns two dimensions via template columns; `auto-fit minmax()` makes responsive tracks for free."),
    ("position relative", "Relative nudges visually while keeping flow space, and anchors absolute children to itself."),
    ("position absolute", "Absolute pins to the nearest positioned ancestor (or page); great for badges, deadly for layouts."),
    ("flex-direction", "`row` vs `column` sets the main axis; `justify-content` then runs along it, `align-items` across."),
    ("justify-content", "Distributes along the main axis (`center`, `space-between`); pairs with `align-items` for full centering."),
    ("align-items", "Aligns across the axis (`stretch` default fills); per-child `align-self` overrides exceptions."),
    ("grid-template-columns", "`1fr 1fr` splits evenly; `repeat(3, 1fr)` scales; `minmax(200px, 1fr)` keeps floors."),
    ("media query", "`@media (min-width: 768px)` enhances upward from mobile base; content-first breakpoints beat device lists."),
    (":hover", "Hover styles need focus equivalents (`:focus-visible`) — touch and keyboard users never hover."),
    ("transition", "Transitions animate state changes (`transition: color .2s`); keep to transform/opacity for smoothness."),
    ("@keyframes", "Keyframes script multi-step loops (spinners, pulses); honor `prefers-reduced-motion` for vestibular safety."),
    ("box-shadow", "Layered shadows (`0 1px 2px, 0 8px 24px`) look natural; animate via pseudo-element opacity, not the property."),
    ("border-radius 50%", "50% on squares makes circles (avatars with `object-fit: cover`); pills use half-height radius."),
]

PHP_TOPICS = [
    ("<?php echo", "`<?php echo $x; ?>` (or `<?= $x ?>`) prints escaped output; all dynamic output needs `htmlspecialchars`."),
    ("$variables", "`$` prefixes all variables (loosely typed); `$_GET/$_POST` carry request input — never trusted raw."),
    ("if else", "Standard branches with `===` strict compare; loose `==` juggling (`'0' == false`) is a classic trap."),
    ("for foreach while", "`foreach ($arr as $k => $v)` is the workhorse; `while` needs manual progress or loops forever."),
    ("function", "Functions take typed params with defaults; `declare(strict_types=1)` per file stops silent coercion."),
    ("arrays assoc", "PHP arrays are ordered maps; `[]` appends, `count()` sizes, `array_*` helpers transform."),
    ("count() sizeof", "`count()` sizes arrays (O(1)); empty check via `=== []` or `empty()` — know their falsy differences."),
    ("strlen strpos", "`strlen` bytes (not chars — use `mb_strlen` for Unicode); `strpos` returns 0-or-false, compare with `!==`."),
    ("str_replace explode", "`str_replace` swaps substrings; `explode(',', $s)` splits, `implode` glues — CSV lines in two calls."),
    ("include require", "`require` fatals on missing (critical code), `include` warns; `_once` variants prevent redeclaration."),
    ("class new", "One class per PSR-4 file; constructor promotion trims boilerplate; visibility guards internals."),
    ("composer install", "Composer resolves `composer.json` into locked `composer.lock`; commit both, never `vendor/`."),
    ("password_hash", "`password_hash` with Argon2/bcrypt handles salt+cost; verify with `password_verify`, rehash on cost bumps."),
    ("PDO prepare", "Prepared placeholders separate data from SQL (injection-proof); set ERRMODE_EXCEPTION and real prepares."),
    ("header Location", "`header('Location: /x')` redirects but must precede any output; follow with `exit` to stop rendering."),
    ("session_start", "Sessions persist per-user state; regenerate id after login, destroy on logout, flag cookies httponly+secure."),
    ("try catch", "Catch specific `Exception` subclasses; log with context and rethrow or render — never swallow silently."),
    ("date() time", "`date('Y-m-d')` formats timestamps; store UTC and convert per user timezone at display."),
    ("file_get_contents", "Reads files/URLs into strings simply; check `=== false` and prefer streams for huge payloads."),
    ("__construct", "Constructors build valid objects (inject deps, no work); heavy lifting belongs in explicit methods, not constructors."),
]

LARAVEL_TOPICS = [
    ("artisan list", "`php artisan list` reveals every command; `--help` details flags — the framework is self-documenting."),
    ("make:controller", "`make:controller X --resource` scaffolds REST methods; `--api` drops create/edit for APIs."),
    ("make:model -m", "`-m` bundles migration (add `-cr` for controller+resource); one command frames full CRUD."),
    ("Route::get post", "Explicit verbs map URLs to handlers; name them (`->name()`) so links survive path changes."),
    ("Route::resource", "Resource routes map seven REST actions to controller methods; `route:list` audits the table."),
    ("blade {{ }}", "`{{ }}` escapes (XSS-safe default); `{!! !!}` raw only for self-generated HTML."),
    ("@if @foreach", "Directives replace PHP tags in templates; `@auth/@guest` branch on login state cleanly."),
    ("Eloquent all find", "`Model::all/find/where()->get()` read fluently; `firstOrFail()` 404s instead of null-checks."),
    ("migrations migrate", "Migrations version the schema in code; `migrate`, `rollback`, `fresh --seed` cycle dev databases."),
    ("db:seed factories", "Factories mint realistic rows; seeders compose them so every dev shares identical starter data."),
    ("request validate", "`$request->validate([...])` auto-redirects with errors and old input; `@error` displays per field."),
    ("auth middleware", "`->middleware('auth')` guards routes; Breeze scaffolds full login/register when starting fresh."),
    (".env config", "Env feeds `config/` files read via `config()`; cache configs in prod and never call `env()` outside config."),
    ("storage:link", "The public disk needs `storage:link` once; uploads go to storage, never the repo."),
    ("queue:work", "Dispatched jobs run async under workers supervised in prod; `failed_jobs` table catches casualties."),
    ("tinker", "`php artisan tinker` is a REPL on your app — inspect models and test snippets without routes."),
    ("serve", "`php artisan serve` boots dev quickly; production needs real servers (nginx + php-fpm), not serve."),
    ("config:cache", "Cache config/routes/views on deploy for free speed; clear and rebuild on every release."),
    ("relationships", "`hasMany/belongsTo` declare links; `with()` eager-loads to kill N+1 query explosions."),
    ("policies gates", "Policies authorize per-model actions (`@can` in Blade); gates cover one-off abilities outside models."),
]

MYSQL_TOPICS = [
    ("SHOW DATABASES", "Lists databases you may access; `USE kedai;` then scopes unqualified tables to it."),
    ("CREATE DATABASE", "`CREATE DATABASE kedai CHARACTER SET utf8mb4;` — charset at creation avoids later conversions."),
    ("USE database", "`USE` sets the session default so short table names resolve; scripts should still qualify in joins."),
    ("SHOW TABLES", "Lists tables in scope; `DESCRIBE t` then details columns, keys and defaults."),
    ("SELECT * WHERE", "`SELECT cols FROM t WHERE cond` filters rows; list columns explicitly instead of `*` in app code."),
    ("ORDER BY LIMIT", "`ORDER BY x DESC LIMIT 20` pages deterministically — but always pair LIMIT with ORDER BY."),
    ("INSERT INTO", "Name columns explicitly (`INSERT INTO t (a,b) VALUES...`); positional inserts rot on schema change."),
    ("UPDATE WHERE", "UPDATE without WHERE rewrites the table — develop the WHERE as SELECT first, then convert."),
    ("DELETE WHERE", "Same discipline as UPDATE; prefer soft-delete flags for business records you may need back."),
    ("CREATE TABLE", "Declare PK, NOT NULLs, sane defaults and InnoDB up front; altering later on big tables is painful."),
    ("primary key", "Auto-increment integer PKs are the default sane choice; UUIDs suit distributed writes at a cost."),
    ("JOIN INNER LEFT", "INNER keeps matches both sides; LEFT keeps all left rows (NULLs where unmatched) — know which you asked."),
    ("GROUP BY COUNT", "Aggregate per group with COUNT/SUM/AVG; HAVING (not WHERE) filters the grouped results."),
    ("CREATE INDEX", "Index WHERE/JOIN/ORDER columns (`(a,b)` composite in query order); EXPLAIN verifies `ref` over `ALL`."),
    ("mysqldump", "`mysqldump --single-transaction db | gzip` backs up consistently; test restores or backups are fiction."),
    ("mysql -u -p", "`mysql -u app -p db` connects interactively (`-h` host, `-P` port); history files can leak — prefer config files."),
    ("GRANT SELECT", "Grant least privilege per app/host (`GRANT SELECT ON db.*`); `FLUSH PRIVILEGES` applies grant-table edits."),
    ("utf8mb4", "utf8mb4 stores full Unicode (emoji included); legacy `utf8` is 3-byte and corrupts the rest."),
    ("slow query log", "Enable slow log (`long_query_time=1`) and mine it with mysqldumpslow; fix with indexes before hardware."),
    ("SHOW PROCESSLIST", "Shows live connections and their queries; `KILL id` stops runaways — diagnose before restarting."),
]

PG_TOPICS = [
    ("psql -U app", "`psql -U app -d kedai` connects; `-h/-p` for remotes; `.pgpass` avoids password prompts in scripts."),
    ("\\l \\c \\dt", "Backslash commands navigate: `\\l` databases, `\\c` switch, `\\dt` tables, `\\d t` columns."),
    ("CREATE DATABASE", "`createdb kedai` or SQL with OWNER set to the app role — migrations then need no superuser."),
    ("CREATE TABLE", "Declare PKs, NOT NULLs and defaults up front; `GENERATED ... AS IDENTITY` beats legacy SERIAL."),
    ("SERIAL IDENTITY UUID", "IDENTITY is the modern auto-key; UUIDs (`gen_random_uuid()`) suit public/distributed ids; JSONB for semi-structure."),
    ("INSERT RETURNING", "`RETURNING id` fetches generated values in one round trip — no follow-up SELECT, no races."),
    ("SELECT WHERE", "Filter early with indexed WHERE; `ILIKE` for case-insensitive, `IN (...)` for sets, `BETWEEN` for ranges."),
    ("UPDATE DELETE", "Same rule as MySQL: develop WHERE as SELECT first; wrap multi-statement changes in transactions."),
    ("JOIN", "INNER/LEFT semantics mirror MySQL; `USING (id)` shortens equi-joins on same-named keys."),
    ("GROUP BY HAVING", "Aggregate then filter groups with HAVING; `FILTER (WHERE ...)` pivots multiple aggregates in one pass."),
    ("ORDER BY LIMIT", "Pair LIMIT with deterministic ORDER BY (add id tiebreak); keyset (`WHERE id >`) beats OFFSET at scale."),
    ("CTE WITH", "WITH names subqueries as readable steps; chain several comma-separated; PG12+ inlines them well."),
    ("window ROW_NUMBER", "Windows compute across related rows without collapsing them — rankings, running totals, lags."),
    ("EXPLAIN ANALYZE", "Shows real plan + timings; seek Seq Scans on big tables, add btree/GIN/BRIN indexes, re-measure."),
    ("CREATE INDEX", "Btree default, GIN for JSONB/full-text, BRIN for append-only time series; partial indexes shrink hot paths."),
    ("pg_dump pg_restore", "Custom format (`-Fc`) enables selective restores; pair base backups with WAL archiving for PITR."),
    ("ALTER TABLE", "Add nullable columns freely; backfill, then add NOT NULL — rewrites need maintenance windows on big tables."),
    ("TRUNCATE", "Instant full clears without row WAL; needs no WHERE by design; `RESTART IDENTITY` resets sequences."),
    ("COPY CSV", "Server-side `\\copy` moves bulk data hundreds of times faster than row INSERTs; mind superuser vs client paths."),
    ("roles pg_hba", "Roles + `pg_hba.conf` (first match wins, prefer scram-sha-256) gate access; `SELECT pg_reload_conf()` applies."),
]

TAILWIND_TOPICS = [
    ("flex", "`flex` lays one dimension with `gap`; `flex-1` children share leftover space proportionally."),
    ("grid", "`grid grid-cols-3 gap-4` aligns two dimensions; `md:` prefixes adapt columns per breakpoint."),
    ("text-center text-xl", "Type scale (`text-sm`..`text-4xl`), weights (`font-bold`) and alignment compose without custom CSS."),
    ("bg-blue-600", "Palette colors with shades (`bg-blue-600 hover:bg-blue-700`); extend theme for brand tokens."),
    ("p-4 m-2", "Spacing scale (`p-4` padding, `m-2` margin, `space-x-4` between children) replaces guesswork pixels."),
    ("rounded-lg shadow", "Radius (`rounded`, `rounded-full` pills) plus layered `shadow` utilities build cards in two classes."),
    ("hover:", "State variants (`hover: focus: disabled:`) compose freely; always pair hover with focus for keyboard users."),
    ("md:", "Mobile-first breakpoints (`sm md lg xl 2xl`) layer enhancements upward from the phone base."),
    ("flex-col flex-row", "Direction switch reflows the same markup; responsive (`flex-col md:flex-row`) without duplicate HTML."),
    ("items-center justify-center", "Cross/main-axis centering pair — the two-class answer to vertical centering."),
    ("gap-4", "`gap` spaces grid/flex children uniformly; replaces margin hacks and last-child exceptions."),
    ("border", "`border border-gray-200` outlines; per-side (`border-t`) and divide utilities for lists."),
    ("w-full h-64", "Sizing scale with fractions (`w-1/2`), viewport units and `max-w-*` containers for readable lines."),
    ("hidden block", "Display swaps per breakpoint (`hidden md:block`); `invisible` keeps layout space unlike `hidden`."),
    ("container mx-auto", "`container mx-auto px-4` centers content with gutters — the standard page shell."),
    ("space-x-4", "`space-x/y` gaps siblings without wrapper classes; RTL-aware variants exist for mirrored layouts."),
    ("divide-y", "`divide-y` draws rules between children only — cleaner than borders on every item."),
    ("dark:", "Dark variants keyed off class strategy toggle whole themes with one ancestor class."),
    ("transition", "`transition` + `duration-200` animate state changes; restrict to transform/opacity for smoothness."),
    ("animate-spin", "Built-in keyframes (`spin ping pulse bounce`) cover loaders; custom ones extend via config."),
]

GENERATORS.update({"html": lambda: gen_cmd(HTML_TOPICS),
                   "css": lambda: gen_cmd(CSS_TOPICS),
                   "php": lambda: gen_cmd(PHP_TOPICS),
                   "laravel": lambda: gen_cmd(LARAVEL_TOPICS),
                   "mysql": lambda: gen_cmd(MYSQL_TOPICS),
                   "postgresql": lambda: gen_cmd(PG_TOPICS),
                   "tailwind": lambda: gen_cmd(TAILWIND_TOPICS)})


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    dry = "--dry-run" in sys.argv
    stacks = args or sorted(GENERATORS)
    total = 0
    for stack in stacks:
        if stack not in GENERATORS:
            print(f"no generator: {stack} (ada: {sorted(GENERATORS)})")
            continue
        dest = DATASETS_DIR / f"{stack}.jsonl"
        seen = set()
        if dest.exists():
            for line in dest.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    seen.add((r.get("instruction", ""), r.get("output", "")))
                except ValueError:
                    pass
        added = 0
        rows = []
        for q, a in GENERATORS[stack]():
            if (q, a) in seen:
                continue
            seen.add((q, a))
            rows.append({"instruction": q, "input": "", "output": a})
        if not dry:
            with dest.open("a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            added = len(rows)
        else:
            added = len(rows)
        total += added
        print(f"{stack}: +{added} rows -> {dest} ({'dry-run' if dry else 'written'})")
    print(f"TOTAL: +{total}")


if __name__ == "__main__":
    main()
