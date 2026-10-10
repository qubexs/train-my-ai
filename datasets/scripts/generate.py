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
    "Apakah fungsi `{c}`?",
    "Beri contoh sebenar `{c}` dalam Bahasa Melayu.",
    "How would you explain `{c}` to a junior developer?",
    "Give a real-world scenario for `{c}`.",
    "What problem does `{c}` solve?",
]

PHRASINGS_CMD = [
    "What does `{c}` do?",
    "Explain `{c}` briefly.",
    "When do I use `{c}`?",
    "Give a practical example of `{c}`.",
    "What is `{c}` for?",
    "Apakah fungsi `{c}`?",
    "Terangkan `{c}` secara ringkas.",
    "Bila perlu guna `{c}`?",
    "Beri contoh penggunaan `{c}`.",
    "`{c}` untuk apa?",
    "How would you explain `{c}` to a junior developer?",
    "Give a real-world scenario for `{c}`.",
    "What problem does `{c}` solve?",
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
    "Apakah SQL bagi: {q}?",
    "Tuliskan ayat SQL: {q}.",
    "Lengkapkan SQL ini: {q}.",
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
    ("history", "`history` recalls past commands; `!42` reruns number 42, `!!` repeats last — verify before rerunning destructive ones."),
    ("alias ll", "`alias ll='ls -la'` shortens favorites; persist in ~/.bashrc, reload with `source ~/.bashrc`."),
    ("export PATH", "`export PATH=$PATH:/opt/bin` extends lookup; verify with `which cmd` and persist in shell rc files."),
    ("echo $?", "`echo $?` prints the last exit code (0 success); scripts branch on it immediately before it gets clobbered."),
    ("whoami id", "`whoami`/`id` show user, groups and ids — first check when permissions mysteriously deny."),
    ("uptime w", "`uptime`/`w` show load averages (1/5/15 min) and logged users; load ≈ CPU count means saturated."),
    ("date", "`date` prints/sets clock (`date -u` UTC); logs in UTC avoid timezone archaeology later."),
    ("df -i", "`df -i` reveals inode exhaustion — full with free space means millions of tiny files (caches, sessions)."),
    ("du --max-depth=1", "`du -h --max-depth=1 | sort -h` drills one level at a time toward the hog without full scans."),
    ("sudo !!", "`sudo !!` reruns the last command as root after permission-denied — the most typed sudo pattern."),
    ("apt update upgrade", "`apt update` refreshes lists, `upgrade` installs; `-y` automates, autoremove cleans strays."),
    ("systemctl enable --now", "`--now` enables plus starts in one shot; `is-enabled`/`is-active` verify each half."),
    ("lsblk", "`lsblk` maps disks, partitions and mountpoints as a tree — partitioning mistakes show up here first."),
    ("mount umount", "`mount /dev/sdb1 /data` attaches storage; `umount` detaches (fails if busy — find culprits with `lsof`)."),
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
    ("select menu", "`select opt in mula henti keluar; do case $opt in ...` builds interactive menus with numbered prompts."),
    ("getopts flags", "`while getopts 'vf:' o; do case $o in v) ...;; f) fail=$OPTARG;;` parses script flags robustly."),
    ("readonly VAR", "`readonly`/`declare -r` freezes constants; typos then fail loudly instead of forking behavior silently."),
    ("mapfile lines", "`mapfile -t arr < file` loads lines into an array fast; beats `while read` loops for big files."),
    ("printf format", "`printf '%-20s %5d\\n' name num` formats columns portably; `echo` flags vary across shells."),
    ("pushd popd", "`pushd dir` stacks directories, `popd` returns — `dirs -v` lists the stack for deep navigation."),
    ("test -e -s -x", "`-e` exists, `-s` non-empty, `-x` executable — probe before acting instead of rescuing failures."),
    ("${s/old/new}", "`${s/old/new}` substitutes first match (`//` for all); pure-bash edits avoid subshell forks."),
    ("${s:0:3}", "Substring slicing (`${s:0:3}`, `${s: -4}`) extracts parts; negative offsets count from the end."),
    ("${VAR:=default}", "`:=` assigns the default permanently when empty — one-shot config with memory."),
    ("source lib.sh", "`source` (or `.`) loads function libraries into the current shell; guard with file existence tests."),
    ("set -x debug", "`set -x` traces every command before running; wrap noisy sections with `set +x` to silence."),
    ("exit codes", "`exit 0` success, nonzero failure (1 generic, 2 misuse, 126/127 exec problems); document yours in usage text."),
    ("jobs wait &", "`cmd &` backgrounds, `jobs` lists, `wait` blocks until done — poor man's parallelism with `wait -n`."),
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
    ("fetch API", "`fetch(url)` returns promises; check `response.ok` since HTTP errors don't reject — only network failures do."),
    ("addEventListener", "One listener per element beats inline `onclick`; `{ once: true }` auto-removes single-shot handlers."),
    ("querySelector", "`querySelector('.item')` grabs first CSS match, `querySelectorAll` all (static NodeList, not live)."),
    ("classList toggle", "`el.classList.toggle('on')` flips classes idempotently; second arg forces state — no string surgery."),
    ("createElement append", "`document.createElement` + `appendChild` builds DOM safely; `innerHTML` with user data invites XSS."),
    ("Array.from", "`Array.from({length: 5}, (_, i) => i)` manufactures ranges; second arg maps during construction."),
    ("flat flatMap", "`flat()` unwraps one level (Infinity for all); `flatMap` maps then flattens — filter+map in one pass."),
    ("sort compare", "`sort()` stringifies by default (`[10, 9]` → wrong); always pass `(a, b) => a - b` for numbers."),
    ("reverse", "`reverse()` flips in place (mutates!); copy first (`[...a].reverse()`) when the original must survive."),
    ("indexOf", "`indexOf` returns position or -1; `includes` reads better for pure membership questions."),
    ("charAt substring", "`charAt(i)`/`substring(a, b)` extract safely (out-of-range yields ''); negative indices need `slice`."),
    ("touPPerCase trim", "`toUpperCase/toLowerCase` normalize case; `trim()` strips edge whitespace before validating input."),
    ("Math.random floor", "`Math.floor(Math.random() * n)` rolls 0..n-1; crypto needs `crypto.getRandomValues`, not Math."),
    ("Date now", "`Date.now()` timestamps in ms for timing; `new Date()` objects format via `toISOString`/`toLocaleString`."),
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
    ("while break continue", "`while` guards with explicit progress; `break` exits, `continue` skips — infinite loops need a visible exit."),
    ("*args **kwargs", "`*args` collects positional extras, `**kwargs` named ones — forward both in wrappers to stay transparent."),
    ("decorators @", "Decorators wrap functions (`@lru_cache`, `@property`); they run at definition, so keep them side-effect free."),
    ("generators yield", "`yield` produces lazily item by item — infinite sequences and huge files stream with constant memory."),
    ("slicing [::-1]", "`s[::-1]` reverses, `s[1:5]` windows; slices copy (safe to mutate) unlike views in other languages."),
    ("sorted key", "`sorted(items, key=lambda r: r[1])` orders by any criterion; `reverse=True` flips without re-sorting."),
    ("any all", "`any()`/`all()` short-circuit across iterables with generator expressions — readable and lazy."),
    ("sum min max", "Builtins beat manual loops (`sum(x)`, `min(x, key=len)`); pass `default=` to survive empty inputs."),
    ("input()", "`input('Nama? ')` reads a line (always string — convert explicitly); strip and validate before trusting."),
    ("type() isinstance", "`type(x)` names the exact class; `isinstance(x, (A, B))` respects inheritance — prefer the latter for checks."),
    ("super()", "`super().__init__()` extends parent construction in subclasses; cooperative MI depends on every link calling it."),
    ("raise custom", "Raise specific builtins or small custom `Exception` subclasses with messages; bare `raise` reraises current."),
    ("assert", "`assert cond, msg` documents invariants in dev/tests; never gate production logic on it (`-O` strips asserts)."),
    ("breakpoint()", "`breakpoint()` drops into pdb where written (respects PYTHONBREAKPOINT); remove or gate before shipping."),
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
    ("Pick", "`Pick<T, 'a' | 'b'>` carves a subtype with named fields; `Omit` drops them — both derive, never redeclare."),
    ("Omit", "`Omit<Response, 'secret'>` strips sensitive fields for public shapes; the source stays canonical."),
    ("Partial", "`Partial<T>` makes every field optional — for updates/patches; `Required<T>` reverses it for completed states."),
    ("Required Readonly", "`Required` fills optionals, `Readonly` freezes writes; compose them (`Readonly<Partial<T>>`) for draft snapshots."),
    ("Exclude Extract", "`Exclude<A, B>` removes members, `Extract` keeps the overlap — set algebra over union types."),
    ("ReturnType Parameters", "`ReturnType<typeof f>`/`Parameters<typeof f>` mirror function shapes so wrappers never drift from implementations."),
    ("non-null !", "Postfix `!` asserts non-null to the checker — a lie the compiler believes; prefer narrowing or explicit throws."),
    ("void", "`void` means returns-nothing-useful (callbacks, setters); `undefined` is a value, `never` is unreachable — three ideas, not one."),
    ("overloads", "Overload signatures give precise call shapes over one implementation; beyond three, an options object reads better."),
    ("const assertions", "`as const` freezes literals (`'get'`, not `string`) for discriminators and config maps checked exhaustively."),
    ("template literal types", "`` `GET /${Resource}` `` types URL patterns at compile; typos in route strings become errors, not 404s."),
    ("conditional types", "`T extends U ? A : B` branches on types — the machinery behind Exclude/ReturnType; distribute over unions carefully."),
    ("index signatures", "`{ [k: string]: number }` types open dictionaries; `Record` is the tidy alias, `Map` the runtime equivalent."),
    ("noUncheckedIndexedAccess", "This flag adds `| undefined` to every index access — array/dict reads must handle missing, killing a whole bug class."),
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
    ("os module", "`os.cpus().length` sizes pools, `os.tmpdir()` finds scratch space, `os.homedir()` locates configs portably."),
    ("path basename extname", "`basename(p)` strips dirs, `extname(p)` grabs `.json` (leading-dot files count as no-extension)."),
    ("process.exit codes", "`process.exit(0)` clean, nonzero failure; flush streams first — abrupt exits truncate piped output."),
    ("cluster fork", "`cluster.fork()` clones one worker per CPU behind one port; the primary only routes, never serves."),
    ("worker threads", "Workers run CPU-bound JS off the loop sharing via `postMessage`; SharedArrayBuffer needs cross-origin isolation headers."),
    ("fs.watch", "`fs.watch` notifies on file change (debounce it — editors emit bursts); chokidar smooths platform quirks."),
    ("zlib gzip", "Compress responses/logs with `zlib.createGzip()` streams; level trades CPU for bytes on hot paths."),
    ("ESM JSON import", "`import data from './x.json' with { type: 'json' }` loads config statically; dynamic needs `fs` + `JSON.parse`."),
    ("npm run env", "`npm run` injects `./node_modules/.bin` into PATH so local CLIs (tsc, vitest) run without global installs."),
    ("NODE_ENV", "`NODE_ENV=production` gates dev-only middleware/logging; read once at boot into a validated config object."),
    ("diagnostics timing", "`console.time('sql')`/`timeEnd` profiles blocks; `process.hrtime.bigint()` nanosecond-diffs hot loops."),
    ("unhandledRejection", "Crash-or-log unhandled rejections explicitly at startup — silent promise failures corrupt state invisibly."),
    ("AbortController fetch", "Pass `signal` into fetch and `abort()` on supersede; catch AbortError separately from real failures."),
    ("readline completer", "`readline` accepts a `completer` for tab-completion in CLIs; persist history with a file-backed interface."),
    ("worker pool queue", "Bounded queues (`p-limit`) cap concurrency on downstream APIs; unbounded `Promise.all` is a self-DDoS."),
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
    ("strong em", "`<strong>`/`<em>` carry meaning (importance/stress) that `<b>`/`<i>` lack; screen readers vocalize the difference."),
    ("blockquote cite", "`<blockquote cite=url>` marks quotations with source; `<cite>` names the work — semantics citation tools consume."),
    ("code pre kbd samp", "`<code>` inline snippets, `<pre>` preserves blocks, `<kbd>` user keystrokes, `<samp>` program output — four tags, four jobs."),
    ("input types", "`type=email|number|date|tel|url|color|range` summon right keyboards and free validation; pick the narrowest that fits."),
    ("label for id", "`<label for=id>` focuses its control on click (bigger hit area, screen-reader bound); wrapping works too but `for` is explicit."),
    ("meter progress", "`<meter>` gauges known quantities (disk), `<progress>` task completion — both expose values to assistive tech unlike div bars."),
    ("datalist", "`<input list=x>` + `<datalist id=x>` suggests options while allowing free text — combobox without JavaScript."),
    ("picture source", "`<picture>` serves AVIF/WebP with `<img>` fallback plus art-directed crops per breakpoint via `media`."),
    ("track subtitles", "`<track kind=subtitles srclang=ms>` captions video natively; transcripts alongside serve search and print."),
    ("canvas", "`<canvas>` is a scriptable bitmap (charts, games) — inherently inaccessible, so mirror state in DOM/ARIA."),
    ("svg inline", "Inline `<svg>` scales perfectly and styles with CSS; complex art belongs in files, icons inline for zero requests."),
    ("template tag", "`<template>` holds inert clonable markup (`content.cloneNode`) — lists and dialogs without string HTML."),
    ("form action method", "`action` targets, `method` verbs (GET idempotent searches, POST mutations); `enctype` multipart for file uploads."),
    ("hidden attribute", "The `hidden` attribute hides semantically (stronger than CSS `display:none` for AT); toggle the property, not styles."),
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
    ("margin auto center", "`margin: 0 auto` centers fixed-width blocks horizontally; needs explicit width — flex/grid center anything."),
    ("padding shorthand", "`padding: 10px 20px` sets vertical/horizontal (1-4 values clockwise from top); margins mirror the pattern."),
    ("outline vs border", "`outline` draws outside without shifting layout (focus rings!); `border` occupies space and rounds."),
    ("display none invisible", "`display:none` removes from layout and AT; `visibility:hidden` hides visually but reserves space."),
    ("flex-wrap", "`flex-wrap: wrap` flows overflow onto rows (add `gap`); nowrap (default) squeezes or overflows."),
    ("flex-grow", "`flex-grow: 2` claims double leftover space vs siblings at 1; `flex-shrink` governs squeeze, `flex-basis` the start size."),
    ("sticky vs fixed", "`sticky` scrolls normally then pins within its parent (needs offset + room); `fixed` pins to viewport always."),
    ("top left offsets", "Offsets need positioned elements (`relative/absolute/fixed/sticky`); static ignores them silently — the classic no-op bug."),
    ("list-style", "`list-style: none` strips bullets (add padding:0 too); custom markers via `::marker` keep semantics."),
    ("line-height", "Unitless `line-height: 1.5` scales with font size (accessible); fixed px breaks zoomed text rhythm."),
    ("cursor pointer", "`cursor: pointer` signals clickability on custom controls; `not-allowed` communicates disabled honestly."),
    ("user-select", "`user-select: none` stops text selection on buttons/labels; never apply to body copy users may need."),
    ("scroll-behavior smooth", "`scroll-behavior: smooth` animates anchor jumps natively; respect reduced-motion with a media query guard."),
    ("filter blur brightness", "Filters (`blur()`, `brightness()`, `grayscale()`) style images live; they spawn stacking contexts, so z-index planning follows."),
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
    ("echo vs print", "`echo` outputs (no return, multiple args); `print` returns 1 — both beat string-concat debugging via error_log."),
    ("comments", "`//` lines, `#` lines, `/** */` docblocks for IDE hints; stale comments lie — delete code instead of commenting it out."),
    ("constants define", "`define('HAD', 100)` or `const HAD = 100` (compile-time, namespaced); magic numbers deserve names."),
    ("string concat .", "`.` concatenates (`'a' . $b`); `.=` appends — PHP has no `+` for strings (that adds numbers)."),
    ("== vs ===", "Loose `==` juggles types (`'0' == false` is true!); strict `===` checks type too — default to strict."),
    ("switch", "`switch` branches scalar equality with `break` per arm (forgetting falls through); `match` returns values strictly."),
    ("do-while", "`do { } while ($c);` runs at least once (menus, retries); plain `while` may run zero times."),
    ("global static vars", "`global` imports scope (avoid), `static` persists across calls (counters, memoization) — both need discipline."),
    ("references &$var", "`&$var` aliases (modify caller data, foreach by reference); unset after loops to avoid spooky reuse."),
    ("null coalescing ??", "`$nama ?? 'tetamu'` defaults null/undefined only; `??=` assigns lazily — cleaner than ternaries for config."),
    ("spaceship <=>", "`$a <=> $b` returns -1/0/1 — the one-line comparator for `usort` callbacks and version checks."),
    ("match expression", "`match($x) { 1 => 'satu', default => 'lain' }` is strict, exhaustive-checked, and returns — switch done right."),
    ("arrow fn fn", "`fn($x) => $x * 2` auto-captures scope (unlike `function() use`) — for short callbacks only."),
    ("named arguments", "`bina(nama: 'Ali', umur: 20)` self-documents calls and survives reordering; pairs with defaults for options bags."),
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
    ("serve artisan", "`php artisan serve` boots dev instantly; production pairs nginx/php-fpm — never serve publicly."),
    ("route model binding", "Type-hinted `{barang}` params auto-resolve models (404 when missing); custom keys via `getRouteKeyName`."),
    ("middleware custom", "`make:middleware SemakUmur` plus `->middleware()` gates routes; global middleware runs on everything — keep it lean."),
    ("FormRequest", "FormRequests move validation+auth out of controllers (`authorize()`, `rules()`); reuse across web and API."),
    ("API resources", "`JsonResource` shapes JSON output (hide internals, nest relations); collections paginate uniformly."),
    ("paginate links", "`Model::paginate(15)` plus `{{ $items->links() }}` renders pages; `simplePaginate` skips total counts for speed."),
    ("soft deletes", "`SoftDeletes` trait + `deleted_at` hides instead of destroying; `withTrashed()/restore()/forceDelete()` manage the lifecycle."),
    ("casts", "`$casts = ['meta' => 'array', 'aktif' => 'boolean']` converts attributes automatically — JSON columns behave natively."),
    ("observers", "Observers hook created/updated/deleted for side effects (cache busting, logs) without fattening models."),
    ("notifications", "`$user->notify(new PesananSiap)` fans out mail/database channels from one call; queue them for speed."),
    ("storage put get", "`Storage::put/get/delete` abstracts local/S3 behind one API; `temporaryUrl()` shares private files safely."),
    ("cache remember", "`Cache::remember('kunci', 3600, fn)` memoizes expensive work; tag-based flush beats blind `flush()` invalidation."),
    ("DB raw select", "`DB::select('... ? ...', [$x])` escapes bindings; raw strings without bindings are injection holes."),
    ("chunk cursor", "`Model::chunk(500, fn)` / `cursor()` stream millions without OOM; never `all()` a big table."),
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
    ("USE database", "`USE kedai;` scopes unqualified tables; scripts should still qualify names in joins for clarity."),
    ("DESCRIBE table", "`DESCRIBE t` (or `SHOW COLUMNS`) reveals types, nullability, keys and defaults — schema archaeology 101."),
    ("information_schema", "Query `information_schema.tables/columns` for cross-DB metadata scripts (audits, doc generators)."),
    ("SHOW CREATE TABLE", "Reveals the full DDL including engine, charset and keys — the authoritative source when docs lie."),
    ("RENAME TABLE", "`RENAME TABLE lama TO baru` is atomic and fast; update code references in the same deploy or break."),
    ("ADD COLUMN", "`ADD COLUMN x INT DEFAULT 0` backfills instantly on modern MySQL; adding NOT NULL without default rewrites."),
    ("DROP COLUMN", "Dropping is instant metadata-wise on new versions but locks old ones — off-peak plus backups first."),
    ("ADD INDEX", "`ADD INDEX (a, b)` speeds WHERE/JOIN/ORDER on those columns in that order; verify with EXPLAIN."),
    ("TRUNCATE", "Instant full clear without row WAL; cannot WHERE, resets auto-increment — staging/test tool, never prod habit."),
    ("REPLACE INTO", "`REPLACE` deletes-then-inserts on key clash (burns auto-increment ids); prefer ON DUPLICATE KEY for upserts."),
    ("ON DUPLICATE KEY", "`... ON DUPLICATE KEY UPDATE bil = bil + VALUES(bil)` upserts atomically — no read-modify-write races."),
    ("LOAD DATA INFILE", "Bulk-loads CSVs orders of magnitude faster than INSERT loops; `LOCAL` reads client-side files."),
    ("OPTIMIZE TABLE", "Reclaims fragmented space after heavy churn (rebuilds table — locks without online DDL); ANALYZE refreshes stats cheaply."),
    ("FOREIGN KEY ADD", "`ADD CONSTRAINT fk FOREIGN KEY (x) REFERENCES t(id)` enforces links; index the child column or every write scans."),
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
    ("\\dt \\du", "`\\dt` tables, `\\du` roles, `\\dv` views, `\\df` functions — meta-commands beat memorizing catalog queries."),
    ("\\x expanded", "`\\x on` pivots wide rows vertical for readable inspection; `\\timing` shows every query's milliseconds."),
    ("\\copy vs COPY", "`\\copy` moves files client-side (no superuser); server `COPY` needs filesystem rights — know which side you're on."),
    ("\\i file.sql", "`\\i` runs script files inside psql (migrations, seeds); `\\o out.txt` redirects all output to a file."),
    ("DISTINCT ON", "`DISTINCT ON (bandar)` keeps one row per group (first by ORDER BY) — Postgres-only expressiveness for latest-per-group."),
    ("FETCH FIRST", "`FETCH FIRST 10 ROWS ONLY` is standard-SQL LIMIT; `OFFSET` pairs the same pagination caveats (prefer keyset)."),
    ("UNION INTERSECT EXCEPT", "Set operators combine query shapes (matching columns); INTERSECT/EXCEPT filter overlaps/differences — UNION dedupes unless ALL."),
    ("generate_series", "`generate_series(1, 100)` fabricates rows for tests/calendars without tables — scaffolding for gaps-and-islands analysis."),
    ("date_trunc", "`date_trunc('month', dibuat)` buckets timestamps; group by it for monthly reports in one pass."),
    (":: cast", "`harga::numeric` / `created::date` convert inline; failing casts abort — use `TRY_CAST`-style CASE guards for dirty data."),
    ("ARRAY unnest", "Native arrays (`text[]`) with `unnest()` explode to rows; GIN indexes them — think twice before normalizing tiny sets."),
    ("information_schema pg", "`information_schema` stays portable; `pg_catalog`/`pg_stat_*` expose Postgres-specific health (bloat, locks, waits)."),
    ("\\password", "`\\password app` rotates role passwords interactively (scram-hashed, never plaintext in history); rotate service accounts quarterly."),
    ("\\q quit", "`\\q` (or Ctrl-D) exits psql; unsaved `\\e` editor buffers warn first — check before killing long sessions."),
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
    ("prose typography", "The typography plugin (`prose`) styles raw HTML/Markdown readably; tune `prose-lg`/`prose-invert` per context."),
    ("line-clamp", "`line-clamp-3` truncates to N lines with ellipsis — cards and previews without JS measuring."),
    ("aspect-video square", "`aspect-video`/`aspect-square` reserve media ratios (no layout shift); arbitrary `aspect-[4/3]` for customs."),
    ("columns break", "`columns-3` magazine-flows content; `break-inside-avoid` keeps cards from splitting across columns."),
    ("ring offset", "`ring-2 ring-offset-2` focus halos that don't shift layout (unlike borders); brand-color the ring."),
    ("placeholder:", "`placeholder:text-gray-400` styles hint text; `placeholder-shown:` variants react to empty inputs."),
    ("file: variant", "`file:mr-4 file:rounded file:bg-blue-50` styles native file buttons — no custom upload widgets needed."),
    ("first last odd even", "Positional variants stripe lists (`odd:bg-gray-50`) and trim edges (`first:pt-0`) without extra classes."),
    ("group-hover", "Parent `group` + child `group-hover:` reveals actions on hover — tooltips and overlays declaratively."),
    ("peer-checked", "Sibling `peer` + `peer-checked:` styles custom radios/toggles from hidden native inputs — accessible by construction."),
    ("has-[:checked]", "`has-[input:checked]:bg-blue-50` styles ancestors by descendant state — parent selectors without JS."),
    ("aria-* variants", "`aria-expanded:`/`aria-disabled:` mirror component state accessibly; pair with real ARIA attributes, not divs."),
    ("data-* variants", "`data-[size=lg]:text-lg` keys styles off data attributes — server-rendered variants without JS toggles."),
    ("open: details", "`open:` styles expanded `<details>` natively (accordions, dropdowns); combine with `group-open:` markers."),
    ("print: variant", "`print:hidden` strips nav/ads on paper while `print:block` reveals print-only summaries — one codebase, two media."),
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
