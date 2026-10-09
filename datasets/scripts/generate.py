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
]

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
