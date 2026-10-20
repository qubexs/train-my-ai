"""Sharded delta trainer untuk xcoder (resume-safe).

Aliran:
  1) SHRED  - pecahkan setiap datasets/<nama>.jsonl besar jadi shard 1k baris
              (finetune/shards/<nama>/part-####.jsonl), dedup, +replay optional.
  2) DELTA  - latih delta LoRA 1 shard sekali, lintas semua fail utama berurutan,
              promote 'master' selepas setiap shard, rekod state (boleh resume).
  3) EXPORT - merge master + tukar GGUF sekali pada penghujung.
  4) LEARN  - selepas shard habis, ambil baris baharu dari app (training.jsonl)
              menerusi store.build_delta_file dan latih sebagai delta tambahan.

Contoh:
  python finetune/train_delta_runner.py --init-master --thinking
  python finetune/train_delta_runner.py --export --self-learn

Setiap peringkat boleh dilangkau dengan flag. Satu runner sahaja pada satu masa
(lock fail). Keluar: 0 ok, 2 tiada GPU, 3 runner lain berjalan, 1 ralat.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOCK = ROOT / "finetune" / ".delta_runner.lock"
MODEL = "Qwen/Qwen3.5-0.8B"
OUT_NAME = "xcoder-gen-0.8b"
SKIP = {"_quarantine"}


def pid_alive(pid):
    if not pid:
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return True
    return str(pid) in out


def acquire_lock():
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii"))
            return fd
        except FileExistsError:
            try:
                pid = int(LOCK.read_text(encoding="utf-8").strip() or 0)
            except (ValueError, OSError):
                pid = 0
            if pid_alive(pid):
                print(f"runner lain sedang berjalan: PID {pid} - keluar")
                sys.exit(3)
            if time.time() - LOCK.stat().st_mtime < 15:
                print("lock baru - mungkin sedang start, cuba lagi nanti")
                sys.exit(3)
            print(f"lock basi (PID {pid}) - buang")
            LOCK.unlink(missing_ok=True)


def load_rows(path, dedup=True):
    rows, seen = [], set()
    try:
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                q = (r.get("instruction") or "").strip()
                a = (r.get("output") or "").strip()
                if len(q) < 2 or len(a) < 2:
                    continue
                key = (q, (r.get("input") or "").strip(), a)
                if dedup and key in seen:
                    continue
                seen.add(key)
                rows.append({"instruction": q, "input": r.get("input") or "",
                             "output": a})
    except OSError:
        return []
    return rows


def load_state(path):
    try:
        st = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(st, dict):
            st.setdefault("done", [])
            return st
    except (OSError, ValueError):
        pass
    return {"done": []}


def save_state(path, st):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def shred(main_dir, shards_root, rows_per, replay, replay_pool):
    main_dir, shards_root = Path(main_dir), Path(shards_root)
    replay_rows = load_rows(replay_pool)[:replay] if (replay and Path(replay_pool).is_file()) else []
    made = 0
    for src in sorted(main_dir.glob("*.jsonl")):
        name = src.stem
        if name in SKIP or name.startswith("all"):
            continue
        rows = load_rows(src)
        if not rows:
            print(f"[shred] {name}: 0 baris - langkau")
            continue
        outdir = shards_root / name
        outdir.mkdir(parents=True, exist_ok=True)
        parts = 0
        for i in range(0, len(rows), rows_per):
            part = i // rows_per + 1
            dest = outdir / f"part-{part:04d}.jsonl"
            if dest.is_file():
                parts += 1
                continue
            chunk = rows[i:i + rows_per] + replay_rows
            tmp = dest.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for r in chunk:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp.replace(dest)
            parts += 1
            made += 1
        print(f"[shred] {name}: {len(rows)} baris -> {parts} shard")
    print(f"[shred] selesai: {made} shard baharu")
    return made


def list_shards(shards_root):
    root = Path(shards_root)
    return sorted(p for p in root.glob("*/part-*.jsonl"))


def row_key(r):
    return ((r.get("instruction") or "").strip(),
            (r.get("input") or "").strip(),
            (r.get("output") or "").strip())


def sha1_file(path):
    import hashlib
    h = hashlib.sha1()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_rows(paths):
    """Gabungan unik (dedup) baris dari senarai fail — untuk audit/dupe."""
    seen, rows = set(), []
    for p in paths:
        for r in load_rows(p, dedup=False):
            k = row_key(r)
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    return rows


def write_full_and_manifest(main_dir, shards_root):
    """Rekod JSONL penuh (union unik sumber) + manifest per shard (audit/dupe)."""
    main_dir, shards_root = Path(main_dir), Path(shards_root)
    srcs = [p for p in sorted(main_dir.glob("*.jsonl"))
            if p.stem not in SKIP and not p.stem.startswith("all")]
    src_rows = unique_rows(srcs)
    shards_root.mkdir(parents=True, exist_ok=True)
    full = shards_root / "_full.jsonl"
    tmp = full.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in src_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(full)
    man = shards_root / "_manifest.jsonl"
    tmp = man.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for shard in list_shards(shards_root):
            rel = shard.relative_to(shards_root)
            f.write(json.dumps({
                "shard": str(rel).replace("\\", "/"),
                "source": shard.parent.name + ".jsonl",
                "part": shard.stem,
                "rows": len(load_rows(shard, dedup=False)),
                "sha1": sha1_file(shard),
            }, ensure_ascii=False) + "\n")
    tmp.replace(man)
    print(f"[audit] full={len(src_rows)} unik dari {len(srcs)} fail -> {full.name}; "
          f"manifest -> {man.name}")
    return full, len(src_rows)


def verify(shards_root, merged_paths):
    """Bandingkan gabungan shard dgn _full + fail merge siap (dupe/hilang/lebih)."""
    shards_root = Path(shards_root)
    shard_keys = {row_key(r) for r in unique_rows(list_shards(shards_root))}
    full = shards_root / "_full.jsonl"
    full_keys = ({row_key(r) for r in load_rows(full, dedup=False)}
                 if full.is_file() else set())
    lines, ok = [], True
    if full_keys:
        miss, extra = full_keys - shard_keys, shard_keys - full_keys
        lines.append(f"shard vs _full: shard={len(shard_keys)} full={len(full_keys)} "
                     f"hilang={len(miss)} lebih={len(extra)}")
        if miss or extra:
            ok = False
    for mp in merged_paths:
        mp = Path(mp)
        if not mp.is_file():
            lines.append(f"merge {mp}: TIADA (langkau)")
            continue
        mrows = load_rows(mp, dedup=False)
        mkeys = {row_key(r) for r in mrows}
        lines.append(
            f"merge {mp.name}: baris={len(mrows)} unik={len(mkeys)} "
            f"dup={len(mrows) - len(mkeys)} sama={len(mkeys & shard_keys)} "
            f"hanya_merge={len(mkeys - shard_keys)} hanya_shard={len(shard_keys - mkeys)}")
    report = shards_root / "_verify.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[verify]\n  " + "\n  ".join(lines) + f"\n[verify] laporan -> {report}")
    return ok


def init_master(master, thinking):
    master = Path(master)
    if (master / "adapter_model.safetensors").is_file():
        print(f"[init-master] sudah ada: {master}")
        return 0
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from qwen_load import load_qwen
    except Exception as e:
        print(f"[init-master] kebergantungan tiada: {e}")
        return 2
    model, tok = load_qwen(MODEL, dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear"))
    master.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(master))
    tok.save_pretrained(str(master))
    tag = "think" if thinking else "base"
    (master / "init.json").write_text(
        json.dumps({"base": MODEL, "mode": tag, "ts": int(time.time())}),
        encoding="utf-8")
    print(f"[init-master] LoRA baharu -> {master} ({tag})")
    return 0


def run_delta(shard, master, out, thinking, a):
    out = Path(out)
    if out.exists() and (out / "adapter_model.safetensors").is_file():
        out = out.with_name(out.name + "-" + time.strftime("%H%M%S"))
    cmd = [PY, str(ROOT / "finetune" / "train_delta_qwen35.py"),
           "--data", str(shard), "--master", str(master), "--out", str(out),
           "--no-export", "--epochs", str(a.epochs), "--lr", str(a.lr),
           "--batch", str(a.batch), "--accum", str(a.accum),
           "--max-len", str(a.max_len)]
    if thinking:
        cmd.append("--thinking")
    print(f"[delta] {Path(shard).parent.name}/{Path(shard).name} -> {out.name}")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def run_export(master, gguf_name, models_dir):
    cmd = [PY, str(ROOT / "finetune" / "train_delta_qwen35.py"),
           "--export-only", "--master", str(master),
           "--gguf-name", gguf_name, "--models-dir", str(models_dir)]
    print(f"[export] master -> models/{gguf_name}.new")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def self_learn(master, rounds, thinking, a):
    sys.path.insert(0, str(ROOT / "app"))
    try:
        import store
    except Exception as e:
        print(f"[learn] store tiada: {e}")
        return 0
    done = 0
    while done < rounds:
        dest = ROOT / "build" / "train" / f"learn-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        path, nd, nr, note = store.build_delta_file(dest=str(dest))
        if not path:
            print(f"[learn] tiada baris baharu ({note}) - tamat")
            break
        print(f"[learn] pusingan {done + 1}: {nd} delta + {nr} replay")
        rc = run_delta(path, master, ROOT / "build" / "train" / f"learn-{done + 1}",
                       thinking, a)
        if rc:
            print(f"[learn] latihan gagal rc={rc} - henti")
            return rc
        store.mark_trained()
        done += 1
    print(f"[learn] selesai: {done} pusingan")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Sharded delta trainer xcoder")
    ap.add_argument("--main-dir", default=str(ROOT / "datasets"))
    ap.add_argument("--shards-root", default=str(ROOT / "finetune" / "shards"))
    ap.add_argument("--rows", type=int, default=1000, help="baris per shard")
    ap.add_argument("--replay", type=int, default=0,
                    help="baris replay anti-lupa ditambah pada setiap shard")
    ap.add_argument("--replay-pool", default=str(ROOT / "datasets" / "general.jsonl"))
    ap.add_argument("--master", default=str(ROOT / "finetune" / "masters" /
                                            "xcoder-gen1-think"))
    ap.add_argument("--init-master", action="store_true",
                    help="cipta master LoRA baharu jika tiada")
    ap.add_argument("--thinking", action="store_true",
                    help="latih dengan medan reasoning (<think>)")
    ap.add_argument("--max-shards", type=int, default=0, help="0 = semua")
    ap.add_argument("--max-minutes", type=float, default=0, help="0 = tiada had")
    ap.add_argument("--shred-only", action="store_true")
    ap.add_argument("--verify", dest="verify", action="store_true", default=True,
                    help="rekod _full.jsonl + manifest & bandingkan (lalai on)")
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    ap.add_argument("--verify-only", action="store_true",
                    help="audit/verify sahaja, tiada latihan")
    ap.add_argument("--merged", nargs="*",
                    default=[str(ROOT / "finetune" / "stacks" / "all" / "dataset.jsonl")],
                    help="fail merge siap untuk dibandingkan")
    ap.add_argument("--export", action="store_true",
                    help="merge + GGUF selepas semua shard selesai")
    ap.add_argument("--self-learn", action="store_true")
    ap.add_argument("--learn-rounds", type=int, default=4)
    ap.add_argument("--gguf-name", default="xcoder-gen1-think-0.8B.Q4_K_M")
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    a = ap.parse_args()

    lockfd = acquire_lock()
    state_p = Path(a.shards_root) / ".state.json"
    try:
        print(f"[runner] mula {time.strftime('%Y-%m-%d %H:%M:%S')}")
        if a.init_master:
            rc = init_master(a.master, a.thinking)
            if rc:
                return rc
        shred(a.main_dir, a.shards_root, a.rows, a.replay, a.replay_pool)
        if a.verify:
            write_full_and_manifest(a.main_dir, a.shards_root)
            verify(a.shards_root, a.merged)
        if a.shred_only or a.verify_only:
            return 0

        shards = list_shards(a.shards_root)
        state = load_state(state_p)
        done = set(state["done"])
        todo = [s for s in shards if str(s.relative_to(a.shards_root)) not in done]
        print(f"[runner] {len(shards)} shard, {len(todo)} belum dilatih")
        if todo and not (Path(a.master) / "adapter_model.safetensors").is_file():
            print("[runner] master tiada - jalankan dengan --init-master dulu")
            return 1

        t0 = time.time()
        trained = 0
        for shard in todo:
            if a.max_shards and trained >= a.max_shards:
                print("[runner] had --max-shards dicapai")
                break
            if a.max_minutes and (time.time() - t0) / 60 >= a.max_minutes:
                print("[runner] had --max-minutes dicapai")
                break
            rel = shard.relative_to(a.shards_root)
            out = ROOT / "build" / "train" / "delta-shards" / shard.parent.name / shard.stem
            rc = run_delta(shard, a.master, out, a.thinking, a)
            if rc:
                print(f"[runner] henti: shard {rel} rc={rc}")
                return rc
            done.add(str(rel))
            state["done"] = sorted(done)
            state["updated"] = int(time.time())
            save_state(state_p, state)
            trained += 1

        remaining = len(shards) - len(done)
        print(f"[runner] {trained} shard dilatih sesi ini; baki {remaining}")

        if a.export and remaining == 0:
            rc = run_export(a.master, a.gguf_name, a.models_dir)
            if rc:
                return rc
        elif a.export:
            print("[runner] --export dilangkau (shard belum habis)")

        if a.self_learn and remaining == 0:
            rc = self_learn(a.master, a.learn_rounds, a.thinking, a)
            if rc:
                return rc
            if a.export:
                rc = run_export(a.master, a.gguf_name, a.models_dir)
                if rc:
                    return rc

        print(f"[runner] tamat {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return 0
    finally:
        os.close(lockfd)
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
