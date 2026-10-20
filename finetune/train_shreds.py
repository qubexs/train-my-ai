"""Train one-by-one over datasets/shred/<stack>/shred-NNNNN.jsonl (Colab T4).

Flow per shard:
  list (GitHub) -> download -> audit -> dedupe -> filter -> clean copy
  -> train 1 epoch -> checkpoint -> next shard -> resume after disconnect.

Usage (Colab, repo cloned or FILE fetched):
  python finetune/train_shreds.py --model myxcoder/xcoder-gen1-0.8b --list-only
  python finetune/train_shreds.py --model myxcoder/xcoder-gen1-0.8b --dry-run
  python finetune/train_shreds.py --model myxcoder/xcoder-gen1-0.8b

Resume: rerun the same command; completed shards are skipped and training
continues from the last saved adapter (state: <out>/state.json).
--model must be HF safetensors (a GGUF cannot be trained).
Checkpoints + cleaned shards live under <out> (mount Drive first on Colab).

Add new knowledge (avoid forgetting):
  mkdir datasets/shred/new          # shard-NNNNN.jsonl, <=5000 rows each
  python finetune/train_shreds.py --model <merged> --stacks new \
      --epochs 2 --lr 5e-5 --replay 0.2 --eval-ratio 0.01
  --replay mixes old rows from <out>/clean into every new shard; --eval-ratio
  holds out a deterministic % and logs eval_loss per shard; --shuffle-shards
  randomises stack order (seeded). Re-run with the same --out to resume.
  Switching to a merged base? Use a NEW --out and --replay-from <old>/clean.
"""
import argparse
import hashlib
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "datasets" / "scripts"))

SYSTEM = ("You are XCoder, a tiny CPU coding assistant. "
          "Answer briefly in the user's language (Bahasa Melayu or English). "
          "Never mention Qwen/Alibaba.")

REPO = "qubexs/train-my-ai"
BRANCH = "master"
SHRED_PREFIX = "datasets/shred/"


def log(msg):
    print(f"[shreds] {msg}", flush=True)


def http_text(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "xcoder-shreds"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def list_shards(repo, branch, local_dir):
    local = Path(local_dir)
    if local.is_dir() and any(local.glob("*/*.jsonl")):
        paths = [p.relative_to(local).as_posix() for p in local.glob("*/*.jsonl")]
        return [f"{SHRED_PREFIX}{p}" for p in sorted(paths)]
    url = (f"https://api.github.com/repos/{repo}/git/trees/{branch}"
           f"?recursive=1")
    try:
        tree = json.loads(http_text(url)).get("tree", [])
        found = sorted(t["path"] for t in tree
                       if t.get("type") == "blob"
                       and t["path"].startswith(SHRED_PREFIX)
                       and t["path"].endswith(".jsonl"))
        if found:
            return found
        raise ValueError("tree kosong")
    except Exception as e:  # noqa: BLE001 - API hadir had kadar (403)
        log(f"amaran: GitHub API gagal ({e}) - guna manifest")
    manifest = (f"https://raw.githubusercontent.com/{repo}/{branch}/"
                f"{SHRED_PREFIX}manifest.txt")
    return [ln.strip() for ln in http_text(manifest).splitlines() if ln.strip()]


def fetch_shard(path, repo, branch, raw_dir, local_dir):
    local = Path(local_dir)
    if local.is_dir():
        cand = local / path[len(SHRED_PREFIX):]
        if cand.exists():
            return cand.read_text(encoding="utf-8")
    dest = Path(raw_dir) / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size == 0:
        url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
        dest.write_text(http_text(url, timeout=120), encoding="utf-8")
    return dest.read_text(encoding="utf-8")


def drop_cached(path, raw_dir):
    p = Path(raw_dir) / path
    if p.exists():
        p.unlink()
        try:
            p.parent.rmdir()
        except OSError:
            pass


def load_limits(a):
    return {"min_q": a.min_q, "max_q": a.max_q,
            "min_a": a.min_a, "max_a": a.max_a}


try:
    from _common import check_row
    from audit import scrub_hits
except Exception:  # noqa: BLE001 - standalone (script fetched without repo)
    def check_row(r):
        return (isinstance(r, dict) and isinstance(r.get("instruction"), str)
                and isinstance(r.get("output"), str)
                and r["instruction"].strip() and r["output"].strip())

    _FORBIDDEN = ["qwen", "alibaba", "tongyi", "dashscope",
                  "as an ai language model", "as a large language model"]
    _DENIAL = ["bukan", "not ", "never", "tiada", "tidak ", "jangan",
               "no. i am", "no, i am", "i am xcoder", "saya xcoder"]

    def scrub_hits(text):
        low = text.lower()
        return [t for t in _FORBIDDEN
                if t in low and not any(m in low for m in _DENIAL)]


def clean_rows(text, seen, limits, stack):
    kept, stats = [], {"rows": 0, "bad": 0, "bounds": 0, "scrub": 0,
                       "dupe": 0, "retag": 0}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        stats["rows"] += 1
        try:
            r = json.loads(line)
        except ValueError:
            stats["bad"] += 1
            continue
        if not check_row(r) or not isinstance(r.get("input", ""), str):
            stats["bad"] += 1
            continue
        q = r["instruction"].strip()
        a = r["output"].strip()
        inp = (r.get("input") or "").strip()
        if not (limits["min_q"] <= len(q) <= limits["max_q"]
                and limits["min_a"] <= len(a) <= limits["max_a"]):
            stats["bounds"] += 1
            continue
        if scrub_hits(q + "\n" + a):
            stats["scrub"] += 1
            continue
        key = (q, inp, a)
        if key in seen:
            stats["dupe"] += 1
            continue
        seen.add(key)
        row = {"instruction": q, "input": inp, "output": a}
        if (r.get("reasoning") or "").strip():
            row["reasoning"] = r["reasoning"].strip()
        kept.append(row)
    return kept, stats


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    out = []
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def rebuild_seen(clean_dir, completed):
    seen = set()
    for path in completed:
        for r in read_jsonl(Path(clean_dir) / path):
            seen.add((r["instruction"], r.get("input", ""), r["output"]))
    return seen


def validate_model(model):
    p = Path(model)
    if p.exists():
        if not (p / "config.json").exists():
            raise SystemExit(f"{model}: tiada config.json (bukan model HF)")
        if not (any(p.glob("*.safetensors")) or any(p.glob("*.bin"))):
            raise SystemExit(f"{model}: tiada pemberat safetensors/bin")
        return
    try:
        info = json.loads(http_text(f"https://huggingface.co/api/models/{model}"))
        files = [s["rfilename"] for s in info.get("siblings", [])]
    except Exception as e:  # noqa: BLE001
        log(f"amaran: tak dapat semak {model} ({e}) - jika private, "
            f"login HF dahulu (huggingface-cli login)")
        return
    has_cfg = "config.json" in files
    has_w = any(f.endswith((".safetensors", ".bin")) for f in files)
    if not (has_cfg and has_w):
        raise SystemExit(
            f"{model} tiada config.json/safetensors (GGUF sahaja) - tidak boleh "
            f"dilatih. Upload merged safetensors ke repo itu, atau guna "
            f"--model Qwen/Qwen3.5-0.8B.")


def load_state(path, model):
    if Path(path).exists():
        st = json.loads(Path(path).read_text(encoding="utf-8"))
        if st.get("model") != model:
            raise SystemExit(f"state model={st.get('model')} != --model {model}; "
                             f"guna --reset atau --out lain")
        return st
    return {"model": model, "completed": [], "last_ckpt": None,
            "rows": 0, "updated": None}


def save_state(path, st):
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(st, indent=1), encoding="utf-8")


def to_text(tok, r, thinking):
    user = r["instruction"]
    if r.get("input"):
        user += "\n" + r["input"]
    asst = {"role": "assistant", "content": r["output"]}
    if thinking:
        asst["reasoning_content"] = (r.get("reasoning") or "").strip()
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}, asst]
    return tok.apply_chat_template(msgs, tokenize=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF safetensors repo/folder (NOT a .gguf)")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--branch", default=BRANCH)
    ap.add_argument("--shred-dir", default=str(ROOT / "datasets" / "shred"))
    ap.add_argument("--out", default="/content/drive/MyDrive/xcoder/shreds")
    ap.add_argument("--max-seq", type=int, default=3072)
    ap.add_argument("--min-q", type=int, default=5)
    ap.add_argument("--max-q", type=int, default=16000)
    ap.add_argument("--min-a", type=int, default=5)
    ap.add_argument("--max-a", type=int, default=16000)
    ap.add_argument("--cap", type=int, default=0, help="max rows per shard")
    ap.add_argument("--stacks", default="",
                    help="comma list, cth 'new' atau 'new,android' (lalai semua)")
    ap.add_argument("--shuffle-shards", action="store_true")
    ap.add_argument("--epochs", type=float, default=1.0,
                    help="epoch per shard (pengetahuan baru: 2-3)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16, help="LoRA r (32-64 utk knowledge)")
    ap.add_argument("--replay", type=float, default=0.0,
                    help="0.2 = campur 20%% row lama setiap shard (anti-lupa)")
    ap.add_argument("--replay-from", default="",
                    help="folder clean/ larian lama (comma) utk replay+dedupe")
    ap.add_argument("--eval-ratio", type=float, default=0.0,
                    help="0.01 = 1%% holdout per shard, log eval_loss")
    ap.add_argument("--eval-rows", type=int, default=100)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list + clean only")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--delete-after", action="store_true",
                    help="padam fail shard yang dimuat turun selepas diproses")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()

    if a.model.lower().endswith(".gguf"):
        raise SystemExit("GGUF ialah format inferens - tidak boleh dilatih. "
                         "Guna repo/folder safetensors (merged gen1).")
    validate_model(a.model)

    out = Path(a.out)
    raw_dir = out / "raw"
    clean_dir = out / "clean"
    state_path = out / "state.json"
    out.mkdir(parents=True, exist_ok=True)
    if a.reset:
        for p in (state_path,):
            if p.exists():
                p.unlink()
        log("state di-reset (checkpoint/clean kekal)")

    shards = list_shards(a.repo, a.branch, a.shred_dir)
    if a.stacks:
        want = {s.strip() for s in a.stacks.split(",") if s.strip()}
        shards = [p for p in shards
                  if p[len(SHRED_PREFIX):].split("/", 1)[0] in want]
        if not shards:
            raise SystemExit(f"tiada shard untuk stack {sorted(want)}")
    if a.shuffle_shards:
        random.Random(a.seed).shuffle(shards)
    log(f"{len(shards)} shard fail dari {a.repo}@{a.branch}")
    if a.list_only:
        for p in shards:
            print(p)
        return

    st = load_state(state_path, a.model)
    done = set(st["completed"])
    seen = rebuild_seen(clean_dir, st["completed"])
    limits = load_limits(a)
    log(f"resume: {len(done)}/{len(shards)} shard siap, {len(seen)} row unik")
    replay_pool = []
    if a.replay > 0:
        for p in st["completed"]:
            replay_pool.extend(read_jsonl(clean_dir / p))
    if a.replay_from:
        for d in a.replay_from.split(","):
            d = d.strip()
            if not d:
                continue
            for p in Path(d).rglob("*.jsonl"):
                replay_pool.extend(read_jsonl(p))
    if a.replay > 0 or a.replay_from:
        log(f"replay pool: {len(replay_pool)} row")

    model = tok = trainer_cls = bf16_ok = None
    if not a.dry_run:
        from unsloth import FastVisionModel, is_bfloat16_supported
        bf16_ok = is_bfloat16_supported()
        log(f"memuat base: {a.model}")
        model, tok = FastVisionModel.from_pretrained(
            a.model, max_seq_length=a.max_seq, load_in_4bit=False)
        if st.get("last_ckpt"):
            from peft import PeftModel
            log(f"resume adapter: {st['last_ckpt']}")
            model = PeftModel.from_pretrained(model, st["last_ckpt"],
                                              is_trainable=True)
        else:
            model = FastVisionModel.get_peft_model(
                model, finetune_vision_layers=False,
                finetune_language_layers=True, finetune_attention_modules=True,
                finetune_mlp_modules=True, r=a.rank, lora_alpha=a.rank,
                lora_dropout=0, bias="none", target_modules="all-linear",
                use_gradient_checkpointing="unsloth")
        from trl import SFTTrainer, SFTConfig
        from unsloth.chat_templates import train_on_responses_only
        trainer_cls = (SFTTrainer, SFTConfig, train_on_responses_only)

    for i, path in enumerate(shards, 1):
        if path in done:
            log(f"[{i}/{len(shards)}] skip {path}")
            if a.delete_after:
                drop_cached(path, raw_dir)
            continue
        text = fetch_shard(path, a.repo, a.branch, raw_dir, a.shred_dir)
        stack = path[len(SHRED_PREFIX):].split("/", 1)[0]
        rows, stats = clean_rows(text, seen, limits, stack)
        if a.cap and len(rows) > a.cap:
            rows = rows[:a.cap]

        new_rows = list(rows)
        eval_ds = None
        if a.eval_ratio > 0 and len(new_rows) > 1 and not a.dry_run:
            band = max(1, int(a.eval_ratio * 1000))

            def _is_eval(r):
                k = f"{r['instruction']}\n{r['output']}"
                return (int(hashlib.md5(k.encode("utf-8")).hexdigest(), 16)
                        % 1000) < band

            ev = [r for r in new_rows if _is_eval(r)][:a.eval_rows]
            if ev:
                new_rows = [r for r in new_rows if not _is_eval(r)]
                from datasets import Dataset
                eval_ds = Dataset.from_list(
                    [{"text": to_text(tok, r, a.thinking)} for r in ev])

        clean_path = clean_dir / path
        write_jsonl(clean_path, rows)
        log(f"[{i}/{len(shards)}] {path} rows={stats['rows']} keep={len(rows)} "
            f"bad={stats['bad']} bounds={stats['bounds']} scrub={stats['scrub']} "
            f"dupe={stats['dupe']} holdout={len(rows) - len(new_rows)}")
        if a.dry_run:
            if a.delete_after:
                drop_cached(path, raw_dir)
            continue
        if not rows:
            st["completed"].append(path)
            save_state(state_path, st)
            if a.delete_after:
                drop_cached(path, raw_dir)
            continue

        train_rows = list(new_rows)
        if a.replay > 0 and replay_pool and train_rows:
            k = int(len(train_rows) * a.replay)
            if k:
                train_rows += random.Random(a.seed + i).sample(
                    replay_pool, min(k, len(replay_pool)))
                random.Random(a.seed).shuffle(train_rows)
                log(f"replay: +{k} row lama")
        if not train_rows:
            st["completed"].append(path)
            save_state(state_path, st)
            if a.delete_after:
                drop_cached(path, raw_dir)
            continue

        from datasets import Dataset
        SFTTrainer, SFTConfig, train_on_responses_only = trainer_cls
        ds = Dataset.from_list([{"text": to_text(tok, r, a.thinking)}
                                for r in train_rows])
        ckpt = out / "ckpt" / f"shard-{i:03d}-{stack}"
        trainer = SFTTrainer(
            model=model, tokenizer=tok, train_dataset=ds, eval_dataset=eval_ds,
            dataset_text_field="text",
            args=SFTConfig(
                output_dir=str(out / "runs"),
                per_device_train_batch_size=a.batch,
                gradient_accumulation_steps=a.accum,
                num_train_epochs=a.epochs, learning_rate=a.lr,
                warmup_steps=5, lr_scheduler_type="linear",
                optim="adamw_8bit", weight_decay=0.01, logging_steps=10,
                save_strategy="no", seed=a.seed, packing=False,
                fp16=not bf16_ok, bf16=bf16_ok,
            ),
        )
        trainer = train_on_responses_only(
            trainer, instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n")
        trainer.train()
        if eval_ds is not None:
            try:
                m = trainer.evaluate()
                log(f"eval_loss={m.get('eval_loss', float('nan')):.4f} "
                    f"n={len(eval_ds)}")
            except Exception as e:  # noqa: BLE001
                log(f"eval gagal: {e}")
        model.save_pretrained(str(ckpt))
        tok.save_pretrained(str(ckpt))
        st["last_ckpt"] = str(ckpt)
        st["completed"].append(path)
        st["rows"] += len(new_rows)
        save_state(state_path, st)
        replay_pool.extend(new_rows)
        log(f"checkpoint -> {ckpt}")
        if a.delete_after:
            drop_cached(path, raw_dir)

    if not a.dry_run and not a.no_export:
        if not st.get("last_ckpt"):
            log("tiada checkpoint - langkau eksport")
        else:
            import gc
            import torch
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()
            from unsloth import FastVisionModel
            log(f"export: muat {st['last_ckpt']}")
            emodel, etok = FastVisionModel.from_pretrained(
                str(st["last_ckpt"]), max_seq_length=a.max_seq,
                load_in_4bit=False)
            emodel.save_pretrained_merged(str(out / "merged"), etok,
                                          save_method="merged_16bit")
            emodel.save_pretrained_gguf(str(out / "gguf"), etok,
                                        quantization_method="q4_k_m")
            log(f"exported {out / 'merged'} + {out / 'gguf'}")
    log("siap")


if __name__ == "__main__":
    main()
