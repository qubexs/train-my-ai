# datasets/ — curated training data per stack (XCoder)

One file per stack, Alpaca JSONL per line:

```json
{"instruction": "tulis Dockerfile Node minima", "input": "", "output": "FROM node:20-alpine ..."}
```

Rules: short + correct answers, identity rows say `XCoder` (never Qwen/Alibaba).
Target: ~5000 rows per stack (python, sql, bash already there; see counts below).

## Workflow

```
chat + /good            # CLI auto-tags domain+stack -> training.jsonl
train --mode sft        # split rows into datasets/<stack>.jsonl (dedupe)
python datasets/scripts/generate.py [stack] [--dry-run]  # template drills (idempotent append)
python datasets/scripts/merge.py [stack] [--eval 0.05]  # combine -> finetune/stacks/<stack>/dataset.jsonl (+eval.jsonl)
python datasets/scripts/validate.py         # check format, all stacks (also eval.jsonl)
python datasets/scripts/audit.py --report   # deep audit: scrub/bounds/dup/retag (+ --fix -> _quarantine)
python datasets/scripts/buckets.py          # coverage vs 10M plan buckets (config/buckets.json)
# bulk imports (P2): HF datasets -> datasets/<stack>.jsonl + provenance ledger
$env:HF_HOME='E:\pr\trainmyai\datasets\raw\hf-cache'
.\.venv-gpu\Scripts\python.exe datasets/scripts/import_hf.py --source all --cap python=1000000
#   sources: oci, commitpackft (--langs ...), taco, gretel, magicoder-oss, magicoder-evol,
#            github-code (jsonl bulk, fetch via curl: --ghc-workers 3 --ghc-shards 400
#                         --ghc-start 400 --ghc-max-chars 10000; rows content-routed:
#                         flask/django/api, nodejs, spring, <sql-engine>, mongodb, redis),
#            sql-create-context, gretel-sql, sqale, apigen, glaive-fc, xlam
#   flags: --dry-run, --limit N, --cap stack=N (new-adds cap), --no-oci-tests
#   license filter: whitelist only (mit/apache-2.0/bsd-2/3-clause/isc/cc0/unlicense/cc-by-4.0)
#   ledger: datasets/manifests/ledger.jsonl (source/license/hash per row; git-ignored)
python datasets/scripts/trim_other.py [--dry]  # cap 'other' via ledger keys -> _quarantine
python datasets/scripts/eval_suite.py models/xcoder-docker-0.5b-q4_k_m.gguf  # prompt suite vs GGUF
# GPU: python finetune/eval_loss.py --model <merged> --eval finetune/stacks/docker/eval.jsonl
# Colab: upload finetune/stacks/<stack>/dataset.jsonl -> LoRA -> GGUF -> /model add
```

`finetune/stacks/*/dataset.jsonl` + `eval.jsonl` are **generated, git-ignored** —
`merge.py` regenerates them byte-deterministically (`train_all.py` falls back to
`datasets/` if missing). Never commit data duplicates.

## SOP: web reference per dataset

Every stack expansion starts from ONE authoritative web reference
(fetched structure, never copied text):

| Dataset | Reference |
|---|---|
| typescript | `typescriptlang.org/docs/handbook` |
| html / css / javascript | `developer.mozilla.org` (MDN) |
| tailwind | `tailwindcss.com/docs` |
| php | `php.net/manual` |
| laravel | `laravel.com/docs` |
| mysql | `dev.mysql.com/doc` |
| postgresql | `postgresql.org/docs` |
| linux | `man7.org` / `linuxjourney.com` |
| nodejs | `nodejs.org/docs` |
| docker | `docs.docker.com` |
| sql | `geeksforgeeks.org/sql/sql-tutorial` ✅ done (6512 rows) |
| general | no single source — existing drills + repo docs |

Depth limit: index page + max 2–4 key sub-pages. Write short original
paraphrase in repo style (Malay/English mix); standard syntax examples are fine,
prose is always rewritten.

If fetching fails (403/timeout/redirect loop/truncation), in order:
1. retry as `text`, then follow the tool-provided redirect URL;
2. switch to a backup source (official docs ↔ MDN ↔ learnxinyminutes);
3. write from built-in knowledge + mark the topic `perlu semak` in a code comment.

## 10M plan (buckets + audit)

The 10M target is split into 12 buckets (`datasets/config/buckets.json`),
mapping every config stack to exactly one bucket. `buckets.py` reports
target/actual/coverage per bucket. Six new stacks form the `engineering`
domain: `algorithms`, `debugging`, `testing`, `docs`, `refactoring`, `api`.

Wave 2/3/4 imports (github-code shards 0-1125 all consumed, OCI 3 passes,
gretel-sql, sqale, sql-create-context, apigen, glaive-fc) leave the plan at
9,666,497 rows vs adjusted target 9,604,485: `git-tooling` was re-scoped to
4,485 (github-code filters boilerplate; no free git source at volume) and the
`other` bucket was trimmed to exactly 500,000 by `trim_other.py` (provenance-
matched ghc rows -> quarantine). Remaining gaps: `apis-backend` -19,700,
`linux-devops` -8,114. Audit reports: `datasets/audit/report_wave2.json`,
`report_final.json` — run audit after imports with `--skip-near` (far faster;
near-dup is report-only anyway).

`audit.py` is the acceptance gate for existing + imported data:
structural, char bounds, identity scrub (Qwen/Alibaba/Tongyi — denials OK),
exact dup -> `--fix` moves rows to `_quarantine.jsonl` (never deleted);
near-dup (simhash64) and retag-mismatch are report-only (phrasing variants
of one answer are intentional). Report: `datasets/audit/report.json`,
review samples: `datasets/audit/samples/<stack>.txt` (`--samples N`).

**One-line invariant:** every JSONL row must be one physical line. All
writers use `_common.dump_row()` which escapes U+2028/U+2029/U+0085 —
readers (app/store.py, finetune/*) use `splitlines()`, which would split
raw ones. Imports get licenses/provenance in `datasets/manifests/ledger.jsonl`
(source, license, hash, bucket) instead of inline fields (`merge_one.py`
strips extras).

## Mistag quarantine

Rows that land in the wrong stack (usually `/good` mis-tags, e.g. iptables
rows in `sql.jsonl`) are MOVED — never deleted — to
`datasets/_quarantine.jsonl`. That file is outside `config/dataset.json`,
so `validate.py`, `merge.py` and `merge_all.py` ignore it. Re-tag later
via `distill.py` or by hand.

## Generalist corpus (xcoder-best)

`python datasets/scripts/merge_all.py [--eval 0.05]` combines all config
stacks into one generalist set for the single-0.5B-generalist track:

- `finetune/stacks/all/dataset.jsonl` (train, ~33k rows)
- `finetune/stacks/all/eval.jsonl` (deterministic md5 held-out — never leaks
  into train, stable across re-runs)

Global dedupe on (instruction, output); deterministic shuffle (seed 77);
50+ `XCoder` identity rows required in train (eval keeps a few to measure
identity retention). Qwen/Alibaba mentions allowed ONLY as denials
("Bukan. Saya XCoder…").

`python datasets/scripts/merge_one.py` fuses ALL config stacks into one
`datasets/all.jsonl` (global dedupe, seed-77 shuffle, refuses to overwrite
without `--force`) for the Qwen3.5-0.8B generalist track
(`finetune/train_unsloth.py --data all`). If `datasets/fable_agent_ms.jsonl`
exists (output of `datasets/scripts/colab_mt.ipynb`), its rows are folded in
automatically. The training script does its own 1% md5 holdout, so no eval
split is written here. Generated (git-ignored); `dedupe.py` skips it by design.

## Scaling to 100k per stack

Three tiers, in order:

1. **Hand seed (done, ~500 rows)** — highest quality, sets voice and facts.
2. **Template drills** — `python datasets/scripts/generate.py [stack]` appends
   correct-by-construction rows (command flags, compose fields, parametric SQL).
   Re-runnable, deduped. Reaches low thousands per stack.
3. **LLM distillation** — `python datasets/scripts/distill.py --stack X --n 5000`
   against a teacher (hosted API for speed, or local `--backend server` for free).
   Topics from `datasets/seeds/<stack>.txt`. Always `validate.py` + spot-check
   before merging — distilled rows carry the teacher's mistakes too.
   Note: raw reasoning teachers (Qwen3.5) think instead of answering —
   fine-tune the teacher first, then distill from it.

Notes at scale: keep answers short (long tails waste LoRA capacity), dedupe
before every train (`merge.py` dedupes; re-running `generate.py` is idempotent),
and split files past ~50MB (`part-*.jsonl` still validate). Quality beats
quantity — 2k clean rows outperform 100k noisy ones on a 0.5B model.

`finetune/stacks/` files are **generated** by `merge.py` — edit data here in
`datasets/`, never in `finetune/` directly. Stack -> domain mapping lives in
`config/dataset.json`. New rows must use the `XCoder` identity.
