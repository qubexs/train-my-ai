# datasets/ — curated training data per stack (XCoder)

One file per stack, Alpaca JSONL per line:

```json
{"instruction": "tulis Dockerfile Node minima", "input": "", "output": "FROM node:20-alpine ..."}
```

Rules: short + correct answers, identity rows say `XCoder` (never Qwen/Alibaba).

## Workflow

```
chat + /good            # CLI auto-tags domain+stack -> training.jsonl
train --mode sft        # split rows into datasets/<stack>.jsonl (dedupe)
python datasets/scripts/merge.py [domain]   # combine stacks -> finetune/<domain>/dataset.jsonl
python datasets/scripts/validate.py         # check format, all stacks
python datasets/scripts/statistics.py       # counts + averages
# Colab: upload finetune/<domain>/dataset.jsonl -> LoRA -> GGUF -> /model add
```

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

Notes at scale: keep answers short (long tails waste LoRA capacity), dedupe
before every train (`merge.py` dedupes; re-running `generate.py` is idempotent),
and split files past ~50MB (`part-*.jsonl` still validate). Quality beats
quantity — 2k clean rows outperform 100k noisy ones on a 0.5B model.

`finetune/<domain>/` files are **generated** by `merge.py` — edit data here in
`datasets/`, never in `finetune/` directly. Stack -> domain mapping lives in
`config/dataset.json`. New rows must use the `XCoder` identity.
