"""store.py: merge + deterministic eval split. Uses tmp dirs only."""
import json
from pathlib import Path

from store import merge_stacks


def _mkroot(tmp_path):
    cfg = tmp_path / "datasets" / "config"
    cfg.mkdir(parents=True)
    (cfg / "dataset.json").write_text(
        json.dumps({"stacks": {"demo": {"file": "demo.jsonl", "domain": "demo"}}}))
    rows = [{"instruction": f"soalan {i}", "input": "", "output": f"jawapan {i}"}
            for i in range(100)]
    (tmp_path / "datasets" / "demo.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return tmp_path


def _read(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines()]


def test_merge_eval_split_no_overlap(tmp_path):
    root = _mkroot(tmp_path)
    out, ev = merge_stacks(root=root, stack="demo", eval_frac=0.1)
    tr = _read(root / "finetune" / "stacks" / "demo" / "dataset.jsonl")
    evr = _read(root / "finetune" / "stacks" / "demo" / "eval.jsonl")
    assert out == {"demo": len(tr)} and ev == {"demo": len(evr)}
    assert 5 <= len(evr) <= 20  # ~10%
    tk = {(r["instruction"], r["output"]) for r in tr}
    ek = {(r["instruction"], r["output"]) for r in evr}
    assert not (tk & ek)
    assert len(tk | ek) == 100


def test_merge_eval_deterministic(tmp_path):
    root = _mkroot(tmp_path)
    _, ev1 = merge_stacks(root=root, stack="demo", eval_frac=0.1)
    e1 = _read(root / "finetune" / "stacks" / "demo" / "eval.jsonl")
    _, ev2 = merge_stacks(root=root, stack="demo", eval_frac=0.1)
    e2 = _read(root / "finetune" / "stacks" / "demo" / "eval.jsonl")
    assert ev1 == ev2 and e1 == e2


def test_merge_no_eval_by_default(tmp_path):
    root = _mkroot(tmp_path)
    out, ev = merge_stacks(root=root, stack="demo")
    assert out == {"demo": 100} and ev == {}
    assert not (root / "finetune" / "stacks" / "demo" / "eval.jsonl").exists()


def test_pretrain_corpus_includes_learned(tmp_path, monkeypatch):
    import store
    monkeypatch.setattr(store, "CHAT_DB", tmp_path / "chat.db")
    monkeypatch.setattr(store, "KNOW_DB", tmp_path / "knowledge.db")
    store.log_training("q", "a")
    learned = tmp_path / "docs" / "learned"
    learned.mkdir(parents=True)
    (learned / "note-1.md").write_text("fakta penting tentang docker",
                                       encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    dest = tmp_path / "corpus.txt"
    n, _where = store.export_pretrain_corpus(dest=str(dest))
    txt = dest.read_text(encoding="utf-8")
    assert n == 1
    assert "### User:" in txt and "fakta penting" in txt
