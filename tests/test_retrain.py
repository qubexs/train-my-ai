"""Delta retrain state + build_delta_file (SQLite, pure python, no torch)."""
import json

import pytest

import store


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "CHAT_DB", tmp_path / "chat.db")
    monkeypatch.setattr(store, "KNOW_DB", tmp_path / "knowledge.db")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docs" / "learned").mkdir(parents=True)
    return tmp_path


def test_train_state_defaults_and_roundtrip(env):
    st = store.train_state()
    assert st["lines_consumed"] == 0 and st["delta_rounds"] == 0
    st["lines_consumed"] = 5
    store.save_train_state(st)
    assert store.train_state()["lines_consumed"] == 5


def test_pending_since_train_thresholds(env):
    for i in range(200):
        store.log_training(f"q{i}", f"a{i}")
    p = store.pending_since_train(min_rows=200, min_notes=50)
    assert p["rows"] == 200 and p["ready"] is True
    store.mark_trained()
    store.log_training("qnew", "anew")
    p = store.pending_since_train(min_rows=200, min_notes=50)
    assert p["rows"] == 1 and p["ready"] is False


def test_pending_notes_trigger(env):
    for i in range(50):
        store.log_learned(f"sha{i}", f"q{i}", f"t{i}", f"body{i}")
    p = store.pending_since_train(min_rows=10 ** 9, min_notes=50)
    assert p["notes"] == 50 and p["ready"] is True
    store.save_train_state({"learned_notes": 50})
    assert store.pending_since_train(min_rows=10 ** 9, min_notes=50)["ready"] is False


def test_mark_trained_consumes_all(env):
    store.log_training("q1", "a1")
    store.log_training("q2", "a2")
    store.log_learned("s1", "qq", "tt", "body")
    st = store.mark_trained()
    assert st["lines_consumed"] == 2 and st["learned_notes"] == 1
    assert st["delta_rounds"] == 1 and st["last_train_ts"] > 0
    assert store.pending_since_train()["rows"] == 0


def test_mark_failed_sets_ts(env):
    st = store.mark_failed()
    assert st["last_failed_ts"] > 0 and st["lines_consumed"] == 0


def test_build_delta_new_rows_only_and_replay(env):
    for i in range(8):
        store.log_training(f"q{i}", f"a{i}")
    store.mark_trained()
    for i in range(8, 10):
        store.log_training(f"q{i}", f"a{i}")
    (env / "datasets").mkdir()
    allrows = [{"instruction": "replay-old-1", "output": "xx"},
               {"instruction": "replay-old-2", "output": "yy"},
               {"instruction": "q3", "output": "a3"}]  # q3 must be deduped
    (env / "datasets" / "all.jsonl").write_text(
        "\n".join(json.dumps(r) for r in allrows) + "\n", encoding="utf-8")
    from store import build_delta_file
    dest, nd, nr, note = build_delta_file(dest=env / "build" / "d.jsonl",
                                          replay_pct=100, replay_cap=10)
    assert dest and nd == 2 and note == ""
    got = [json.loads(x) for x in dest.read_text(encoding="utf-8").splitlines()]
    qs = [g["instruction"] for g in got]
    assert qs[:2] == ["q8", "q9"]
    assert "q3" not in qs and "replay-old-1" in qs and "replay-old-2" in qs
    assert nr == 2


def test_build_delta_empty_when_consumed(env):
    store.log_training("q", "a")
    store.mark_trained()
    from store import build_delta_file
    dest, nd, nr, note = build_delta_file(dest=env / "build" / "d.jsonl")
    assert dest is None and nd == 0 and note


def test_build_delta_skips_bad_and_junk(env):
    store.log_training("q-good", "answer")
    store.log_training("q-bad", "a", rating="bad")
    store.log_training("q", "")  # short output -> filtered
    from store import build_delta_file
    dest, nd, nr, note = build_delta_file(dest=env / "build" / "d.jsonl")
    assert dest and nd == 1
    got = [json.loads(x) for x in dest.read_text(encoding="utf-8").splitlines()]
    assert [g["instruction"] for g in got] == ["q-good"]


def test_swap_pending_models(tmp_path):
    import experts
    (tmp_path / "m.gguf").write_bytes(b"GGUF" + b"\0" * 100)
    (tmp_path / "m.gguf.new").write_bytes(b"GGUF" + b"\0" * 200)
    swapped = experts.swap_pending_models(tmp_path)
    assert "m" in swapped
    assert not (tmp_path / "m.gguf.new").exists()
    assert (tmp_path / "m.gguf").stat().st_size == 204
    assert (tmp_path / "m.gguf.bak").stat().st_size == 104
