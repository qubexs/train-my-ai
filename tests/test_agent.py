"""agent.py: extraction + loop + truncation. Fake backends, no models."""
import agent as agent_mod
from agent import (extract_tool_call, extract_tool_calls, run_agent, sanitize,
                   strip_tool_blocks, system_prompt)

B = "```"


def test_extract_multi():
    t = f"x {B}tool:read a.md{B} y {B}tool:list docs{B}"
    assert extract_tool_calls(t) == [("read", "a.md"), ("list", "docs")]
    assert extract_tool_call(t) == ("read", "a.md")
    assert extract_tool_call("no tools here") is None
    assert strip_tool_blocks(t).strip() == "x  y"


class FakeMulti:
    def __init__(self):
        self.n = 0

    def chat_ex(self, s, m, max_tokens=None):
        self.n += 1
        if self.n == 1:
            return (f"go {B}tool:run py print(6*7){B} and "
                    f"{B}tool:run js console.log(1+1){B}", 99, "stop")
        return "JAWAPAN AKHIR done", 99, "stop"


def test_run_agent_executes_all_calls_one_turn():
    got = []
    ans, tps, trace = run_agent(FakeMulti(), "kira", tools_on=True,
                                on_tool=lambda n, a, r: got.append((n, r.strip())))
    assert len(trace) == 2
    assert ("run", "42") in got and ("run", "2") in got
    assert tps == 99


class FakeCut:
    def __init__(self):
        self.n = 0

    def chat_ex(self, s, m, max_tokens=None):
        self.n += 1
        if self.n == 1:
            return "JAWAPAN AKHIR separuh", 50, "length"
        return "sambungan akhir", 50, "stop"


def test_run_agent_continues_on_length():
    be = FakeCut()
    ans, _, _ = run_agent(be, "cerita", tools_on=False)
    assert be.n == 2
    assert "separuh" in ans and "akhir" in ans


class LegacyBackend:
    """chat() without chat_ex still works (finish unknown)."""

    def chat(self, system, messages, max_tokens=None):
        return "JAWAPAN AKHIR ok", 10


def test_run_agent_legacy_backend():
    ans, tps, trace = run_agent(LegacyBackend(), "hai", tools_on=False)
    assert "ok" in ans and tps == 10 and trace == []


class CountingBackend:
    """Implements both chat and chat_ex: thinking pre-call must NOT fire."""

    def __init__(self):
        self.chat_calls = 0
        self.ex_calls = 0

    def chat(self, system, messages, max_tokens=None):
        self.chat_calls += 1
        return "fikiran lama", 1

    def chat_ex(self, system, messages, max_tokens=None):
        self.ex_calls += 1
        return "pengguna mahu jawapan ringkas\nJAWAPAN AKHIR siap", 20, "stop"


def test_no_thinking_precall():
    be = CountingBackend()
    seen = []
    ans, _, _ = run_agent(be, "hai", tools_on=False,
                          on_think=lambda t: seen.append(t))
    assert be.chat_calls == 0  # tiada panggilan thinking berasingan
    assert be.ex_calls == 1  # satu generasi sahaja
    assert seen and "ringkas" in seen[0]
    assert "siap" in ans


def test_system_prompt_online_vs_offline():
    ms_on = system_prompt("XCoder", "ms", online=True)
    ms_off = system_prompt("XCoder", "ms", online=False)
    assert "Sentiasa sebut sumber" in ms_on
    assert "Sentiasa sebut sumber" not in ms_off
    assert "offline tanpa internet" in ms_off
    en_on = system_prompt("XCoder", "en", online=True)
    assert "cite sources" in en_on.lower()
    en_off = system_prompt("XCoder", "en", tools_on=False, online=False)
    assert "fully offline" in en_off.lower()


def test_sanitize_strips_online_prompt_echo():
    assert sanitize("Anda boleh guna tool:search <soalan> untuk maklumat") == ""
    assert sanitize("You can use tool:search for up-to-date information") == ""


class FakeSearchBackend:
    def __init__(self):
        self.n = 0

    def chat_ex(self, s, m, max_tokens=None):
        self.n += 1
        if self.n == 1:
            return f"cari dulu {B}tool:search docker volume{B}", 10, "stop"
        return "JAWAPAN AKHIR guna named volume", 10, "stop"


def test_run_agent_online_autolearn(monkeypatch):
    import tools as tools_mod
    monkeypatch.setattr(tools_mod, "tool_search", lambda a, ctx=None:
                        "HASIL CARIAN: docker volume\n"
                        "1. Volumes - https://docs.docker.com/storage/volumes/\n"
                        "   Volumes menyimpan data.\n"
                        "SUMBER: https://docs.docker.com/storage/volumes/")
    saved = {}

    class P:
        name = "20260101-nota.md"

    def fake_learn(ctx, q, res, sources=None, lang="ms"):
        saved["q"] = q
        saved["res"] = res
        return P()
    monkeypatch.setattr(agent_mod, "maybe_learn", fake_learn)
    got = []
    ans, _, trace = run_agent(FakeSearchBackend(), "apa itu docker volume",
                              tools_on=True, online=True,
                              on_tool=lambda n, a, r: got.append(n))
    assert saved.get("q") == "docker volume"
    assert any(t["tool"] == "learn" for t in trace)
    assert "learn" in got
    assert "named volume" in ans


def test_run_agent_offline_no_autolearn(monkeypatch):
    import tools as tools_mod
    monkeypatch.setattr(tools_mod, "tool_search", lambda a, ctx=None:
                        "HASIL CARIAN: x\nSUMBER: https://x.example/")
    called = []
    monkeypatch.setattr(agent_mod, "maybe_learn",
                        lambda *a, **k: called.append(1))
    run_agent(FakeSearchBackend(), "soalan", tools_on=True, online=False)
    assert called == []
