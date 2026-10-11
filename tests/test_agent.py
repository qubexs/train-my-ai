"""agent.py: extraction + loop + truncation. Fake backends, no models."""
from agent import extract_tool_call, extract_tool_calls, run_agent, strip_tool_blocks

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
