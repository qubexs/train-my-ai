"""hardware.py: physical core detection sane on any box."""
from hardware import auto_budget, cpu_count, physical_cores


def test_physical_cores_sane():
    assert isinstance(physical_cores(), int)
    assert 1 <= physical_cores() <= cpu_count()


def test_budget_uses_physical_threads():
    ram, toks, ctx, threads = auto_budget()
    assert threads == max(2, physical_cores())
    assert toks > 0 and ctx >= 1024
