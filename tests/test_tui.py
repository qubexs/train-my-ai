"""tui.py: keymaps + TAB plumbing. No live terminal needed."""
import inspect

from tui import CHAR_KEYMAP, InputBox, UI, run_tui


def test_char_keymap_tab_eof():
    assert CHAR_KEYMAP["\t"] == "tab"
    assert CHAR_KEYMAP["\x04"] == "eof"
    assert CHAR_KEYMAP["\x03"] == "ctrlc"


def test_inputbox_ignores_tab():
    box = InputBox()
    assert box.handle("tab") is None
    assert box.text() == ""


def test_ui_on_tab_plumbing():
    seen = []
    ui = UI(None, "t", "s", on_tab=lambda: seen.append(1) or "[tools on]")
    assert ui.on_tab is not None
    assert ui.on_tab() == "[tools on]" and seen == [1]
    ui2 = UI(None, "t", "s")
    assert ui2.on_tab is None


def test_run_tui_accepts_on_tab():
    assert "on_tab" in inspect.signature(run_tui).parameters
