"""Fullscreen opencode-style TUI — stdlib only, Windows + POSIX.

Viewport (messages) + input box + status bar, alternate screen, no deps.
The existing line REPL runs UNCHANGED: sys.stdin becomes the input box,
sys.stdout feeds the viewport. Falls back to classic REPL when not a TTY.
"""
import os
import re
import shutil
import sys
import textwrap

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s):
    return ANSI_RE.sub("", s)


def _enable_windows_ansi():
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        h = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
            kernel32.SetConsoleMode(h, mode.value | 0x0004)
    except Exception:
        pass


class Screen:
    """Alternate screen + hidden cursor. Writes go to the REAL stdout."""

    def __enter__(self):
        self.real = sys.__stdout__
        self.write("\x1b[?1049h\x1b[?25l")
        return self

    def __exit__(self, *exc):
        self.write("\x1b[?25h\x1b[?1049l")
        try:
            self.real.flush()
        except Exception:
            pass

    def write(self, s):
        try:
            self.real.write(s)
            self.real.flush()
        except Exception:
            pass


class Viewport:
    def __init__(self):
        self.lines = []
        self.scroll = 0  # lines up from bottom
        self._partial = ""

    def add(self, text):
        data = self._partial + text
        parts = data.split("\n")
        self._partial = parts.pop()
        self.lines.extend(parts)
        if len(self.lines) > 2000:
            del self.lines[:len(self.lines) - 2000]
        self.scroll = 0

    def flush_partial(self):
        if self._partial:
            self.lines.append(self._partial)
            self._partial = ""
            self.scroll = 0

    def page(self, width, height):
        buf = self.lines + ([self._partial] if self._partial else [])
        wrapped = []
        for ln in buf:
            plain = _strip_ansi(ln)
            if not plain.strip():
                wrapped.append(ln)
                continue
            parts = textwrap.wrap(plain, width=width) or [""]
            if len(parts) == 1:
                wrapped.append(ln)
            else:
                # keep ANSI on first visual line only; rest plain
                wrapped.append(ln)
                wrapped.extend(parts[1:])
        if self.scroll:
            end = max(0, len(wrapped) - self.scroll)
        else:
            end = len(wrapped)
        start = max(0, end - height)
        return wrapped[start:end]


class InputBox:
    def __init__(self):
        self.buf = []
        self.pos = 0
        self.history = []
        self.hidx = None

    def text(self):
        return "".join(self.buf)

    def set_text(self, s):
        self.buf = list(s)
        self.pos = len(self.buf)

    def handle(self, key):
        """Returns 'submit' or None. Mutates buffer/history."""
        if key == "enter":
            return "submit"
        if key == "backspace":
            if self.pos > 0:
                del self.buf[self.pos - 1]
                self.pos -= 1
        elif key == "delete":
            if self.pos < len(self.buf):
                del self.buf[self.pos]
        elif key == "left":
            self.pos = max(0, self.pos - 1)
        elif key == "right":
            self.pos = min(len(self.buf), self.pos + 1)
        elif key == "home":
            self.pos = 0
        elif key == "end":
            self.pos = len(self.buf)
        elif key == "up":
            if self.history:
                if self.hidx is None:
                    self.hidx = len(self.history) - 1
                    self._saved = self.text()
                elif self.hidx > 0:
                    self.hidx -= 1
                self.set_text(self.history[self.hidx])
        elif key == "down":
            if self.history and self.hidx is not None:
                if self.hidx < len(self.history) - 1:
                    self.hidx += 1
                    self.set_text(self.history[self.hidx])
                else:
                    self.hidx = None
                    self.set_text(getattr(self, "_saved", ""))
        elif key == "clearline":
            self.buf = []
            self.pos = 0
        elif key == "killline":
            del self.buf[self.pos:]
        elif key == "delword":
            while self.pos > 0 and self.buf[self.pos - 1] == " ":
                del self.buf[self.pos - 1]
                self.pos -= 1
            while self.pos > 0 and self.buf[self.pos - 1] != " ":
                del self.buf[self.pos - 1]
                self.pos -= 1
        elif isinstance(key, str) and len(key) == 1 and key.isprintable():
            self.buf.insert(self.pos, key)
            self.pos += 1
        return None

    def commit(self):
        t = self.text()
        if t.strip():
            self.history.append(t)
            if len(self.history) > 200:
                del self.history[0]
        self.buf = []
        self.pos = 0
        self.hidx = None
        return t


class WindowsReader:
    def __init__(self):
        import msvcrt
        self.msvcrt = msvcrt

    def read_key(self):
        ch = self.msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            ch2 = self.msvcrt.getwch()
            return {"H": "up", "P": "down", "K": "left", "M": "right",
                    "G": "home", "O": "end", "I": "pgup", "Q": "pgdn",
                    "S": "delete"}.get(ch2, "")
        return {"\r": "enter", "\n": "enter", "\x08": "backspace",
                "\x7f": "backspace", "\x1b": "esc", "\x03": "ctrlc",
                "\x04": "eof", "\x01": "home", "\x05": "end",
                "\x0b": "killline", "\x15": "clearline",
                "\x17": "delword"}.get(ch, ch)


class PosixReader:
    def __init__(self):
        import termios
        import tty
        self.termios = termios
        self.tty = tty
        self.fd = sys.stdin.fileno()

    def __enter__(self):
        self.old = self.termios.tcgetattr(self.fd)
        self.tty.setraw(self.fd)
        return self

    def __exit__(self, *exc):
        try:
            self.termios.tcsetattr(self.fd, self.termios.TCSADRAIN, self.old)
        except Exception:
            pass

    def _read_char(self):
        b = os.read(self.fd, 1)
        n = len(b)
        lead = b[0]
        if lead < 0x80:
            need = 0
        elif (lead & 0xE0) == 0xC0:
            need = 1
        elif (lead & 0xF0) == 0xE0:
            need = 2
        else:
            need = 3
        while n <= need:
            more = os.read(self.fd, need - n + 1)
            if not more:
                break
            b += more
            n = len(b)
        try:
            return b.decode("utf-8")
        except Exception:
            return ""

    def read_key(self):
        ch = self._read_char()
        if ch == "\x1b":
            seq = self._read_char()
            if seq != "[":
                return "esc"
            code = self._read_char()
            if code == "A":
                return "up"
            if code == "B":
                return "down"
            if code == "C":
                return "right"
            if code == "D":
                return "left"
            if code == "H":
                return "home"
            if code == "F":
                return "end"
            if code in "12345678":
                rest = self._read_char()
                while rest not in ("~", "") and len(rest) < 4:
                    if rest == "~":
                        break
                    rest = self._read_char()
                return {"1": "home", "3": "delete", "4": "end",
                        "5": "pgup", "6": "pgdn", "7": "home",
                        "8": "end"}.get(code, "")
            return ""
        return {"\r": "enter", "\n": "enter", "\x7f": "backspace",
                "\x08": "backspace", "\x03": "ctrlc", "\x04": "eof",
                "\x01": "home", "\x05": "end", "\x0b": "killline",
                "\x15": "clearline", "\x17": "delword"}.get(ch, ch)


class UI:
    def __init__(self, screen, title, status):
        self.screen = screen
        self.title = title
        self.status = status
        self.view = Viewport()
        self.box = InputBox()
        self.label = ""

    def size(self):
        s = shutil.get_terminal_size((80, 24))
        return max(20, s.columns), max(10, s.lines)

    def draw(self):
        w, h = self.size()
        msg_h = max(1, h - 6)
        out = []
        out.append(("\x1b[7m" + self.title.ljust(w) + "\x1b[0m")[:w + 9])
        for ln in self.view.page(w, msg_h):
            out.append(ln[:w] if len(_strip_ansi(ln)) <= w else ln)
        while len(out) < 1 + msg_h:
            out.append("")
        # input box (2 lines: label + text with cursor)
        label = (self.label or "anda> ")[:w]
        text = self.box.text()
        cx = len(self.label or "anda> ") + self.box.pos
        # horizontal scroll for long lines
        if cx >= w:
            cut = cx - w + 1
            shown = (self.label or "anda> ") + text
            shown = shown[cut:cut + w]
            cx = w - 1
        else:
            shown = (self.label or "anda> ") + text
        out.append(label)
        out.append(shown + " " * max(0, w - len(_strip_ansi(shown))))
        out.append(("\x1b[2m" + self.status.ljust(w) + "\x1b[0m")[:w + 9])
        self.screen.write("\x1b[H" + "\n".join(out) + f"\x1b[{h};{cx + 1}H")
        try:
            self.screen.write("\x1b[?25h")
        except Exception:
            pass

    def read_line(self, label=""):
        self.label = label
        reader = WindowsReader() if os.name == "nt" else PosixReader()
        if os.name != "nt":
            reader.__enter__()
        try:
            while True:
                self.draw()
                try:
                    key = reader.read_key()
                except Exception:
                    continue
                if key == "eof":
                    if not self.box.text():
                        raise EOFError
                    continue
                if key == "ctrlc":
                    self.box.set_text("")
                    self.draw()
                    continue
                if key == "esc":
                    continue
                if key == "pgup":
                    self.view.scroll += 10
                    continue
                if key == "pgdn":
                    self.view.scroll = max(0, self.view.scroll - 10)
                    continue
                if self.box.handle(key) == "submit":
                    line = self.box.commit()
                    self.label = ""
                    self.view.add(f"anda> {line}")
                    self.draw()
                    return line
        finally:
            if os.name != "nt":
                reader.__exit__()


class TuiStdin:
    def __init__(self, ui):
        self.ui = ui

    def readline(self):
        return self.ui.read_line() + "\n"

    def isatty(self):
        return False


class TuiStdout:
    def __init__(self, ui):
        self.ui = ui

    def write(self, s):
        if s:
            self.ui.view.add(s)
            self.ui.draw()

    def flush(self):
        self.ui.draw()

    def isatty(self):
        return False


def run_tui(repl_fn, title, status):
    """Run repl_fn() inside the fullscreen TUI. Returns when repl exits."""
    _enable_windows_ansi()
    with Screen() as screen:
        ui = UI(screen, title, status)
        old_in, old_out = sys.stdin, sys.stdout
        sys.stdin = TuiStdin(ui)
        sys.stdout = TuiStdout(ui)
        try:
            ui.draw()
            repl_fn()
        finally:
            sys.stdin, sys.stdout = old_in, old_out
