"""Runner terjemahan fable - memegang lock O_EXCL sepanjang hayat proses.

Menjamin SATU proses sahaja walaupun di-launch berulang kali. Log:
datasets/.fable_mt_run.log. Start: mt_start.bat atau:
  python datasets/scripts/mt_runner.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "datasets" / ".fable_mt.lock"
LOG = ROOT / "datasets" / ".fable_mt_run.log"


def pid_alive(pid):
    if not pid:
        return False
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FI", "IMAGENAME eq python.exe"],
            capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return True
    return "python.exe" in out.lower() and str(pid) in out


def acquire():
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii"))
            return fd
        except FileExistsError:
            try:
                pid = int(LOCK.read_text(encoding="utf-8").strip() or 0)
            except (ValueError, OSError):
                pid = 0
            if pid_alive(pid):
                print(f"sudah berjalan: PID {pid} - keluar")
                sys.exit(0)
            age = time.time() - LOCK.stat().st_mtime
            if age < 15:
                print(f"lock baru ({age:.0f}s) - mungkin sedang start, cuba lagi")
                sys.exit(0)
            print(f"lock basi (PID {pid}) - buang")
            LOCK.unlink(missing_ok=True)


def run():
    fd = acquire()
    log = open(LOG, "w", encoding="utf-8", buffering=1)
    sys.stdout = log
    sys.stderr = log
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "datasets" / "scripts"))
    print(f"[runner] PID {os.getpid()} mula {time.strftime('%Y-%m-%d %H:%M:%S')}")
    rc = 1
    try:
        import import_fable
        sys.argv = ["import_fable.py", "translate",
                    "--in", str(ROOT / "datasets" / "fable_agent.jsonl"),
                    "--out", str(ROOT / "datasets" / "fable_agent_ms.jsonl"),
                    "--server", "http://127.0.0.1:8080", "--model", "gemma",
                    "--workers", "2", "--translate-output", "--verbose",
                    "--cache", str(ROOT / "datasets" / ".fable_mt_cache.jsonl")]
        rc = import_fable.main()
    except SystemExit as e:
        rc = int(e.code or 0)
    finally:
        print(f"[runner] tamat rc={rc} {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log.close()
        os.close(fd)
        LOCK.unlink(missing_ok=True)
    return rc


if __name__ == "__main__":
    sys.exit(run())
