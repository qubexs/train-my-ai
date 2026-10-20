"""Runner training lokal - memegang lock O_EXCL sepanjang hayat proses.

Menjamin SATU proses training sahaja (selamat walau di-launch berulang).
Log: finetune/.train_run.log. Start: finetune\\train_start.bat atau:
  python finetune/train_runner.py
Terputus/PC off -> run semula, auto-resume dari checkpoint terakhir.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "finetune" / ".train_run.lock"
LOG = ROOT / "finetune" / ".train_run.log"


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
                print(f"training sudah berjalan: PID {pid} - keluar")
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
    print(f"[runner] training PID {os.getpid()} mula "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
    rc = 1
    try:
        import runpy
        sys.argv = ["train_local_qwen35.py", "--data", "all"]
        runpy.run_path(str(ROOT / "finetune" / "train_local_qwen35.py"),
                       run_name="__main__")
        rc = 0
    except SystemExit as e:
        rc = int(e.code or 0)
    except Exception:
        import traceback
        traceback.print_exc()
        rc = 1
    finally:
        print(f"[runner] tamat rc={rc} {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log.close()
        os.close(fd)
        LOCK.unlink(missing_ok=True)
    return rc


if __name__ == "__main__":
    sys.exit(run())
