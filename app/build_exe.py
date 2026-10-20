"""Build single-file executable for current OS (win/linux/macos). Stdlib-only app -> small exe.
  pip install pyinstaller
  python app/build_exe.py
Output: dist/ezcodex-<os>-<arch>[.exe]
For all OS: push a tag and let .github/workflows/build-exe.yml build the matrix.
"""
import platform
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

def main():
    try:
        import PyInstaller  # noqa
    except ImportError:
        sys.exit("pip install pyinstaller, then re-run")
    s = platform.system().lower()
    arch = platform.machine().lower()
    name = f"ezcodex-{s}-{arch}"
    cmd = [sys.executable, "-m", "PyInstaller", "--onefile", "--console",
           "--name", name, str(APP_DIR / "ezcodex.py")]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"OK: dist/{name}{'.exe' if s == 'windows' else ''}")

if __name__ == "__main__":
    main()
