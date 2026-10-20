@echo off
cd /d "%~dp0\.."

rem Auto-detect interpreter: prefer GPU venv, fall back to gguf venv, then PATH.
set "PY="
for %%V in (.venv-gpu .venv-gguf) do (
  if not defined PY if exist "%%V\Scripts\python.exe" set "PY=%%V\Scripts\python.exe"
)
if not defined PY (
  for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"
)
if not defined PY (
  echo [train_start] tiada python dijumpai. Cipta .venv-gpu dahulu:
  echo   py -m venv .venv-gpu
  echo   .venv-gpu\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu118
  pause
  exit /b 1
)

echo Menggunakan interpreter: %PY%
powershell -NoProfile -Command "Start-Process -FilePath '%PY%' -ArgumentList '-u','finetune\train_runner.py' -WorkingDirectory (Get-Location).Path -WindowStyle Hidden"
echo Training dimulakan (jika belum berjalan).
echo Pantau dengan train_status.bat
pause
