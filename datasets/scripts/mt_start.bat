@echo off
cd /d "%~dp0\..\.."
powershell -NoProfile -Command "Start-Process -FilePath '.venv-gpu\Scripts\python.exe' -ArgumentList '-u','datasets\scripts\mt_runner.py' -WorkingDirectory (Get-Location).Path -WindowStyle Hidden"
echo Terjemahan dimulakan (jika belum berjalan).
echo Pantau dengan mt_status.bat
pause
