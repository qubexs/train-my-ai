@echo off
cd /d "%~dp0\.."
powershell -NoProfile -Command "Start-Process -FilePath '.venv-gpu\Scripts\python.exe' -ArgumentList '-u','finetune\train_runner.py' -WorkingDirectory (Get-Location).Path -WindowStyle Hidden"
echo Training dimulakan (jika belum berjalan).
echo Pantau dengan train_status.bat
pause
