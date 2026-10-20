@echo off
title Repair Fable MT (auto-refresh 60s)
set DIR=C:\Users\tokki\AppData\Local\Temp\opencode\fable-5k
:loop
cls
echo === repair fable MT - %date% %time% ===
echo.
echo === progres terakhir (repair_log.txt) ===
powershell -NoProfile -Command "Get-Content '%DIR%\repair_log.txt' -Encoding UTF8 | Select-Object -Last 8"
echo.
echo === unit siap dalam cache ===
powershell -NoProfile -Command "(Get-Content '%DIR%\repair_cache.jsonl' -ErrorAction SilentlyContinue | Measure-Object -Line).Lines"
echo.
echo === proses py berjalan ===
powershell -NoProfile -Command "(Get-Process py -ErrorAction SilentlyContinue | Measure-Object).Count"
echo.
echo === GPU ===
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo.
powershell -NoProfile -Command "if ((Get-Process py -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0) { exit 9 }"
if errorlevel 9 (
  echo === REPAIR SELESAI - proses py tiada lagi ===
  pause
  exit /b
)
echo (Ctrl+C untuk keluar - refresh setiap 60 saat)
ping -n 61 127.0.0.1 >nul
goto loop
