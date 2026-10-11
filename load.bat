@echo off
REM load.bat [model] [cuda] - hidupkan llama-server resident di 127.0.0.1:8080
REM Lalai: models\xcoder-general-*.gguf (CPU). Guna bersama: ezcodex.bat --backend server
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set FLAVOR=cpu
if /i "%~1"=="cuda" set FLAVOR=cuda
if /i "%~2"=="cuda" set FLAVOR=cuda

set SERVER=bin\llama-server.exe
if "%FLAVOR%"=="cuda" set SERVER=bin-cuda\llama-server.exe
if not exist "%SERVER%" set SERVER=app\bin\llama-server.exe
if not exist "%SERVER%" (
  echo [load] llama-server tiada. Jalankan dahulu:
  echo   powershell -ExecutionPolicy Bypass -File app\install.ps1
  exit /b 1
)

set MODEL=%~1
if /i "%MODEL%"=="cuda" set MODEL=
if "!MODEL!"=="" (
  for %%F in (models\xcoder-general-*.gguf) do if "!MODEL!"=="" set MODEL=%%F
  for %%F in (models\xcoder-*.gguf) do if "!MODEL!"=="" set MODEL=%%F
  for %%F in (models\*.gguf) do if "!MODEL!"=="" set MODEL=%%F
) else (
  if not exist "%MODEL%" if exist "models\%MODEL%" set MODEL=models\%MODEL%
)
if "%MODEL%"=="" (
  echo [load] tiada model GGUF dalam models\ - latih/import dahulu.
  exit /b 1
)
if not exist "%MODEL%" (
  echo [load] model tiada: %MODEL%
  exit /b 1
)

echo [load] %MODEL% -^> http://127.0.0.1:8080 (%FLAVOR%)
"%SERVER%" -m "%MODEL%" -c 4096 --host 127.0.0.1 --port 8080
