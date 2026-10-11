@echo off
REM ezcodex-gpu.bat - XCoder mod GPU (server resident + offload penuh ke GTX 1070)
REM Guna: ezcodex-gpu.bat [--lang ms] [arahan lain...]
REM Disyor untuk gemma4-e4b (pengurus): ~36 T/s vs ~10 T/s di CPU.
setlocal
cd /d "%~dp0"
set EXE=
for %%F in (ezcodex-*.exe) do set EXE=%%F
if not "%EXE%"=="" (
  "%EXE%" --backend server --gpu cuda %*
) else (
  py "%~dp0app\ezcodex.py" --backend server --gpu cuda %*
)
