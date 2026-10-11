@echo off
REM ezcodex-cpu.bat - XCoder mod CPU (server resident CPU, inferens jimat kuasa)
REM Guna: ezcodex-cpu.bat [--lang ms] [arahan lain...]
setlocal
cd /d "%~dp0"
set EXE=
for %%F in (ezcodex-*.exe) do set EXE=%%F
if not "%EXE%"=="" (
  "%EXE%" --backend auto %*
) else (
  py "%~dp0app\ezcodex.py" --backend auto %*
)
