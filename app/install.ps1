# One-command installer for XCoder standalone app (Windows).
# Run from the working folder:
#   powershell -ExecutionPolicy Bypass -File app/install.ps1
$ErrorActionPreference = "Stop"
$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $AppDir

$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command python -ErrorAction SilentlyContinue }
if (-not $py) {
    Write-Error "Python 3.9+ not found. Install from https://www.python.org/downloads/ and re-run."
    exit 1
}
$ver = & $py.Source -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
if ([version]$ver -lt [version]"3.9") {
    Write-Error "Python $ver found, need 3.9+. Update Python and re-run."
    exit 1
}
Write-Output "Python $ver OK, fetching engine + model (once only)..."
& $py.Source "$AppDir\xcoder.py" --setup

$bat = Join-Path $Root "xcoder.bat"
Set-Content -LiteralPath $bat -Value "@echo off`r`npy `"%~dp0app\xcoder.py`" %*`r`n" -Encoding Ascii
Write-Output ""
Write-Output "Done. Run the AI with:"
Write-Output "  xcoder.bat"
Write-Output "  or: py app\xcoder.py --lang ms"
