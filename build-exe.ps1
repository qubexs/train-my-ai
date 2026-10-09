# Local Windows build: pyinstaller --onefile app/xcoder.py
# Cross-OS (Linux/macOS): use GitHub Actions (push a tag v* or Run workflow).
$ErrorActionPreference = "Stop"
py -m pip install -q pyinstaller
py -m PyInstaller --onefile --name xcoder --console app/xcoder.py --clean -y
Write-Output ""
Write-Output "Built: dist\xcoder.exe"
.\dist\xcoder.exe --help
Write-Output ""
Write-Output "For Linux/macOS exes: git tag v0.1.0; git push origin v0.1.0"
Write-Output "Then check Actions -> build-xcoder -> Artifacts / Releases."
