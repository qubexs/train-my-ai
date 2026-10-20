# XCoder one-command installer (Windows).
# Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File install.ps1
# Options: -Minimal (base+general only, ~1GB) | -DryRun (show URLs, download nothing)
param([switch]$Minimal, [switch]$DryRun)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$HF = "https://huggingface.co/myxcoder/xcoder/resolve/main"

$Base = @("qwen2.5-0.5b-instruct-q4_k_m.gguf", "xcoder-general-0.5b-q4_k_m.gguf")
$Rest = @("xcoder-bash", "xcoder-css", "xcoder-docker", "xcoder-html",
  "xcoder-javascript", "xcoder-laravel", "xcoder-linux", "xcoder-mysql",
  "xcoder-nodejs", "xcoder-php", "xcoder-postgresql", "xcoder-python",
  "xcoder-sql", "xcoder-tailwind", "xcoder-typescript", "xcoder-web") |
  ForEach-Object { "$_-0.5b-q4_k_m.gguf" }
$Files = if ($Minimal) { $Base } else { $Base + $Rest }

$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command python -ErrorAction SilentlyContinue }
if (-not $py) {
  Write-Error "Python 3.9+ tiada. Pasang dari https://www.python.org/downloads/ dan jalan semula."
  exit 1
}
$ver = & $py.Source -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
if ([version]$ver -lt [version]"3.9") {
  Write-Error "Python $ver ditemui, perlu 3.9+. Kemas kini Python dan jalan semula."
  exit 1
}
Write-Output "Python $ver OK."

function Get-Model($name) {
  $dest = Join-Path $Root "models\$name"
  if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 300MB)) {
    $magic = [System.IO.File]::ReadAllBytes($dest)[0..3]
    if ([System.Text.Encoding]::ASCII.GetString($magic) -eq "GGUF") {
      Write-Output "  [sedia] $name"; return
    }
  }
  $url = "$HF/$name"
  if ($DryRun) { Write-Output "  [dryrun] $url"; return }
  Write-Output "  [muat] $name"
  New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
  try {
    Start-BitsTransfer -Source $url -Destination $dest -Description $name -ErrorAction Stop
  } catch {
    Invoke-WebRequest -Uri $url -OutFile $dest
  }
}

Write-Output "Model dari myxcoder/xcoder ($($Files.Count) fail)..."
$Files | ForEach-Object { Get-Model $_ }
if ($DryRun) { Write-Output "DryRun siap."; exit 0 }

Write-Output "Enjin llama.cpp (sekali sahaja)..."
& $py.Source "$Root\app\ezcodex.py" --setup

Write-Output ""
Write-Output "Siap. Jalan dengan:"
Write-Output "  .\ezcodex-cpu.bat --lang ms   (harian, CPU)"
Write-Output "  .\ezcodex-gpu.bat --lang ms   (GTX 1070, gemma pantas)"
Write-Output "Model pengurus gemma4-e4b (pilihan, 4.6GB): muat turun"
Write-Output "  https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF (gemma-4-E4B-it-Q4_K_M.gguf)"
Write-Output "ke models\, kemudian: /model add models\gemma-4-E4B-it-Q4_K_M.gguf --name gemma4-e4b --domains general"
