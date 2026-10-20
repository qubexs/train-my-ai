$ErrorActionPreference = 'SilentlyContinue'
Set-Location (Join-Path $PSScriptRoot '..')

$run = 'finetune\.train_run.log'
$outdir = 'finetune\runs\xcoder-best-0.8b'

while ($true) {
    Clear-Host
    $now = Get-Date

    $procs = Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'train_runner|train_local' }
    if ($procs) {
        $state = 'BERJALAN (PID ' + (($procs | Select-Object -ExpandProperty ProcessId) -join ', ') + ')'
    } else {
        $state = 'TIDAK BERJALAN - double-click train_start.bat untuk sambung'
    }

    Write-Host ''
    Write-Host ('  Masa    : ' + $now.ToString('HH:mm:ss'))
    Write-Host ("  Training: $state") -ForegroundColor Cyan

    $cks = Get-ChildItem "$outdir\checkpoint-*" -Directory | Sort-Object { [int]($_.Name -split '-')[-1] }
    if ($cks) {
        $last = $cks[-1]
        $ts = Get-Content (Join-Path $last.FullName 'trainer_state.json') -Raw | ConvertFrom-Json
        $pct = 0
        if ($ts.max_steps -gt 0) { $pct = [math]::Round($ts.global_step / $ts.max_steps * 100, 1) }
        Write-Host ('  Langkah : checkpoint-{0} / {1}  ({2}%)  epoch {3}' -f $ts.global_step, $ts.max_steps, $pct, [math]::Round($ts.epoch, 2))
        Write-Host ('  Ckpt    : {0} set (keep 2)' -f $cks.Count)
    } else {
        Write-Host '  Langkah : belum ada checkpoint (tunggu step 100)'
    }

    $loss = Get-Content $run | Select-String 'loss' | Select-Object -Last 1
    if ($loss) { Write-Host ("  Loss    : " + $loss.Line.Trim()) -ForegroundColor DarkGray }

    Write-Host '  GPU     : ' -NoNewline
    nvidia-smi --query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader

    Write-Host '  Log terkini:' -ForegroundColor DarkGray
    Get-Content $run -Tail 2 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    Write-Host ''

    Start-Sleep -Seconds 60
}
