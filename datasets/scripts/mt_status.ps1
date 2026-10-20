param([switch]$Once)

$ErrorActionPreference = 'SilentlyContinue'
Set-Location (Join-Path $PSScriptRoot '..\..')

$cache = 'datasets\.fable_mt_cache.jsonl'
$log = 'datasets\.fable_mt_run.log'
$total = 2405

$prev = -1
$prevTime = Get-Date

if (-not $Once) {
    Write-Host 'Pantau terjemahan fable - ulang setiap 60s. Ctrl+C untuk keluar.' -ForegroundColor Yellow
}

while ($true) {
    if (-not $Once) { Clear-Host }

    $n = 0
    if (Test-Path $cache) {
        $n = (Get-Content $cache | Where-Object { $_.Trim() -ne '' } | Measure-Object).Count
    }
    $now = Get-Date

    $procs = Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'mt_runner|import_fable' }
    if ($procs) {
        $state = 'BERJALAN (PID ' + (($procs | Select-Object -ExpandProperty ProcessId) -join ', ') + ')'
    } else {
        $state = 'TIDAK BERJALAN - double-click mt_start.bat'
    }

    Write-Host ''
    Write-Host ('  Masa             : ' + $now.ToString('HH:mm:ss'))
    Write-Host ('  Terjemahan Fable : {0}/{1}  ({2}%)' -f $n, $total, [math]::Round($n / $total * 100, 1)) -ForegroundColor Cyan
    Write-Host ("  Status           : $state")

    if ($prev -ge 0 -and $n -ge $prev) {
        $mins = ($now - $prevTime).TotalMinutes
        if ($mins -gt 0) {
            $rate = ($n - $prev) / $mins
            Write-Host ('  Laju (sesi ini)  : ~{0} segmen/minit' -f [math]::Round($rate, 1))
            if ($rate -gt 0) {
                $left = ($total - $n) / $rate
                Write-Host ('  Anggaran siap    : ~{0} jam lagi (sekitar {1})' -f [math]::Round($left / 60, 1), $now.AddMinutes($left).ToString('HH:mm'))
            }
        }
    }

    Write-Host '  GPU              : ' -NoNewline
    nvidia-smi --query-gpu=power.draw,temperature.gpu,utilization.gpu --format=csv,noheader

    Write-Host '  Log terkini:' -ForegroundColor DarkGray
    Get-Content $log -Tail 2 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    Write-Host ''

    $prev = $n
    $prevTime = $now

    if ($Once) { break }
    Start-Sleep -Seconds 60
}
