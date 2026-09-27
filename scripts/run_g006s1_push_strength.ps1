[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][ValidateSet('A1', 'A2', 'B')][string]$Phase,
  [Parameter(Mandatory = $true)][double[]]$Grid,
  [string]$IsaacLabRoot = (Join-Path $HOME 'IsaacLab'),
  [string]$LeaseScript = (Join-Path $HOME 'CodexSignals\Use-GpuLease.ps1'),
  [int]$VramLimitMiB = 9830,
  [int]$WaitTimeoutSeconds = 3600
)

# G006S1: evaluate existing G006 checkpoints at a stronger push grid through the GPU lease.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repo = Split-Path $PSScriptRoot -Parent
$contractPath = Join-Path $repo 'configs\g006s1_push_strength_sweep.json'
$contract = Get-Content $contractPath -Raw | ConvertFrom-Json
$queue = Get-Content (Join-Path $repo 'reports\runs\g006_queue_state.json') -Raw | ConvertFrom-Json
$python = Join-Path $IsaacLabRoot '_isaac_sim\python.bat'
$runLogPath = Join-Path $repo 'reports\runs\g006s1_run_log.json'
if ($Grid.Count -ne 3) { throw 'Grid requires three magnitudes' }

$gridJson = & python (Join-Path $repo 'scripts\g006s1_push_strength.py') grid --grid $Grid[0] $Grid[1] $Grid[2]
if ($LASTEXITCODE -ne 0) { throw 'grid manifest materialization failed' }
$gridManifest = Join-Path $repo (($gridJson | ConvertFrom-Json).grid_manifest)

$targets = if ($Phase -eq 'B') { $contract.checkpoints } else { @($contract.checkpoints | Where-Object { $_.id -eq $contract.calibration.checkpoint }) }

function Read-RunLog {
  if (Test-Path $runLogPath) { return @(Get-Content $runLogPath -Raw | ConvertFrom-Json) }
  return @()
}

function Get-GpuUsedMiB {
  $line = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1
  if ($line) { return [int]$line.Trim() }
  return $null
}

foreach ($target in $targets) {
  $job = $queue.jobs | Where-Object { $_.id -eq $target.queue_job }
  $checkpoint = $job.push_command[2].Replace('%USERPROFILE%', $HOME)
  $output = Join-Path $repo "reports\runs\g006s1_${Phase}_$($target.variant)_s$($target.training_seed)_push.json"
  if (Test-Path $output) {
    $existing = Get-Content $output -Raw | ConvertFrom-Json
    if ($existing.status -eq 'complete') { Write-Host "skip complete $output"; continue }
    throw "existing non-complete report blocks rerun: $output"
  }
  $baselineVram = Get-GpuUsedMiB
  foreach ($path in @($python, $repo, $checkpoint)) {
    if ($path -match '\s') { throw "paths passed through cmd.exe must not contain whitespace: $path" }
  }
  $arguments = @(
    '/d', '/s', '/c',
    "call $python $(Join-Path $repo 'scripts\evaluate_g006s1_push_strength.py') --sweep-contract $contractPath --phase $Phase --checkpoint $checkpoint --variant $($target.variant) --training-seed $($target.training_seed) --mode push --protocol $gridManifest --output $output --headless"
  )
  $sampler = Start-Job -ScriptBlock {
    while ($true) {
      $value = & nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1
      if ($value) { "$([DateTime]::UtcNow.ToString('o')),$value" }
      Start-Sleep -Seconds 2
    }
  }
  $started = [DateTime]::UtcNow
  Push-Location $IsaacLabRoot
  try {
    & $LeaseScript -Owner 'G006S1' -FilePath 'C:\Windows\System32\cmd.exe' -ArgumentList $arguments -WaitTimeoutSeconds $WaitTimeoutSeconds
    $exitCode = $LASTEXITCODE
  } finally {
    Pop-Location
    Stop-Job $sampler
    $samples = @(Receive-Job $sampler)
    Remove-Job $sampler
  }
  $finished = [DateTime]::UtcNow
  $parsed = @($samples | ForEach-Object { $parts = $_.Split(','); [pscustomobject]@{ mib = [int]$parts[1].Trim(); util = [int]$parts[2].Trim(); temp = [int]$parts[3].Trim() } })
  Start-Sleep -Seconds 5
  $afterVram = Get-GpuUsedMiB
  $report = if (Test-Path $output) { Get-Content $output -Raw | ConvertFrom-Json } else { $null }
  $peak = if ($parsed.Count) { ($parsed | Measure-Object mib -Maximum).Maximum } else { $null }
  $entry = [ordered]@{
    phase = $Phase
    checkpoint = $target.id
    grid_mps = $Grid
    started_utc = $started.ToString('o')
    finished_utc = $finished.ToString('o')
    wall_time_s = [math]::Round(($finished - $started).TotalSeconds, 3)
    lease_exit_code = $exitCode
    report_status = if ($report) { $report.status } else { 'missing' }
    report_sha256 = if (Test-Path $output) { (Get-FileHash $output -Algorithm SHA256).Hash.ToLower() } else { $null }
    gpu_samples = $parsed.Count
    vram_before_mib = $baselineVram
    peak_vram_mib = $peak
    peak_gpu_util_pct = if ($parsed.Count) { ($parsed | Measure-Object util -Maximum).Maximum } else { $null }
    peak_temp_c = if ($parsed.Count) { ($parsed | Measure-Object temp -Maximum).Maximum } else { $null }
    vram_after_exit_mib = $afterVram
    vram_limit_mib = $VramLimitMiB
    gpu_released = ($null -ne $afterVram -and $null -ne $baselineVram -and $afterVram -le ($baselineVram + 256))
  }
  $log = @(Read-RunLog) + @([pscustomobject]$entry)
  [IO.File]::WriteAllText($runLogPath, (ConvertTo-Json -InputObject $log -Depth 5) + "`n")
  Write-Host (ConvertTo-Json $entry -Compress)
  if ($exitCode -ne 0 -or -not $report -or $report.status -ne 'complete') { throw "stop: $($target.id) exit=$exitCode status=$($entry.report_status)" }
  if ($peak -and $peak -gt $VramLimitMiB) { throw "stop: peak VRAM $peak MiB exceeds $VramLimitMiB" }
  if (-not $entry.gpu_released) { throw "stop: GPU memory not released after $($target.id)" }
}
