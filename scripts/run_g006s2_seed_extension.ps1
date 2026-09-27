[CmdletBinding()]
param(
  [string]$IsaacLabRoot = (Join-Path $HOME 'IsaacLab'),
  [string]$LeaseScript = (Join-Path $HOME 'CodexSignals\Use-GpuLease.ps1'),
  [int]$TrainingLeaseWaitSeconds = 21600,
  [int]$EvaluationLeaseWaitSeconds = 3600
)

# G006S2: train seeds 45/46 for both G006 variants, gate each run against G006, then evaluate at the
# G006S1 push grid. Resumable: completed training (gate PASS) and complete evaluations are skipped.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repo = Split-Path $PSScriptRoot -Parent
$contractPath = Join-Path $repo 'configs\g006s2_seed_extension.json'
$contract = Get-Content $contractPath -Raw | ConvertFrom-Json
$tool = Join-Path $repo 'scripts\g006s2_seed_extension.py'
$python = Join-Path $IsaacLabRoot '_isaac_sim\python.bat'
$pwsh = (Get-Process -Id $PID).Path
$runLogPath = Join-Path $repo 'reports\runs\g006s2_run_log.json'

function Add-RunLog($entry) {
  $log = if (Test-Path $runLogPath) { @(Get-Content $runLogPath -Raw | ConvertFrom-Json) } else { @() }
  $log = @($log) + @([pscustomobject]$entry)
  [IO.File]::WriteAllText($runLogPath, (ConvertTo-Json -InputObject $log -Depth 6) + "`n")
  Write-Host ("[{0}] {1}" -f [DateTime]::UtcNow.ToString('o'), (ConvertTo-Json $entry -Compress -Depth 6))
}

function Invoke-Gate([string]$variant, [int]$seed) {
  $json = & python $tool gate --variant $variant --seed $seed
  return ($json | ConvertFrom-Json)
}

$gridJson = & python (Join-Path $repo 'scripts\g006s1_push_strength.py') grid --contract configs/g006s2_seed_extension.json --grid 2.0 2.5 3.0
if ($LASTEXITCODE -ne 0) { throw 'grid manifest materialization failed' }
$gridManifest = Join-Path $repo (($gridJson | ConvertFrom-Json).grid_manifest)

foreach ($id in $contract.training.order) {
  $variant, $seedText = $id -split '-s'
  $seed = [int]$seedText
  $entry = $contract.training_reports | Where-Object { $_.variant -eq $variant -and $_.seed -eq $seed }
  $trainingReport = Join-Path $repo ($entry.path -replace '/', '\')
  $runName = $contract.training.run_name.Replace('{variant}', $variant).Replace('{seed}', "$seed")
  $evalReport = Join-Path $repo "reports\runs\g006s2_S2_${variant}_s${seed}_push.json"

  if (Test-Path $evalReport) {
    $existing = Get-Content $evalReport -Raw | ConvertFrom-Json
    if ($existing.status -eq 'complete') { Write-Host "skip complete $id"; continue }
    throw "existing non-complete evaluation blocks rerun: $evalReport"
  }

  if (Test-Path $trainingReport) {
    $gate = Invoke-Gate $variant $seed
    if (-not $gate.passed) { throw "existing training report fails gate: $id $(ConvertTo-Json $gate.checks -Compress)" }
    Write-Host "reuse gated training $id"
  } else {
    $bundleBefore = (& python $tool bundle | ConvertFrom-Json)
    if (-not $bundleBefore.matches) { throw "training source bundle differs from G006 before $id" }
    $started = [DateTime]::UtcNow
    $arguments = @(
      '-NoProfile', '-File', (Join-Path $repo 'scripts\run_training.ps1'),
      '-Task', $contract.training.tasks.$variant,
      '-NumEnvs', "$($contract.training.num_envs)",
      '-MaxIterations', "$($contract.training.max_iterations)",
      '-Seed', "$seed",
      '-RunName', $runName,
      '-IsaacLabPath', $IsaacLabRoot,
      '-ReportPath', $trainingReport,
      '-TrainingEntrypointPath', (Join-Path $repo 'scripts\bootstrap_train_g006.py')
    )
    Push-Location $IsaacLabRoot
    try {
      & $LeaseScript -Owner 'G006S2' -FilePath $pwsh -ArgumentList $arguments -WaitTimeoutSeconds $TrainingLeaseWaitSeconds
      $exitCode = $LASTEXITCODE
    } finally { Pop-Location }
    $bundleAfter = (& python $tool bundle | ConvertFrom-Json)
    $gate = if (Test-Path $trainingReport) { Invoke-Gate $variant $seed } else { $null }
    Add-RunLog ([ordered]@{
      step = 'train'; id = $id; run_name = $runName
      started_utc = $started.ToString('o'); finished_utc = [DateTime]::UtcNow.ToString('o')
      lease_exit_code = $exitCode
      bundle_before_matches = $bundleBefore.matches; bundle_after_matches = $bundleAfter.matches
      gate = $gate
    })
    if (-not $gate -or -not $gate.passed -or -not $bundleAfter.matches) { throw "stop: training gate failed for $id (exit=$exitCode)" }
  }

  $checkpoint = (Get-Content $trainingReport -Raw | ConvertFrom-Json).artifacts.checkpoint.Replace('%USERPROFILE%', $HOME)
  foreach ($path in @($python, $repo, $checkpoint)) {
    if ($path -match '\s') { throw "paths passed through cmd.exe must not contain whitespace: $path" }
  }
  $evalArguments = @(
    '/d', '/s', '/c',
    "call $python $(Join-Path $repo 'scripts\evaluate_g006s1_push_strength.py') --sweep-contract $contractPath --phase S2 --checkpoint $checkpoint --variant $variant --training-seed $seed --mode push --protocol $gridManifest --output $evalReport --headless"
  )
  $started = [DateTime]::UtcNow
  Push-Location $IsaacLabRoot
  try {
    & $LeaseScript -Owner 'G006S2' -FilePath 'C:\Windows\System32\cmd.exe' -ArgumentList $evalArguments -WaitTimeoutSeconds $EvaluationLeaseWaitSeconds
    $exitCode = $LASTEXITCODE
  } finally { Pop-Location }
  $report = if (Test-Path $evalReport) { Get-Content $evalReport -Raw | ConvertFrom-Json } else { $null }
  Add-RunLog ([ordered]@{
    step = 'evaluate'; id = $id
    started_utc = $started.ToString('o'); finished_utc = [DateTime]::UtcNow.ToString('o')
    lease_exit_code = $exitCode
    report_status = if ($report) { $report.status } else { 'missing' }
    report_sha256 = if (Test-Path $evalReport) { (Get-FileHash $evalReport -Algorithm SHA256).Hash.ToLower() } else { $null }
    recovery_rate = if ($report -and $report.status -eq 'complete') { $report.aggregate.recovery_rate } else { $null }
  })
  if (-not $report -or $report.status -ne 'complete') { throw "stop: evaluation failed for $id (exit=$exitCode)" }
}

& python $tool summarize
if ($LASTEXITCODE -ne 0) { throw 'summary failed' }
Write-Host 'G006S2 queue complete'
