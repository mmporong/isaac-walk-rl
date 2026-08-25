[CmdletBinding(DefaultParameterSetName='Production')]
param(
    [string]$ConfigPath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'configs\g006_rough_push.json'),
    [string]$StatePath,
    [string]$ScaleStatePath,
    [string]$ReportRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'reports\runs'),
    [string]$IsaacLabPath = "$HOME\IsaacLab",
    [string]$TrainingHarness = (Join-Path $PSScriptRoot 'run_training.ps1'),
    [string]$TrainingEntrypointPath = (Join-Path $PSScriptRoot 'bootstrap_train_g006.py'),
    [string]$EvaluationScript = (Join-Path $PSScriptRoot 'evaluate_push_recovery.py'),
    [string]$SummaryScript = (Join-Path $PSScriptRoot 'summarize_g006.py'),
    [switch]$Resume,
    [switch]$Smoke,
    [switch]$CurriculumSmoke,
    [switch]$ScaleLadderOnly,
    [ValidateRange(1,65536)][int]$SelectedNumEnvs
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = Split-Path -Parent $PSScriptRoot
$script:labRoot = $null

function Full([string]$p) {
  [IO.Path]::GetFullPath($p)
}

function IsWithin([string]$path, [string]$root) {
  $fullPath = Full $path
  $fullRoot = (Full $root).TrimEnd(
    [IO.Path]::DirectorySeparatorChar,
    [IO.Path]::AltDirectorySeparatorChar
  )
  return (
    $fullPath -eq $fullRoot -or
    $fullPath.StartsWith($fullRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)
  )
}

function Portable([string]$p) {
  $full = Full $p
  $homePath = (Full $HOME).TrimEnd('\')
  if (IsWithin $full $homePath) {
    return '%USERPROFILE%' + $full.Substring($homePath.Length)
  }
  $repositoryPath = (Full $repoRoot).TrimEnd('\')
  if (IsWithin $full $repositoryPath) {
    return '%REPO_ROOT%' + $full.Substring($repositoryPath.Length)
  }
  if (-not [string]::IsNullOrWhiteSpace($script:labRoot)) {
    $isaacLabPath = (Full $script:labRoot).TrimEnd('\')
    if (IsWithin $full $isaacLabPath) {
      return '%ISAACLAB_ROOT%' + $full.Substring($isaacLabPath.Length)
    }
  }
  throw 'portable_path_outside_allowed_roots'
}

function ResolvePortable([string]$p) {
  if ($p.Contains('%')) {
    $match = [regex]::Match(
      $p,
      '^(?<token>%USERPROFILE%|%REPO_ROOT%|%ISAACLAB_ROOT%)(?<suffix>(?:[\\/].*)?)$',
      [Text.RegularExpressions.RegexOptions]::IgnoreCase
    )
    if (-not $match.Success) { throw 'portable_token_invalid' }
    $token = $match.Groups['token'].Value.ToUpperInvariant()
    $selectedRoot = switch ($token) {
      '%USERPROFILE%' { $HOME }
      '%REPO_ROOT%' { $repoRoot }
      '%ISAACLAB_ROOT%' {
        if ([string]::IsNullOrWhiteSpace($script:labRoot)) {
          throw 'isaaclab_root_not_initialized'
        }
        $script:labRoot
      }
      default { throw 'portable_token_invalid' }
    }
    $suffix = $match.Groups['suffix'].Value.TrimStart('\', '/')
    $resolved = if ([string]::IsNullOrEmpty($suffix)) {
      Full $selectedRoot
    }
    else {
      Full (Join-Path $selectedRoot $suffix)
    }
    if (-not (IsWithin $resolved $selectedRoot)) { throw 'portable_token_escape' }
    return $resolved
  }

  if (-not [IO.Path]::IsPathFullyQualified($p)) {
    $resolved = Full (Join-Path $repoRoot $p)
    if (IsWithin $resolved $repoRoot) {
      return $resolved
    }
    throw 'resolved_path_outside_allowed_roots'
  }

  $resolved = Full $p
  foreach ($root in @($HOME, $repoRoot, $script:labRoot)) {
    if (-not [string]::IsNullOrWhiteSpace([string]$root) -and (IsWithin $resolved ([string]$root))) {
      return $resolved
    }
  }
  throw 'resolved_path_outside_allowed_roots'
}

function HashFile([string]$p) {
  (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()
}

function HashText([string]$s) {
  $bytes = [Text.Encoding]::UTF8.GetBytes($s)
  [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant()
}

function GetSourceBundle([string[]]$relativePaths) {
  $ordered = @($relativePaths | ForEach-Object { $_.Replace('\', '/') } | Sort-Object -Unique)
  $algorithm = [Security.Cryptography.HashAlgorithmName]::SHA256
  $hash = [Security.Cryptography.IncrementalHash]::CreateHash($algorithm)
  $files = @()
  try {
    foreach ($relative in $ordered) {
      $full = Full (Join-Path $repoRoot $relative)
      $boundary = (Full $repoRoot).TrimEnd('\') + '\'
      if (
        -not $full.StartsWith($boundary, [StringComparison]::OrdinalIgnoreCase) -or
        -not (Test-Path -LiteralPath $full -PathType Leaf)
      ) {
        throw "source_bundle_missing_or_outside:$relative"
      }
      $bytes = [IO.File]::ReadAllBytes($full)
      $pathBytes = [Text.Encoding]::UTF8.GetBytes($relative)
      $hash.AppendData($pathBytes)
      $hash.AppendData([byte[]]@(0))
      $hash.AppendData($bytes)
      $hash.AppendData([byte[]]@(0))
      $files += , [ordered]@{
        path = $relative
        sha256 = [Convert]::ToHexString(
          [Security.Cryptography.SHA256]::HashData($bytes)
        ).ToLowerInvariant()
      }
    }
    $digest = [Convert]::ToHexString($hash.GetHashAndReset()).ToLowerInvariant()
  }
  finally {
    $hash.Dispose()
  }
  return [ordered]@{ sha256 = $digest; files = $files }
}

function AssertOutputPath([string]$path) {
  $full = ResolvePortable $path
  $boundary = $reportRootFull.TrimEnd('\') + '\'
  if (-not $full.StartsWith($boundary, [StringComparison]::OrdinalIgnoreCase)) {
    throw "output_outside_report_root:$path"
  }
  return $full
}

function SameJson($left, $right) {
  return (($left | ConvertTo-Json -Depth 20 -Compress) -eq ($right | ConvertTo-Json -Depth 20 -Compress))
}

function Prop($object, [string]$name) {
  $property = $object.PSObject.Properties[$name]
  if ($null -eq $property) {
    return $null
  }
  return $property.Value
}

function GpuUsed() {
  try {
    $value = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null |
      Select-Object -First 1
    if ($LASTEXITCODE -eq 0 -and $value -match '^\s*\d+\s*$') {
      return [int]$value
    }
  }
  catch {}
  return $null
}

function QuoteArg([string]$s) {
  if ($s -notmatch '[\s"]') {
    return $s
  }
  return '"' + ($s -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}

function WriteJson($value, [string]$path) {
  $tempPath = $path + '.' + [guid]::NewGuid().ToString('N') + '.tmp'
  [IO.File]::WriteAllText(
    $tempPath,
    ($value | ConvertTo-Json -Depth 30),
    [Text.UTF8Encoding]::new($false)
  )
  [IO.File]::Move($tempPath, $path, $true)
}

function ReadJson([string]$p) {
  if (-not (Test-Path -LiteralPath $p -PathType Leaf)) {
    return $null
  }
  try {
    Get-Content -LiteralPath $p -Raw | ConvertFrom-Json
  }
  catch {
    throw "corrupt_json:$p"
  }
}

function AddFailure($state, $job, [string]$phase, [string]$reason, [Nullable[int]]$exit) {
  $fingerprint = HashText("$phase`n$reason")
  $job.attempts = @($job.attempts) + @([ordered]@{
    at = (Get-Date).ToString('o')
    phase = $phase
    reason = $reason
    exit_code = $exit
    failure_fingerprint = $fingerprint
  })
  $sameFailures = @($job.attempts | Where-Object failure_fingerprint -eq $fingerprint).Count
  $job.status = 'failed'
  $job.hard_blocked = ($sameFailures -ge 3)
  $state.status = 'failed'
  $state.updated_at = (Get-Date).ToString('o')
  WriteJson $state $script:statePath
  return $fingerprint
}

function AddStateFailure($state, [string]$phase, [string]$reason, [Nullable[int]]$exit) {
  $state.failures = @($state.failures) + @([ordered]@{
    at = (Get-Date).ToString('o')
    phase = $phase
    reason = $reason
    exit_code = $exit
    failure_fingerprint = HashText("$phase`n$reason")
  })
  $state.status = 'failed'
  $state.updated_at = (Get-Date).ToString('o')
  WriteJson $state $script:statePath
}

function BindTrainingReport($report, [string]$path) {
  $report | Add-Member -NotePropertyName training_source_bundle_sha256 `
    -NotePropertyValue $script:trainingBundle.sha256 -Force
  $report | Add-Member -NotePropertyName training_source_bundle_files `
    -NotePropertyValue @($script:trainingBundle.files) -Force
  $report | Add-Member -NotePropertyName isaaclab_commit `
    -NotePropertyValue $script:isaacLabCommit -Force
  $report | Add-Member -NotePropertyName agent_entry_point `
    -NotePropertyValue ([string]$cfg.training.agent_learning_config.entry_point) -Force
  $report | Add-Member -NotePropertyName normalized_env_diff_allowlist `
    -NotePropertyValue @('events.push_robot') -Force
  WriteJson $report (AssertOutputPath $path)
}
function ValidateTraining($job, $report) {
  if ($null -eq $report) { return 'missing_json' }
  if ($report.passed -ne $true) { return 'report_not_passed' }
  if (
    $report.task -ne $job.task -or
    [int]$report.seed -ne [int]$job.seed -or
    [int]$report.num_envs -ne [int]$job.num_envs -or
    [int]$report.max_iterations -ne [int]$job.max_iterations
  ) { return 'identity_mismatch' }
  if ($report.training_entrypoint.sha256 -ne $script:entryHash) {
    return 'entrypoint_hash_mismatch'
  }
  if (
    $report.training_source_bundle_sha256 -ne $script:trainingBundle.sha256 -or
    $job.training_source_bundle_sha256 -ne $script:trainingBundle.sha256 -or
    -not (SameJson (@($report.training_source_bundle_files)) (@($script:trainingBundle.files)))
  ) { return 'training_source_bundle_mismatch' }
  if (
    $report.isaaclab_commit -ne $script:isaacLabCommit -or
    $report.agent_entry_point -ne $cfg.training.agent_learning_config.entry_point
  ) { return 'runtime_provenance_mismatch' }

  $checkpoint = ResolvePortable ([string]$report.artifacts.checkpoint)
  if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) { return 'missing_checkpoint' }
  if ((HashFile $checkpoint) -ne $report.artifacts.checkpoint_sha256) {
    return 'checkpoint_hash_mismatch'
  }
  foreach ($name in @('raw_stdout', 'raw_stderr')) {
    $path = ResolvePortable ([string]$report.artifacts.$name)
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return "missing_$name" }
  }
  $tensorboard = ResolvePortable ([string]$report.artifacts.tensorboard_directory)
  if (-not (Test-Path -LiteralPath $tensorboard -PathType Container)) {
    return 'missing_tensorboard_directory'
  }
  $trainingGates = @(
    'gpu_measurement_complete', 'gpu_recovered_to_baseline',
    'tensorboard_exists', 'checkpoint_exists'
  )
  foreach ($gate in $trainingGates) {
    if ($report.success_checks.$gate -ne $true) { return "training_$gate`_failed" }
  }
  if (@($report.fatal_patterns_found).Count -ne 0) { return 'training_fatal_patterns' }
  return 'passed'
}

function ValidateLogger($report, [string]$variant, [string]$validationMode) {
  if ($null -eq $report.tensorboard) { return 'missing_tensorboard_scalars' }
  $prefix = 'Curriculum/g006_state/'
  $expected = @($cfg.logger.terrain_keys) + @($cfg.logger.push_keys)
  $tags = @($report.tensorboard.tags)
  foreach ($key in $expected) {
    if ($tags -notcontains ($prefix + $key)) { return "missing_logger_key:$key" }
  }
  $latest = $report.tensorboard.latest
  if ($variant -eq 'baseline') {
    if ([int]$latest.($prefix + 'stage') -ne -1) { return 'baseline_stage_not_minus_one' }
    foreach ($index in 0..2) {
      if (
        [int]$latest.($prefix + "events_stage_$index") -ne 0 -or
        [double]$latest.($prefix + "magnitude_mean_stage_$index") -ne 0 -or
        [double]$latest.($prefix + "magnitude_min_stage_$index") -ne 0 -or
        [double]$latest.($prefix + "magnitude_max_stage_$index") -ne 0
      ) { return 'baseline_push_evidence_nonzero' }
    }
    return 'passed'
  }

  $requiredStages = if ($validationMode -eq 'production') {
    @(0, 1, 2)
  }
  elseif ($validationMode -eq 'curriculum_smoke') {
    @(0)
  }
  else {
    @()
  }
  foreach ($index in $requiredStages) {
    $count = [int]$latest.($prefix + "events_stage_$index")
    $minimum = [double]$latest.($prefix + "magnitude_min_stage_$index")
    $mean = [double]$latest.($prefix + "magnitude_mean_stage_$index")
    $maximum = [double]$latest.($prefix + "magnitude_max_stage_$index")
    $contract = $cfg.push_training.stages[$index].magnitude_mps
    if ($count -le 0) { return "push_stage_${index}_count_zero" }
    if (
      $minimum -lt [double]$contract[0] -or
      $maximum -gt [double]$contract[1] -or
      $minimum -gt $mean -or
      $mean -gt $maximum
    ) { return "push_stage_${index}_magnitude_out_of_range" }
  }
  if ($validationMode -eq 'production' -and [int]$latest.($prefix + 'stage') -ne 2) {
    return 'production_push_stage_not_two'
  }
  return 'passed'
}

function AssertStateSpecs($state, $specs) {
  if (@($state.jobs).Count -ne @($specs).Count) { throw 'resume_job_count_mismatch' }
  $ids = @($state.jobs.id)
  if (@($ids | Sort-Object -Unique).Count -ne $ids.Count) { throw 'resume_duplicate_job_id' }
  foreach ($spec in $specs) {
    $jobs = @($state.jobs | Where-Object id -eq $spec.id)
    if ($jobs.Count -ne 1) { throw "resume_job_identity_missing_or_duplicate:$($spec.id)" }
    $job = $jobs[0]
    $fields = @(
      'id', 'variant', 'task', 'seed', 'num_envs', 'max_iterations',
      'run_name', 'report_path', 'push_report_path', 'guardrail_report_path'
    )
    foreach ($field in $fields) {
      if ([string]$job.$field -ne [string]$spec.$field) {
        throw "resume_job_spec_mismatch:$($spec.id):$field"
      }
    }
    foreach ($pathField in @('report_path', 'push_report_path', 'guardrail_report_path')) {
      AssertOutputPath ([string]$job.$pathField) | Out-Null
    }
    if (
      $job.training_source_bundle_sha256 -ne $script:trainingBundle.sha256 -or
      $job.evaluation_source_bundle_sha256 -ne $script:evaluationBundle.sha256
    ) { throw "resume_job_source_bundle_mismatch:$($spec.id)" }
  }
}

function EvalBinding($job, $training, [string]$mode, [string]$output) {
  $executionArguments = @(
    $script:evaluator, '--checkpoint', (ResolvePortable ([string]$training.artifacts.checkpoint)),
    '--variant', $job.variant, '--training-seed', "$($job.seed)", '--mode', $mode,
    '--protocol', $script:config, '--output', (ResolvePortable $output), '--headless'
  )
  $durableArguments = @(
    (Portable $script:evaluator), '--checkpoint', (Portable (ResolvePortable ([string]$training.artifacts.checkpoint))),
    '--variant', $job.variant, '--training-seed', "$($job.seed)", '--mode', $mode,
    '--protocol', (Portable $script:config), '--output', (Portable (ResolvePortable $output)), '--headless'
  )
  [ordered]@{
    execution_args = $executionArguments
    durable_args = $durableArguments
    sha256 = HashText ($durableArguments -join "`n")
    script_sha256 = HashFile $script:evaluator
    source_bundle_sha256 = $script:evaluationBundle.sha256
  }
}

function ValidateEval($job, $training, [string]$mode, [string]$path) {
  $path = AssertOutputPath $path
  $report = ReadJson $path
  if ($null -eq $report) { return 'missing_json' }
  if (
    $report.goal -ne 'G006' -or $report.status -ne 'complete' -or
    $report.mode -ne $mode -or $report.variant -ne $job.variant -or
    [int]$report.training_seed -ne [int]$job.seed
  ) { return 'identity_or_status_mismatch' }
  if ($report.protocol_compliant -ne $true -or $report.experimental_use -ne 'g006_production_evaluation') {
    return 'not_production_protocol_evidence'
  }
  if ($report.protocol.sha256 -ne $script:protocolHash) { return 'protocol_hash_mismatch' }
  if (
    $report.evaluation_source_bundle_sha256 -ne $script:evaluationBundle.sha256 -or
    -not (SameJson (@($report.evaluation_source_bundle_files)) (@($script:evaluationBundle.files)))
  ) { return 'evaluation_source_bundle_mismatch' }
  if (
    $report.checkpoint.sha256 -ne $training.artifacts.checkpoint_sha256 -or
    (HashFile (ResolvePortable ([string]$report.checkpoint.path))) -ne $report.checkpoint.sha256
  ) { return 'checkpoint_binding_mismatch' }
  foreach ($key in @('runtime', 'terrain_evidence', 'trials', 'cells', 'aggregate', 'warnings')) {
    if ($null -eq $report.$key) { return "missing_$key" }
  }
  $expectedTrials = if ($mode -eq 'push') { 1080 } else { 90 }
  $expectedCells = if ($mode -eq 'push') { 108 } else { 9 }
  if (
    @($report.trials).Count -ne $expectedTrials -or
    @($report.cells).Count -ne $expectedCells -or
    [int]$report.aggregate.trial_count -ne $expectedTrials
  ) { return 'fixed_trial_cell_denominator_mismatch' }
  if (
    @($report.trials.trial_id | Sort-Object -Unique).Count -ne $expectedTrials -or
    @($report.cells | Where-Object { [int]$_.trial_count -ne 10 }).Count -ne 0
  ) { return 'duplicate_trials_or_cell_denominator_mismatch' }
  if (
    [int]$report.aggregate.boundary_violation_count -ne 0 -or
    [int]$report.aggregate.auto_reset_excluded_count -ne 0
  ) { return 'boundary_or_auto_reset_violation' }
  foreach ($gate in @(
    'app_close_completed', 'finalized_after_process_exit', 'gpu_measurement_complete',
    'gpu_recovered_to_baseline', 'process_recovered'
  )) {
    if ($report.runtime.$gate -ne $true) { return "runtime_$gate`_failed" }
  }
  if (
    $report.runtime.preliminary -ne $false -or
    $report.runtime.exit_code -ne 0 -or
    @($report.runtime.fatal_patterns).Count -ne 0
  ) { return 'runtime_exit_or_fatal_failed' }
  return 'passed'
}

function ValidateCompleteJob($job) {
  $trainingPath = AssertOutputPath ([string]$job.report_path)
  $training = ReadJson $trainingPath
  $valid = ValidateTraining $job $training
  if ($valid -ne 'passed') { return "training:$valid" }
  if ((HashFile $trainingPath) -ne (Prop $job 'training_report_sha256')) {
    return 'training:report_file_hash_mismatch'
  }
  $loggerMode = if ($mode -eq 'production') {
    'production'
  }
  elseif ($mode -eq 'curriculum_smoke') {
    'curriculum_smoke'
  }
  else {
    'smoke'
  }
  if ($mode -ne 'scale') {
    $logger = ValidateLogger $training $job.variant $loggerMode
    if ($logger -ne 'passed') { return "logger:$logger" }
  }
  if ($mode -ne 'production') { return 'passed' }

  $portableMigrations = @()
  foreach ($evalMode in @('push', 'guardrail')) {
    $path = [string](Prop $job "${evalMode}_report_path")
    $binding = EvalBinding $job $training $evalMode $path
    if ((Prop $job "${evalMode}_script_sha256") -ne $binding.script_sha256) {
      return "${evalMode}:script_hash_mismatch"
    }
    if ($job.evaluation_source_bundle_sha256 -ne $binding.source_bundle_sha256) {
      return "${evalMode}:bundle_hash_mismatch"
    }
    $storedCommand = @((Prop $job "${evalMode}_command"))
    $storedHash = Prop $job "${evalMode}_command_sha256"
    $canonicalCommand = SameJson $storedCommand @($binding.durable_args)
    $canonicalHash = ($storedHash -eq $binding.sha256)
    $legacyCommand = SameJson $storedCommand @($binding.execution_args)
    $legacyHash = ($storedHash -eq (HashText (@($binding.execution_args) -join "`n")))
    if (-not (($canonicalCommand -and $canonicalHash) -or ($legacyCommand -and $legacyHash))) {
      return "${evalMode}:command_or_hash_mismatch"
    }
    $evalValid = ValidateEval $job $training $evalMode $path
    if ($evalValid -ne 'passed') { return "${evalMode}:$evalValid" }
    if ((HashFile (AssertOutputPath $path)) -ne (Prop $job "${evalMode}_report_sha256")) {
      return "${evalMode}:report_file_hash_mismatch"
    }
    if ($legacyCommand -and $legacyHash) {
      $portableMigrations += , [ordered]@{ mode = $evalMode; binding = $binding }
    }
  }
  foreach ($migration in $portableMigrations) {
    $evalMode = [string]$migration.mode
    $job."${evalMode}_command" = @($migration.binding.durable_args)
    $job."${evalMode}_command_sha256" = $migration.binding.sha256
  }
  if ($portableMigrations.Count -gt 0) { return 'passed_migrated' }
  return 'passed'
}
function InvokeEvaluator($binding, [string]$output, [string]$runName, [string]$mode) {
  $output = ResolvePortable $output
  $stdout = Join-Path $reportRootFull "$runName`_$mode.stdout.log"
  $stderr = Join-Path $reportRootFull "$runName`_$mode.stderr.log"
  $baseline = GpuUsed
  $peak = $baseline
  $line = (@($binding.execution_args) | ForEach-Object { QuoteArg ([string]$_) }) -join ' '
  $process = Start-Process `
    -FilePath $python -ArgumentList $line -WorkingDirectory $lab `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
    -WindowStyle Hidden -PassThru
  while (-not $process.HasExited) {
    $used = GpuUsed
    if ($null -ne $used -and ($null -eq $peak -or $used -gt $peak)) {
      $peak = $used
    }
    Start-Sleep -Seconds 2
    $process.Refresh()
  }
  $process.WaitForExit()
  $exit = $process.ExitCode
  $after = $null
  $deadline = (Get-Date).AddSeconds(30)
  do {
    $after = GpuUsed
    if ($null -ne $after -and $null -ne $baseline -and $after -le $baseline + 128) { break }
    Start-Sleep -Seconds 1
  } while ((Get-Date) -lt $deadline)

  $stdoutText = ''
  if (Test-Path $stdout) { $stdoutText = [string](Get-Content $stdout -Raw) }
  $stderrText = ''
  if (Test-Path $stderr) { $stderrText = [string](Get-Content $stderr -Raw) }
  [string]$text = $stdoutText + $stderrText
  $fatal = @('Traceback (most recent call last)', '[Error]') |
    Where-Object { $text.Contains($_) }
  $report = ReadJson $output
  if ($null -ne $report) {
    $runtime = [ordered]@{}
    foreach ($property in $report.runtime.PSObject.Properties) {
      $runtime[$property.Name] = $property.Value
    }
    $finalize = [ordered]@{
      preliminary = $false
      exit_code = $exit
      app_close_completed = ($exit -eq 0)
      finalized_after_process_exit = $true
      gpu_measurement_complete = ($null -ne $baseline -and $null -ne $peak -and $null -ne $after)
      gpu_recovered_to_baseline = ($null -ne $baseline -and $null -ne $after -and $after -le $baseline + 128)
      process_recovered = ($null -eq (Get-Process -Id $process.Id -ErrorAction SilentlyContinue))
      fatal_patterns = @($fatal)
      gpu_baseline_mib = $baseline
      gpu_peak_mib = $peak
      gpu_after_mib = $after
      stdout_path = Portable $stdout
      stderr_path = Portable $stderr
    }
    foreach ($key in $finalize.Keys) {
      $runtime[$key] = $finalize[$key]
    }
    $report.runtime = $runtime
    WriteJson $report $output
  }
  return $exit
}

$script:config = Full $ConfigPath
$reportRootFull = Full $ReportRoot
New-Item -ItemType Directory -Path $reportRootFull -Force | Out-Null
if ([string]::IsNullOrWhiteSpace($StatePath)) {
  $StatePath = Join-Path $reportRootFull 'g006_queue_state.json'
}
$script:statePath = Full $StatePath
$script:training = Full $TrainingHarness
$script:entry = Full $TrainingEntrypointPath
$script:evaluator = Full $EvaluationScript
$script:summaryPath = Full $SummaryScript
$lab = Full $IsaacLabPath
$script:labRoot = $lab
$python = Join-Path $lab '_isaac_sim\python.bat'
foreach ($path in @($script:config, $script:training, $script:entry, $python)) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "missing_file:$path" }
}
$cfg = ReadJson $script:config
if (
  $cfg.goal -ne 'G006' -or (@($cfg.variants.name) -join ',') -ne 'baseline,push_curriculum' -or
  (@($cfg.training.seeds) -join ',') -ne '42,43,44' -or
  [int]$cfg.training.max_iterations -ne 1500
) { throw 'manifest_contract_mismatch' }
$expectedSuccess = [ordered]@{
  lin_vel_error_mps_max = 0.30
  yaw_rate_error_radps_max = 0.30
  roll_abs_rad_max = 0.35
  pitch_abs_rad_max = 0.35
  consecutive_post_push_samples = 25
  recovery_completed_step_start = 201
  recovery_completed_step_end = 450
  horizon_completed_step = 600
  push_injection_completed_steps = 200
  base_contact_allowed = $false
  survival_to_horizon_required = $true
}
if (-not (SameJson $cfg.evaluation_protocol.success_criteria $expectedSuccess)) {
  throw 'success_criteria_contract_mismatch'
}
$trainingPaths = @(
  'scripts/bootstrap_train_g006.py',
  'src/isaac_walk_g006/__init__.py',
  'src/isaac_walk_g006/registry.py',
  'src/isaac_walk_g006/rough_env_cfg.py',
  'src/isaac_walk_g006/mdp/__init__.py',
  'src/isaac_walk_g006/mdp/events.py',
  'src/isaac_walk_g006/mdp/curriculums.py'
)
$evaluationPaths = @('scripts/evaluate_push_recovery.py') + @(
  Get-ChildItem `
    -LiteralPath (Join-Path $repoRoot 'src\isaac_walk_g006\evaluation') `
    -Filter '*.py' -File -Recurse |
      ForEach-Object { $_.FullName.Substring((Full $repoRoot).Length + 1).Replace('\', '/') }
)
$script:trainingBundle = GetSourceBundle $trainingPaths
$script:evaluationBundle = GetSourceBundle $evaluationPaths
$script:isaacLabCommit = ([string](& git -C $lab rev-parse HEAD 2>$null)).Trim()
if ($LASTEXITCODE -ne 0 -or $script:isaacLabCommit -ne $cfg.isaaclab.commit) {
  throw "isaaclab_commit_mismatch:$($script:isaacLabCommit)"
}
$canonical="import hashlib,json,sys;d=json.load(open(sys.argv[1],encoding='utf-8'));c=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest();print(json.dumps({'config':c(d),'protocol':c(d['evaluation_protocol']),'agent':c(d['training']['agent_learning_config'])}))"
$hashOut = @(& $python -c $canonical $script:config 2>&1)
if ($LASTEXITCODE -ne 0) { throw "hash_failed:$($hashOut -join ' ')" }
$hashes = $hashOut[-1] | ConvertFrom-Json
$script:configHash = [string]$hashes.config
$script:protocolHash = [string]$hashes.protocol
$script:entryHash = HashFile $script:entry
if ($hashes.agent -ne $cfg.training.agent_learning_config_sha256) {
  throw 'agent_learning_hash_mismatch'
}
$mode = if ($ScaleLadderOnly) {
  'scale'
}
elseif ($CurriculumSmoke) {
  'curriculum_smoke'
}
elseif ($Smoke) {
  'smoke'
}
else {
  'production'
}
$lockPath = Join-Path $reportRootFull '.g006.queue.lock'
$lock = $null
try {
  $lock = [IO.File]::Open(
    $lockPath,
    [IO.FileMode]::OpenOrCreate,
    [IO.FileAccess]::ReadWrite,
    [IO.FileShare]::None
  )
}
catch {
  throw "report_root_locked:$reportRootFull"
}
try {
  $variants = @($cfg.variants)
  $seeds = if ($Smoke -or $CurriculumSmoke -or $ScaleLadderOnly) {
    @(42)
  }
  else {
    @($cfg.training.seeds | ForEach-Object { [int]$_ })
  }
  if ($ScaleLadderOnly) {
    $envs = @($cfg.training.scale_ladder | ForEach-Object { [int]$_ })
    $iterations = 10
  }
  else {
    $envs = @(if ($Smoke -or $CurriculumSmoke) { 64 } elseif ($SelectedNumEnvs) { $SelectedNumEnvs } else { 0 })
    $iterations = if ($Smoke) { 1 } elseif ($CurriculumSmoke) { 30 } else { 1500 }
  }
  if (-not $ScaleLadderOnly -and $envs[0] -eq 0) {
    if ([string]::IsNullOrWhiteSpace($ScaleStatePath)) {
      throw 'production_requires_ScaleStatePath_or_SelectedNumEnvs'
    }
    $scaleState = ReadJson (Full $ScaleStatePath)
    if (
      $null -eq $scaleState -or $scaleState.goal -ne 'G006' -or
      $scaleState.mode -ne 'scale' -or $scaleState.status -ne 'complete' -or
      $scaleState.config_sha256 -ne $script:configHash -or -not $scaleState.selected_num_envs
    ) { throw 'invalid_scale_ladder_state' }
    $envs = @([int]$scaleState.selected_num_envs)
  }

  $specs = @()
  foreach ($envCount in $envs) {
    foreach ($variant in $variants) {
      if ($CurriculumSmoke -and $variant.name -ne 'push_curriculum') { continue }
      foreach ($seed in $seeds) {
        $run = "g006_$mode`_$($variant.name)_e$envCount`_i$iterations`_s$seed"
        $report = Join-Path $reportRootFull "$run.json"
        $specs += , [ordered]@{
          id = "$($variant.name)-e$envCount-s$seed"
          variant = $variant.name
          task = $variant.task
          seed = $seed
          num_envs = $envCount
          max_iterations = $iterations
          run_name = $run
          report_path = Portable $report
          push_report_path = Portable (Join-Path $reportRootFull "$run`_push.json")
          guardrail_report_path = Portable (Join-Path $reportRootFull "$run`_guardrail.json")
        }
      }
    }
  }

  $state = ReadJson $script:statePath
  if ($null -ne $state) {
    if (-not $Resume) { throw "state_exists_use_resume:$($script:statePath)" }
    if (
      $state.config_sha256 -ne $script:configHash -or $state.mode -ne $mode -or
      $state.training_entrypoint_sha256 -ne $script:entryHash -or
      $state.training_source_bundle_sha256 -ne $script:trainingBundle.sha256 -or
      $state.evaluation_source_bundle_sha256 -ne $script:evaluationBundle.sha256 -or
      -not (SameJson $state.source_bundles.training $script:trainingBundle) -or
      -not (SameJson $state.source_bundles.evaluation $script:evaluationBundle) -or
      $state.isaaclab_commit -ne $script:isaacLabCommit -or
      $state.agent_entry_point -ne $cfg.training.agent_learning_config.entry_point
    ) { throw 'resume_config_mode_source_or_provenance_mismatch' }
    AssertStateSpecs $state $specs
  }
  else {
    $jobs = @($specs | ForEach-Object {
      [ordered]@{
        id = $_.id; variant = $_.variant; task = $_.task; seed = $_.seed
        num_envs = $_.num_envs; max_iterations = $_.max_iterations; run_name = $_.run_name
        status = 'pending'; report_path = $_.report_path
        checkpoint_sha256 = $null; training_report_sha256 = $null
        push_report_path = $_.push_report_path; push_report_sha256 = $null
        push_command = @(); push_command_sha256 = $null; push_script_sha256 = $null
        guardrail_report_path = $_.guardrail_report_path; guardrail_report_sha256 = $null
        guardrail_command = @(); guardrail_command_sha256 = $null; guardrail_script_sha256 = $null
        training_source_bundle_sha256 = $script:trainingBundle.sha256
        evaluation_source_bundle_sha256 = $script:evaluationBundle.sha256
        attempts = @(); hard_blocked = $false
      }
    })
    $state = [ordered]@{
      schema_version = 2; goal = 'G006'; mode = $mode; status = 'pending'
      config_sha256 = $script:configHash; protocol_sha256 = $script:protocolHash
      training_entrypoint_sha256 = $script:entryHash
      training_source_bundle_sha256 = $script:trainingBundle.sha256
      evaluation_source_bundle_sha256 = $script:evaluationBundle.sha256
      source_bundles = [ordered]@{ training = $script:trainingBundle; evaluation = $script:evaluationBundle }
      isaaclab_commit = $script:isaacLabCommit
      agent_entry_point = [string]$cfg.training.agent_learning_config.entry_point
      normalized_env_diff = [ordered]@{ baseline = @(); push_curriculum = @('events.push_robot') }
      selected_num_envs = if ($ScaleLadderOnly) { $null } else { $envs[0] }
      summary_path = $null; summary_sha256 = $null; failures = @()
      created_at = (Get-Date).ToString('o'); updated_at = (Get-Date).ToString('o'); jobs = $jobs
    }
    WriteJson $state $script:statePath
  }

  $pwsh = (Get-Process -Id $PID).Path
  foreach ($spec in $specs) {
    $job = @($state.jobs | Where-Object id -eq $spec.id)[0]
    if ($null -eq $job) { throw "resume_missing_job:$($spec.id)" }
    if ($job.hard_blocked) { throw "hard_blocked:$($job.id)" }
    $trainingPath = AssertOutputPath ([string]$spec.report_path)
    $report = ReadJson $trainingPath
    $reuseExistingTraining = ($job.status -ne 'complete')

    if ($job.status -eq 'complete') {
      $completeValid = ValidateCompleteJob $job
      if ($completeValid -eq 'passed_migrated') {
        $state.updated_at = (Get-Date).ToString('o')
        WriteJson $state $script:statePath
        continue
      }
      if ($completeValid -eq 'passed') { continue }
      $job.attempts = @($job.attempts) + @([ordered]@{
        at = (Get-Date).ToString('o')
        phase = 'complete_integrity_restart'
        reason = $completeValid
        exit_code = $null
        failure_fingerprint = HashText("complete_integrity_restart`n$completeValid")
      })
      $job.status = 'pending'
      $job.checkpoint_sha256 = $null
      $job.training_report_sha256 = $null
      $job.push_report_sha256 = $null
      $job.guardrail_report_sha256 = $null
      WriteJson $state $script:statePath
    }
    $valid = if ($reuseExistingTraining) {
      ValidateTraining $job $report
    }
    else {
      'complete_integrity_restart'
    }
    if ($valid -ne 'passed') {
      $job.status = 'training'
      $state.status = 'training'
      $state.updated_at = (Get-Date).ToString('o')
      WriteJson $state $script:statePath
      & $pwsh -NoProfile -File $script:training `
        -Task $job.task -NumEnvs $job.num_envs -MaxIterations $job.max_iterations `
        -Seed $job.seed -RunName $job.run_name -IsaacLabPath $lab `
        -ReportPath $trainingPath -TrainingEntrypointPath $script:entry
      $exit = $LASTEXITCODE
      $report = ReadJson $trainingPath
      if ($null -ne $report) {
        BindTrainingReport $report $trainingPath
        $report = ReadJson $trainingPath
      }
      $valid = ValidateTraining $job $report
      if ($exit -ne 0 -or $valid -ne 'passed') {
        AddFailure $state $job 'training' "exit=$exit;$valid" $exit | Out-Null
        throw "training_failed:$($job.id)"
      }
    }

    $loggerMode = if ($mode -eq 'production') {
      'production'
    }
    elseif ($mode -eq 'curriculum_smoke') {
      'curriculum_smoke'
    }
    else {
      'smoke'
    }
    if ($mode -ne 'scale') {
      $loggerValid = ValidateLogger $report $job.variant $loggerMode
      if ($loggerValid -ne 'passed') {
        AddFailure $state $job 'logger_validation' $loggerValid 0 | Out-Null
        throw "logger_failed:$($job.id):$loggerValid"
      }
    }
    $job.checkpoint_sha256 = $report.artifacts.checkpoint_sha256
    $job.training_report_sha256 = HashFile $trainingPath
    $job.status = 'trained'
    $state.updated_at = (Get-Date).ToString('o')
    WriteJson $state $script:statePath
    if ($ScaleLadderOnly -or $Smoke -or $CurriculumSmoke) {
      $job.status = 'complete'
      $state.updated_at = (Get-Date).ToString('o')
      WriteJson $state $script:statePath
      continue
    }
    foreach ($evalMode in @('push', 'guardrail')) {
      $output = [string]$job."${evalMode}_report_path"
      $binding = EvalBinding $job $report $evalMode $output
      $job."${evalMode}_command" = @($binding.durable_args)
      $job."${evalMode}_command_sha256" = $binding.sha256
      $job."${evalMode}_script_sha256" = $binding.script_sha256
      $job.status = "evaluating_$evalMode"
      WriteJson $state $script:statePath
      $evalExit = InvokeEvaluator $binding $output $job.run_name $evalMode
      $evalValid = ValidateEval $job $report $evalMode $output
      if ($evalExit -ne 0 -or $evalValid -ne 'passed') {
        AddFailure $state $job "evaluation_$evalMode" "exit=$evalExit;$evalValid" $evalExit | Out-Null
        throw "evaluation_failed:$($job.id):${evalMode}:exit=${evalExit}:$evalValid"
      }
      $job."${evalMode}_report_sha256" = HashFile (AssertOutputPath $output)
    }
    $job.status = 'complete'
    $state.updated_at = (Get-Date).ToString('o')
    WriteJson $state $script:statePath
  }
  if ($ScaleLadderOnly) {
    $totalText = & nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null |
      Select-Object -First 1
    if ($LASTEXITCODE -ne 0 -or $totalText -notmatch '^\s*\d+\s*$') {
      throw 'gpu_total_measurement_failed'
    }
    $total = [double]$totalText
    $safe = @()
    foreach ($envCount in $envs) {
      $rungJobs = @($state.jobs | Where-Object num_envs -eq $envCount)
      $rungValid = ($rungJobs.Count -eq 2 -and @($rungJobs.variant | Sort-Object -Unique).Count -eq 2)
      $peaks = @()
      foreach ($rungJob in $rungJobs) {
        $row = ReadJson (AssertOutputPath ([string]$rungJob.report_path))
        if (
          (ValidateTraining $rungJob $row) -ne 'passed' -or
          $rungJob.status -ne 'complete' -or @($row.fatal_patterns_found).Count -ne 0 -or
          $row.gpu.measurement_complete -ne $true -or $row.gpu.recovered_to_baseline -ne $true
        ) {
          $rungValid = $false
        }
        else {
          $peaks += [double]$row.gpu.peak_used_mib
        }
      }
      $peak = ($peaks | Measure-Object -Maximum).Maximum
      if (
        $rungValid -and $peaks.Count -eq 2 -and
        $peak -lt $total * [double]$cfg.training.vram_limit_fraction
      ) { $safe += $envCount }
    }
    if ($safe.Count -eq 0) { throw 'no_common_safe_scale' }
    $state.selected_num_envs = ($safe | Measure-Object -Maximum).Maximum
  }
  if ($mode -eq 'production') {
    foreach ($path in @($script:evaluator, $script:summaryPath)) {
      if (-not (Test-Path $path -PathType Leaf)) { throw "missing_file:$path" }
    }
    $summaryOutput = Join-Path $reportRootFull 'g006_summary.json'
    $state.summary_path = $null
    $state.summary_sha256 = $null
    $state.status = 'summarizing'
    $state.updated_at = (Get-Date).ToString('o')
    WriteJson $state $script:statePath
    & $python $script:summaryPath `
      --manifest $script:config --queue-state $script:statePath `
      --isaaclab-root $lab --output $summaryOutput
    $summaryExit = $LASTEXITCODE
    $summaryReport = ReadJson $summaryOutput
    if ($summaryExit -ne 0 -or $null -eq $summaryReport -or $summaryReport.status -ne 'complete') {
      AddStateFailure $state 'summarizing' "exit=$summaryExit;status=$($summaryReport.status)" $summaryExit
      throw 'summary_failed'
    }
    $state.summary_path = Portable $summaryOutput
    $state.summary_sha256 = HashFile $summaryOutput
  }
  $state.status = 'complete'
  $state.updated_at = (Get-Date).ToString('o')
  WriteJson $state $script:statePath
  Write-Host "G006 queue PASS: mode=$mode jobs=$(@($state.jobs).Count) selected_envs=$($state.selected_num_envs)"
}
finally {
  if ($null -ne $lock) {
    $lock.Dispose()
  }
}
