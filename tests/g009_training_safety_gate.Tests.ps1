[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Assert {
    param([bool]$Condition, [string]$Message)

    if (-not $Condition) {
        throw "ASSERT FAIL: $Message"
    }
}

$root = Split-Path -Parent $PSScriptRoot
$harness = Join-Path $root 'scripts\run_training.ps1'
$pwsh = (Get-Command pwsh -ErrorAction Stop).Source
$currentPowerShell = (Get-Process -Id $PID -ErrorAction Stop).Path
$actualGit = (Get-Command git -ErrorAction Stop).Source
$tempRoot = Join-Path $PSScriptRoot ('.g009-training-safety-' + [guid]::NewGuid().ToString('N'))
$fakeLab = Join-Path $tempRoot 'IsaacLab'
$fakePythonDirectory = Join-Path $fakeLab '_isaac_sim'
$fakeLogRoot = Join-Path $tempRoot 'logs'
$mockBin = Join-Path $tempRoot 'bin'
$originalPath = $env:PATH
$originalCase = [Environment]::GetEnvironmentVariable('G009_FAKE_SAFETY_CASE', 'Process')
$originalLogRoot = [Environment]::GetEnvironmentVariable('G009_FAKE_LOG_ROOT', 'Process')
$originalGitCase = [Environment]::GetEnvironmentVariable('G009_FAKE_GIT_CASE', 'Process')

function Invoke-QualificationGitCase {
    param([string]$CaseName)

    $env:G009_FAKE_GIT_CASE = $CaseName
    $arguments = @(
        '-NoProfile', '-File', $harness,
        '-Task', 'Isaac-G009-Recover-Flat-Go2-R0-Matrix-v0',
        '-NumEnvs', '1024',
        '-MaxIterations', '300',
        '-Seed', '42',
        '-RunName', "g009_git_$CaseName",
        '-Qualification',
        '-IsaacLabPath', $fakeLab,
        '-ReportPath', (Join-Path $tempRoot "git-$CaseName.json"),
        '-TrainingEntrypointPath', (Join-Path $root 'scripts\bootstrap_train_g009.py'),
        '-SourceBindingPaths', 'configs/g009_r0_rev26_qualification.json'
    )
    $output = @(& $pwsh @arguments 2>&1)
    return [pscustomobject]@{ ExitCode = $LASTEXITCODE; Output = $output -join "`n" }
}

function Invoke-TrainingCase {
    param(
        [string]$CaseName,
        [switch]$SafetyGate
    )

    $env:G009_FAKE_SAFETY_CASE = $CaseName
    $reportPath = Join-Path $tempRoot "$CaseName.json"
    $arguments = @(
        '-NoProfile', '-File', $harness,
        '-Task', 'Isaac-G009-Recover-Flat-Go2-R0-v0',
        '-NumEnvs', '1',
        '-MaxIterations', '1',
        '-Seed', '42',
        '-RunName', "g009_safety_$CaseName",
        '-IsaacLabPath', $fakeLab,
        '-ReportPath', $reportPath,
        '-TrainingEntrypointPath', $PSCommandPath,
        '-GpuSampleIntervalSeconds', '1'
    )
    if ($SafetyGate) {
        $arguments += '-RequireZeroTrainingSafetyTerminations'
    }

    $output = @(& $pwsh @arguments 2>&1)
    $exitCode = $LASTEXITCODE
    $report = if (Test-Path -LiteralPath $reportPath) {
        Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
    }
    else {
        $null
    }
    return [pscustomobject]@{
        ExitCode = $exitCode
        Output = $output -join "`n"
        Report = $report
    }
}

function Invoke-EntropySmokeSingleLinePreflightCase {
    $env:G009_FAKE_SAFETY_CASE = 'entropy_preflight_single_line'
    Remove-Item Env:G009_FAKE_GIT_CASE -ErrorAction SilentlyContinue
    $entropySourcePaths = @(
        (Get-Content -LiteralPath (Join-Path $root 'configs\g009_r0_rev28_entropy_smoke.json') -Raw |
            ConvertFrom-Json).source_binding_paths
    )
    $entropyRepo = Join-Path $tempRoot 'entropy-repo'
    & $actualGit clone --quiet --no-hardlinks $root $entropyRepo
    if ($LASTEXITCODE -ne 0) { throw 'entropy fixture clone failed' }
    foreach ($relativePath in $entropySourcePaths) {
        $destination = Join-Path $entropyRepo $relativePath
        New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $root $relativePath) -Destination $destination -Force
    }
    & $actualGit -C $entropyRepo add -- $entropySourcePaths
    & $actualGit -C $entropyRepo -c user.name=g009-fixture -c user.email=g009-fixture@example.invalid commit --quiet -m 'fixture'
    if ($LASTEXITCODE -ne 0) { throw 'entropy fixture commit failed' }
    $entropyHarness = Join-Path $entropyRepo 'scripts\run_training.ps1'
    $wrapperPath = Join-Path $tempRoot 'invoke-entropy-single-line.ps1'
    $quotedSourcePaths = ($entropySourcePaths | ForEach-Object {
        "'" + ([string]$_).Replace("'", "''") + "'"
    }) -join ', '
    $wrapper = @"
& '$($entropyHarness.Replace("'", "''"))' ``
    -Task 'Isaac-G009-Recover-Flat-Go2-R0-Matrix-v0' ``
    -NumEnvs 1024 -MaxIterations 50 -Seed 42 ``
    -RunName 'g009_entropy_single_line_preflight' -EntropySmoke ``
    -IsaacLabPath '$($fakeLab.Replace("'", "''"))' ``
    -ReportPath '$((Join-Path $tempRoot 'entropy-single-line.json').Replace("'", "''"))' ``
    -TrainingEntrypointPath '$((Join-Path $entropyRepo 'scripts\bootstrap_train_g009.py').Replace("'", "''"))' ``
    -SourceBindingPaths @($quotedSourcePaths)
"@
    [IO.File]::WriteAllText($wrapperPath, $wrapper, [Text.UTF8Encoding]::new($true))
    $nvidiaOnlyBin = Join-Path $tempRoot 'nvidia-only-bin'
    New-Item -ItemType Directory -Path $nvidiaOnlyBin -Force | Out-Null
    Copy-Item -LiteralPath $nvidiaSmiFixture -Destination (Join-Path $nvidiaOnlyBin 'nvidia-smi.cmd') -Force
    $fixturePath = $env:PATH
    try {
        $env:PATH = $nvidiaOnlyBin + [IO.Path]::PathSeparator + $originalPath
        $output = @(& $currentPowerShell -NoProfile -File $wrapperPath 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $env:PATH = $fixturePath
    }
    return [pscustomobject]@{ ExitCode = $exitCode; Output = $output -join "`n" }
}

New-Item -ItemType Directory -Path $fakePythonDirectory, $fakeLogRoot, $mockBin -Force | Out-Null
$fakePythonBat = Join-Path $fakePythonDirectory 'python.bat'
$fakePythonHelper = Join-Path $fakePythonDirectory 'fake_python.ps1'
$nvidiaSmiFixture = Join-Path $mockBin 'nvidia-smi.cmd'
$gitFixture = Join-Path $mockBin 'git.cmd'
$gitFixtureHelper = Join-Path $mockBin 'git-fixture.ps1'

[IO.File]::WriteAllLines(
    $fakePythonBat,
    @(
        '@echo off',
        'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0fake_python.ps1" %*',
        'exit /b %ERRORLEVEL%'
    ),
    [Text.Encoding]::ASCII
)
[IO.File]::WriteAllText(
    $fakePythonHelper,
    @'
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Remaining)
$ErrorActionPreference = 'Stop'

if ($Remaining.Count -gt 0 -and [IO.Path]::GetFileName($Remaining[0]) -eq 'validate_g009_r0_rev28_entropy_smoke.py') {
    Write-Output '{"status":"pass","canonical_static_readback":{"entropy_coef":0.0}}'
    exit 0
}

if ($Remaining.Count -gt 0 -and $Remaining[0] -eq '-c') {
    $hardMaximum = if ($env:G009_FAKE_SAFETY_CASE -like 'nonzero*') { 1.0 } else { 0.0 }
    $hardSummary = [ordered]@{
        sample_count = 2
        latest = $hardMaximum
        minimum = 0.0
        maximum = $hardMaximum
        mean = $hardMaximum / 2.0
        nonzero_sample_count = if ($hardMaximum -eq 0.0) { 0 } else { 1 }
    }
    if ($env:G009_FAKE_SAFETY_CASE -eq 'missing_maximum') {
        [void]$hardSummary.Remove('maximum')
    }
    elseif ($env:G009_FAKE_SAFETY_CASE -eq 'null_maximum') {
        $hardSummary.maximum = $null
    }
    elseif ($env:G009_FAKE_SAFETY_CASE -eq 'nan_maximum') {
        $hardSummary.maximum = 'NaN'
    }
    elseif ($env:G009_FAKE_SAFETY_CASE -eq 'nan_mean') {
        $hardSummary.mean = 'NaN'
    }
    elseif ($env:G009_FAKE_SAFETY_CASE -eq 'zero_samples') {
        $hardSummary.sample_count = 0
    }
    elseif ($env:G009_FAKE_SAFETY_CASE -eq 'nonzero_count_mismatch') {
        $hardSummary.nonzero_sample_count = 1
    }
    elseif ($env:G009_FAKE_SAFETY_CASE -eq 'string_zero_fields') {
        foreach ($field in @('sample_count', 'latest', 'minimum', 'maximum', 'mean', 'nonzero_sample_count')) {
            $hardSummary[$field] = [Convert]::ToString(
                $hardSummary[$field],
                [System.Globalization.CultureInfo]::InvariantCulture
            )
        }
    }
    $summary = [ordered]@{
        'Episode_Termination/hard_joint_limit' = [ordered]@{
        }
    }
    $summary['Episode_Termination/hard_joint_limit'] = $hardSummary
    if ($env:G009_FAKE_SAFETY_CASE -ne 'missing') {
        $summary['Episode_Termination/numeric_invalid'] = [ordered]@{
            sample_count = 2
            latest = 0.0
            minimum = 0.0
            maximum = 0.0
            mean = 0.0
            nonzero_sample_count = 0
        }
    }
    [ordered]@{
        tags = @($summary.Keys)
        latest = [ordered]@{}
        series_summary = $summary
    } | ConvertTo-Json -Depth 8 -Compress
    exit 0
}

function Get-ArgumentValue([string]$Name) {
    $index = [Array]::IndexOf($Remaining, $Name)
    if ($index -lt 0 -or $index + 1 -ge $Remaining.Count) {
        throw "missing argument: $Name"
    }
    return $Remaining[$index + 1]
}

$runName = Get-ArgumentValue '--run_name'
$maxIterations = [int](Get-ArgumentValue '--max_iterations')
$timestamp = '2026-08-28_120000'
$logDirectory = Join-Path $env:G009_FAKE_LOG_ROOT ($timestamp + '_' + $runName)
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
if ($env:G009_FAKE_SAFETY_CASE -eq 'ambiguous_timestamp') {
    New-Item -ItemType Directory -Path (Join-Path $env:G009_FAKE_LOG_ROOT ($timestamp + '-alt_' + $runName)) -Force | Out-Null
}
[IO.File]::WriteAllText((Join-Path $logDirectory 'events.out.tfevents.fake'), 'fake')
[IO.File]::WriteAllText((Join-Path $logDirectory ("model_$($maxIterations - 1).pt")), 'fake checkpoint')
Write-Output "[INFO] Logging experiment in directory: $env:G009_FAKE_LOG_ROOT"
if ($env:G009_FAKE_SAFETY_CASE -notin @('no_timestamp', 'ambiguous_timestamp')) {
    Write-Output "Exact experiment name requested from command line: $timestamp"
}
Write-Output "Learning iteration  $($maxIterations - 1)/$maxIterations"
Write-Output 'Computation: 100 steps/s'
Write-Output 'Mean reward: 1.0'
Write-Output 'Mean episode length: 1.0'
exit 0
'@,
    [Text.UTF8Encoding]::new($false)
)
[IO.File]::WriteAllLines(
    $nvidiaSmiFixture,
    @(
        '@echo off',
        'if "%G009_FAKE_SAFETY_CASE%"=="entropy_preflight_single_line" exit /b 19',
        'echo 1000, 10, 50, 20, 12288',
        'exit /b 0'
    ),
    [Text.Encoding]::ASCII
)
[IO.File]::WriteAllText(
    $gitFixtureHelper,
    @"
param([Parameter(ValueFromRemainingArguments = `$true)][string[]]`$Remaining)
`$fakeLab = '$fakeLab'
`$cIndex = [Array]::IndexOf(`$Remaining, '-C')
`$targetsFakeLab = `$cIndex -ge 0 -and `$cIndex + 1 -lt `$Remaining.Count -and `$Remaining[`$cIndex + 1] -eq `$fakeLab
if (`$targetsFakeLab -and `$Remaining -contains 'rev-parse' -and `$Remaining -contains 'HEAD') {
    if (`$env:G009_FAKE_GIT_CASE -eq 'commit_failure_status_clean') {
        [Console]::Error.WriteLine('fatal: fixture-commit-failure')
        exit 23
    }
    Write-Output '90b79bb2d44feb8d833f260f2bf37da3487180ba'
    exit 0
}

if (`$targetsFakeLab -and `$Remaining -contains 'status' -and `$Remaining -contains '--untracked-files=no') {
    if (`$env:G009_FAKE_GIT_CASE -eq 'status_failure') {
        [Console]::Error.WriteLine('fatal: fixture-status-failure')
        exit 31
    }
    if (`$env:G009_FAKE_GIT_CASE -eq 'tracked_line') {
        Write-Output ' M scripts/reinforcement_learning/rsl_rl/train.py'
    }
    exit 0
}
& '$actualGit' @Remaining
exit `$LASTEXITCODE
"@,
    [Text.UTF8Encoding]::new($false)
)
[IO.File]::WriteAllLines(
    $gitFixture,
    @('@echo off', 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0git-fixture.ps1" %*', 'exit /b %ERRORLEVEL%'),
    [Text.Encoding]::ASCII
)

$env:PATH = $mockBin + [IO.Path]::PathSeparator + $originalPath
$env:G009_FAKE_LOG_ROOT = $fakeLogRoot
try {
    foreach ($validatorFixture in @(
        [pscustomobject]@{ Content = ''; ExpectedCount = 0 },
        [pscustomobject]@{ Content = '{"status":"pass"}'; ExpectedCount = 1 },
        [pscustomobject]@{ Content = "first`nsecond"; ExpectedCount = 2 }
    )) {
        $validatorOutputPath = Join-Path $tempRoot "validator-$($validatorFixture.ExpectedCount).stdout"
        [IO.File]::WriteAllText($validatorOutputPath, $validatorFixture.Content, [Text.UTF8Encoding]::new($false))
        $validatorLines = @(
            if (Test-Path -LiteralPath $validatorOutputPath) {
                Get-Content -LiteralPath $validatorOutputPath -ErrorAction Stop
            }
            else { }
        )
        Assert ($validatorLines -is [array]) 'validator stdout capture must preserve the array type'
        Assert ($validatorLines.Count -eq $validatorFixture.ExpectedCount) "validator stdout count mismatch: expected=$($validatorFixture.ExpectedCount) actual=$($validatorLines.Count)"
    }

    $entropyPreflight = Invoke-EntropySmokeSingleLinePreflightCase
    Assert ($entropyPreflight.Output.Contains('nvidia-smi')) "single-line validator JSON must pass preflight and reach the GPU boundary; actual=$($entropyPreflight.Output)"
    Assert (-not $entropyPreflight.Output.Contains('validator')) 'single-line validator JSON must not fail before the GPU boundary'
    Assert (-not $entropyPreflight.Output.Contains("Property 'Count' cannot be found")) 'single-line validator stdout must remain an array under StrictMode'

    $cleanGit = Invoke-QualificationGitCase -CaseName 'clean_empty'
    Assert ($cleanGit.ExitCode -ne 0) "clean git fixture still stops at unrelated qualification prerequisites; actual=$($cleanGit.Output)"
    Assert (-not $cleanGit.Output.Contains('git status 실패')) 'clean empty status must not be reported as git failure'
    Assert (-not $cleanGit.Output.Contains('tracked 변경 감지')) 'clean empty status must not be reported as tracked changes'

    $staleExit = Invoke-QualificationGitCase -CaseName 'commit_failure_status_clean'
    $staleText = (($staleExit.Output -replace '\s*\|\s*', ' ') -replace '\s+', ' ')
    Assert ($staleText.Contains('git rev-parse 실패: exit=23')) "commit failure exit code must be captured immediately; actual=$($staleExit.Output)"
    Assert ($staleText.Contains('fixture-commit-failure')) 'commit failure stderr must be preserved'
    Assert (-not $staleExit.Output.Contains('tracked 변경 감지')) 'prior nonzero exit must not poison clean status result'

    $gitFailure = Invoke-QualificationGitCase -CaseName 'status_failure'
    $gitFailureText = (($gitFailure.Output -replace '\s*\|\s*', ' ') -replace '\s+', ' ')
    Assert ($gitFailureText.Contains('git status 실패: exit=31')) 'status failure exit code must be distinct'
    Assert ($gitFailureText.Contains('fixture-status-failure')) 'status failure stderr must be preserved'
    Assert (-not $gitFailure.Output.Contains('tracked 변경 감지')) 'git failure must not be mislabeled as tracked changes'

    $trackedLine = Invoke-QualificationGitCase -CaseName 'tracked_line'
    $trackedLineText = (($trackedLine.Output -replace '\s*\|\s*', ' ') -replace '\s+', ' ')
    Assert ($trackedLineText.Contains('tracked 변경 감지: count=1')) 'one normalized tracked line must report exact count'
    Assert ($trackedLineText.Contains('M scripts/reinforcement_learning/rsl_rl/train.py')) 'tracked line details must be preserved'

    Remove-Item Env:G009_FAKE_GIT_CASE -ErrorAction SilentlyContinue
    $pass = Invoke-TrainingCase -CaseName 'pass' -SafetyGate
    Assert ($pass.ExitCode -eq 0) 'zero-valued required series must pass'
    Assert ($pass.Report.passed -eq $true) 'pass report must be passed'
    Assert ($pass.Report.training_safety_gate.requested -eq $true) 'report must record requested gate'
    Assert ($pass.Report.training_safety_gate.passed -eq $true) 'report must record passing verdict'
    $passReportHash = (Get-FileHash -LiteralPath (Join-Path $tempRoot 'pass.json') -Algorithm SHA256).Hash
    $passCollision = Invoke-TrainingCase -CaseName 'pass' -SafetyGate
    Assert ($passCollision.ExitCode -ne 0) 'existing raw logs/report must be rejected before execution'
    Assert ($passCollision.Output.Contains('덮어쓸 수 없습니다')) 'no-overwrite rejection must be explicit'
    Assert ((Get-FileHash -LiteralPath (Join-Path $tempRoot 'pass.json') -Algorithm SHA256).Hash -eq $passReportHash) 'existing report must remain unchanged'
    $rawCollisionDirectory = Join-Path $fakeLab 'logs\harness'
    New-Item -ItemType Directory -Path $rawCollisionDirectory -Force | Out-Null
    $rawCollisionPath = Join-Path $rawCollisionDirectory 'g009_safety_raw_collision.stdout.log'
    [IO.File]::WriteAllText($rawCollisionPath, 'preserve-me', [Text.UTF8Encoding]::new($false))
    $rawCollision = Invoke-TrainingCase -CaseName 'raw_collision'
    Assert ($rawCollision.ExitCode -ne 0) 'existing raw stdout must be rejected before execution'
    Assert ($rawCollision.Output.Contains('덮어쓸 수 없습니다')) 'raw stdout no-overwrite rejection must be explicit'
    Assert ((Get-Content -LiteralPath $rawCollisionPath -Raw) -eq 'preserve-me') 'existing raw stdout must remain unchanged'

    $nonzero = Invoke-TrainingCase -CaseName 'nonzero' -SafetyGate
    Assert ($nonzero.ExitCode -eq 1) 'nonzero hard_joint_limit maximum must fail closed'
    Assert ($nonzero.Report.passed -eq $false) 'nonzero report must fail'
    Assert ($nonzero.Report.training_safety_gate.passed -eq $false) 'nonzero verdict must be explicit'

    $missing = Invoke-TrainingCase -CaseName 'missing' -SafetyGate
    Assert ($missing.ExitCode -eq 1) 'missing numeric_invalid series must fail closed'
    Assert ($missing.Report.training_safety_gate.numeric_invalid_series_present -eq $false) 'missing series must be explicit'
    Assert ($missing.Report.training_safety_gate.passed -eq $false) 'missing series verdict must fail'

    foreach ($malformedCase in @(
        'missing_maximum',
        'null_maximum',
        'nan_maximum',
        'nan_mean',
        'zero_samples',
        'nonzero_count_mismatch',
        'string_zero_fields'
    )) {
        $malformed = Invoke-TrainingCase -CaseName $malformedCase -SafetyGate
        Assert ($malformed.ExitCode -eq 1) "$malformedCase maximum must fail closed"
        Assert ($malformed.Report.training_safety_gate.passed -eq $false) "$malformedCase verdict must fail"
    }

    $default = Invoke-TrainingCase -CaseName 'nonzero_default'
    Assert ($default.ExitCode -eq 0) 'default execution must not enforce the diagnostic gate'
    Assert ($default.Report.training_safety_gate.requested -eq $false) 'default report must record gate not requested'
    Assert ($null -eq $default.Report.training_safety_gate.passed) 'default diagnostic verdict must remain null'

    $noTimestamp = Invoke-TrainingCase -CaseName 'no_timestamp'
    Assert ($noTimestamp.ExitCode -eq 0) 'official benchmark-style output without exact timestamp must pass'
    Assert ($noTimestamp.Report.success_checks.log_directory_exists -eq $true) 'log directory fallback must resolve the run-name directory'
    Assert (-not [string]::IsNullOrWhiteSpace($noTimestamp.Report.artifacts.checkpoint)) 'log directory fallback must bind the checkpoint'

    $ambiguousTimestamp = Invoke-TrainingCase -CaseName 'ambiguous_timestamp'
    Assert ($ambiguousTimestamp.ExitCode -eq 1) 'ambiguous benchmark-style log directories must fail closed'
    Assert ($ambiguousTimestamp.Report.log_directory_resolution.mode -eq 'ambiguous_new_run_name_directories') 'ambiguity reason must be recorded'
    Assert ($ambiguousTimestamp.Report.success_checks.log_directory_exists -eq $false) 'ambiguous fallback must not select a directory'

    $resumeOutput = @(& $pwsh -NoProfile -File $harness `
        -Task 'Isaac-G009-Recover-Flat-Go2-R0-v0' -NumEnvs 1 -MaxIterations 1 -Seed 42 `
        -RunName 'g009_safety_resume' -RequireZeroTrainingSafetyTerminations -Resume `
        -LoadRun 'prior' -ResumeCheckpoint 'model_1.pt' 2>&1)
    Assert ($LASTEXITCODE -ne 0) 'diagnostic gate plus resume must be rejected'
    Assert (($resumeOutput -join "`n") -match 'scratch.*Resume') 'resume rejection must explain scratch requirement'

    $overrideOutput = @(& $pwsh -NoProfile -File $harness `
        -Task 'Isaac-G009-Recover-Flat-Go2-R0-v0' -NumEnvs 1 -MaxIterations 1 -Seed 42 `
        -RunName 'g009_safety_override' -RequireZeroTrainingSafetyTerminations `
        -HydraOverrides 'agent.algorithm.gamma=0.95' 2>&1)
    Assert ($LASTEXITCODE -ne 0) 'diagnostic gate plus Hydra override must be rejected'
    Assert (($overrideOutput -join "`n") -match 'Hydra override') 'override rejection must explain fixed scratch semantics'

    $qualificationOutput = @(& $pwsh -NoProfile -File $harness `
        -Task 'Isaac-G009-Recover-Flat-Go2-R0-Matrix-v0' -NumEnvs 512 -MaxIterations 300 -Seed 42 `
        -RunName 'g009_qualification_compatibility' -Qualification `
        -RequireZeroTrainingSafetyTerminations 2>&1)
    Assert ($LASTEXITCODE -ne 0) 'noncanonical qualification budget must remain rejected'
    $qualificationText = $qualificationOutput -join "`n"
    Assert (
        $qualificationText.Contains('num_envs=1024') -and
        $qualificationText.Contains('max_iterations=300') -and
        $qualificationText.Contains('seed=42')
    ) 'qualification fixed budget guard must remain intact'

    $harnessText = Get-Content -LiteralPath $harness -Raw
    Assert ($harnessText.Contains("`$qualificationTemperatureC = 90.0")) 'qualification temperature threshold must remain 90C'
    Assert ($harnessText.Contains("`$qualificationSustainedTemperatureSamples = 3")) 'sustained temperature must require three samples'
    Assert ($harnessText.Contains("Join-Path `$actualLogDirectory 'model_299.pt'")) 'qualification must bind exact model_299.pt'
    Assert ($harnessText.Contains('qualification_gpu_safety')) 'qualification GPU safety must remain a success check'
    Assert ($harnessText.Contains('--untracked-files=no')) 'qualification must verify Isaac Lab tracked cleanliness'
    Assert ($harnessText.Contains('taskkill.exe')) 'qualification abort must terminate the Windows process tree'
    Assert ($harnessText.Contains('descendants_exited')) 'qualification report must record descendant exit verification'
    Assert ($harnessText.Contains('[System.IO.File]::Move($stdoutCapturePath, $stdoutPath)')) 'raw stdout must publish with a no-overwrite move'
    Assert ($harnessText.Contains('[System.IO.File]::Move($stderrCapturePath, $stderrPath)')) 'raw stderr must publish with a no-overwrite move'
    foreach ($fatalToken in @('CUDA\s+out\s+of\s+memory', 'out\s+of\s+memory', '\bXid\b', 'driver\s+reset', 'device\s+lost')) {
        Assert ($harnessText.Contains($fatalToken)) "qualification fatal detector must include $fatalToken"
    }

    $childPidPath = Join-Path $tempRoot 'descendant.pid'
    $processTreeFixture = Join-Path $tempRoot 'process-tree-fixture.ps1'
    [IO.File]::WriteAllText(
        $processTreeFixture,
        "`$child=Start-Process -FilePath '$pwsh' -ArgumentList '-NoProfile','-Command','Start-Sleep -Seconds 120' -PassThru;[IO.File]::WriteAllText('$childPidPath',[string]`$child.Id);Start-Sleep -Seconds 120",
        [Text.UTF8Encoding]::new($false)
    )
    $fixtureParent = Start-Process -FilePath $pwsh -ArgumentList '-NoProfile', '-File', $processTreeFixture -PassThru -WindowStyle Hidden
    try {
        $pidDeadline = (Get-Date).AddSeconds(10)
        while (-not (Test-Path -LiteralPath $childPidPath) -and (Get-Date) -lt $pidDeadline) {
            Start-Sleep -Milliseconds 100
        }
        Assert (Test-Path -LiteralPath $childPidPath) 'child-survivor fixture must publish descendant PID'
        $fixtureChildId = [int](Get-Content -LiteralPath $childPidPath -Raw)
        & taskkill.exe /PID $fixtureParent.Id /T /F *> $null
        $exitDeadline = (Get-Date).AddSeconds(10)
        while (
            ((Get-Process -Id $fixtureParent.Id -ErrorAction SilentlyContinue) -or (Get-Process -Id $fixtureChildId -ErrorAction SilentlyContinue)) -and
            (Get-Date) -lt $exitDeadline
        ) {
            Start-Sleep -Milliseconds 100
        }
        Assert ($null -eq (Get-Process -Id $fixtureParent.Id -ErrorAction SilentlyContinue)) 'tree termination must stop parent'
        Assert ($null -eq (Get-Process -Id $fixtureChildId -ErrorAction SilentlyContinue)) 'tree termination must stop surviving descendant'
    }
    finally {
        if (-not $fixtureParent.HasExited) {
            & taskkill.exe /PID $fixtureParent.Id /T /F *> $null
        }
    }

    Write-Host 'G009 training safety gate assertions PASS'
}
finally {
    $env:PATH = $originalPath
    if ($null -eq $originalCase) {
        Remove-Item Env:G009_FAKE_SAFETY_CASE -ErrorAction SilentlyContinue
    }
    else {
        $env:G009_FAKE_SAFETY_CASE = $originalCase
    }
    if ($null -eq $originalLogRoot) {
        Remove-Item Env:G009_FAKE_LOG_ROOT -ErrorAction SilentlyContinue
    }
    else {
        $env:G009_FAKE_LOG_ROOT = $originalLogRoot
    }
    if ($null -eq $originalGitCase) {
        Remove-Item Env:G009_FAKE_GIT_CASE -ErrorAction SilentlyContinue
    }
    else {
        $env:G009_FAKE_GIT_CASE = $originalGitCase
    }
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}
