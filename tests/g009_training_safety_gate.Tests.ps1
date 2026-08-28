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
$tempRoot = Join-Path $PSScriptRoot ('.g009-training-safety-' + [guid]::NewGuid().ToString('N'))
$fakeLab = Join-Path $tempRoot 'IsaacLab'
$fakePythonDirectory = Join-Path $fakeLab '_isaac_sim'
$fakeLogRoot = Join-Path $tempRoot 'logs'
$mockBin = Join-Path $tempRoot 'bin'
$originalPath = $env:PATH
$originalCase = [Environment]::GetEnvironmentVariable('G009_FAKE_SAFETY_CASE', 'Process')
$originalLogRoot = [Environment]::GetEnvironmentVariable('G009_FAKE_LOG_ROOT', 'Process')

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

New-Item -ItemType Directory -Path $fakePythonDirectory, $fakeLogRoot, $mockBin -Force | Out-Null
$fakePythonBat = Join-Path $fakePythonDirectory 'python.bat'
$fakePythonHelper = Join-Path $fakePythonDirectory 'fake_python.ps1'
$nvidiaSmiFixture = Join-Path $mockBin 'nvidia-smi.cmd'

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

if ($Remaining.Count -gt 0 -and $Remaining[0] -eq '-c') {
    $hardMaximum = if ($env:G009_FAKE_SAFETY_CASE -eq 'nonzero') { 1.0 } else { 0.0 }
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
[IO.File]::WriteAllText((Join-Path $logDirectory 'events.out.tfevents.fake'), 'fake')
[IO.File]::WriteAllText((Join-Path $logDirectory ("model_$($maxIterations - 1).pt")), 'fake checkpoint')
Write-Output "[INFO] Logging experiment in directory: $env:G009_FAKE_LOG_ROOT"
Write-Output "Exact experiment name requested from command line: $timestamp"
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
    @('@echo off', 'echo 1000', 'exit /b 0'),
    [Text.Encoding]::ASCII
)

$env:PATH = $mockBin + [IO.Path]::PathSeparator + $originalPath
$env:G009_FAKE_LOG_ROOT = $fakeLogRoot
try {
    $pass = Invoke-TrainingCase -CaseName 'pass' -SafetyGate
    Assert ($pass.ExitCode -eq 0) 'zero-valued required series must pass'
    Assert ($pass.Report.passed -eq $true) 'pass report must be passed'
    Assert ($pass.Report.training_safety_gate.requested -eq $true) 'report must record requested gate'
    Assert ($pass.Report.training_safety_gate.passed -eq $true) 'report must record passing verdict'

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

    $default = Invoke-TrainingCase -CaseName 'nonzero'
    Assert ($default.ExitCode -eq 0) 'default execution must not enforce the diagnostic gate'
    Assert ($default.Report.training_safety_gate.requested -eq $false) 'default report must record gate not requested'
    Assert ($null -eq $default.Report.training_safety_gate.passed) 'default diagnostic verdict must remain null'

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
        -Task 'Isaac-G009-Recover-Flat-Go2-R0-v0' -NumEnvs 512 -MaxIterations 300 -Seed 42 `
        -RunName 'g009_qualification_compatibility' -Qualification `
        -RequireZeroTrainingSafetyTerminations 2>&1)
    Assert ($LASTEXITCODE -ne 0) 'noncanonical qualification budget must remain rejected'
    $qualificationText = $qualificationOutput -join "`n"
    Assert (
        $qualificationText.Contains('num_envs=1024') -and
        $qualificationText.Contains('max_iterations=300') -and
        $qualificationText.Contains('seed=42')
    ) 'qualification fixed budget guard must remain intact'

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
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}
