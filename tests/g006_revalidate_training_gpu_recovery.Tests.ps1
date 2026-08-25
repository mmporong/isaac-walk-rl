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
$scriptPath = Join-Path $root 'scripts\revalidate_training_gpu_recovery.ps1'
$reportRoot = Join-Path $root 'reports\runs'
$tempRoot = Join-Path $reportRoot ('g006-revalidation-test-' + [guid]::NewGuid().ToString('N'))

function New-RevalidationFixture {
    param([string]$CaseName)

    $caseRoot = Join-Path $tempRoot $CaseName
    $tensorboard = Join-Path $caseRoot 'tensorboard'
    New-Item -ItemType Directory -Path $tensorboard -Force | Out-Null

    $checkpoint = Join-Path $caseRoot 'model_1499.pt'
    $stdout = Join-Path $caseRoot 'stdout.log'
    $stderr = Join-Path $caseRoot 'stderr.log'
    [IO.File]::WriteAllText($checkpoint, 'checkpoint')
    [IO.File]::WriteAllText($stdout, 'training complete')
    [IO.File]::WriteAllText($stderr, '')

    $reportPath = Join-Path $caseRoot 'training.json'
    $report = [ordered]@{
        passed = $false
        run_name = "g006_$CaseName"
        exit_code = 0
        last_iteration = 1499
        iteration_target = 1500
        fatal_patterns_found = @()
        artifacts = [ordered]@{
            checkpoint = $checkpoint
            checkpoint_sha256 = (Get-FileHash $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
            raw_stdout = $stdout
            raw_stderr = $stderr
            tensorboard_directory = $tensorboard
        }
        gpu = [ordered]@{
            baseline_used_mib = 1000
            measurement_complete = $true
            recovered_to_baseline = $false
            recovery_samples = @([ordered]@{ used_mib = 1400 })
        }
        success_checks = [ordered]@{
            gpu_measurement_complete = $true
            gpu_recovered_to_baseline = $false
            tensorboard_exists = $true
            checkpoint_exists = $true
        }
    }
    [IO.File]::WriteAllText($reportPath, ($report | ConvertTo-Json -Depth 12))

    return [pscustomobject]@{
        ReportPath = $reportPath
        Checkpoint = $checkpoint
    }
}

function Invoke-RevalidationFailure {
    param([string]$ReportPath)

    try {
        & $scriptPath `
            -ReportPath $ReportPath `
            -SampleCount 3 `
            -SampleIntervalSeconds 1
        return 'unexpected_success'
    }
    catch {
        return $_.Exception.Message
    }
}

$previousCimFunction = Get-Item Function:\global:Get-CimInstance -ErrorAction SilentlyContinue
$previousSleepFunction = Get-Item Function:\global:Start-Sleep -ErrorAction SilentlyContinue
$originalPath = $env:PATH
$originalGpuMiB = [Environment]::GetEnvironmentVariable('G006_TEST_GPU_MIB', 'Process')
$global:G006TestProcesses = @()
$global:G006TestGpuSamples = [Collections.Queue]::new()

function global:Get-CimInstance {
    param([string]$ClassName)

    return @($global:G006TestProcesses)
}

function global:Start-Sleep {
    param([int]$Seconds)

    if ($global:G006TestGpuSamples.Count -gt 0) {
        $env:G006_TEST_GPU_MIB = [string]$global:G006TestGpuSamples.Dequeue()
    }
}

function Set-GpuSamples {
    param([int[]]$Samples)

    $global:G006TestGpuSamples.Clear()
    $env:G006_TEST_GPU_MIB = [string]$Samples[0]
    foreach ($sample in @($Samples | Select-Object -Skip 1)) {
        $global:G006TestGpuSamples.Enqueue($sample)
    }
}

New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
$mockBin = Join-Path $tempRoot 'bin'
New-Item -ItemType Directory -Path $mockBin -Force | Out-Null
$nvidiaSmiFixture = Join-Path $mockBin 'nvidia-smi.cmd'
[IO.File]::WriteAllLines(
    $nvidiaSmiFixture,
    @('@echo off', 'echo %G006_TEST_GPU_MIB%', 'exit /b 0'),
    [Text.Encoding]::ASCII
)
$env:PATH = $mockBin + [IO.Path]::PathSeparator + $originalPath
try {
    $eligibility = New-RevalidationFixture 'eligibility'
    $eligibilityReport = Get-Content $eligibility.ReportPath -Raw | ConvertFrom-Json
    $eligibilityReport.success_checks.tensorboard_exists = $false
    [IO.File]::WriteAllText($eligibility.ReportPath, ($eligibilityReport | ConvertTo-Json -Depth 12))
    $eligibilityError = Invoke-RevalidationFailure $eligibility.ReportPath
    Assert ($eligibilityError -match '^unsupported_failed_checks:') 'unsupported eligibility must fail closed'

    $tamper = New-RevalidationFixture 'tamper'
    [IO.File]::AppendAllText($tamper.Checkpoint, 'tampered')
    $tamperError = Invoke-RevalidationFailure $tamper.ReportPath
    Assert ($tamperError -eq 'checkpoint_hash_mismatch') 'checkpoint tamper must fail closed'

    $running = New-RevalidationFixture 'running'
    $global:G006TestProcesses = @([pscustomobject]@{
        Name = 'python.exe'
        CommandLine = 'python train.py --run-name g006_running'
        ProcessId = 4242
        ParentProcessId = 100
    })
    $runningError = Invoke-RevalidationFailure $running.ReportPath
    Assert ($runningError -eq 'training_process_still_running:4242') 'matching training process must block revalidation'

    $global:G006TestProcesses = @()
    $unstable = New-RevalidationFixture 'unstable'
    Set-GpuSamples @(1000, 1400, 1000)
    $unstableError = Invoke-RevalidationFailure $unstable.ReportPath
    Assert (
        $unstableError -eq 'gpu_memory_not_stable:min=1000;max=1400;tolerance=128'
    ) 'unstable GPU samples must fail closed'

    $stable = New-RevalidationFixture 'stable'
    Set-GpuSamples @(1010, 1010, 1010)
    & $scriptPath `
        -ReportPath $stable.ReportPath `
        -SampleCount 3 `
        -SampleIntervalSeconds 1
    $stableReport = Get-Content $stable.ReportPath -Raw | ConvertFrom-Json
    Assert ($stableReport.passed -eq $true) 'stable evidence must pass revalidation'
    Assert ($stableReport.gpu.recovery_revalidation.passed -eq $true) 'stable evidence must record attestation'

    Write-Host 'G006 GPU recovery revalidation assertions PASS'
}
finally {
    foreach ($mock in @(
        @{ Name = 'Get-CimInstance'; Previous = $previousCimFunction },
        @{ Name = 'Start-Sleep'; Previous = $previousSleepFunction }
    )) {
        $functionPath = 'Function:\global:' + $mock.Name
        if ($null -eq $mock.Previous) {
            Remove-Item $functionPath -ErrorAction SilentlyContinue
        }
        else {
            Set-Item $functionPath -Value $mock.Previous.ScriptBlock
        }
    }
    Remove-Variable G006TestProcesses -Scope Global -ErrorAction SilentlyContinue
    Remove-Variable G006TestGpuSamples -Scope Global -ErrorAction SilentlyContinue
    $env:PATH = $originalPath
    if ($null -eq $originalGpuMiB) {
        Remove-Item Env:G006_TEST_GPU_MIB -ErrorAction SilentlyContinue
    }
    else {
        $env:G006_TEST_GPU_MIB = $originalGpuMiB
    }
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}
