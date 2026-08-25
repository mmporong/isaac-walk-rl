[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ReportPath,

    [ValidateRange(3, 60)]
    [int]$SampleCount = 5,

    [ValidateRange(1, 60)]
    [int]$SampleIntervalSeconds = 1,

    [ValidateRange(0, 1024)]
    [int]$StabilityToleranceMiB = 128
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Resolve-PortablePath {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw 'artifact_path_missing'
    }
    if ($Path.StartsWith('%USERPROFILE%', [System.StringComparison]::OrdinalIgnoreCase)) {
        return [System.IO.Path]::GetFullPath($HOME + $Path.Substring('%USERPROFILE%'.Length))
    }
    return [System.IO.Path]::GetFullPath($Path)
}

function Get-GpuMemoryUsedMiB {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        throw 'nvidia_smi_missing'
    }

    $output = @(& $nvidiaSmi.Source --query-gpu=memory.used --format=csv,noheader,nounits 2>$null)
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia_smi_failed:$LASTEXITCODE"
    }

    $total = 0
    $parsed = 0
    foreach ($line in $output) {
        if ($line -match '^\s*(\d+)\s*$') {
            $total += [int]$Matches[1]
            $parsed++
        }
    }
    if ($parsed -eq 0) {
        throw 'nvidia_smi_unparseable'
    }
    return $total
}

$repoRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$reportRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot 'reports\runs')).TrimEnd('\') + '\'
$reportFullPath = [System.IO.Path]::GetFullPath($ReportPath)
if (-not $reportFullPath.StartsWith($reportRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "report_outside_canonical_root:$reportFullPath"
}
if (-not (Test-Path -LiteralPath $reportFullPath -PathType Leaf)) {
    throw "report_missing:$reportFullPath"
}

$originalReportSha256 = (Get-FileHash -LiteralPath $reportFullPath -Algorithm SHA256).Hash.ToLowerInvariant()
$report = Get-Content -LiteralPath $reportFullPath -Raw | ConvertFrom-Json
$failedChecks = @(
    $report.success_checks.PSObject.Properties |
        Where-Object { $_.Value -ne $true } |
        ForEach-Object { $_.Name }
)
if ($failedChecks.Count -ne 1 -or $failedChecks[0] -ne 'gpu_recovered_to_baseline') {
    throw "unsupported_failed_checks:$($failedChecks -join ',')"
}
if (
    [int]$report.exit_code -ne 0 -or
    [int]$report.last_iteration -ne ([int]$report.iteration_target - 1) -or
    @($report.fatal_patterns_found).Count -ne 0 -or
    $report.gpu.measurement_complete -ne $true -or
    $report.gpu.recovered_to_baseline -ne $false
) {
    throw 'training_evidence_not_eligible_for_gpu_revalidation'
}

$checkpoint = Resolve-PortablePath ([string]$report.artifacts.checkpoint)
$stdout = Resolve-PortablePath ([string]$report.artifacts.raw_stdout)
$stderr = Resolve-PortablePath ([string]$report.artifacts.raw_stderr)
$tensorboard = Resolve-PortablePath ([string]$report.artifacts.tensorboard_directory)
foreach ($artifact in @($checkpoint, $stdout, $stderr)) {
    if (-not (Test-Path -LiteralPath $artifact -PathType Leaf)) {
        throw "artifact_missing:$artifact"
    }
}
if (-not (Test-Path -LiteralPath $tensorboard -PathType Container)) {
    throw "tensorboard_missing:$tensorboard"
}
$checkpointHash = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
if ($checkpointHash -ne [string]$report.artifacts.checkpoint_sha256) {
    throw 'checkpoint_hash_mismatch'
}

$fatalPatterns = @(
    'Traceback (most recent call last)',
    '[Error]',
    'CUDA out of memory',
    'OutOfMemoryError',
    'Fatal Python error',
    'Segmentation fault'
)
$stdoutText = [string](Get-Content -LiteralPath $stdout -Raw)
$stderrText = [string](Get-Content -LiteralPath $stderr -Raw)
$combined = $stdoutText + [Environment]::NewLine + $stderrText
$fatalMatches = @($fatalPatterns | Where-Object { $combined.Contains($_) })
if ($fatalMatches.Count -ne 0) {
    throw "fatal_training_log:$($fatalMatches -join ',')"
}

$runNamePattern = [regex]::Escape([string]$report.run_name)
$matchingProcesses = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match '^(python|kit)(\.exe)?$' -and
            [string]$_.CommandLine -match $runNamePattern
        } |
        ForEach-Object {
            [ordered]@{
                process_id = [int]$_.ProcessId
                parent_process_id = [int]$_.ParentProcessId
                name = [string]$_.Name
                command_line = [string]$_.CommandLine
            }
        }
)
if ($matchingProcesses.Count -ne 0) {
    throw "training_process_still_running:$($matchingProcesses.process_id -join ',')"
}

$samples = [System.Collections.Generic.List[object]]::new()
for ($index = 0; $index -lt $SampleCount; $index++) {
    $samples.Add([ordered]@{
        captured_at = (Get-Date).ToString('o')
        used_mib = Get-GpuMemoryUsedMiB
    })
    if ($index -lt ($SampleCount - 1)) {
        Start-Sleep -Seconds $SampleIntervalSeconds
    }
}
$usedValues = @($samples | ForEach-Object { [int]$_.used_mib })
$minimumUsedMiB = [int](($usedValues | Measure-Object -Minimum).Minimum)
$maximumUsedMiB = [int](($usedValues | Measure-Object -Maximum).Maximum)
$stable = ($maximumUsedMiB - $minimumUsedMiB) -le $StabilityToleranceMiB
if (-not $stable) {
    throw "gpu_memory_not_stable:min=$minimumUsedMiB;max=$maximumUsedMiB;tolerance=$StabilityToleranceMiB"
}

$initialFinalUsedMiB = if (@($report.gpu.recovery_samples).Count -gt 0) {
    [int]$report.gpu.recovery_samples[-1].used_mib
}
else {
    $null
}
$attestation = [ordered]@{
    schema_version = 1
    reason = 'aggregate_gpu_baseline_shift_recheck'
    initial_baseline_used_mib = [int]$report.gpu.baseline_used_mib
    initial_final_used_mib = $initialFinalUsedMiB
    initial_recovered_to_baseline = $false
    recheck_samples = $samples
    recheck_min_used_mib = $minimumUsedMiB
    recheck_max_used_mib = $maximumUsedMiB
    stability_tolerance_mib = $StabilityToleranceMiB
    matching_training_processes = $matchingProcesses
    checkpoint_sha256 = [string]$report.artifacts.checkpoint_sha256
    original_report_sha256 = $originalReportSha256
    raw_logs_fatal_patterns = $fatalMatches
    script_path = '%USERPROFILE%\isaac-walk-rl\scripts\revalidate_training_gpu_recovery.ps1'
    script_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    passed = $true
    revalidated_at = (Get-Date).ToString('o')
}
$report.gpu | Add-Member -NotePropertyName recovery_revalidation -NotePropertyValue $attestation -Force
$report.gpu.recovered_to_baseline = $true
$report.success_checks.gpu_recovered_to_baseline = $true
$report.passed = $true

$tempPath = $reportFullPath + '.' + [guid]::NewGuid().ToString('N') + '.tmp'
try {
    [System.IO.File]::WriteAllText(
        $tempPath,
        ($report | ConvertTo-Json -Depth 30),
        [System.Text.UTF8Encoding]::new($false)
    )
    [System.IO.File]::Move($tempPath, $reportFullPath, $true)
}
finally {
    if (Test-Path -LiteralPath $tempPath -PathType Leaf) {
        [System.IO.File]::Delete($tempPath)
    }
}

Write-Host (
    "Training GPU recovery revalidation PASS: " +
    "run=$($report.run_name) min_mib=$minimumUsedMiB max_mib=$maximumUsedMiB"
)
