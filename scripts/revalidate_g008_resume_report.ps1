[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ReportPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Resolve-PortablePath {
    param([Parameter(Mandatory = $true)][string]$Path)

    if ($Path.StartsWith('%USERPROFILE%', [System.StringComparison]::OrdinalIgnoreCase)) {
        return [System.IO.Path]::GetFullPath($HOME + $Path.Substring('%USERPROFILE%'.Length))
    }
    return [System.IO.Path]::GetFullPath($Path)
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$fullReportPath = [System.IO.Path]::GetFullPath($ReportPath)
if (-not $fullReportPath.StartsWith($repoRoot + '\', [System.StringComparison]::OrdinalIgnoreCase) -or
    [System.IO.Path]::GetExtension($fullReportPath) -ne '.json') {
    throw "ReportPath는 저장소 안의 JSON이어야 합니다: $fullReportPath"
}
if (-not (Test-Path -LiteralPath $fullReportPath -PathType Leaf)) {
    throw "보고서가 없습니다: $fullReportPath"
}

$report = Get-Content -LiteralPath $fullReportPath -Raw | ConvertFrom-Json
if ($report.schema_version -ne 1 -or $report.resume.enabled -ne $true) {
    throw 'schema_version 1의 resume 보고서가 필요합니다.'
}
if ($report.resume.checkpoint -notmatch '^model_([0-9]+)\.pt$') {
    throw 'resume.checkpoint 형식이 잘못됐습니다.'
}
$checkpointIteration = [int]$Matches[1]
# RSL-RL resumes from the loaded iteration inclusively.
$expectedIteration = $checkpointIteration + [int]$report.max_iterations - 1

$stdoutPath = Resolve-PortablePath $report.artifacts.raw_stdout
$checkpointPath = Resolve-PortablePath $report.artifacts.checkpoint
$tensorboardPath = Resolve-PortablePath $report.artifacts.tensorboard_directory
$sourceCheckpointPath = Join-Path (Split-Path -Parent $tensorboardPath) (
    Join-Path ([string]$report.resume.load_run) ([string]$report.resume.checkpoint)
)
if (-not (Test-Path -LiteralPath $stdoutPath -PathType Leaf)) {
    throw "stdout가 없습니다: $stdoutPath"
}
if (-not (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
    throw "checkpoint가 없습니다: $checkpointPath"
}
if (-not (Test-Path -LiteralPath $tensorboardPath -PathType Container)) {
    throw "TensorBoard 디렉터리가 없습니다: $tensorboardPath"
}
if (-not (Test-Path -LiteralPath $sourceCheckpointPath -PathType Leaf)) {
    throw "resume 원본 checkpoint가 없습니다: $sourceCheckpointPath"
}

$stdout = Get-Content -LiteralPath $stdoutPath -Raw
$iterationMatches = [regex]::Matches($stdout, 'Learning iteration\s+(\d+)/(\d+)')
if ($iterationMatches.Count -eq 0) {
    throw 'stdout에서 learning iteration을 찾지 못했습니다.'
}
$lastMatch = $iterationMatches[$iterationMatches.Count - 1]
$lastIteration = [int]$lastMatch.Groups[1].Value
$iterationTarget = [int]$lastMatch.Groups[2].Value
$checkpointHash = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
$sourceCheckpointHash = (Get-FileHash -LiteralPath $sourceCheckpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
$tensorboardExists = $null -ne (
    Get-ChildItem -LiteralPath $tensorboardPath -Filter 'events.out.tfevents.*' -File | Select-Object -First 1
)

$iterationPass = $lastIteration -eq $expectedIteration -and $iterationTarget -eq ($expectedIteration + 1)
$checkpointPass = $checkpointHash -eq $report.artifacts.checkpoint_sha256
if (-not $iterationPass -or -not $checkpointPass -or -not $tensorboardExists) {
    throw (
        'resume 재검증 실패: iteration={0}/{1} expected={2}, checkpoint_hash={3}, tensorboard={4}' -f
        $lastIteration, $iterationTarget, $expectedIteration, $checkpointPass, $tensorboardExists
    )
}

$report.last_iteration = $lastIteration
$report.iteration_target = $iterationTarget
$report.success_checks.requested_iteration_reached = $true
$report.passed = -not (@($report.success_checks.psobject.Properties.Value) -contains $false)
$report | Add-Member -Force -NotePropertyName resume_revalidation -NotePropertyValue ([ordered]@{
    status = if ($report.passed) { 'pass' } else { 'fail' }
    expected_iteration = $expectedIteration
    stdout_iteration = $lastIteration
    stdout_iteration_target = $iterationTarget
    checkpoint_sha256_verified = $checkpointPass
    source_checkpoint = [ordered]@{
        path = '%USERPROFILE%' + $sourceCheckpointPath.Substring(
            [System.IO.Path]::GetFullPath($HOME).TrimEnd('\').Length
        )
        sha256 = $sourceCheckpointHash
    }
    tensorboard_exists = $tensorboardExists
    validator_sha256 = (Get-FileHash -LiteralPath $MyInvocation.MyCommand.Path -Algorithm SHA256).Hash.ToLowerInvariant()
    revalidated_at = (Get-Date).ToString('o')
})

$temporary = Join-Path (Split-Path -Parent $fullReportPath) ('.' + [System.IO.Path]::GetFileName($fullReportPath) + '.tmp')
try {
    [System.IO.File]::WriteAllText(
        $temporary,
        ($report | ConvertTo-Json -Depth 10),
        [System.Text.UTF8Encoding]::new($false)
    )
    [System.IO.File]::Move($temporary, $fullReportPath, $true)
}
finally {
    if (Test-Path -LiteralPath $temporary -PathType Leaf) {
        Remove-Item -LiteralPath $temporary -Force
    }
}

Write-Host "G008 resume report revalidation: passed=$($report.passed) iteration=$lastIteration/$iterationTarget"
if (-not $report.passed) {
    exit 1
}
