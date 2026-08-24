[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Task,

    [ValidateRange(1, 65536)]
    [int]$NumEnvs,

    [ValidateRange(1, 1000000)]
    [int]$MaxIterations,

    [int]$Seed = 42,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z0-9][a-z0-9_-]+$')]
    [string]$RunName,

    [string]$IsaacLabPath = "$HOME\IsaacLab",

    [string]$ReportPath,

    [ValidateRange(1, 60)]
    [int]$GpuSampleIntervalSeconds = 2
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Convert-ToPortablePath {
    param([AllowNull()][string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }

    $homePath = [System.IO.Path]::GetFullPath($HOME).TrimEnd('\')
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if (
        $fullPath.Equals($homePath, [System.StringComparison]::OrdinalIgnoreCase) -or
        $fullPath.StartsWith($homePath + '\', [System.StringComparison]::OrdinalIgnoreCase)
    ) {
        return '%USERPROFILE%' + $fullPath.Substring($homePath.Length)
    }
    return $fullPath
}

function Get-GpuMemoryUsedMiB {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        return $null
    }

    $sample = & $nvidiaSmi.Source --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $sample) {
        return $null
    }

    $total = 0
    $parsedCount = 0
    foreach ($line in @($sample)) {
        if ($line -match '^\s*(\d+)\s*$') {
            $total += [int]$Matches[1]
            $parsedCount++
        }
    }
    if ($parsedCount -eq 0) {
        return $null
    }
    return $total
}

function Convert-ToWindowsCommandLineArgument {
    param([AllowEmptyString()][string]$Argument)

    if ($Argument -notmatch '[\s"]') {
        return $Argument
    }

    $builder = [System.Text.StringBuilder]::new()
    [void]$builder.Append('"')
    $backslashCount = 0
    foreach ($character in $Argument.ToCharArray()) {
        if ($character -eq '\') {
            $backslashCount++
            continue
        }

        if ($character -eq '"') {
            [void]$builder.Append(('\' * (($backslashCount * 2) + 1)))
            [void]$builder.Append('"')
        }
        else {
            if ($backslashCount -gt 0) {
                [void]$builder.Append(('\' * $backslashCount))
            }
            [void]$builder.Append($character)
        }
        $backslashCount = 0
    }
    if ($backslashCount -gt 0) {
        [void]$builder.Append(('\' * ($backslashCount * 2)))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Get-LastMatchValue {
    param(
        [string]$Text,
        [string]$Pattern,
        [int]$Group = 1
    )

    $matches = [regex]::Matches($Text, $Pattern, [System.Text.RegularExpressions.RegexOptions]::Multiline)
    if ($matches.Count -eq 0) {
        return $null
    }
    return $matches[$matches.Count - 1].Groups[$Group].Value
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$isaacLabFullPath = [System.IO.Path]::GetFullPath($IsaacLabPath)
$pythonBat = Join-Path $isaacLabFullPath '_isaac_sim\python.bat'
$trainScript = Join-Path $isaacLabFullPath 'scripts\reinforcement_learning\rsl_rl\train.py'
$rawLogRoot = Join-Path $isaacLabFullPath 'logs\harness'

if (-not (Test-Path -LiteralPath $pythonBat -PathType Leaf)) {
    throw "Isaac Sim bundled python.bat을 찾을 수 없습니다: $pythonBat"
}
if (-not (Test-Path -LiteralPath $trainScript -PathType Leaf)) {
    throw "RSL-RL train.py를 찾을 수 없습니다: $trainScript"
}

if ([string]::IsNullOrWhiteSpace($ReportPath)) {
    $ReportPath = Join-Path $repoRoot "reports\runs\$RunName.json"
}
$reportFullPath = [System.IO.Path]::GetFullPath($ReportPath)
$reportDirectory = Split-Path -Parent $reportFullPath
New-Item -ItemType Directory -Path $reportDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $rawLogRoot -Force | Out-Null

$stdoutPath = Join-Path $rawLogRoot "$RunName.stdout.log"
$stderrPath = Join-Path $rawLogRoot "$RunName.stderr.log"
Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue

$arguments = @(
    $trainScript,
    '--task', $Task,
    '--num_envs', $NumEnvs,
    '--max_iterations', $MaxIterations,
    '--seed', $Seed,
    '--run_name', $RunName,
    '--headless'
)
$argumentLine = ($arguments | ForEach-Object {
    Convert-ToWindowsCommandLineArgument ([string]$_)
}) -join ' '

$baselineGpuMiB = Get-GpuMemoryUsedMiB
if ($null -eq $baselineGpuMiB) {
    throw 'GPU 메모리 baseline 측정 실패: nvidia-smi 실행 가능 여부와 출력을 확인하세요. 학습은 시작되지 않았습니다.'
}
$gpuSamples = [System.Collections.Generic.List[object]]::new()
$gpuMeasurementFailureCount = 0
$startedAt = Get-Date
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

$process = Start-Process -FilePath $pythonBat `
    -ArgumentList $argumentLine `
    -WorkingDirectory $isaacLabFullPath `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden `
    -PassThru

while (-not $process.HasExited) {
    $usedMiB = Get-GpuMemoryUsedMiB
    if ($null -eq $usedMiB) {
        $gpuMeasurementFailureCount++
    }
    $gpuSamples.Add([ordered]@{
        elapsed_seconds = [math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
        used_mib = $usedMiB
    })
    Start-Sleep -Seconds $GpuSampleIntervalSeconds
    $process.Refresh()
}
$process.WaitForExit()
$stopwatch.Stop()
$endedAt = Get-Date

$gpuRecoverySamples = [System.Collections.Generic.List[object]]::new()
$gpuRecoveryDeadline = (Get-Date).AddSeconds(30)
do {
    $recoveryUsedMiB = Get-GpuMemoryUsedMiB
    if ($null -eq $recoveryUsedMiB) {
        $gpuMeasurementFailureCount++
    }
    $gpuRecoverySamples.Add([ordered]@{
        elapsed_seconds = [math]::Round(((Get-Date) - $endedAt).TotalSeconds, 3)
        used_mib = $recoveryUsedMiB
    })
    if ($null -ne $recoveryUsedMiB -and $recoveryUsedMiB -le ($baselineGpuMiB + 128)) {
        break
    }
    Start-Sleep -Seconds 1
} while ((Get-Date) -lt $gpuRecoveryDeadline)

$stdout = if (Test-Path -LiteralPath $stdoutPath) { Get-Content -LiteralPath $stdoutPath -Raw } else { '' }
$stderr = if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath -Raw } else { '' }
$combined = $stdout + [Environment]::NewLine + $stderr

$logRootMatch = Get-LastMatchValue -Text $combined -Pattern '\[INFO\] Logging experiment in directory:\s*(.+?)\s*$'
$timestampMatch = Get-LastMatchValue -Text $combined -Pattern '^Exact experiment name requested from command line:\s*(\S+)\s*$'
$actualLogDirectory = $null
if ($logRootMatch -and $timestampMatch) {
    $actualLogDirectory = Join-Path $logRootMatch.Trim() ($timestampMatch.Trim() + '_' + $RunName)
}

$iterationMatches = [regex]::Matches($combined, 'Learning iteration\s+(\d+)/(\d+)')
$lastIteration = if ($iterationMatches.Count -gt 0) { [int]$iterationMatches[$iterationMatches.Count - 1].Groups[1].Value } else { $null }
$iterationTarget = if ($iterationMatches.Count -gt 0) { [int]$iterationMatches[$iterationMatches.Count - 1].Groups[2].Value } else { $null }

$stepsPerSecond = @([regex]::Matches($combined, 'Computation:\s+(\d+)\s+steps/s') | ForEach-Object { [int]$_.Groups[1].Value })
$meanStepsPerSecond = if ($stepsPerSecond.Count -gt 0) { [math]::Round(($stepsPerSecond | Measure-Object -Average).Average, 2) } else { $null }
$medianStepsPerSecond = $null
if ($stepsPerSecond.Count -gt 0) {
    $sortedSteps = @($stepsPerSecond | Sort-Object)
    $middle = [math]::Floor($sortedSteps.Count / 2)
    if ($sortedSteps.Count % 2 -eq 0) {
        $medianStepsPerSecond = [math]::Round(($sortedSteps[$middle - 1] + $sortedSteps[$middle]) / 2.0, 2)
    }
    else {
        $medianStepsPerSecond = $sortedSteps[$middle]
    }
}

$finalRewardText = Get-LastMatchValue -Text $combined -Pattern '^\s*Mean reward:\s+(-?[0-9.]+)\s*$'
$finalEpisodeLengthText = Get-LastMatchValue -Text $combined -Pattern '^\s*Mean episode length:\s+([0-9.]+)\s*$'
$finalReward = if ($null -ne $finalRewardText) { [double]::Parse($finalRewardText, [System.Globalization.CultureInfo]::InvariantCulture) } else { $null }
$finalEpisodeLength = if ($null -ne $finalEpisodeLengthText) { [double]::Parse($finalEpisodeLengthText, [System.Globalization.CultureInfo]::InvariantCulture) } else { $null }

$checkpoint = $null
if ($actualLogDirectory -and (Test-Path -LiteralPath $actualLogDirectory -PathType Container)) {
    $checkpoint = Get-ChildItem -LiteralPath $actualLogDirectory -Filter 'model_*.pt' -File |
        Sort-Object LastWriteTimeUtc |
        Select-Object -Last 1
}
$checkpointHash = if ($checkpoint) { (Get-FileHash -LiteralPath $checkpoint.FullName -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
$tensorboardExists = $false
if ($actualLogDirectory -and (Test-Path -LiteralPath $actualLogDirectory -PathType Container)) {
    $tensorboardExists = $null -ne (Get-ChildItem -LiteralPath $actualLogDirectory -Filter 'events.out.tfevents.*' -File | Select-Object -First 1)
}

$gpuValues = @($gpuSamples | ForEach-Object { $_.used_mib } | Where-Object { $null -ne $_ })
$peakGpuMiB = if ($gpuValues.Count -gt 0) { ($gpuValues | Measure-Object -Maximum).Maximum } else { $baselineGpuMiB }
$finalGpuMiB = if ($gpuRecoverySamples.Count -gt 0) { $gpuRecoverySamples[$gpuRecoverySamples.Count - 1].used_mib } else { Get-GpuMemoryUsedMiB }
$gpuMeasurementComplete = ($gpuMeasurementFailureCount -eq 0 -and $null -ne $finalGpuMiB)
$recoveredToBaseline = ($gpuMeasurementComplete -and $finalGpuMiB -le ($baselineGpuMiB + 128))
$fatalPatterns = @('Traceback (most recent call last)', '[Error]')
$fatalMatches = @($fatalPatterns | Where-Object { $combined.Contains($_) })
$expectedLastIteration = $MaxIterations - 1
$successChecks = [ordered]@{
    process_exit_zero = ($process.ExitCode -eq 0)
    no_traceback_or_error = ($fatalMatches.Count -eq 0)
    requested_iteration_reached = ($lastIteration -eq $expectedLastIteration -and $iterationTarget -eq $MaxIterations)
    log_directory_exists = ($actualLogDirectory -and (Test-Path -LiteralPath $actualLogDirectory -PathType Container))
    tensorboard_exists = $tensorboardExists
    checkpoint_exists = ($null -ne $checkpoint)
    gpu_measurement_complete = $gpuMeasurementComplete
    gpu_recovered_to_baseline = $recoveredToBaseline
}
$passed = -not ($successChecks.Values -contains $false)

$report = [ordered]@{
    schema_version = 1
    run_name = $RunName
    task = $Task
    num_envs = $NumEnvs
    max_iterations = $MaxIterations
    seed = $Seed
    headless = $true
    command = @(
        (Convert-ToPortablePath $pythonBat),
        (Convert-ToPortablePath $trainScript),
        '--task', $Task,
        '--num_envs', $NumEnvs,
        '--max_iterations', $MaxIterations,
        '--seed', $Seed,
        '--run_name', $RunName,
        '--headless'
    )
    started_at = $startedAt.ToString('o')
    ended_at = $endedAt.ToString('o')
    wall_time_seconds = [math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
    exit_code = $process.ExitCode
    last_iteration = $lastIteration
    iteration_target = $iterationTarget
    gpu = [ordered]@{
        baseline_used_mib = $baselineGpuMiB
        peak_used_mib = $peakGpuMiB
        delta_used_mib = $peakGpuMiB - $baselineGpuMiB
        sample_interval_seconds = $GpuSampleIntervalSeconds
        samples = $gpuSamples
        recovery_samples = $gpuRecoverySamples
        measurement_failure_count = $gpuMeasurementFailureCount
        measurement_complete = $gpuMeasurementComplete
        recovered_to_baseline = $recoveredToBaseline
    }
    performance = [ordered]@{
        metric_source = 'stdout; TensorBoard event file existence verified'
        steps_per_second_samples = $stepsPerSecond
        mean_steps_per_second = $meanStepsPerSecond
        median_steps_per_second = $medianStepsPerSecond
        final_mean_reward = $finalReward
        final_mean_episode_length = $finalEpisodeLength
    }
    artifacts = [ordered]@{
        raw_stdout = Convert-ToPortablePath $stdoutPath
        raw_stderr = Convert-ToPortablePath $stderrPath
        tensorboard_directory = Convert-ToPortablePath $actualLogDirectory
        checkpoint = if ($checkpoint) { Convert-ToPortablePath $checkpoint.FullName } else { $null }
        checkpoint_sha256 = $checkpointHash
    }
    fatal_patterns_found = $fatalMatches
    success_checks = $successChecks
    passed = $passed
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportFullPath -Encoding utf8
Write-Host "Report: $(Convert-ToPortablePath $reportFullPath)"
Write-Host "Result: passed=$passed exit=$($process.ExitCode) iteration=$lastIteration/$iterationTarget peak_vram_mib=$peakGpuMiB"

if (-not $passed) {
    exit 1
}
