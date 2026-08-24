[CmdletBinding()]
param(
    [string]$IsaacLabPath = "$HOME\IsaacLab",

    [int]$Seed = 42,

    [ValidateRange(1, 1000)]
    [int]$MaxIterations = 10,

    [ValidateRange(1, 60)]
    [int]$GpuSampleIntervalSeconds = 2,

    [string]$SummaryPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Get-Median {
    param([object[]]$Values)

    if ($Values.Count -eq 0) {
        return $null
    }
    $sorted = @($Values | Sort-Object)
    $middle = [math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2 -eq 0) {
        return [math]::Round(($sorted[$middle - 1] + $sorted[$middle]) / 2.0, 2)
    }
    return [double]$sorted[$middle]
}

function Get-PortablePath {
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

function Get-TotalVramMiB {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        throw 'nvidia-smi를 찾을 수 없어 scale ladder를 시작하지 않습니다.'
    }
    $values = & $nvidiaSmi.Source --query-gpu=memory.total --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $values) {
        throw 'nvidia-smi에서 총 VRAM을 읽지 못해 scale ladder를 시작하지 않습니다.'
    }
    $total = 0
    foreach ($value in @($values)) {
        if ($value -notmatch '^\s*(\d+)\s*$') {
            throw "해석할 수 없는 VRAM 값: $value"
        }
        $total += [int]$Matches[1]
    }
    return $total
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$trainingHarness = Join-Path $PSScriptRoot 'run_training.ps1'
if (-not (Test-Path -LiteralPath $trainingHarness -PathType Leaf)) {
    throw "학습 하네스를 찾을 수 없습니다: $trainingHarness"
}
$pwshPath = (Get-Process -Id $PID).Path
if (
    [string]::IsNullOrWhiteSpace($pwshPath) -or
    -not (Test-Path -LiteralPath $pwshPath -PathType Leaf) -or
    [System.IO.Path]::GetFileName($pwshPath) -notin @('pwsh', 'pwsh.exe')
) {
    throw "현재 PowerShell 7 실행 파일을 확인할 수 없습니다: $pwshPath"
}

if ([string]::IsNullOrWhiteSpace($SummaryPath)) {
    $SummaryPath = Join-Path $repoRoot 'reports\runs\g004_go2_scale_summary.json'
}
$summaryFullPath = [System.IO.Path]::GetFullPath($SummaryPath)
New-Item -ItemType Directory -Path (Split-Path -Parent $summaryFullPath) -Force | Out-Null

$manifestPath = Join-Path $repoRoot 'reports\environment_manifest.json'
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "환경 매니페스트가 없습니다: $manifestPath"
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$manifestVramMiB = [int](($manifest.gpu.devices | Measure-Object -Property vram_mib -Sum).Sum)
$measuredTotalVramMiB = Get-TotalVramMiB
if ($manifestVramMiB -ne $measuredTotalVramMiB) {
    throw "환경 매니페스트 VRAM($manifestVramMiB MiB)과 nvidia-smi($measuredTotalVramMiB MiB)가 다릅니다."
}
if ($measuredTotalVramMiB -ne 12288) {
    throw "이 호스트의 검증 기준인 12288 MiB와 실제 총 VRAM($measuredTotalVramMiB MiB)이 다릅니다."
}

$safeLimitPercent = 80.0
$safeLimitMiB = [math]::Round($measuredTotalVramMiB * ($safeLimitPercent / 100.0), 1)
$mandatoryRungs = @(64, 256, 512, 1024, 2048)
$allRungs = [System.Collections.Generic.List[int]]::new()
foreach ($rung in $mandatoryRungs) {
    $allRungs.Add($rung)
}
$runStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$results = [System.Collections.Generic.List[object]]::new()
$stoppedAfterFailure = $false
$gate4096 = [ordered]@{
    evaluated = $false
    allowed = $false
    reason = '2048 environments 결과 대기'
}

for ($index = 0; $index -lt $allRungs.Count; $index++) {
    $numEnvs = $allRungs[$index]
    $runName = "go2_flat_scale_e${numEnvs}_i${MaxIterations}_s${Seed}_${runStamp}"
    $reportPath = Join-Path $repoRoot "reports\runs\$runName.json"
    Write-Host "[G004] rung 시작: envs=$numEnvs iterations=$MaxIterations seed=$Seed run=$runName"

    $processStartInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $processStartInfo.FileName = $pwshPath
    $processStartInfo.UseShellExecute = $false
    foreach ($argument in @(
        '-NoProfile',
        '-NonInteractive',
        '-File', $trainingHarness,
        '-Task', 'Isaac-Velocity-Flat-Unitree-Go2-v0',
        '-NumEnvs', $numEnvs,
        '-MaxIterations', $MaxIterations,
        '-Seed', $Seed,
        '-RunName', $runName,
        '-IsaacLabPath', $IsaacLabPath,
        '-ReportPath', $reportPath,
        '-GpuSampleIntervalSeconds', $GpuSampleIntervalSeconds
    )) {
        $processStartInfo.ArgumentList.Add([string]$argument)
    }

    $harnessLaunchError = $null
    try {
        $harnessProcess = [System.Diagnostics.Process]::Start($processStartInfo)
        $harnessProcess.WaitForExit()
        $harnessExitCode = $harnessProcess.ExitCode
        $harnessProcess.Dispose()
    }
    catch {
        $harnessExitCode = -1
        $harnessLaunchError = $_.Exception.Message
    }

    if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
        $results.Add([ordered]@{
            num_envs = $numEnvs
            run_name = $runName
            report = Get-PortablePath $reportPath
            harness_exit_code = $harnessExitCode
            passed = $false
            safe = $false
            performance = $null
            error = if ($harnessLaunchError) {
                "학습 하네스 자식 프로세스 시작 실패: $harnessLaunchError"
            }
            else {
                "학습 하네스가 JSON 보고서를 생성하지 않음 (exit=$harnessExitCode, IsaacLabPath=$(Get-PortablePath $IsaacLabPath)); 자식 프로세스 출력을 확인하세요."
            }
        })
        $stoppedAfterFailure = $true
        break
    }

    $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
    $samples = @($report.performance.steps_per_second_samples)
    $steadySamples = if ($samples.Count -gt 1) { @($samples | Select-Object -Skip 1) } else { @() }
    $peakPercent = [math]::Round(([double]$report.gpu.peak_used_mib / $measuredTotalVramMiB) * 100.0, 2)
    $safe = (
        [bool]$report.passed -and
        [bool]$report.gpu.measurement_complete -and
        [bool]$report.gpu.recovered_to_baseline -and
        [double]$report.gpu.peak_used_mib -le $safeLimitMiB
    )
    $result = [ordered]@{
        num_envs = $numEnvs
        run_name = $runName
        report = Get-PortablePath $reportPath
        harness_exit_code = $harnessExitCode
        passed = [bool]$report.passed
        safe = $safe
        wall_time_seconds = $report.wall_time_seconds
        gpu = [ordered]@{
            baseline_used_mib = $report.gpu.baseline_used_mib
            peak_used_mib = $report.gpu.peak_used_mib
            delta_used_mib = $report.gpu.delta_used_mib
            peak_percent_total = $peakPercent
            measurement_complete = [bool]$report.gpu.measurement_complete
            recovered_to_baseline = [bool]$report.gpu.recovered_to_baseline
        }
        performance = [ordered]@{
            samples = $samples
            mean_steps_per_second = $report.performance.mean_steps_per_second
            median_steps_per_second = $report.performance.median_steps_per_second
            steady_state_sample_count = $steadySamples.Count
            steady_state_median_steps_per_second = Get-Median $steadySamples
            final_mean_reward = $report.performance.final_mean_reward
            final_mean_episode_length = $report.performance.final_mean_episode_length
        }
        artifacts = [ordered]@{
            tensorboard_directory = $report.artifacts.tensorboard_directory
            checkpoint = $report.artifacts.checkpoint
            checkpoint_sha256 = $report.artifacts.checkpoint_sha256
        }
        error = if ([bool]$report.passed) { $null } else { '하네스 success_checks 실패; 개별 보고서 확인' }
    }
    $results.Add($result)
    Write-Host "[G004] rung 결과: envs=$numEnvs passed=$($result.passed) safe=$safe peak=$($report.gpu.peak_used_mib)MiB ($peakPercent%) median=$($report.performance.median_steps_per_second) steps/s"

    if (-not [bool]$report.passed -or $harnessExitCode -ne 0) {
        $stoppedAfterFailure = $true
        break
    }

    if ($numEnvs -eq 2048) {
        $gate4096.evaluated = $true
        $gate4096.allowed = $safe
        $gate4096.reason = if ($safe) {
            "2048 PASS, GPU 측정 complete/recovered, peak <= $safeLimitMiB MiB"
        }
        else {
            "2048 결과가 4096 게이트를 충족하지 않음: passed=$($report.passed), complete=$($report.gpu.measurement_complete), recovered=$($report.gpu.recovered_to_baseline), peak=$($report.gpu.peak_used_mib) MiB"
        }
        if ($safe) {
            $allRungs.Add(4096)
        }
    }
}

$passedRungs = @($results | Where-Object { $_.passed })
$safeRungs = @($results | Where-Object { $_.safe })
$highestOperational = if ($passedRungs.Count -gt 0) { [int](($passedRungs.num_envs | Measure-Object -Maximum).Maximum) } else { $null }
$highestSafe = if ($safeRungs.Count -gt 0) { [int](($safeRungs.num_envs | Measure-Object -Maximum).Maximum) } else { $null }

$summary = [ordered]@{
    schema_version = 1
    goal = 'G004'
    task = 'Isaac-Velocity-Flat-Unitree-Go2-v0'
    seed = $Seed
    max_iterations_per_rung = $MaxIterations
    run_name_rule = 'go2_flat_scale_e{num_envs}_i{iterations}_s{seed}_{yyyyMMdd-HHmmss}'
    total_vram = [ordered]@{
        nvidia_smi_mib = $measuredTotalVramMiB
        environment_manifest_mib = $manifestVramMiB
        values_match = ($measuredTotalVramMiB -eq $manifestVramMiB)
        safe_limit_percent = $safeLimitPercent
        safe_limit_mib = $safeLimitMiB
    }
    highest_operational = $highestOperational
    highest_safe = $highestSafe
    stopped_after_failure = $stoppedAfterFailure
    gate_4096 = $gate4096
    mujoco_reference = [ordered]@{
        user_provided_steps_per_second = 51000
        comparable_benchmark = $false
        warning = '사용자 제공 MuJoCo 51k steps/s는 환경, 물리 설정, rollout 길이와 측정법이 통제된 동일 조건 벤치마크가 아니므로 직접 성능 비교가 불가능하다. 아래 비율은 참고 계산일 뿐 우열의 근거가 아니다.'
        ratios = @($results | Where-Object { $null -ne $_.performance } | ForEach-Object {
            [ordered]@{
                num_envs = $_.num_envs
                isaac_mean_divided_by_reference = [math]::Round([double]$_.performance.mean_steps_per_second / 51000.0, 6)
                isaac_steady_median_divided_by_reference = if ($null -ne $_.performance.steady_state_median_steps_per_second) {
                    [math]::Round([double]$_.performance.steady_state_median_steps_per_second / 51000.0, 6)
                } else { $null }
            }
        })
    }
    runs = $results
}

$summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $summaryFullPath -Encoding utf8
Write-Host "[G004] summary: $(Get-PortablePath $summaryFullPath)"
Write-Host "[G004] highest_operational=$highestOperational highest_safe=$highestSafe stopped_after_failure=$stoppedAfterFailure"

if ($stoppedAfterFailure) {
    exit 1
}
exit 0
