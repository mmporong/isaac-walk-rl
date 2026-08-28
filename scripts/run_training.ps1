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

    [ValidateNotNull()]
    [string[]]$HydraOverrides = @(),

    [string]$HydraOverridesBase64,

    [string]$TrainingEntrypointPath,

    [ValidateNotNull()]
    [string[]]$SourceBindingPaths = @(),

    [switch]$Qualification,

    [switch]$Resume,

    [string]$LoadRun,

    [string]$ResumeCheckpoint,

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

function Get-TextSha256 {
    param([Parameter(Mandatory = $true)][string]$Text)

    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
        return ([System.BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$isaacLabFullPath = [System.IO.Path]::GetFullPath($IsaacLabPath)
$pythonBat = Join-Path $isaacLabFullPath '_isaac_sim\python.bat'
$trainScript = Join-Path $isaacLabFullPath 'scripts\reinforcement_learning\rsl_rl\train.py'
$rawLogRoot = Join-Path $isaacLabFullPath 'logs\harness'

if (-not [string]::IsNullOrWhiteSpace($TrainingEntrypointPath)) {
    $candidateEntrypoint = [System.IO.Path]::GetFullPath($TrainingEntrypointPath)
    $repoBoundary = [System.IO.Path]::GetFullPath($repoRoot).TrimEnd('\') + '\'
    if (-not $candidateEntrypoint.StartsWith($repoBoundary, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "TrainingEntrypointPath는 저장소 내부 파일이어야 합니다: $candidateEntrypoint"
    }
    $trainScript = $candidateEntrypoint
}

if (-not [string]::IsNullOrWhiteSpace($HydraOverridesBase64)) {
    if ($HydraOverrides.Count -gt 0) {
        throw 'HydraOverrides와 HydraOverridesBase64는 동시에 사용할 수 없습니다.'
    }
    try {
        $decodedOverrideJson = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($HydraOverridesBase64))
        $HydraOverrides = @($decodedOverrideJson | ConvertFrom-Json)
    }
    catch {
        throw "Hydra override Base64/JSON 해석 실패: $($_.Exception.Message)"
    }
}

foreach ($override in $HydraOverrides) {
    if ([string]::IsNullOrWhiteSpace($override) -or $override.Contains("`r") -or $override.Contains("`n")) {
        throw 'Hydra override는 비어 있거나 줄바꿈을 포함할 수 없습니다.'
    }
    if ($override -notmatch '^[A-Za-z0-9_.:/+-]+=[A-Za-z0-9_.:/+-]+$' -or
        $override.IndexOf('=') -ne $override.LastIndexOf('=')) {
        throw "안전하지 않은 Hydra override 형식입니다: $override"
    }
}

if ($Resume) {
    if ([string]::IsNullOrWhiteSpace($LoadRun) -or $LoadRun -notmatch '^[A-Za-z0-9_.-]+$') {
        throw 'Resume에는 안전한 LoadRun 이름이 필요합니다.'
    }
    if ([string]::IsNullOrWhiteSpace($ResumeCheckpoint) -or $ResumeCheckpoint -notmatch '^model_[0-9]+\.pt$') {
        throw 'ResumeCheckpoint는 model_<iteration>.pt 형식이어야 합니다.'
    }
}
elseif (-not [string]::IsNullOrWhiteSpace($LoadRun) -or -not [string]::IsNullOrWhiteSpace($ResumeCheckpoint)) {
    throw 'LoadRun과 ResumeCheckpoint는 -Resume과 함께 사용해야 합니다.'
}
if ($Qualification -and $Resume) {
    throw 'Qualification 학습은 scratch 실행만 허용합니다.'
}
if ($Qualification -and $HydraOverrides.Count -gt 0) {
    throw 'Qualification 학습은 런타임 의미를 바꾸는 Hydra override를 허용하지 않습니다.'
}
$g009QualificationTask = 'Isaac-G009-Recover-Flat-Go2-R0-v0'
if (
    $Qualification -and
    $Task -eq $g009QualificationTask -and
    ($NumEnvs -ne 1024 -or $MaxIterations -ne 300 -or $Seed -ne 42)
) {
    throw 'G009 R0 Qualification은 num_envs=1024, max_iterations=300, seed=42만 허용합니다.'
}

if (-not (Test-Path -LiteralPath $pythonBat -PathType Leaf)) {
    throw "Isaac Sim bundled python.bat을 찾을 수 없습니다: $pythonBat"
}
if (-not (Test-Path -LiteralPath $trainScript -PathType Leaf)) {
    throw "RSL-RL train.py를 찾을 수 없습니다: $trainScript"
}
$trainingEntrypointHash = (Get-FileHash -LiteralPath $trainScript -Algorithm SHA256).Hash.ToLowerInvariant()
$sourceBindingFiles = [ordered]@{}
$repoBoundary = [System.IO.Path]::GetFullPath($repoRoot).TrimEnd('\') + '\'
foreach ($sourcePath in @($SourceBindingPaths | Sort-Object -Unique)) {
    $fullSourcePath = if ([System.IO.Path]::IsPathRooted($sourcePath)) {
        [System.IO.Path]::GetFullPath($sourcePath)
    }
    else {
        [System.IO.Path]::GetFullPath((Join-Path $repoRoot $sourcePath))
    }
    if (-not $fullSourcePath.StartsWith($repoBoundary, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "SourceBindingPaths는 저장소 내부 파일이어야 합니다: $fullSourcePath"
    }
    if (-not (Test-Path -LiteralPath $fullSourcePath -PathType Leaf)) {
        throw "SourceBindingPaths 파일을 찾을 수 없습니다: $fullSourcePath"
    }
    $relativeSourcePath = [System.IO.Path]::GetRelativePath($repoRoot, $fullSourcePath).Replace('\', '/')
    $sourceBindingFiles[$relativeSourcePath] = (
        Get-FileHash -LiteralPath $fullSourcePath -Algorithm SHA256
    ).Hash.ToLowerInvariant()
}
$sourceBundlePayload = (
    $sourceBindingFiles.GetEnumerator() | ForEach-Object { "$($_.Key):$($_.Value)" }
) -join "`n"
$sourceBundleHash = if ($sourceBindingFiles.Count -gt 0) {
    Get-TextSha256 $sourceBundlePayload
}
else {
    $null
}
$repositoryCommit = $null
$repositoryDirty = $null
$sourceBundleMatchesHead = $null
$gitCommand = Get-Command git -ErrorAction SilentlyContinue
if ($null -ne $gitCommand) {
    $commitOutput = & $gitCommand.Source -C $repoRoot rev-parse HEAD 2>$null
    if ($LASTEXITCODE -eq 0 -and $commitOutput) {
        $repositoryCommit = ([string]$commitOutput).Trim()
        $statusOutput = @(& $gitCommand.Source -C $repoRoot status --porcelain=v1 2>$null)
        if ($LASTEXITCODE -eq 0) {
            $repositoryDirty = $statusOutput.Count -gt 0
        }
    }
    if ($sourceBindingFiles.Count -gt 0 -and $null -ne $repositoryCommit) {
        $allSourceFilesTracked = $true
        foreach ($relativeSourcePath in $sourceBindingFiles.Keys) {
            & $gitCommand.Source -C $repoRoot ls-files --error-unmatch -- $relativeSourcePath *> $null
            if ($LASTEXITCODE -ne 0) {
                $allSourceFilesTracked = $false
                break
            }
        }
        if ($allSourceFilesTracked) {
            $diffArguments = @('-C', $repoRoot, 'diff', '--quiet', 'HEAD', '--') + @($sourceBindingFiles.Keys)
            & $gitCommand.Source @diffArguments
            $sourceBundleMatchesHead = $LASTEXITCODE -eq 0
        }
        else {
            $sourceBundleMatchesHead = $false
        }
    }
}

$qualificationPreflightPassed = $null
if ($Qualification) {
    $qualificationFailures = [System.Collections.Generic.List[string]]::new()
    if ($sourceBindingFiles.Count -eq 0) {
        $qualificationFailures.Add('source bundle이 비어 있음')
    }
    if ($null -eq $repositoryCommit -or $repositoryCommit -notmatch '^[0-9a-f]{40}$') {
        $qualificationFailures.Add('유효한 repository commit을 읽지 못함')
    }
    if ($repositoryDirty -ne $false) {
        $qualificationFailures.Add('repository가 clean 상태가 아님')
    }
    if ($sourceBundleMatchesHead -ne $true) {
        $qualificationFailures.Add('source bundle이 현재 HEAD와 일치하지 않음')
    }
    if ([string]::IsNullOrWhiteSpace($TrainingEntrypointPath)) {
        $qualificationFailures.Add('repository 내부 training entrypoint가 지정되지 않음')
    }
    if ($Task -eq $g009QualificationTask) {
        $requiredG009SourcePaths = @(
            'configs/g009_r0.json',
            'scripts/bootstrap_train_g009.py',
            'scripts/run_training.ps1',
            'src/isaac_walk_g009/agent_cfg.py',
            'src/isaac_walk_g009/mdp/__init__.py',
            'src/isaac_walk_g009/mdp/events.py',
            'src/isaac_walk_g009/mdp/recover.py',
            'src/isaac_walk_g009/recover_contracts.py',
            'src/isaac_walk_g009/recover_env_cfg.py',
            'src/isaac_walk_g009/registry.py'
        )
        foreach ($requiredPath in $requiredG009SourcePaths) {
            if (-not $sourceBindingFiles.Contains($requiredPath)) {
                $qualificationFailures.Add("G009 R0 필수 source binding 누락: $requiredPath")
            }
        }
        $expectedG009Entrypoint = [System.IO.Path]::GetFullPath(
            (Join-Path $repoRoot 'scripts\bootstrap_train_g009.py')
        )
        if (-not $trainScript.Equals($expectedG009Entrypoint, [System.StringComparison]::OrdinalIgnoreCase)) {
            $qualificationFailures.Add('G009 R0 training entrypoint가 bootstrap_train_g009.py와 일치하지 않음')
        }
    }
    if ($qualificationFailures.Count -gt 0) {
        throw ('Qualification 사전 검증 실패: ' + ($qualificationFailures -join '; '))
    }
    $qualificationPreflightPassed = $true
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
if ($Resume) {
    $arguments += @('--resume', '--load_run', $LoadRun, '--checkpoint', $ResumeCheckpoint)
}
$arguments += @($HydraOverrides)
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
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
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
$tensorboardScalars = $null
if ($actualLogDirectory -and (Test-Path -LiteralPath $actualLogDirectory -PathType Container)) {
    $tensorboardExists = $null -ne (Get-ChildItem -LiteralPath $actualLogDirectory -Filter 'events.out.tfevents.*' -File | Select-Object -First 1)
    if ($tensorboardExists) {
        $scalarCode = "import json,sys;from tensorboard.backend.event_processing.event_accumulator import EventAccumulator;e=EventAccumulator(sys.argv[1],size_guidance={'scalars':0});e.Reload();tags=e.Tags().get('scalars',[]);series={t:[float(x.value) for x in e.Scalars(t)] for t in tags};summary={t:{'sample_count':len(v),'latest':v[-1],'minimum':min(v),'maximum':max(v),'mean':sum(v)/len(v),'nonzero_sample_count':sum(abs(x)>1.0e-12 for x in v)} for t,v in series.items() if v};print(json.dumps({'tags':tags,'latest':{t:v[-1] for t,v in series.items() if v},'series_summary':summary}))"
        $scalarOutput = @(& $pythonBat -c $scalarCode $actualLogDirectory 2>$null)
        if ($LASTEXITCODE -eq 0 -and $scalarOutput.Count -gt 0) {
            try { $tensorboardScalars = $scalarOutput[-1] | ConvertFrom-Json } catch { $tensorboardScalars = $null }
        }
    }
}

$hardJointLimitSummary = $null
$numericInvalidSummary = $null
if ($tensorboardScalars -and $tensorboardScalars.series_summary) {
    $hardJointLimitProperty = $tensorboardScalars.series_summary.PSObject.Properties['Episode_Termination/hard_joint_limit']
    $numericInvalidProperty = $tensorboardScalars.series_summary.PSObject.Properties['Episode_Termination/numeric_invalid']
    if ($hardJointLimitProperty) { $hardJointLimitSummary = $hardJointLimitProperty.Value }
    if ($numericInvalidProperty) { $numericInvalidSummary = $numericInvalidProperty.Value }
}
$qualificationTrainingSafetyPassed = if ($Qualification) {
    $null -ne $hardJointLimitSummary -and
    $null -ne $numericInvalidSummary -and
    [double]$hardJointLimitSummary.maximum -eq 0.0 -and
    [double]$numericInvalidSummary.maximum -eq 0.0
}
else {
    $null
}

$gpuValues = @($gpuSamples | ForEach-Object { $_.used_mib } | Where-Object { $null -ne $_ })
$peakGpuMiB = if ($gpuValues.Count -gt 0) { ($gpuValues | Measure-Object -Maximum).Maximum } else { $baselineGpuMiB }
$finalGpuMiB = if ($gpuRecoverySamples.Count -gt 0) { $gpuRecoverySamples[$gpuRecoverySamples.Count - 1].used_mib } else { Get-GpuMemoryUsedMiB }
$gpuMeasurementComplete = ($gpuMeasurementFailureCount -eq 0 -and $null -ne $finalGpuMiB)
$recoveredToBaseline = ($gpuMeasurementComplete -and $finalGpuMiB -le ($baselineGpuMiB + 128))
$fatalPatterns = @('Traceback (most recent call last)', '[Error]')
$fatalMatches = @($fatalPatterns | Where-Object { $combined.Contains($_) })
$expectedLastIteration = if ($Resume) {
    # RSL-RL includes the loaded iteration in the resumed learning range.
    # model_N plus M iterations therefore ends at model_(N + M - 1).
    [int]([regex]::Match($ResumeCheckpoint, '^model_([0-9]+)\.pt$').Groups[1].Value) + $MaxIterations - 1
}
else {
    $MaxIterations - 1
}
$expectedIterationTarget = if ($Resume) { $expectedLastIteration + 1 } else { $MaxIterations }
$successChecks = [ordered]@{
    process_exit_zero = ($process.ExitCode -eq 0)
    no_traceback_or_error = ($fatalMatches.Count -eq 0)
    requested_iteration_reached = (
        $lastIteration -eq $expectedLastIteration -and $iterationTarget -eq $expectedIterationTarget
    )
    log_directory_exists = ($actualLogDirectory -and (Test-Path -LiteralPath $actualLogDirectory -PathType Container))
    tensorboard_exists = $tensorboardExists
    checkpoint_exists = ($null -ne $checkpoint)
    gpu_measurement_complete = $gpuMeasurementComplete
    gpu_recovered_to_baseline = $recoveredToBaseline
    qualification_training_safety_zero = $qualificationTrainingSafetyPassed
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
    qualification_mode = [ordered]@{
        enabled = [bool]$Qualification
        preflight_passed = $qualificationPreflightPassed
        policy_qualification_status = 'not_run'
    }
    command = @(
        (Convert-ToPortablePath $pythonBat),
        (Convert-ToPortablePath $trainScript),
        '--task', $Task,
        '--num_envs', $NumEnvs,
        '--max_iterations', $MaxIterations,
        '--seed', $Seed,
        '--run_name', $RunName,
        '--headless'
    ) + $(if ($Resume) { @('--resume', '--load_run', $LoadRun, '--checkpoint', $ResumeCheckpoint) } else { @() }) + @($HydraOverrides)
    resume = [ordered]@{
        enabled = [bool]$Resume
        load_run = if ($Resume) { $LoadRun } else { $null }
        checkpoint = if ($Resume) { $ResumeCheckpoint } else { $null }
    }
    effective_hydra_overrides = @($HydraOverrides)
    training_entrypoint = [ordered]@{
        path = Convert-ToPortablePath $trainScript
        sha256 = $trainingEntrypointHash
        repository_internal = -not [string]::IsNullOrWhiteSpace($TrainingEntrypointPath)
    }
    repository = [ordered]@{
        commit = $repositoryCommit
        dirty = $repositoryDirty
    }
    source_bundle = [ordered]@{
        sha256 = $sourceBundleHash
        files = $sourceBindingFiles
        matches_repository_commit = $sourceBundleMatchesHead
    }
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
    training_safety_aggregate = [ordered]@{
        metric_semantics = 'TensorBoard statistics over per-reset-batch termination counts across every logged scalar sample'
        scalar_reservoir = 'unbounded (size_guidance scalars=0)'
        hard_joint_limit = $hardJointLimitSummary
        numeric_invalid = $numericInvalidSummary
        qualification_requires_both_maximum_counts_zero = [bool]$Qualification
        qualification_passed = $qualificationTrainingSafetyPassed
        unavailable_fields = @(
            'hard_limit_total_count',
            'hard_limit_max_excess_rad',
            'hard_limit_by_joint',
            'hard_limit_by_pose'
        )
        unavailable_reason = 'RSL-RL episode summaries expose rates only; joint/pose attribution requires dedicated runtime instrumentation'
    }
    artifacts = [ordered]@{
        raw_stdout = Convert-ToPortablePath $stdoutPath
        raw_stderr = Convert-ToPortablePath $stderrPath
        tensorboard_directory = Convert-ToPortablePath $actualLogDirectory
        checkpoint = if ($checkpoint) { Convert-ToPortablePath $checkpoint.FullName } else { $null }
        checkpoint_sha256 = $checkpointHash
    }
    tensorboard = $tensorboardScalars
    fatal_patterns_found = $fatalMatches
    success_checks = $successChecks
    run_health_passed = $passed
    qualification_passed = $null
    passed = $passed
}

$reportJson = $report | ConvertTo-Json -Depth 8
$reportTempPath = Join-Path $reportDirectory ('.' + [System.IO.Path]::GetFileName($reportFullPath) + '.' + [guid]::NewGuid().ToString('N') + '.tmp')
try {
    [System.IO.File]::WriteAllText($reportTempPath, $reportJson, [System.Text.UTF8Encoding]::new($false))
    [System.IO.File]::Move($reportTempPath, $reportFullPath, $true)
}
finally {
    if (Test-Path -LiteralPath $reportTempPath -PathType Leaf) {
        [System.IO.File]::Delete($reportTempPath)
    }
}
Write-Host "Report: $(Convert-ToPortablePath $reportFullPath)"
Write-Host "Result: passed=$passed exit=$($process.ExitCode) iteration=$lastIteration/$iterationTarget peak_vram_mib=$peakGpuMiB"

if (-not $passed) {
    exit 1
}
