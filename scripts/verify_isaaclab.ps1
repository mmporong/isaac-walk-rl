[CmdletBinding()]
param(
    [string]$OutputPath,
    [string]$IsaacLabPath,
    [string]$IsaacSimPath = 'E:\IsaacSim\isaac-sim-4.5.0'
)

$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$expectedCommit = '90b79bb2d44feb8d833f260f2bf37da3487180ba'
$expectedTag = 'v2.1.1'
$expectedTasks = @(
    'Isaac-Velocity-Flat-Anymal-C-v0',
    'Isaac-Velocity-Rough-Anymal-C-v0',
    'Isaac-Velocity-Flat-Unitree-Go2-v0',
    'Isaac-Velocity-Rough-Unitree-Go2-v0'
)

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $repoRoot 'reports\isaaclab_verification.json'
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath = Join-Path $repoRoot $OutputPath
}

if ([string]::IsNullOrWhiteSpace($IsaacLabPath)) {
    $IsaacLabPath = Join-Path $HOME 'IsaacLab'
}

function Convert-ToPortablePath {
    param([Parameter(Mandatory)][string]$Path)

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $homePath = [System.IO.Path]::GetFullPath($HOME).TrimEnd('\')
    if ($fullPath.Equals($homePath, [System.StringComparison]::OrdinalIgnoreCase)) {
        return '%USERPROFILE%'
    }
    if ($fullPath.StartsWith($homePath + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        return '%USERPROFILE%' + $fullPath.Substring($homePath.Length)
    }
    return $fullPath
}

function Convert-ToPortableText {
    param([Parameter(Mandatory)][string]$Text)

    $portable = [regex]::Replace(
        $Text,
        [regex]::Escape([System.IO.Path]::GetFullPath($HOME).TrimEnd('\')),
        '%USERPROFILE%',
        [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
    )
    $homeWithForwardSlashes = [System.IO.Path]::GetFullPath($HOME).TrimEnd('\').Replace('\', '/')
    return [regex]::Replace(
        $portable,
        [regex]::Escape($homeWithForwardSlashes),
        '%USERPROFILE%',
        [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
    )
}

function Get-MarkerJson {
    param(
        [Parameter(Mandatory)][object[]]$Output,
        [Parameter(Mandatory)][string]$Prefix
    )

    $line = $Output |
        ForEach-Object { $_.ToString() } |
        Where-Object { $_.StartsWith($Prefix, [System.StringComparison]::Ordinal) } |
        Select-Object -Last 1
    if ($null -eq $line) {
        return $null
    }
    return $line.Substring($Prefix.Length) | ConvertFrom-Json
}

$failures = [System.Collections.Generic.List[string]]::new()
$warnings = [System.Collections.Generic.List[string]]::new()
$warnings.Add('Isaac Lab v2.1.1 Windows isaaclab.bat -p는 nested batch를 call 없이 실행해 성공 후에도 exit 1을 전달할 수 있으므로 runtime 검증에는 bundled python.bat 직접 실행을 사용함')
if (-not (Test-Path -LiteralPath $IsaacLabPath -PathType Container)) {
    $failures.Add('Isaac Lab 디렉터리가 없음')
}
if (-not (Test-Path -LiteralPath $IsaacSimPath -PathType Container)) {
    $failures.Add('Isaac Sim 디렉터리가 없음')
}

$isaacLabBat = Join-Path $IsaacLabPath 'isaaclab.bat'
$pythonBat = Join-Path $IsaacLabPath '_isaac_sim\python.bat'
$linkPath = Join-Path $IsaacLabPath '_isaac_sim'
foreach ($requiredFile in @($isaacLabBat, $pythonBat)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        $failures.Add("필수 실행 파일 없음: $(Convert-ToPortablePath -Path $requiredFile)")
    }
}

$sourceCommit = $null
$sourceTag = $null
if (Test-Path -LiteralPath (Join-Path $IsaacLabPath '.git') -PathType Container) {
    $sourceCommit = ((& git -C $IsaacLabPath rev-parse HEAD 2>$null) | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        $failures.Add('Isaac Lab commit 확인 실패')
    }
    $sourceTag = ((& git -C $IsaacLabPath describe --tags --exact-match HEAD 2>$null) | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        $failures.Add('Isaac Lab exact tag 확인 실패')
    }
}
else {
    $failures.Add('Isaac Lab Git 저장소가 아님')
}
if ($sourceCommit -ne $expectedCommit) {
    $failures.Add("Isaac Lab commit 불일치: $sourceCommit")
}
if ($sourceTag -ne $expectedTag) {
    $failures.Add("Isaac Lab tag 불일치: $sourceTag")
}

$linkType = $null
$linkTarget = $null
$resolvedTarget = $null
if (Test-Path -LiteralPath $linkPath) {
    $linkItem = Get-Item -LiteralPath $linkPath -Force
    $linkType = $linkItem.LinkType
    $linkTarget = @($linkItem.Target) | Select-Object -First 1
    try {
        $resolvedTarget = $linkItem.ResolveLinkTarget($true).FullName
    }
    catch {
        $failures.Add("_isaac_sim 링크 해석 실패: $($_.Exception.Message)")
    }
}
else {
    $failures.Add('_isaac_sim 링크가 없음')
}
if ($linkType -notin @('Junction', 'SymbolicLink')) {
    $failures.Add("_isaac_sim 링크 유형이 올바르지 않음: $linkType")
}
if ($null -ne $resolvedTarget -and
    -not ([System.IO.Path]::GetFullPath($resolvedTarget).Equals(
        [System.IO.Path]::GetFullPath($IsaacSimPath),
        [System.StringComparison]::OrdinalIgnoreCase
    ))) {
    $failures.Add("_isaac_sim 대상 불일치: $resolvedTarget")
}

$versionProbe = $null
$versionExitCode = $null
if (Test-Path -LiteralPath $pythonBat -PathType Leaf) {
    $versionScriptPath = Join-Path ([System.IO.Path]::GetTempPath()) ("verify_isaaclab_versions_{0}.py" -f [guid]::NewGuid().ToString('N'))
    $versionCode = @'
import importlib.metadata as metadata
import json
import platform

import torch

result = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "rsl_rl": metadata.version("rsl-rl-lib"),
    "isaaclab": metadata.version("isaaclab"),
}
print("ISAACLAB_VERSION_JSON=" + json.dumps(result, sort_keys=True), flush=True)
'@
    try {
        [System.IO.File]::WriteAllText($versionScriptPath, $versionCode, [System.Text.UTF8Encoding]::new($false))
        $versionOutput = @(& $pythonBat $versionScriptPath 2>&1)
        $versionExitCode = $LASTEXITCODE
        if ($versionOutput.Count -gt 0) {
            try {
                $versionProbe = Get-MarkerJson -Output $versionOutput -Prefix 'ISAACLAB_VERSION_JSON='
            }
            catch {
                $failures.Add("버전 probe JSON 파싱 실패: $($_.Exception.Message)")
            }
        }
    }
    finally {
        if (Test-Path -LiteralPath $versionScriptPath -PathType Leaf) {
            Remove-Item -LiteralPath $versionScriptPath -Force
        }
    }
    if ($versionExitCode -ne 0) {
        $failures.Add("버전/import probe 종료 코드: $versionExitCode")
    }
    if ($null -eq $versionProbe) {
        $failures.Add('버전/import probe marker가 없음')
    }
}

if ($null -ne $versionProbe) {
    if (-not $versionProbe.python.StartsWith('3.10.')) {
        $failures.Add("Python 버전 불일치: $($versionProbe.python)")
    }
    if ($versionProbe.torch -ne '2.7.0+cu128') {
        $failures.Add("PyTorch 버전 불일치: $($versionProbe.torch)")
    }
    if (-not $versionProbe.cuda_available) {
        $failures.Add('torch.cuda.is_available()이 false')
    }
    if ($versionProbe.rsl_rl -ne '2.3.3') {
        $failures.Add("RSL-RL 버전 불일치: $($versionProbe.rsl_rl)")
    }
    if ($versionProbe.isaaclab -ne '0.41.3') {
        $failures.Add("Isaac Lab package 버전 불일치: $($versionProbe.isaaclab)")
    }
}

$pipCheckExitCode = $null
if (Test-Path -LiteralPath $pythonBat -PathType Leaf) {
    $null = @(& $pythonBat -m pip check 2>&1)
    $pipCheckExitCode = $LASTEXITCODE
    if ($pipCheckExitCode -ne 0) {
        $warnings.Add(
            "pip check 종료 코드 $pipCheckExitCode`: Isaac Sim extension별 pip_prebundle metadata와 Isaac Lab의 starlette==0.45.3 고정으로 알려진 불일치가 있으나 패키지는 수정하지 않음; 핵심 RL imports, AppLauncher, task registry는 runtime probe로 별도 판정함"
        )
    }
}

$runtimeProbe = $null
$runtimeExitCode = $null
$runtimeLogSummary = @()
if (Test-Path -LiteralPath $pythonBat -PathType Leaf) {
    $runtimeScriptPath = Join-Path ([System.IO.Path]::GetTempPath()) ("verify_isaaclab_{0}.py" -f [guid]::NewGuid().ToString('N'))
    $runtimeCode = @'
import json

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
try:
    import gymnasium as gym
    import isaaclab
    import isaaclab_assets
    import isaaclab_rl
    import isaaclab_tasks

    expected = [
        "Isaac-Velocity-Flat-Anymal-C-v0",
        "Isaac-Velocity-Rough-Anymal-C-v0",
        "Isaac-Velocity-Flat-Unitree-Go2-v0",
        "Isaac-Velocity-Rough-Unitree-Go2-v0",
    ]
    tasks = []
    for task_id in expected:
        spec = gym.registry.get(task_id)
        tasks.append({
            "id": task_id,
            "registered": spec is not None,
            "entry_point": str(spec.entry_point) if spec is not None else None,
            "config": str(spec.kwargs.get("env_cfg_entry_point")) if spec is not None else None,
        })
    result = {
        "app_initialized": True,
        "imports": ["isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks"],
        "task_count": len([task for task in tasks if task["registered"]]),
        "tasks": tasks,
    }
    print("ISAACLAB_RUNTIME_JSON=" + json.dumps(result, sort_keys=True), flush=True)
finally:
    app.close()
'@
    try {
        [System.IO.File]::WriteAllText($runtimeScriptPath, $runtimeCode, [System.Text.UTF8Encoding]::new($false))
        # Isaac Lab v2.1.1의 Windows isaaclab.bat -p 경로는 다른 .bat를 call 없이
        # 실행해 성공한 Python 프로세스의 종료 코드를 1로 전달할 수 있다.
        # 실제 runtime 판정은 동일한 공식 bundled Python을 직접 실행한다.
        $runtimeOutput = @(& $pythonBat $runtimeScriptPath 2>&1)
        $runtimeExitCode = $LASTEXITCODE
        $runtimeLogSummary = @($runtimeOutput |
            ForEach-Object { $_.ToString() } |
            Where-Object {
                $_ -match '^\[INFO\]\[AppLauncher\]' -or
                $_ -match 'Simulation App Startup Complete' -or
                $_ -match 'Simulation App Shutting Down' -or
                $_ -match '^Traceback' -or
                $_ -match '\[Error\]'
            })
        $runtimeErrorLines = @($runtimeOutput |
            ForEach-Object { $_.ToString() } |
            Where-Object { $_ -match '^Traceback' -or $_ -match '\[Error\]' } |
            Select-Object -Unique)
        foreach ($runtimeErrorLine in $runtimeErrorLines) {
            $failures.Add("runtime 오류 로그: $(Convert-ToPortableText -Text $runtimeErrorLine)")
        }
        try {
            $runtimeProbe = Get-MarkerJson -Output $runtimeOutput -Prefix 'ISAACLAB_RUNTIME_JSON='
        }
        catch {
            $failures.Add("runtime probe JSON 파싱 실패: $($_.Exception.Message)")
        }
    }
    finally {
        if (Test-Path -LiteralPath $runtimeScriptPath -PathType Leaf) {
            Remove-Item -LiteralPath $runtimeScriptPath -Force
        }
    }
    if ($runtimeExitCode -ne 0) {
        $failures.Add("headless AppLauncher runtime 종료 코드: $runtimeExitCode")
    }
    if ($null -eq $runtimeProbe) {
        $failures.Add('headless AppLauncher runtime marker가 없음')
    }
}

if ($null -ne $runtimeProbe) {
    if (-not $runtimeProbe.app_initialized) {
        $failures.Add('headless AppLauncher 초기화 실패')
    }
    foreach ($taskId in $expectedTasks) {
        $task = @($runtimeProbe.tasks | Where-Object { $_.id -eq $taskId }) | Select-Object -First 1
        if ($null -eq $task -or -not $task.registered) {
            $failures.Add("Gym task 미등록: $taskId")
        }
    }
}

$report = [ordered]@{
    schema_version = 1
    status = if ($failures.Count -eq 0) { 'pass' } else { 'fail' }
    source = [ordered]@{
        path = Convert-ToPortablePath -Path $IsaacLabPath
        tag = $sourceTag
        commit = $sourceCommit
    }
    simulator_link = [ordered]@{
        path = Convert-ToPortablePath -Path $linkPath
        type = $linkType
        configured_target = Convert-ToPortablePath -Path $IsaacSimPath
        link_target = if ($null -ne $linkTarget) { Convert-ToPortablePath -Path $linkTarget } else { $null }
        resolved_target = if ($null -ne $resolvedTarget) { Convert-ToPortablePath -Path $resolvedTarget } else { $null }
    }
    versions = if ($null -ne $versionProbe) {
        [ordered]@{
            python = $versionProbe.python
            torch = $versionProbe.torch
            cuda_available = [bool]$versionProbe.cuda_available
            cuda_device = $versionProbe.cuda_device
            rsl_rl = $versionProbe.rsl_rl
            isaaclab = $versionProbe.isaaclab
            process_exit_code = $versionExitCode
        }
    }
    else {
        $null
    }
    runtime = [ordered]@{
        launcher = 'isaac_sim_bundled_python_direct'
        process_exit_code = $runtimeExitCode
        probe = $runtimeProbe
        log_summary = @($runtimeLogSummary | ForEach-Object { Convert-ToPortableText -Text $_ })
    }
    warnings = @($warnings)
    failures = @($failures)
}

$outputDirectory = Split-Path -Parent $OutputPath
[System.IO.Directory]::CreateDirectory($outputDirectory) | Out-Null
$json = $report | ConvertTo-Json -Depth 10
[System.IO.File]::WriteAllText(
    [System.IO.Path]::GetFullPath($OutputPath),
    $json + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Host "Isaac Lab 검증 보고서 생성: $([System.IO.Path]::GetFullPath($OutputPath))"
if ($failures.Count -gt 0) {
    Write-Error ("Isaac Lab 검증 실패 ({0}건):`n- {1}" -f $failures.Count, ($failures -join "`n- ")) -ErrorAction Continue
    exit 1
}

Write-Host 'Isaac Lab 검증 PASS: source, link, versions, CUDA, imports, headless task registry'
exit 0
