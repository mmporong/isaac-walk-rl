[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$failures = [System.Collections.Generic.List[string]]::new()

function Add-Failure {
    param([Parameter(Mandatory)][string]$Message)
    $script:failures.Add($Message)
}

$requiredFiles = @(
    '.gitignore',
    'AGENTS.md',
    'README.md',
    'PROJECT_BRIEF.md',
    'PROMPT_WINDOWS.md',
    'NOTES.md',
    'RUN_NOTES.md',
    'docs\VALIDATION_MATRIX.md',
    'scripts\collect_environment.ps1',
    'scripts\validate_repository.ps1',
    'reports\environment_manifest.json',
    '.omx\ultragoal\brief.md',
    '.omx\ultragoal\goals.json',
    '.omx\ultragoal\ledger.jsonl'
)

foreach ($relativePath in $requiredFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $relativePath) -PathType Leaf)) {
        Add-Failure "필수 파일 없음: $relativePath"
    }
}

$manifestPath = Join-Path $repoRoot 'reports\environment_manifest.json'
if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
    try {
        $manifestText = Get-Content -LiteralPath $manifestPath -Raw
        $manifest = $manifestText | ConvertFrom-Json
        if ($manifest.schema_version -ne 1) {
            Add-Failure '환경 매니페스트 schema_version이 1이 아님'
        }
        if ($null -eq $manifest.repository.state) {
            Add-Failure '환경 매니페스트에 repository.state가 없음'
        }
        $homePath = [System.IO.Path]::GetFullPath($HOME).TrimEnd('\')
        if ($manifestText.IndexOf($homePath, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
            Add-Failure '환경 매니페스트에 사용자 홈 절대 경로가 노출됨'
        }
    }
    catch {
        Add-Failure "환경 매니페스트 JSON 파싱 실패: $($_.Exception.Message)"
    }
}

$forbiddenDependencyDirectories = @('_isaac_sim', 'IsaacLab')
foreach ($directory in $forbiddenDependencyDirectories) {
    if (Test-Path -LiteralPath (Join-Path $repoRoot $directory)) {
        Add-Failure "외부 의존성 디렉터리가 저장소 안에 있음: $directory"
    }
}

$forbiddenExtensions = @(
    '.ckpt', '.pt', '.pth', '.onnx',
    '.mp4', '.avi', '.mov',
    '.usd', '.usda', '.usdc'
)
$maxFileBytes = 10MB
$repositoryFiles = Get-ChildItem -LiteralPath $repoRoot -Recurse -Force -File |
    Where-Object { $_.FullName -notlike "$repoRoot\.git\*" }

foreach ($file in $repositoryFiles) {
    $relativePath = $file.FullName.Substring($repoRoot.Length).TrimStart('\')
    if ($forbiddenExtensions -contains $file.Extension.ToLowerInvariant()) {
        Add-Failure "금지된 산출물 확장자: $relativePath"
    }
    if ($file.Name -like 'events.out.tfevents.*') {
        Add-Failure "TensorBoard 원시 로그 포함: $relativePath"
    }
    if ($file.Length -gt $maxFileBytes) {
        Add-Failure "10 MiB 초과 파일: $relativePath ($($file.Length) bytes)"
    }
}

$reparsePoints = Get-ChildItem -LiteralPath $repoRoot -Recurse -Force |
    Where-Object {
        $_.FullName -notlike "$repoRoot\.git\*" -and
        ($_.Attributes -band [System.IO.FileAttributes]::ReparsePoint)
    }
foreach ($item in $reparsePoints) {
    $relativePath = $item.FullName.Substring($repoRoot.Length).TrimStart('\')
    Add-Failure "외부 내용을 포함할 수 있는 링크/재분석 지점: $relativePath"
}

$requiredIgnoreRules = @(
    '.omx/state/',
    '.omx/tmux-hook.json',
    '.omx/runtime/',
    '.omx/*.lock',
    '.omx/**/*.lock'
)
$gitIgnoreLines = @()
$gitIgnorePath = Join-Path $repoRoot '.gitignore'
if (Test-Path -LiteralPath $gitIgnorePath -PathType Leaf) {
    $gitIgnoreLines = Get-Content -LiteralPath $gitIgnorePath
}
foreach ($rule in $requiredIgnoreRules) {
    if ($gitIgnoreLines -notcontains $rule) {
        Add-Failure "필수 .gitignore 규칙 없음: $rule"
    }
}

$durableOmxArtifacts = @(
    '.omx/ultragoal/brief.md',
    '.omx/ultragoal/goals.json',
    '.omx/ultragoal/ledger.jsonl'
)
foreach ($artifact in $durableOmxArtifacts) {
    & git -C $repoRoot check-ignore -q -- $artifact
    if ($LASTEXITCODE -eq 0) {
        Add-Failure "durable OMX artifact가 ignore됨: $artifact"
    }
}

$rawRunProbe = 'reports/runs/ignore-boundary-probe.raw.log'
$jsonRunProbe = 'reports/runs/ignore-boundary-probe.json'
& git -C $repoRoot check-ignore -q --no-index -- $rawRunProbe
if ($LASTEXITCODE -ne 0) {
    Add-Failure "reports/runs 비JSON 파일이 ignore되지 않음: $rawRunProbe"
}
& git -C $repoRoot check-ignore -q --no-index -- $jsonRunProbe
if ($LASTEXITCODE -eq 0) {
    Add-Failure "reports/runs JSON 파일이 ignore됨: $jsonRunProbe"
}

$diffCheck = & git -C $repoRoot diff --check 2>&1
if ($LASTEXITCODE -ne 0) {
    Add-Failure "git diff --check 실패: $($diffCheck -join ' ')"
}
$cachedDiffCheck = & git -C $repoRoot diff --cached --check 2>&1
if ($LASTEXITCODE -ne 0) {
    Add-Failure "git diff --cached --check 실패: $($cachedDiffCheck -join ' ')"
}

if ($failures.Count -gt 0) {
    Write-Error ("저장소 검증 실패 ({0}건):`n- {1}" -f $failures.Count, ($failures -join "`n- "))
    exit 1
}

Write-Host "저장소 검증 PASS: 필수 파일, 의존성 경계, 산출물 제한, reports/runs JSON-only 경계, OMX 추적 경계, git diff 검사"
exit 0
