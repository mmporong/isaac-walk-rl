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
    '.gitattributes',
    '.gitignore',
    'AGENTS.md',
    'README.md',
    'PROJECT_BRIEF.md',
    'PROMPT_WINDOWS.md',
    'NOTES.md',
    'RUN_NOTES.md',
    'docs\VALIDATION_MATRIX.md',
    'docs\G006_ROUGH_PUSH_RECOVERY.md',
    'docs\G008_COMMAND_FRICTION_LINK_MASS.md',
    'docs\G008_PERIODIC_FRICTION_AND_LINK_MASS_LIMITS.md',
    'docs\G008_VISUAL_EVIDENCE.md',
    'docs\media\g008\g008_direction_commands.gif',
    'docs\media\g008\g008_direction_contact_sheet.png',
    'docs\media\g008\g008_policy_comparison.gif',
    'docs\media\g008\g008_policy_comparison_contact_sheet.png',
    'docs\media\g008\g008_stage_periodic_friction.gif',
    'docs\media\g008\g008_stage_periodic_friction_contact_sheet.png',
    'docs\media\g008\g008_stage_link_mass_groups.gif',
    'docs\media\g008\g008_stage_link_mass_groups_contact_sheet.png',
    'docs\G007_RBQ_COMPATIBILITY_SPIKE.md',
    'configs\g006_rough_push.json',
    'configs\g008_locomotion_dynamics.json',
    'configs\g007_rbq_asset_manifest.json',
    'scripts\collect_environment.ps1',
    'scripts\run_g006_experiment.ps1',
    'scripts\revalidate_training_gpu_recovery.ps1',
    'scripts\summarize_g006.py',
    'scripts\bootstrap_train_g008.py',
    'scripts\evaluate_g008_directions.py',
    'scripts\evaluate_g008_periodic_friction.py',
    'scripts\aggregate_g008_periodic_friction.py',
    'scripts\evaluate_g008_link_mass_sensitivity.py',
    'scripts\probe_g008_dynamics.py',
    'scripts\record_g008_directions.py',
    'scripts\record_g008_policy_comparison.py',
    'scripts\build_g008_comparison_media.py',
    'scripts\record_g008_stage_evidence.py',
    'scripts\build_g008_stage_media.py',
    'scripts\revalidate_g008_resume_report.ps1',
    'scripts\run_g008_stage.ps1',
    'scripts\validate_rbq_assets.py',
    'scripts\validate_repository.ps1',
    'tests\test_g008_resume_revalidation.py',
    'tests\test_g008_visual_evidence.py',
    'tests\test_g008_policy_comparison_visual_evidence.py',
    'tests\test_g008_stage_capture.py',
    'tests\test_g008_stage_visual_evidence.py',
    'tests\test_g008_dynamics_stress_reports.py',
    'reports\environment_manifest.json',
    'reports\g007_rbq_compatibility_spike.json',
    'reports\runs\g006_queue_state.json',
    'reports\runs\g006_summary.json',
    'reports\runs\g008_command_smoke_e64_i1_s42.json',
    'reports\runs\g008_friction_s1_smoke_e64_i1_s42.json',
    'reports\runs\g008_leg_mass_s1_smoke_e64_i1_s42.json',
    'reports\runs\g008_directional_qualification_g006_s42.json',
    'reports\runs\g008_direction_visual_evidence.json',
    'reports\runs\g008_policy_command_capture.json',
    'reports\runs\g008_policy_friction_s1_capture.json',
    'reports\runs\g008_policy_leg_mass_s1_capture.json',
    'reports\runs\g008_policy_comparison_visual_evidence.json',
    'reports\runs\g008_stage_periodic_friction_capture.json',
    'reports\runs\g008_stage_periodic_friction_visual_evidence.json',
    'reports\runs\g008_stage_link_mass_hip_capture.json',
    'reports\runs\g008_stage_link_mass_thigh_capture.json',
    'reports\runs\g008_stage_link_mass_calf_capture.json',
    'reports\runs\g008_stage_link_mass_foot_capture.json',
    'reports\runs\g008_stage_link_mass_visual_evidence.json',
    'reports\runs\g008_friction_s1_finetune_command_s42_e1024_i300.json',
    'reports\runs\g008_directional_qualification_friction_s1_s42_randomized_plane.json',
    'reports\runs\g008_directional_qualification_friction_s1_s42_nominal_plane.json',
    'reports\runs\g008_leg_mass_s1_finetune_command_s42_e1024_i300.json',
    'reports\runs\g008_directional_qualification_leg_mass_s1_s42_randomized_plane.json',
    'reports\runs\g008_directional_qualification_leg_mass_s1_s42_nominal_plane.json',
    'reports\runs\g008_periodic_friction_sweep_command_vs_friction_s1_e32_h500_s20260826.json',
    'reports\runs\g008_periodic_friction_case_mixed_010_005_e32_h500_s20260826_failure.json',
    'reports\runs\g008_link_mass_sensitivity_command_vs_leg_mass_s1_e800_h300_s20260826.json',
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
    '.usd', '.usda', '.usdc',
    '.urdf', '.stl', '.dae', '.obj'
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

$requiredAttributeRules = @(
    '/.gitattributes text eol=lf',
    '/configs/g007_rbq_asset_manifest.json text eol=lf',
    '/reports/g007_rbq_compatibility_spike.json text eol=lf',
    '/scripts/validate_rbq_assets.py text eol=lf'
)
$gitAttributeLines = @()
$gitAttributePath = Join-Path $repoRoot '.gitattributes'
if (Test-Path -LiteralPath $gitAttributePath -PathType Leaf) {
    $gitAttributeLines = Get-Content -LiteralPath $gitAttributePath
    $gitAttributeBytes = [System.IO.File]::ReadAllBytes($gitAttributePath)
    $expectedAttributeContent = ($requiredAttributeRules -join "`n") + "`n"
    $expectedAttributeBytes = [System.Text.UTF8Encoding]::new($false).GetBytes($expectedAttributeContent)
    if ($gitAttributeBytes -contains [byte]13) {
        Add-Failure '.gitattributes에 CR 바이트가 있음'
    }
    if ([System.Convert]::ToBase64String($gitAttributeBytes) -ne
        [System.Convert]::ToBase64String($expectedAttributeBytes)) {
        Add-Failure '.gitattributes 내용이 필수 4개 규칙의 순서·UTF-8·최종 LF 계약과 다름'
    }
}
foreach ($rule in $requiredAttributeRules) {
    if ($gitAttributeLines -notcontains $rule) {
        Add-Failure "필수 .gitattributes 규칙 없음: $rule"
    }
}
$requiredAttributeRuleByPath = @{}
foreach ($rule in $requiredAttributeRules) {
    $requiredAttributeRuleByPath[($rule -split '\s+')[0]] = $rule
}
foreach ($line in $gitAttributeLines) {
    $trimmedLine = $line.Trim()
    if ($trimmedLine.Length -eq 0 -or $trimmedLine.StartsWith('#')) {
        continue
    }
    $attributePath = ($trimmedLine -split '\s+')[0]
    if ($requiredAttributeRuleByPath.ContainsKey($attributePath) -and
        $trimmedLine -ne $requiredAttributeRuleByPath[$attributePath]) {
        Add-Failure "허용되지 않은 .gitattributes 규칙: $trimmedLine"
    }
}

$requiredAttributeTargets = @(
    'configs/g007_rbq_asset_manifest.json',
    'reports/g007_rbq_compatibility_spike.json',
    'scripts/validate_rbq_assets.py'
)
foreach ($target in $requiredAttributeTargets) {
    $attributeOutput = @(& git -C $repoRoot check-attr text eol -- $target 2>&1)
    if ($LASTEXITCODE -ne 0) {
        Add-Failure "git check-attr 실패: $target ($($attributeOutput -join ' '))"
        continue
    }
    $attributes = @{}
    $attributeOutputValid = $true
    foreach ($outputLine in $attributeOutput) {
        if ($outputLine -notmatch '^(?<path>.*): (?<attribute>[^:]+): (?<value>.*)$') {
            $attributeOutputValid = $false
            break
        }
        $outputPath = $Matches.path.Replace('\', '/')
        if ($outputPath -ne $target -or $attributes.ContainsKey($Matches.attribute)) {
            $attributeOutputValid = $false
            break
        }
        $attributes[$Matches.attribute] = $Matches.value
    }
    if (-not $attributeOutputValid -or $attributes.Count -ne 2 -or
        $attributes.text -ne 'set' -or $attributes.eol -ne 'lf') {
        Add-Failure "G007 파일 Git 속성이 text=set/eol=lf가 아님: $target ($($attributeOutput -join ' '))"
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
