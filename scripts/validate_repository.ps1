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
    'docs\G008_IRREGULAR_ROAD.md',
    'docs\G008_REWARD_AND_ROAD_CURRICULUM.md',
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
    'docs\media\g008\g008_road_g0_vs_turn_air.gif',
    'docs\media\g008\g008_road_g0_vs_turn_air_contact_sheet.png',
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
    'scripts\aggregate_g008_road_curriculum.py',
    'scripts\build_g008_road_curriculum_media.py',
    'scripts\record_g008_road_curriculum.py',
    'scripts\report_g008_reward_contract.py',
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
    'tests\test_g008_road_curriculum_reports.py',
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
    'reports\runs\g008_reward_contract_s20260826.json',
    'reports\runs\g008_road_curriculum_summary_s20260826.json',
    'reports\runs\g008_road_curriculum_visual_evidence.json',
    'reports\runs\g008_road_g0_inherited_capture.json',
    'reports\runs\g008_road_g0_turn_air_i2100_capture.json',
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
$oversizeEvidenceSha256 = @{
    'reports\runs\g009_r0_rev16_arm_a_cpu_rep01_retry04_s42.json' = 'edb650cbbcb3309adbfc3141ff511418d5c2780bf0ae926008df1e4189ae56b1'
    'reports\runs\g009_r0_rev16_arm_a_cpu_rep01_retry05_s42.json' = '2c9622b2c6fe266c0d059ed9e97e817add0c8e2181a4a7fc9fa829ea2cd546ab'
    'reports\runs\g009_r0_rev16_arm_a_cpu_rep01_retry06_s42.json' = '1c9d54948ac019bbe3e03edb1a5b642cd7fcda85dca2647052aba95f4df4066d'
    'reports\runs\g009_r0_rev16_arm_a_cpu_rep02_retry01_s42.json' = '2340f3314e04b6457e3e3c99cafd6610a68e0ec8e95f88dc34e1bce3814bd052'
    'reports\runs\g009_r0_rev16_arm_a_cpu_rep02_retry02_s42.json' = '8cc0a4dc44ccd6bf24e52a02643d489da03c07a27e36eadc775446ae4d9bfd1c'
    'reports\runs\g009_r0_rev16_arm_a_cpu_rep02_s42.json' = '67a149c0cf9266f5b0bdbb0225dd747bde4f16ea951ce138a5a418422a6f7eb2'
    'reports\runs\g009_r0_rev16_arm_a_cpu_rep03_retry01_s42.json' = 'bebcd799befa852098ca46da2ecf5b27287e7110d8c83ce565008c9e6f99defd'
    'reports\runs\g009_r0_rev16_arm_a_cpu_rep03_retry02_s42.json' = '7fc468c0ec0e4477d36b41912d70b067f99428544e6bbbe9fdefb5d656eba3a5'
    'reports\runs\g009_r0_rev16_arm_a_cpu_rep03_s42.json' = 'e372586255f9dbfdf908707de1c01c652a56c93c79dd32751b61bc37be8e19a4'
    'reports\runs\g009_r0_rev16_arm_b_cpu_rep01_retry01_s42.json' = '6d3466cb43f46035278f11da13bf202bb7a727c7e3243e053525c1c835b7c47e'
    'reports\runs\g009_r0_rev16_arm_b_cpu_rep02_retry01_s42.json' = '3335c5f898f1c392823f1b6f019d228e5f411cb943259671dc8cc3a3fc55c9ac'
    'reports\runs\g009_r0_rev16_arm_b_cpu_rep03_retry01_s42.json' = 'd8fc1b96a284cd51a147ff9aabecc42d8083dddc918c58433db1263e9c458c22'
    'reports\runs\g009_r0_rev20_terrain_contact_matrix_cpu_rep01_s42.json' = 'd4f8a371edd77c69fb74994c56d629c3e27dd122907ade90f931eeb546c41c29'
    'reports\runs\g009_r0_rev20_terrain_contact_matrix_cpu_rep02_s42.json' = '63d19f42e7c79cc77846f09b5245c2dc46e77630ce97020dfb29be0837375e6c'
    'reports\runs\g009_r0_rev23_matrix_observation_adapter_cpu_rep01_s42.json' = '1f01963e09574ec1388669dac75ed44cddd787f18f8dc7d806c72b11951d3660'
    'reports\runs\g009_r0_rev23_matrix_observation_adapter_cpu_rep02_s42.json' = '6ccd4ce4cdf681524505c5122a2a0282097afd31f23129e0a30101e27f632844'
}
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
        $expectedSha256 = $oversizeEvidenceSha256[$relativePath]
        $actualSha256 = if ($null -ne $expectedSha256) {
            (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        else {
            $null
        }
        if ($null -eq $expectedSha256 -or $actualSha256 -ne $expectedSha256) {
            Add-Failure "허용 목록에 없거나 hash가 달라진 10 MiB 초과 파일: $relativePath ($($file.Length) bytes)"
        }
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
    if ($gitAttributeBytes -contains [byte]13) {
        Add-Failure '.gitattributes에 CR 바이트가 있음'
    }
    if ($gitAttributeBytes.Length -eq 0 -or $gitAttributeBytes[$gitAttributeBytes.Length - 1] -ne [byte]10) {
        Add-Failure '.gitattributes가 최종 LF로 끝나지 않음'
    }
    if ($gitAttributeBytes.Length -ge 3 -and
        $gitAttributeBytes[0] -eq [byte]0xEF -and
        $gitAttributeBytes[1] -eq [byte]0xBB -and
        $gitAttributeBytes[2] -eq [byte]0xBF) {
        Add-Failure '.gitattributes에 UTF-8 BOM이 있음'
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
