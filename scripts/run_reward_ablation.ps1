[CmdletBinding()]
param(
    [string]$ConfigPath, [string]$IsaacLabPath = "$HOME\IsaacLab", [string]$StatePath,
    [string]$TrainingHarness, [string]$EvaluationScript, [string]$ReportRoot,
    [switch]$Resume, [switch]$Smoke, [switch]$AdoptExistingSmokeTraining
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Get-Sha256Text { param([Parameter(Mandatory)][string]$Text)
    [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($Text))).ToLowerInvariant() }
function Get-PortablePath { param([AllowNull()][string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return $null }
    $homePath = [IO.Path]::GetFullPath($HOME).TrimEnd('\'); $fullPath = [IO.Path]::GetFullPath($Path)
    if ($fullPath.Equals($homePath, [StringComparison]::OrdinalIgnoreCase) -or $fullPath.StartsWith($homePath + '\', [StringComparison]::OrdinalIgnoreCase)) {
        return '%USERPROFILE%' + $fullPath.Substring($homePath.Length) }
    $fullPath }
function Resolve-PortablePath { param([Parameter(Mandatory)][string]$Path)
    if ($Path.StartsWith('%USERPROFILE%', [StringComparison]::OrdinalIgnoreCase)) { return [IO.Path]::GetFullPath($HOME + $Path.Substring(13)) }
    [IO.Path]::GetFullPath($Path) }
function Write-JsonAtomic { param([Parameter(Mandatory)][object]$Value, [Parameter(Mandatory)][string]$Path)
    $directory = Split-Path -Parent $Path; New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $temporary = Join-Path $directory ('.' + [IO.Path]::GetFileName($Path) + '.' + [guid]::NewGuid().ToString('N') + '.tmp')
    try { [IO.File]::WriteAllText($temporary, ($Value | ConvertTo-Json -Depth 16), [Text.UTF8Encoding]::new($false)); [IO.File]::Move($temporary, $Path, $true) }
    finally { if (Test-Path -LiteralPath $temporary -PathType Leaf) { [IO.File]::Delete($temporary) } } }
function Read-JsonResult { param([Parameter(Mandatory)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return [ordered]@{ valid=$false; value=$null; reason="missing_json:$([IO.Path]::GetFileName($Path))" } }
    try { [ordered]@{ valid=$true; value=(Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json); reason=$null } }
    catch { [ordered]@{ valid=$false; value=$null; reason="corrupt_json:$([IO.Path]::GetFileName($Path)):$($_.Exception.GetType().Name)" } } }
function Add-FailedAttempt { param([object]$State,[object]$Job,[string]$Phase,[string]$Reason,[AllowNull()][object]$ExitCode)
    $fingerprint = Get-Sha256Text "$Phase|$Reason"
    $Job.attempts = @($Job.attempts) + @([ordered]@{ phase=$Phase; at=(Get-Date).ToString('o'); exit_code=$ExitCode; validation_reason=$Reason; failure_fingerprint=$fingerprint })
    $Job.hard_blocked = (@($Job.attempts | Where-Object failure_fingerprint -eq $fingerprint).Count -ge 3)
    $Job.status='failed'; $Job.updated_at=(Get-Date).ToString('o'); $State.status='failed'; $State.updated_at=$Job.updated_at
    Write-JsonAtomic $State $script:stateFullPath; $script:failureRecorded=$true; $fingerprint }

function Assert-Manifest { param([object]$Config)
    $variants=@('baseline','no_torque','no_action_rate','no_feet_air_time')
    if ($Config.schema_version-ne 1 -or $Config.task-ne'Isaac-Velocity-Flat-Unitree-Go2-v0' -or $Config.num_envs-ne 4096 -or $Config.max_iterations-ne 300 -or
        (@($Config.seeds)-join ',')-ne'42,43,44' -or (@($Config.variants.name)-join ',')-ne($variants-join ',')) { throw 'G005 4x3/task/4096 env/300 iter 계약이 다릅니다.' }
    if ($Config.reward_keys.torque-ne'env.rewards.dof_torques_l2.weight' -or $Config.reward_keys.action_rate-ne'env.rewards.action_rate_l2.weight' -or
        $Config.reward_keys.feet_air_time-ne'env.rewards.feet_air_time.weight') { throw 'G005 reward key 계약이 다릅니다.' }
    $baseline=$Config.variants[0].weights
    if ($baseline.torque-ne -0.0002 -or $baseline.action_rate-ne -0.01 -or $baseline.feet_air_time-ne 0.25) { throw 'baseline weight가 다릅니다.' }
    foreach($variant in @($Config.variants|Select-Object -Skip 1)) { $changed=@(@('torque','action_rate','feet_air_time')|Where-Object{$variant.weights.$_ -ne $baseline.$_})
        if($changed.Count-ne 1 -or $variant.weights.($changed[0])-ne 0.0){throw "$($variant.name)은 정확히 한 reward만 0.0이어야 합니다."} }
    $p=$Config.evaluation_protocol
    if($p.task-ne'Isaac-Velocity-Flat-Unitree-Go2-Play-v0' -or $p.seed-ne 20260824 -or $p.num_envs-ne 260 -or $p.horizon_steps-ne 1000 -or $p.step_dt-ne 0.02 -or
       $p.command_grid_conditions-ne 26 -or $p.environments_per_condition-ne 10 -or (@($p.command_grid.vx_mps)-join ',')-ne'-1,0,1' -or
       (@($p.command_grid.vy_mps)-join ',')-ne'-0.5,0,0.5' -or (@($p.command_grid.yaw_rate_radps)-join ',')-ne'-0.5,0,0.5' -or
       (@($p.command_grid.exclude[0])-join ',')-ne'0,0,0' -or $p.command_grid.environments_per_condition-ne 10){throw 'G005 evaluation protocol/grid 계약이 다릅니다.'}
    foreach($name in $variants){if($Config.variant_sha256.$name-notmatch'^[0-9a-f]{64}$'){throw "variant SHA256 오류: $name"}} }
function Format-HydraFloat { param([double]$Value) if($Value-eq 0.0){'0.0'}else{$Value.ToString([Globalization.CultureInfo]::InvariantCulture)} }
function New-Overrides { param([object]$Config,[object]$Variant) @(
    "$($Config.reward_keys.torque)=$(Format-HydraFloat $Variant.weights.torque)",
    "$($Config.reward_keys.action_rate)=$(Format-HydraFloat $Variant.weights.action_rate)",
    "$($Config.reward_keys.feet_air_time)=$(Format-HydraFloat $Variant.weights.feet_air_time)","agent.experiment_name=$($Config.experiment_name)") }

function Test-TrainingReport { param([string]$Path,[object]$Spec)
    $read=Read-JsonResult $Path; if(-not $read.valid){return [ordered]@{valid=$false;report=$null;reason=$read.reason}}
    try{$report=$read.value;if(-not $report.passed){throw'passed_false'}
        foreach($check in @('gpu_measurement_complete','gpu_recovered_to_baseline','tensorboard_exists','checkpoint_exists')){if($report.success_checks.$check-ne$true){throw "check_false:$check"}}
        if($report.task-ne$Spec.task-or$report.num_envs-ne$Spec.num_envs-or$report.max_iterations-ne$Spec.max_iterations-or$report.seed-ne$Spec.seed-or$report.run_name-ne$Spec.run_name){throw'training_identity_or_budget_mismatch'}
        if((@($report.effective_hydra_overrides)-join"`n")-ne(@($Spec.overrides)-join"`n")){throw'effective_overrides_mismatch'}
        $checkpoint=Resolve-PortablePath ([string]$report.artifacts.checkpoint);if(-not(Test-Path -LiteralPath $checkpoint -PathType Leaf)){throw'checkpoint_missing'}
        $hash=(Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant();if($hash-ne$report.artifacts.checkpoint_sha256){throw'checkpoint_hash_mismatch'}
        [ordered]@{valid=$true;report=$report;checkpoint=$checkpoint;reason=$null}
    }catch{[ordered]@{valid=$false;report=$null;reason="training_validation:$($_.Exception.Message)"}} }
function Get-EvaluationBinding { param([object]$Job,[object]$Training,[object]$Config,[string]$OutputPath)
    $scriptHash=(Get-FileHash -LiteralPath $script:evaluationFullPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $args=@((Get-PortablePath $script:pythonBat),(Get-PortablePath $script:evaluationFullPath),'--checkpoint',(Get-PortablePath $Training.checkpoint),'--variant',$Job.variant,
      '--training-seed',"$($Job.seed)",'--eval-seed',"$($Config.evaluation_protocol.seed)",'--num-envs',"$($Config.evaluation_protocol.num_envs)",
      '--protocol',(Get-PortablePath $script:configFullPath),'--output',(Get-PortablePath $OutputPath),'--headless',
      '--finalize-runtime-command',(Get-PortablePath $script:pythonBat),(Get-PortablePath $script:evaluationFullPath),'--finalize-runtime','--output',(Get-PortablePath $OutputPath),
      "script_sha256:$scriptHash")
    [ordered]@{command=$args;command_sha256=Get-Sha256Text($args-join"`n");script_sha256=$scriptHash} }
function Test-CloseNumber { param([AllowNull()][object]$Actual,[AllowNull()][object]$Expected)
    if($null-eq$Actual-or$null-eq$Expected){return $null-eq$Actual-and$null-eq$Expected}
    $a=[double]$Actual;$e=[double]$Expected;return [math]::Abs($a-$e)-le([math]::Max(1e-6,1e-5*[math]::Abs($e))) }
function Get-ExpectedCommands { param([object]$Config)
    $result=@{};$culture=[Globalization.CultureInfo]::InvariantCulture
    foreach($vx in $Config.evaluation_protocol.command_grid.vx_mps){foreach($vy in $Config.evaluation_protocol.command_grid.vy_mps){foreach($yaw in $Config.evaluation_protocol.command_grid.yaw_rate_radps){
      if($vx-eq 0-and$vy-eq 0-and$yaw-eq 0){continue};$id='vx'+([double]$vx).ToString('+0.0;-0.0;+0.0',$culture)+'_vy'+([double]$vy).ToString('+0.0;-0.0;+0.0',$culture)+'_yaw'+([double]$yaw).ToString('+0.0;-0.0;+0.0',$culture)
      $result[$id]=[ordered]@{id=$id;vx_mps=[double]$vx;vy_mps=[double]$vy;yaw_rate_radps=[double]$yaw}}}};$result }
function Test-EvaluationReport { param([string]$Path,[object]$Job,[object]$Training,[object]$Config)
    $read=Read-JsonResult $Path;if(-not$read.valid){return [ordered]@{valid=$false;reason=$read.reason}}
    try{$r=$read.value
      if($r.schema_version-ne 1-or$r.protocol_compliant-ne$true){throw'schema_or_protocol_compliant'}
      if($r.variant-ne$Job.variant-or$r.training_seed-ne$Job.seed-or$r.evaluation_seed-ne$Config.evaluation_protocol.seed){throw'identity_mismatch'}
      if($r.config_sha256-ne$script:canonicalConfigHash-or$r.config_file_sha256-ne$script:configFileHash-or$r.variant_config_sha256-ne$Job.variant_config_sha256-or$r.protocol_sha256-ne$script:protocolHash){throw'config_or_protocol_hash_mismatch'}
      if($r.checkpoint_sha256-ne$Training.report.artifacts.checkpoint_sha256){throw'evaluation_checkpoint_hash_mismatch'}
      if($r.checkpoint.sha256-ne$Training.report.artifacts.checkpoint_sha256-or(Resolve-PortablePath ([string]$r.checkpoint.reference))-ne$Training.checkpoint){throw'evaluation_checkpoint_reference_mismatch'}
      if($r.task-ne$Config.evaluation_protocol.task-or$r.num_envs-ne$Config.evaluation_protocol.num_envs-or$r.horizon_steps-ne$Config.evaluation_protocol.horizon_steps-or$r.step_dt-ne$Config.evaluation_protocol.step_dt){throw'protocol_value_mismatch'}
      $variant=@($Config.variants|Where-Object name -eq $Job.variant)[0];foreach($key in @('torque','action_rate','feet_air_time')){if($r.effective_weights.$key-ne$variant.weights.$key){throw "effective_weight_mismatch:$key"}}
      foreach($name in @('sample_count','fall_trial_rate','survival_rate','trials_started','fall_timeout_overlap_count')){if($r.denominators.$name -isnot [string] -or [string]::IsNullOrWhiteSpace($r.denominators.$name)){throw "missing_denominator:$name"}}
      $meanMetrics=@('lin_vel_rmse_mps','yaw_rate_rmse_radps','torque_l2_mean','absolute_mechanical_power_w','action_rate_l2_mean','feet_air_time_raw_mean','mean_air_time_at_first_contact_s','fall_trial_rate','survival_rate')
      $countMetrics=@('sample_count','first_contact_count','fall_count','timeout_count','reset_count','fall_timeout_overlap_count','trials_started')
      $blocks=@($r.metrics.overall)+@($r.metrics.by_command);foreach($block in $blocks){foreach($field in $meanMetrics){if($block.PSObject.Properties.Name-notcontains$field){throw "missing_metric:$field"};if($null-ne$block.$field-and([double]::IsNaN([double]$block.$field)-or[double]::IsInfinity([double]$block.$field))){throw "nonfinite_metric:$field"}}
        foreach($field in $countMetrics){if($block.$field -isnot [int] -and $block.$field -isnot [long]){throw "invalid_count_type:$field"};if($block.$field-lt 0){throw "negative_count:$field"}}
        if($block.sample_count-le 0-or$block.reset_count-gt($block.fall_count+$block.timeout_count)-or$block.reset_count-lt[math]::Max($block.fall_count,$block.timeout_count)-or[math]::Max($block.fall_count,[math]::Max($block.timeout_count,$block.reset_count)) -gt $block.trials_started){throw'trial_count_inconsistent'}
        if($block.fall_timeout_overlap_count-ne($block.fall_count+$block.timeout_count-$block.reset_count)){throw'fall_timeout_overlap_inconsistent'}
        $fallRate=if($block.trials_started){$block.fall_count/[double]$block.trials_started}else{$null};if(-not(Test-CloseNumber $block.fall_trial_rate $fallRate)-or-not(Test-CloseNumber $block.survival_rate $(if($null-eq$fallRate){$null}else{1-$fallRate}))){throw'trial_rate_inconsistent'}
        if($block.first_contact_count-gt 0-and$null-eq$block.mean_air_time_at_first_contact_s){throw'mean_air_time_missing'}}
      $byCommand=@($r.metrics.by_command);$expected=Get-ExpectedCommands $Config;if($byCommand.Count-ne 26-or$expected.Count-ne 26){throw'metric_grid_count'};$seen=@{}
      foreach($condition in $byCommand){$id=[string]$condition.command.id;if(-not$expected.ContainsKey($id)-or$seen.ContainsKey($id)){throw'duplicate_missing_or_unknown_command'};$want=$expected[$id]
        if($condition.command.vx_mps-ne$want.vx_mps-or$condition.command.vy_mps-ne$want.vy_mps-or$condition.command.yaw_rate_radps-ne$want.yaw_rate_radps){throw'command_identity_mismatch'};$seen[$id]=$true
        if($condition.trials_started-ne 10){throw'condition_trial_total_mismatch'}}
      if($seen.Count-ne$expected.Count){throw'command_coverage_mismatch'};$overall=$r.metrics.overall
      foreach($field in $countMetrics){if($overall.$field-ne($byCommand|Measure-Object -Property $field -Sum).Sum){throw "overall_additive_mismatch:$field"}}
      $samples=[double]$overall.sample_count;foreach($field in @('torque_l2_mean','absolute_mechanical_power_w','action_rate_l2_mean','feet_air_time_raw_mean')){$weighted=($byCommand|ForEach-Object{$_.($field)*$_.sample_count}|Measure-Object -Sum).Sum/$samples;if(-not(Test-CloseNumber $overall.$field $weighted)){throw "overall_weighted_mismatch:$field"}}
      foreach($field in @('lin_vel_rmse_mps','yaw_rate_rmse_radps')){$weighted=[math]::Sqrt(($byCommand|ForEach-Object{$_.($field)*$_.($field)*$_.sample_count}|Measure-Object -Sum).Sum/$samples);if(-not(Test-CloseNumber $overall.$field $weighted)){throw "overall_rmse_mismatch:$field"}}
      $contacts=[double]$overall.first_contact_count;$air=if($contacts){($byCommand|Where-Object first_contact_count -gt 0|ForEach-Object{$_.mean_air_time_at_first_contact_s*$_.first_contact_count}|Measure-Object -Sum).Sum/$contacts}else{$null};if(-not(Test-CloseNumber $overall.mean_air_time_at_first_contact_s $air)){throw'overall_air_time_mismatch'}
      foreach($field in @('fall_trial_rate','survival_rate')){$weighted=($byCommand|ForEach-Object{$_.($field)*$_.trials_started}|Measure-Object -Sum).Sum/$overall.trials_started;if(-not(Test-CloseNumber $overall.$field $weighted)){throw "overall_rate_mismatch:$field"}}
      $runtime=$r.runtime_evidence
      if($runtime.exit_code-ne 0-or$runtime.app_close_completed-ne$true-or$runtime.finalized_after_process_exit-ne$true-or
         $runtime.gpu_recovered_to_baseline-ne$true-or$runtime.process_recovered-ne$true-or$runtime.gpu_after.measurement_complete-ne$true-or
         $runtime.fatal_scan.measurement_complete-ne$true-or$runtime.fatal_scan.count-ne 0-or@($runtime.fatal_scan.patterns).Count-ne 0){throw'runtime_evidence_invalid'}
      [ordered]@{valid=$true;reason=$null;report=$r}
    }catch{[ordered]@{valid=$false;reason="evaluation_validation:$($_.Exception.Message)";report=$null}} }
function Write-PreflightFailure { param([string]$Reason)
    if([string]::IsNullOrWhiteSpace($script:stateFullPath)){return}
    try{$evidencePath=$script:stateFullPath+'.preflight_failure.json';$failure=[ordered]@{schema_version=1;status='failed';phase='preflight';state_path=Get-PortablePath $script:stateFullPath;updated_at=(Get-Date).ToString('o');lock_owner=[ordered]@{pid=$PID;started_at=$script:queueStartedAt};
        attempts=@([ordered]@{phase='preflight';at=(Get-Date).ToString('o');validation_reason=$Reason;failure_fingerprint=Get-Sha256Text "preflight|$Reason"})}
      if(Test-Path -LiteralPath $script:stateFullPath -PathType Leaf){$failure.source_state_sha256=(Get-FileHash -LiteralPath $script:stateFullPath -Algorithm SHA256).Hash.ToLowerInvariant()}
      Write-JsonAtomic $failure $evidencePath;$script:failureRecorded=$true}catch{} }

$script:failureRecorded=$false;$script:stateFullPath=$null;$script:lockStream=$null;$script:queueStartedAt=(Get-Date).ToString('o')
try{
  $repoRoot=Split-Path -Parent $PSScriptRoot;$mode=if($Smoke){'smoke'}else{'production'}
  if($AdoptExistingSmokeTraining-and-not$Smoke){throw 'AdoptExistingSmokeTraining은 -Smoke와 함께만 사용할 수 있습니다.'}
  if([string]::IsNullOrWhiteSpace($ReportRoot)){$ReportRoot=Join-Path $repoRoot 'reports\runs'};$reportRootFullPath=[IO.Path]::GetFullPath($ReportRoot);New-Item -ItemType Directory -Path $reportRootFullPath -Force|Out-Null
  if([string]::IsNullOrWhiteSpace($StatePath)){$StatePath=Join-Path $repoRoot "reports\runs\g005_reward_ablation_$mode`_state.json"}
  try{$script:stateFullPath=[IO.Path]::GetFullPath($StatePath)}catch{throw "state path preflight 오류: $($_.Exception.Message)"}
  $stateDirectory=Split-Path -Parent $script:stateFullPath;New-Item -ItemType Directory -Path $stateDirectory -Force|Out-Null;$lockPath=Join-Path $reportRootFullPath '.g005_reward_ablation.queue.lock'
  try{$script:lockStream=[IO.File]::Open($lockPath,[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None);$script:lockStream.SetLength(0)
    $lockBytes=[Text.Encoding]::UTF8.GetBytes((@{pid=$PID;started_at=$script:queueStartedAt}|ConvertTo-Json -Compress));$script:lockStream.Write($lockBytes,0,$lockBytes.Length);$script:lockStream.Flush($true)
  }catch{throw "queue_lock_active:${lockPath}:$($_.Exception.GetType().Name)"}
  if([string]::IsNullOrWhiteSpace($ConfigPath)){$ConfigPath=Join-Path $repoRoot 'configs\g005_reward_ablation.json'}
  if([string]::IsNullOrWhiteSpace($TrainingHarness)){$TrainingHarness=Join-Path $PSScriptRoot 'run_training.ps1'}
  if([string]::IsNullOrWhiteSpace($EvaluationScript)){$EvaluationScript=Join-Path $PSScriptRoot 'evaluate_go2_policy.py'}
  $script:configFullPath=[IO.Path]::GetFullPath($ConfigPath);$trainingFullPath=[IO.Path]::GetFullPath($TrainingHarness);$script:evaluationFullPath=[IO.Path]::GetFullPath($EvaluationScript)
  $labFullPath=[IO.Path]::GetFullPath($IsaacLabPath);$script:pythonBat=Join-Path $labFullPath '_isaac_sim\python.bat'
  foreach($required in @($script:configFullPath,$trainingFullPath,$script:pythonBat)){if(-not(Test-Path -LiteralPath $required -PathType Leaf)){throw "preflight_missing_file:$required"}}
  if(-not$Smoke-and-not(Test-Path -LiteralPath $script:evaluationFullPath -PathType Leaf)){throw "preflight_missing_file:$($script:evaluationFullPath)"}
  $configRead=Read-JsonResult $script:configFullPath;if(-not$configRead.valid){throw$configRead.reason};$config=$configRead.value;Assert-Manifest $config
  $canonicalCode="import hashlib,json,sys;d=json.load(open(sys.argv[1],encoding='utf-8'));c=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest();print(json.dumps({'config':c(d),'protocol':c(d['evaluation_protocol']),'variants':{v['name']:c(v) for v in d['variants']}}))"
  $canonicalOutput=@(& $script:pythonBat -c $canonicalCode $script:configFullPath 2>&1);if($LASTEXITCODE-ne 0){throw "canonical_hash_failed:$($canonicalOutput-join' ')"};$hashes=$canonicalOutput[-1]|ConvertFrom-Json
  $script:canonicalConfigHash=[string]$hashes.config;$script:protocolHash=[string]$hashes.protocol;$script:configFileHash=(Get-FileHash -LiteralPath $script:configFullPath -Algorithm SHA256).Hash.ToLowerInvariant()
  foreach($variant in @($config.variants)){if($hashes.variants.($variant.name)-ne$config.variant_sha256.($variant.name)){throw "variant_hash_mismatch:$($variant.name)"}}
  $pwshPath=(Get-Process -Id $PID).Path;$numEnvs=if($Smoke){64}else{[int]$config.num_envs};$iterations=if($Smoke){1}else{[int]$config.max_iterations};$seeds=if($Smoke){@([int]$config.seeds[0])}else{@($config.seeds|ForEach-Object{[int]$_})}
  $specs=[Collections.Generic.List[object]]::new();foreach($variant in @($config.variants)){foreach($seed in $seeds){$runName="g005_$mode`_$($variant.name)_s$seed";$reportPath=Join-Path $reportRootFullPath "$runName.json";$evalPath=Join-Path $reportRootFullPath "$runName`_evaluation.json"
    $overrides=@(New-Overrides $config $variant);$base64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($overrides|ConvertTo-Json -Compress)))
    $command=@((Get-PortablePath $pwshPath),'-NoProfile','-File',(Get-PortablePath $trainingFullPath),'-Task',$config.task,'-NumEnvs',"$numEnvs",'-MaxIterations',"$iterations",'-Seed',"$seed",'-RunName',$runName,'-IsaacLabPath',(Get-PortablePath $labFullPath),'-ReportPath',(Get-PortablePath $reportPath),'-HydraOverridesBase64',$base64)
    $specs.Add([ordered]@{id="$($variant.name)-s$seed";variant=$variant.name;seed=$seed;run_name=$runName;task=$config.task;num_envs=$numEnvs;max_iterations=$iterations;report_path=Get-PortablePath $reportPath;evaluation_report_path=Get-PortablePath $evalPath;overrides=$overrides;override_base64=$base64;training_command=$command;training_command_sha256=Get-Sha256Text($command-join"`n");variant_config_sha256=[string]$config.variant_sha256.($variant.name)})}}
  $stateRead=Read-JsonResult $script:stateFullPath
  if((Test-Path -LiteralPath $script:stateFullPath -PathType Leaf)-and-not$AdoptExistingSmokeTraining){if(-not$Resume){throw "state_exists_use_resume:$($script:stateFullPath)"};if(-not$stateRead.valid){Write-PreflightFailure $stateRead.reason;throw$stateRead.reason};$state=$stateRead.value
    if($state.config_sha256-ne$script:canonicalConfigHash-or$state.config_file_sha256-ne$script:configFileHash-or$state.protocol_sha256-ne$script:protocolHash-or$state.mode-ne$mode){throw'resume_config_protocol_or_mode_mismatch'}}
  else{$jobs=@($specs|ForEach-Object{[ordered]@{id=$_.id;variant=$_.variant;seed=$_.seed;run_name=$_.run_name;status='pending';config_sha256=$script:canonicalConfigHash;protocol_sha256=$script:protocolHash;variant_config_sha256=$_.variant_config_sha256;overrides=$_.overrides;training_command=$_.training_command;training_command_sha256=$_.training_command_sha256;evaluation_command=$null;evaluation_command_sha256=$null;evaluation_script_sha256=$null;report_path=$_.report_path;evaluation_report_path=$_.evaluation_report_path;checkpoint_sha256=$null;attempts=@();hard_blocked=$false;updated_at=(Get-Date).ToString('o')}})
    foreach($job in $jobs){$job.config_file_sha256=$script:configFileHash}
    $state=[ordered]@{schema_version=2;config_path=Get-PortablePath $script:configFullPath;config_sha256=$script:canonicalConfigHash;config_file_sha256=$script:configFileHash;protocol_sha256=$script:protocolHash;mode=$mode;status='pending';lock_owner=[ordered]@{pid=$PID;started_at=$script:queueStartedAt};created_at=(Get-Date).ToString('o');updated_at=(Get-Date).ToString('o');jobs=$jobs};Write-JsonAtomic $state $script:stateFullPath}
  if($AdoptExistingSmokeTraining){foreach($spec in $specs){$job=@($state.jobs|Where-Object id -eq $spec.id)[0];$training=Test-TrainingReport (Resolve-PortablePath $job.report_path) $spec;if(-not$training.valid){Add-FailedAttempt $state $job 'smoke_adoption' $training.reason $null|Out-Null;throw "smoke_adoption_failed:$($job.id)"};$job.checkpoint_sha256=$training.report.artifacts.checkpoint_sha256;$job.status='trained';$job.updated_at=(Get-Date).ToString('o')};$state.status='trained';$state.updated_at=(Get-Date).ToString('o');Write-JsonAtomic $state $script:stateFullPath;Write-Host "G005 smoke state adopted: jobs=$(@($state.jobs).Count)";return}
  foreach($spec in $specs){$job=@($state.jobs|Where-Object id -eq $spec.id)[0];if($null-eq$job){throw "resume_missing_job:$($spec.id)"};if($job.hard_blocked-eq$true){throw "hard_blocked:$($job.id)"}
    if($job.training_command_sha256-ne$spec.training_command_sha256-or(@($job.training_command)-join"`n")-ne(@($spec.training_command)-join"`n")){Add-FailedAttempt $state $job 'resume_validation' 'training_command_or_hash_mismatch' $null|Out-Null;throw "resume_validation_failed:$($job.id)"}
    $trainingPath=Resolve-PortablePath $job.report_path;$evalPath=Resolve-PortablePath $job.evaluation_report_path;$training=Test-TrainingReport $trainingPath $spec;$lastFailurePhase=if(@($job.attempts).Count){[string]@($job.attempts)[-1].phase}else{$null}
    if($job.status-eq'complete'){if(-not$training.valid){Add-FailedAttempt $state $job 'complete_resume_validation' $training.reason $null|Out-Null;throw "complete_training_invalid:$($job.id)"};$binding=Get-EvaluationBinding $job $training $config $evalPath
      if($job.evaluation_command_sha256-ne$binding.command_sha256-or$job.evaluation_script_sha256-ne$binding.script_sha256-or(@($job.evaluation_command)-join"`n")-ne(@($binding.command)-join"`n")){Add-FailedAttempt $state $job 'complete_resume_validation' 'evaluation_command_path_args_or_script_hash_mismatch' $null|Out-Null;throw "complete_evaluation_binding_invalid:$($job.id)"}
      $evaluation=Test-EvaluationReport $evalPath $job $training $config;if(-not$evaluation.valid){Add-FailedAttempt $state $job 'complete_resume_validation' $evaluation.reason $null|Out-Null;throw "complete_evaluation_invalid:$($job.id)"};continue}
    $reuseTraining=$training.valid-and$job.status-in@('trained','evaluating','failed')-and$lastFailurePhase-notin@('training_execution')
    if(-not$reuseTraining){if($job.status-ne'pending'-and$lastFailurePhase-ne'training_execution'-and-not$training.valid){Add-FailedAttempt $state $job 'training_resume_validation' $training.reason $null|Out-Null;throw "training_artifact_invalid:$($job.id)"}
      $job.status='training';$job.updated_at=(Get-Date).ToString('o');$state.status='training';$state.updated_at=$job.updated_at;Write-JsonAtomic $state $script:stateFullPath
      & $pwshPath -NoProfile -File $trainingFullPath -Task $config.task -NumEnvs $numEnvs -MaxIterations $iterations -Seed $job.seed -RunName $job.run_name -IsaacLabPath $labFullPath -ReportPath $trainingPath -HydraOverridesBase64 $spec.override_base64;$exitCode=$LASTEXITCODE;$training=Test-TrainingReport $trainingPath $spec
      if($exitCode-ne 0-or-not$training.valid){$reason="exit=$exitCode;$($training.reason)";Add-FailedAttempt $state $job 'training_execution' $reason $exitCode|Out-Null;throw "training_failed:$($job.id):$reason"}
      $job.checkpoint_sha256=$training.report.artifacts.checkpoint_sha256;$job.status='trained';$job.updated_at=(Get-Date).ToString('o');$state.updated_at=$job.updated_at;Write-JsonAtomic $state $script:stateFullPath}else{$job.checkpoint_sha256=$training.report.artifacts.checkpoint_sha256}
    if($Smoke){continue};$binding=Get-EvaluationBinding $job $training $config $evalPath
    if($null-ne$job.evaluation_command_sha256-and($job.evaluation_command_sha256-ne$binding.command_sha256-or$job.evaluation_script_sha256-ne$binding.script_sha256-or(@($job.evaluation_command)-join"`n")-ne(@($binding.command)-join"`n"))){Add-FailedAttempt $state $job 'evaluation_resume_validation' 'evaluation_command_path_args_or_script_hash_mismatch' $null|Out-Null;throw "evaluation_binding_changed:$($job.id)"}
    $job.evaluation_command=$binding.command;$job.evaluation_command_sha256=$binding.command_sha256;$job.evaluation_script_sha256=$binding.script_sha256;$job.status='evaluating';$job.updated_at=(Get-Date).ToString('o');$state.status='evaluating';$state.updated_at=$job.updated_at;Write-JsonAtomic $state $script:stateFullPath
    $evalArgs=@($script:evaluationFullPath,'--checkpoint',$training.checkpoint,'--variant',$job.variant,'--training-seed',"$($job.seed)",'--eval-seed',"$($config.evaluation_protocol.seed)",'--num-envs',"$($config.evaluation_protocol.num_envs)",'--protocol',$script:configFullPath,'--output',$evalPath,'--headless')
    & $script:pythonBat @evalArgs;$exitCode=$LASTEXITCODE;$finalizeExitCode=-1
    if($exitCode-eq 0-and(Test-Path -LiteralPath $evalPath -PathType Leaf)){& $script:pythonBat $script:evaluationFullPath --finalize-runtime --output $evalPath;$finalizeExitCode=$LASTEXITCODE}
    $evaluation=Test-EvaluationReport $evalPath $job $training $config
    if($exitCode-ne 0-or$finalizeExitCode-ne 0-or-not$evaluation.valid){$reason="evaluate_exit=$exitCode;finalize_exit=$finalizeExitCode;$($evaluation.reason)";Add-FailedAttempt $state $job 'evaluation_execution' $reason $exitCode|Out-Null;throw "evaluation_failed:$($job.id):$reason"}
    $job.attempts=@($job.attempts)+@([ordered]@{phase='evaluation_execution';at=(Get-Date).ToString('o');exit_code=0;validation_reason='passed';failure_fingerprint=$null});$job.status='complete';$job.updated_at=(Get-Date).ToString('o');$state.updated_at=$job.updated_at;Write-JsonAtomic $state $script:stateFullPath}
  $state.status=if($Smoke){'trained'}else{'complete'};$state.updated_at=(Get-Date).ToString('o');Write-JsonAtomic $state $script:stateFullPath;Write-Host "G005 $mode queue: $($state.status), jobs=$(@($state.jobs).Count), state=$(Get-PortablePath $script:stateFullPath)"
}catch{if(-not$script:failureRecorded-and$null-ne$script:lockStream){Write-PreflightFailure $_.Exception.Message};throw}
finally{if($null-ne$script:lockStream){$script:lockStream.Dispose();$script:lockStream=$null}}
