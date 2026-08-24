[CmdletBinding()]param()
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
function Assert-True{param([bool]$Condition,[string]$Message)if(-not$Condition){throw"ASSERT FAIL: $Message"}}
function Invoke-Queue{param([string]$Root,[switch]$Resume,[switch]$Smoke,[string]$Evaluator=$script:mockEval,[string]$StatePath)
  if([string]::IsNullOrWhiteSpace($StatePath)){$StatePath=Join-Path $Root 'state.json'}
  $args=@('-NoProfile','-File',$script:queue,'-ConfigPath',$script:configPath,'-StatePath',$StatePath,'-ReportRoot',$Root,'-TrainingHarness',$script:mockTrain,'-EvaluationScript',$Evaluator,'-IsaacLabPath',$script:lab)
  if($Resume){$args+='-Resume'};if($Smoke){$args+='-Smoke'};$null=& $script:pwsh @args 2>$null;return $LASTEXITCODE}
function New-Scenario{param([string]$Name)$path=Join-Path $script:testRoot $Name;New-Item -ItemType Directory -Path $path|Out-Null;$path}

$repoRoot=Split-Path -Parent $PSScriptRoot;$script:queue=Join-Path $repoRoot 'scripts\run_reward_ablation.ps1';$runTraining=Join-Path $repoRoot 'scripts\run_training.ps1'
$script:configPath=Join-Path $repoRoot 'configs\g005_reward_ablation.json';$script:pwsh=(Get-Process -Id $PID).Path;$script:lab=Join-Path $HOME 'IsaacLab'
$script:testRoot=Join-Path ([IO.Path]::GetTempPath()) ('isaac-walk-g005-review-'+[guid]::NewGuid().ToString('N'));New-Item -ItemType Directory -Path $script:testRoot|Out-Null
try{
  $script:mockTrain=Join-Path $script:testRoot 'mock_training.ps1'
  $trainSource=@'
param([string]$Task,[int]$NumEnvs,[int]$MaxIterations,[int]$Seed,[string]$RunName,[string]$IsaacLabPath,[string]$ReportPath,[string]$HydraOverridesBase64)
$overrides=@([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($HydraOverridesBase64))|ConvertFrom-Json)
if($env:G005_TRAIN_COUNT){$n=if(Test-Path $env:G005_TRAIN_COUNT){[int](Get-Content $env:G005_TRAIN_COUNT -Raw)}else{0};[IO.File]::WriteAllText($env:G005_TRAIN_COUNT,"$($n+1)")}
if($env:G005_TRAIN_FAIL -eq'1'){exit 19}
$dir=Join-Path (Split-Path -Parent $ReportPath) ('artifact-'+$RunName);New-Item -ItemType Directory -Path $dir -Force|Out-Null;$checkpoint=Join-Path $dir 'model.pt';[IO.File]::WriteAllText($checkpoint,$RunName)
$hash=(Get-FileHash $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant();$report=[ordered]@{passed=$true;task=$Task;num_envs=$NumEnvs;max_iterations=$MaxIterations;seed=$Seed;run_name=$RunName;effective_hydra_overrides=$overrides;success_checks=[ordered]@{gpu_measurement_complete=$true;gpu_recovered_to_baseline=$true;tensorboard_exists=$true;checkpoint_exists=$true};artifacts=[ordered]@{checkpoint=$checkpoint;checkpoint_sha256=$hash};fatal_patterns_found=@()}
$tmp=$ReportPath+'.tmp';[IO.File]::WriteAllText($tmp,($report|ConvertTo-Json -Depth 6));[IO.File]::Move($tmp,$ReportPath,$true);exit 0
'@;[IO.File]::WriteAllText($script:mockTrain,$trainSource,[Text.UTF8Encoding]::new($false))
  $script:mockEval=Join-Path $script:testRoot 'mock_eval.py'
  $evalSource=@'
import argparse,hashlib,json,os,pathlib,sys
if '--finalize-runtime' in sys.argv:
 o=pathlib.Path(sys.argv[sys.argv.index('--output')+1]);r=json.loads(o.read_text());r['runtime_evidence'].update({'app_close_completed':True,'finalized_after_process_exit':True,'gpu_after':{'measurement_complete':True},'gpu_recovered_to_baseline':True,'process_recovered':True,'fatal_scan':{'measurement_complete':True,'patterns':[],'count':0}});t=o.with_suffix(o.suffix+'.tmp');t.write_text(json.dumps(r));os.replace(t,o);sys.exit(0)
p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--variant');p.add_argument('--training-seed',type=int);p.add_argument('--eval-seed',type=int);p.add_argument('--num-envs',type=int);p.add_argument('--protocol');p.add_argument('--output');p.add_argument('--headless',action='store_true');a=p.parse_args()
count=os.environ.get('G005_EVAL_COUNT')
if count:
 q=pathlib.Path(count);q.write_text(str((int(q.read_text()) if q.exists() else 0)+1))
if os.environ.get('G005_EVAL_FAIL')=='1': sys.exit(17)
d=json.load(open(a.protocol,encoding='utf-8'));c=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest();v=next(x for x in d['variants'] if x['name']==a.variant)
metric={'sample_count':10000,'lin_vel_rmse_mps':1.0,'yaw_rate_rmse_radps':0.5,'torque_l2_mean':2.0,'absolute_mechanical_power_w':3.0,'action_rate_l2_mean':4.0,'feet_air_time_raw_mean':0.1,'mean_air_time_at_first_contact_s':0.2,'first_contact_count':20,'fall_count':0,'timeout_count':0,'reset_count':0,'fall_timeout_overlap_count':0,'fall_trial_rate':0.0,'survival_rate':1.0,'trials_started':10}
commands=[{'id':f'vx{vx:+.1f}_vy{vy:+.1f}_yaw{yaw:+.1f}','vx_mps':vx,'vy_mps':vy,'yaw_rate_radps':yaw} for vx in [-1.0,0.0,1.0] for vy in [-0.5,0.0,0.5] for yaw in [-0.5,0.0,0.5] if [vx,vy,yaw]!=[0.0,0.0,0.0]]
overall=dict(metric,sample_count=260000,first_contact_count=520,trials_started=260);ch=hashlib.sha256(pathlib.Path(a.checkpoint).read_bytes()).hexdigest();r={'schema_version':1,'variant':a.variant,'training_seed':a.training_seed,'evaluation_seed':a.eval_seed,'config_sha256':c(d),'config_file_sha256':hashlib.sha256(pathlib.Path(a.protocol).read_bytes()).hexdigest(),'variant_config_sha256':c(v),'protocol_sha256':c(d['evaluation_protocol']),'checkpoint_sha256':ch,'checkpoint':{'reference':str(pathlib.Path(a.checkpoint).resolve()),'sha256':ch},'protocol_compliant':True,'task':d['evaluation_protocol']['task'],'num_envs':a.num_envs,'horizon_steps':d['evaluation_protocol']['horizon_steps'],'step_dt':d['evaluation_protocol']['step_dt'],'effective_weights':v['weights'],'denominators':{x:x+' definition' for x in ['sample_count','fall_trial_rate','survival_rate','trials_started','fall_timeout_overlap_count']},'metrics':{'overall':overall,'by_command':[dict(metric,command=cmd) for cmd in commands]},'runtime_evidence':{'exit_code':0,'app_close_completed':False,'finalized_after_process_exit':False,'gpu_recovered_to_baseline':None,'process_recovered':None,'gpu_after':None,'fatal_scan':None}}
o=pathlib.Path(a.output);t=o.with_suffix(o.suffix+'.tmp');t.write_text(json.dumps(r),encoding='utf-8');os.replace(t,o)
'@;[IO.File]::WriteAllText($script:mockEval,$evalSource,[Text.UTF8Encoding]::new($false))

  $config=Get-Content $script:configPath -Raw|ConvertFrom-Json;Assert-True((@($config.variants).Count*@($config.seeds).Count)-eq12)'manifest 4x3'

  $success=New-Scenario 'success';$env:G005_TRAIN_COUNT=Join-Path $success 'train.count';$env:G005_EVAL_COUNT=Join-Path $success 'eval.count'
  Assert-True((Invoke-Queue $success)-eq0)'production mock success';$state=Get-Content (Join-Path $success 'state.json')-Raw|ConvertFrom-Json
  Assert-True($state.status -eq 'complete' -and @($state.jobs|Where-Object { $_.status -eq 'complete' }).Count -eq 12)'12 jobs complete'
  Assert-True($state.config_sha256-eq'3e8455a9efba77f67b2ac436d5eef41421dfeac10f9e67ab9620c6775b6c2576' -and $state.config_file_sha256-match'^[0-9a-f]{64}$' -and $state.protocol_sha256-match'^[0-9a-f]{64}$')'state canonical/raw/protocol hash contract'
  Assert-True([int](Get-Content $env:G005_TRAIN_COUNT -Raw)-eq12 -and [int](Get-Content $env:G005_EVAL_COUNT -Raw)-eq12)'12 train/eval calls'
  $summaryPath=Join-Path $success 'summary.json';$null=& (Join-Path $script:lab '_isaac_sim\python.bat') (Join-Path $repoRoot 'scripts\summarize_reward_ablation.py') --manifest $script:configPath --queue (Join-Path $success 'state.json') --output $summaryPath 2>$null
  Assert-True($LASTEXITCODE-eq0 -and (Get-Content $summaryPath -Raw|ConvertFrom-Json).job_completeness.complete-eq12)'actual queue to summarizer E2E'
  Assert-True((Invoke-Queue $success -Resume)-eq0)'complete resume valid';Assert-True([int](Get-Content $env:G005_TRAIN_COUNT -Raw)-eq12)'complete resume training skip'
  $firstEval=Join-Path $success 'g005_production_baseline_s42_evaluation.json';$evalOriginal=[IO.File]::ReadAllText($firstEval);$tampered=$evalOriginal|ConvertFrom-Json;$tampered.metrics.overall.PSObject.Properties.Remove('mean_air_time_at_first_contact_s');[IO.File]::WriteAllText($firstEval,($tampered|ConvertTo-Json -Depth 12))
  Assert-True((Invoke-Queue $success -Resume)-ne0)'minimal eval missing required metric rejected';[IO.File]::WriteAllText($firstEval,$evalOriginal);Assert-True((Invoke-Queue $success -Resume)-eq0)'minimal eval recovery'
  $evalOriginal=[IO.File]::ReadAllText($firstEval);$tampered=$evalOriginal|ConvertFrom-Json;$tampered.metrics.by_command[1].command=$tampered.metrics.by_command[0].command;[IO.File]::WriteAllText($firstEval,($tampered|ConvertTo-Json -Depth 12))
  Assert-True((Invoke-Queue $success -Resume)-ne0)'duplicate command identity rejected';[IO.File]::WriteAllText($firstEval,$evalOriginal);Assert-True((Invoke-Queue $success -Resume)-eq0)'command identity recovery'
  $evalOriginal=[IO.File]::ReadAllText($firstEval);$tampered=$evalOriginal|ConvertFrom-Json;$tampered.metrics.overall.torque_l2_mean=99.0;[IO.File]::WriteAllText($firstEval,($tampered|ConvertTo-Json -Depth 12))
  Assert-True((Invoke-Queue $success -Resume)-ne0)'overall weighted inconsistency rejected';[IO.File]::WriteAllText($firstEval,$evalOriginal);Assert-True((Invoke-Queue $success -Resume)-eq0)'weighted consistency recovery'
  $evalOriginal=[IO.File]::ReadAllText($firstEval);$tampered=$evalOriginal|ConvertFrom-Json;$tampered.protocol_compliant=$false;[IO.File]::WriteAllText($firstEval,($tampered|ConvertTo-Json -Depth 12))
  Assert-True((Invoke-Queue $success -Resume)-ne0)'tampered complete eval rejected';[IO.File]::WriteAllText($firstEval,$evalOriginal);Assert-True((Invoke-Queue $success -Resume)-eq0)'tampered eval recovery reuses training'
  $evalOriginal=[IO.File]::ReadAllText($firstEval);[IO.File]::WriteAllText($firstEval,'{bad');Assert-True((Invoke-Queue $success -Resume)-ne0)'corrupt complete eval rejected';[IO.File]::WriteAllText($firstEval,$evalOriginal);Assert-True((Invoke-Queue $success -Resume)-eq0)'corrupt eval recovery'
  [IO.File]::Delete($firstEval);Assert-True((Invoke-Queue $success -Resume)-ne0)'deleted complete eval rejected'
  $state=Get-Content (Join-Path $success 'state.json')-Raw|ConvertFrom-Json;Assert-True($state.jobs[0].attempts[-1].validation_reason-like'missing_json*')'missing eval durable reason'

  $resume=New-Scenario 'eval-resume';$env:G005_TRAIN_COUNT=Join-Path $resume 'train.count';$env:G005_EVAL_COUNT=Join-Path $resume 'eval.count';$env:G005_EVAL_FAIL='1'
  Assert-True((Invoke-Queue $resume)-ne0)'eval first failure';Remove-Item Env:G005_EVAL_FAIL
  Assert-True((Invoke-Queue $resume -Resume)-eq0)'eval resume success';Assert-True([int](Get-Content $env:G005_TRAIN_COUNT -Raw)-eq12)'eval resume reused valid training'

  $strike=New-Scenario 'eval-strike';$env:G005_TRAIN_COUNT=Join-Path $strike 'train.count';$env:G005_EVAL_COUNT=Join-Path $strike 'eval.count';$env:G005_EVAL_FAIL='1'
  1..3|ForEach-Object{Assert-True((Invoke-Queue $strike -Resume:($_-gt1))-ne0)"eval strike $_"};Remove-Item Env:G005_EVAL_FAIL
  $s=Get-Content (Join-Path $strike 'state.json')-Raw|ConvertFrom-Json;Assert-True($s.jobs[0].hard_blocked-eq$true)'third eval fingerprint hard block';Assert-True(@($s.jobs[0].attempts|Where-Object failure_fingerprint -eq $s.jobs[0].attempts[-1].failure_fingerprint).Count-eq3)'same eval fingerprint count3'

  $corrupt=New-Scenario 'corrupt-training';$env:G005_TRAIN_COUNT=Join-Path $corrupt 'train.count';Remove-Item Env:G005_EVAL_COUNT -ErrorAction SilentlyContinue
  Assert-True((Invoke-Queue $corrupt -Smoke)-eq0)'smoke for corrupt training';[IO.File]::WriteAllText((Join-Path $corrupt 'g005_smoke_baseline_s42.json'),'{bad')
  Assert-True((Invoke-Queue $corrupt -Smoke -Resume)-ne0)'corrupt training durable fail';$s=Get-Content (Join-Path $corrupt 'state.json')-Raw|ConvertFrom-Json;Assert-True($s.jobs[0].attempts[-1].validation_reason-like'corrupt_json*')'corrupt training reason'

  $missing=New-Scenario 'missing-training';$env:G005_TRAIN_COUNT=Join-Path $missing 'train.count';Assert-True((Invoke-Queue $missing -Smoke)-eq0)'smoke for missing training';[IO.File]::Delete((Join-Path $missing 'g005_smoke_baseline_s42.json'))
  Assert-True((Invoke-Queue $missing -Smoke -Resume)-ne0)'missing training durable fail';$s=Get-Content (Join-Path $missing 'state.json')-Raw|ConvertFrom-Json;Assert-True($s.jobs[0].attempts[-1].validation_reason-like'missing_json*')'missing training reason'

  $binding=New-Scenario 'binding';$env:G005_TRAIN_COUNT=Join-Path $binding 'train.count';$env:G005_EVAL_COUNT=Join-Path $binding 'eval.count';$env:G005_EVAL_FAIL='1';Assert-True((Invoke-Queue $binding)-ne0)'binding seeded';Remove-Item Env:G005_EVAL_FAIL
  [IO.File]::AppendAllText($script:mockEval,"`n# changed")
  Assert-True((Invoke-Queue $binding -Resume)-ne0)'evaluator script change rejected';$s=Get-Content (Join-Path $binding 'state.json')-Raw|ConvertFrom-Json;Assert-True($s.jobs[0].attempts[-1].validation_reason-eq'evaluation_command_path_args_or_script_hash_mismatch')'binding reason durable'

  $lock=New-Scenario 'lock';$stateA=Join-Path $lock 'state-a.json';$stateB=Join-Path $lock 'state-b.json';[IO.File]::WriteAllText($stateA,'{"sentinel":true}');$before=(Get-FileHash $stateA).Hash;$commonLock=Join-Path $lock '.g005_reward_ablation.queue.lock';$stream=[IO.File]::Open($commonLock,[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
  try{Assert-True((Invoke-Queue $lock -Resume -StatePath $stateB)-ne0)'same ReportRoot different StatePath concurrent lock reject';Assert-True((Get-FileHash $stateA).Hash-eq$before -and -not(Test-Path $stateB))'namespace lock preserves both states'}finally{$stream.Dispose()}

  $preflight=New-Scenario 'preflight';$badState=Join-Path $preflight 'state.json';[IO.File]::WriteAllText($badState,'{corrupt-original');$before=(Get-FileHash $badState).Hash;$null=& $script:pwsh -NoProfile -File $script:queue -Resume -ConfigPath $script:configPath -StatePath $badState -ReportRoot $preflight -TrainingHarness $script:mockTrain -EvaluationScript $script:mockEval -IsaacLabPath $script:lab 2>$null
  $pfPath=$badState+'.preflight_failure.json';Assert-True($LASTEXITCODE-ne0 -and (Get-FileHash $badState).Hash-eq$before -and (Test-Path $pfPath))'corrupt state preserved with sidecar evidence';$pf=Get-Content $pfPath -Raw|ConvertFrom-Json;Assert-True($pf.phase-eq'preflight' -and $pf.source_state_sha256-eq$before)'preflight sidecar source hash'
  [IO.File]::Delete($badState);Assert-True((Invoke-Queue $preflight -Smoke -StatePath $badState)-eq0)'operator cleanup allows normal recovery after sidecar evidence'

  $safe=@('env.rewards.dof_torques_l2.weight=-0.0002','env.rewards.action_rate_l2.weight=-0.01','env.rewards.feet_air_time.weight=0.25','agent.experiment_name=g005_go2_flat_reward_ablation')|ConvertTo-Json -Compress;$safe64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($safe))
  $safeOutput=@(& $script:pwsh -NoProfile -File $runTraining -Task x -NumEnvs 1 -MaxIterations 1 -RunName safe_override -IsaacLabPath 'Z:\missing' -HydraOverridesBase64 $safe64 2>&1);Assert-True($LASTEXITCODE-ne0 -and ($safeOutput-join' ')-match'python.bat')'normal override accepted through safety gate'
  $emptyOutput=@(& $script:pwsh -NoProfile -File $runTraining -Task x -NumEnvs 1 -MaxIterations 1 -RunName no_override -IsaacLabPath 'Z:\missing' 2>&1);Assert-True($LASTEXITCODE-ne0 -and ($emptyOutput-join' ')-match'python.bat')'no-override accepted through safety gate'

  foreach($meta in @('&','|','<','>','^','%','!','"',"`t")){$ov=@("agent.experiment_name=x${meta}bad")|ConvertTo-Json -Compress;$b64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($ov));
    $maliciousOutput=@(& $script:pwsh -NoProfile -File $runTraining -Task x -NumEnvs 1 -MaxIterations 1 -RunName malicious_test -IsaacLabPath $script:lab -HydraOverridesBase64 $b64 2>&1);Assert-True($LASTEXITCODE-ne0 -and ($maliciousOutput-join' ')-match'안전하지 않은 Hydra override')"malicious override exact rejection: $meta"}
  Write-Host 'G005 review assertions PASS: canonical hashes, independent resume, complete integrity, 3-strike, corrupt JSON, binding hash, lock, Hydra meta'
}finally{Remove-Item Env:G005_TRAIN_COUNT,Env:G005_EVAL_COUNT,Env:G005_EVAL_FAIL,Env:G005_TRAIN_FAIL -ErrorAction SilentlyContinue;if(Test-Path $script:testRoot){Remove-Item $script:testRoot -Recurse -Force}}
