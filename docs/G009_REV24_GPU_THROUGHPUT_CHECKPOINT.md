# G009 R0 rev24 GPU throughput 실행 인계

- 체크포인트 날짜: 2026-09-01
- 증거 ID: `G009-5-E017`
- 구현 커밋: `134b6ee31c5ef220048eaaf632f7872a86122258`
- 실행 상태: 소스·사전등록·검증 완료, `1024/2048 env` 실제 실행 미시작
- 다음 세션의 중단 해제 조건: 원격 `main`과 일치하는 clean worktree

## 지금 멈춘 위치

rev24는 GPU에서 짧은 scratch PPO를 실행하기 직전까지 준비했다. 실제 Isaac Sim 프로세스, PPO rollout, optimizer update는 이번 체크포인트에서 시작하지 않았다. 새 checkpoint, TensorBoard 로그, MP4, GIF, PNG도 만들지 않았다. Garden 글은 실행 결과와 미디어가 생긴 뒤 다시 판단한다.

| 항목 | 현재 상태 |
| --- | --- |
| active R0 계약 | `g009_r0_recover_rev24` |
| active 계약 SHA-256 | `64eb108bb736d9ba8c1727c3a56ddc3fefaafaba25a98f93c1f8505704c5dd91` |
| solver | position `8`, velocity `0` |
| max depenetration velocity | `1.0m/s` |
| action | scale `0.70`, EMA alpha `0.2` |
| PPO 초기 noise | `0.5` |
| throughput protocol | `1024 env` PASS 뒤에만 `2048 env` |
| 실제 throughput 결과 | 없음 |
| policy qualification | `not_run` |
| recovery success claim | `false` |

기존 rev15 계약은 active baseline이 아니다. [`g009_r0_rev15.json`](../configs/g009_r0_rev15.json)에 historical snapshot으로 보존했고, 당시 계약 SHA-256은 `5f29ba19458404b5009d3734294c57e79294efecc7fe03bf8c71c71656129832`다. rev15의 solver position `16`은 GPU non-foot force gate 실패로 기각된 값이다.

## 이번에 고정한 실행 경로

실행은 저장소 wrapper [`bootstrap_benchmark_g009.py`](../scripts/bootstrap_benchmark_g009.py)에서 G009 task를 등록한 뒤 Isaac Lab 공식 `scripts/benchmarks/benchmark_rsl_rl.py`로 넘긴다. wrapper는 실행 전에 아래 세 조건을 확인한다.

1. Isaac Lab commit이 `90b79bb2d44feb8d833f260f2bf37da3487180ba`인지 확인한다.
2. 공식 benchmark SHA-256이 `2d5a88b9c07bfb38852082a0b9bf00f4213043b16ce0294776646ab06d351c82`인지 확인한다.
3. Isaac Lab의 추적 소스가 clean인지 확인한다.

저장소 쪽 실행 보고서는 [`run_training.ps1`](../scripts/run_training.ps1)이 만든다. 각 실행에 저장소 HEAD, 필수 source file별 SHA-256, source bundle SHA-256, 명령, 로그 디렉터리, checkpoint, TensorBoard, GPU 메모리·사용률·온도·소비전력과 recovery sample을 기록한다. 최종 판정은 [`summarize_g009_r0_rev24_gpu_throughput.py`](../scripts/summarize_g009_r0_rev24_gpu_throughput.py)가 맡는다.

## 실험 크기와 주장 범위

두 rung 모두 seed `42`, headless, scratch, `5 iterations`, `24 steps/env`, PPO learning epoch `5`, mini-batch `4`로 고정했다. resume와 Hydra override는 허용하지 않는다.

| rung | rollout transitions | optimizer mini-batch update | 용도 |
| ---: | ---: | ---: | --- |
| `1024 env` | `1024 × 24 × 5 = 122,880` | `5 × 5 × 4 = 100` | 첫 GPU 실행 건강·throughput gate |
| `2048 env` | `2048 × 24 × 5 = 245,760` | `5 × 5 × 4 = 100` | 1024 PASS 뒤 한 단계 확장 |

이 실행에는 PPO update가 실제로 들어간다. 다만 iteration이 5회뿐인 throughput smoke라서 복구 성능이나 정책 학습 완료를 평가하지 않는다. reward 상승, 우연한 upright frame, checkpoint 생성만으로 전복 복구 성공을 주장하지 않는다. matrix Gate01과 정식 R0 qualification은 별도 실험이다.

## PASS 조건

각 rung은 아래 조건을 전부 통과해야 한다.

- 프로세스 exit code `0`, traceback·error 없음
- 요청한 iteration 도달
- 새 로그 디렉터리를 정확히 하나로 판별
- TensorBoard event와 checkpoint 존재
- `steps/s` 표본이 정확히 5개이며 모두 양의 유한값
- visible GPU가 정확히 1개
- GPU total memory, peak used memory, utilization, temperature, power draw 측정 가능
- peak VRAM이 total VRAM의 `90%` 이하
- 종료 뒤 GPU memory가 baseline으로 회복
- 실행 보고서의 repository commit이 현재 HEAD와 일치
- 필수 source 12개와 source bundle이 해당 HEAD와 일치
- `numeric_invalid`가 보고되면 maximum이 정확히 `0`; metric이 없으면 unavailable로 기록

1024가 하나라도 실패하면 2048은 실행하지 않는다. 2048이 PASS해도 matrix Gate01이나 policy qualification을 자동 승인하지 않는다.

## 다음 세션 실행 순서

PowerShell 하나로 진행한다. 먼저 원격과 동기화된 clean 상태를 확인한다.

```powershell
cd $HOME\isaac-walk-rl
git pull --ff-only origin main
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
```

두 commit이 같고 `git status --short`가 비어 있을 때만 source binding을 고정한다.

```powershell
$sourceBindings = @(
  'configs/g009_r0.json',
  'configs/g009_r0_rev24_gpu_throughput.json',
  'scripts/bootstrap_benchmark_g009.py',
  'scripts/run_training.ps1',
  'scripts/summarize_g009_r0_rev24_gpu_throughput.py',
  'src/isaac_walk_g009/agent_cfg.py',
  'src/isaac_walk_g009/mdp/__init__.py',
  'src/isaac_walk_g009/mdp/events.py',
  'src/isaac_walk_g009/mdp/recover.py',
  'src/isaac_walk_g009/recover_contracts.py',
  'src/isaac_walk_g009/recover_env_cfg.py',
  'src/isaac_walk_g009/registry.py'
)
```

1024 rung을 먼저 실행한다.

```powershell
.\scripts\run_training.ps1 `
  -Task 'Isaac-G009-Recover-Flat-Go2-R0-v0' `
  -NumEnvs 1024 `
  -MaxIterations 5 `
  -Seed 42 `
  -RunName 'g009_r0_rev24_throughput_1024_s42' `
  -IsaacLabPath "$HOME\IsaacLab" `
  -ReportPath "$HOME\isaac-walk-rl\reports\runs\g009_r0_rev24_gpu_throughput_1024env_5iter_s42.json" `
  -TrainingEntrypointPath "$HOME\isaac-walk-rl\scripts\bootstrap_benchmark_g009.py" `
  -SourceBindingPaths $sourceBindings `
  -GpuSampleIntervalSeconds 1

py .\scripts\summarize_g009_r0_rev24_gpu_throughput.py `
  --input-1024 .\reports\runs\g009_r0_rev24_gpu_throughput_1024env_5iter_s42.json
```

partial synthesis의 outcome이 `awaiting_2048_input`인지 확인한다. 2048 실행 보고서도 clean launch를 기록해야 하므로, 1024 report와 partial synthesis는 잠시 저장소 밖으로 옮긴다. staging 경로가 이미 있으면 덮어쓰지 말고 이전 시도의 provenance부터 확인한다.

```powershell
$stagingPath = "$HOME\IsaacLab\logs\harness\g009_r0_rev24_staging_s42"
if (Test-Path -LiteralPath $stagingPath) { throw "staging path already exists: $stagingPath" }
New-Item -ItemType Directory -Path $stagingPath | Out-Null
Move-Item -LiteralPath .\reports\runs\g009_r0_rev24_gpu_throughput_1024env_5iter_s42.json -Destination $stagingPath
Move-Item -LiteralPath .\reports\runs\g009_r0_rev24_gpu_throughput_synthesis_s42.json -Destination $stagingPath
git status --short
```

worktree가 다시 clean일 때만 2048 rung을 실행한다.

```powershell
.\scripts\run_training.ps1 `
  -Task 'Isaac-G009-Recover-Flat-Go2-R0-v0' `
  -NumEnvs 2048 `
  -MaxIterations 5 `
  -Seed 42 `
  -RunName 'g009_r0_rev24_throughput_2048_s42' `
  -IsaacLabPath "$HOME\IsaacLab" `
  -ReportPath "$HOME\isaac-walk-rl\reports\runs\g009_r0_rev24_gpu_throughput_2048env_5iter_s42.json" `
  -TrainingEntrypointPath "$HOME\isaac-walk-rl\scripts\bootstrap_benchmark_g009.py" `
  -SourceBindingPaths $sourceBindings `
  -GpuSampleIntervalSeconds 1

Move-Item -LiteralPath "$stagingPath\g009_r0_rev24_gpu_throughput_1024env_5iter_s42.json" -Destination .\reports\runs

py .\scripts\summarize_g009_r0_rev24_gpu_throughput.py `
  --input-1024 .\reports\runs\g009_r0_rev24_gpu_throughput_1024env_5iter_s42.json `
  --input-2048 .\reports\runs\g009_r0_rev24_gpu_throughput_2048env_5iter_s42.json
```

staging에 남은 partial synthesis는 1024 단독 판정 원본으로 보존한다. 최종 synthesis가 두 입력의 순서와 시간 경계를 검증한다. 1024 종료 시각이 2048 시작 시각보다 늦으면 fail-closed한다.

## 미디어와 공개 기록

실제 rung을 실행한 뒤 단계마다 자료를 만든다.

| 번호 | 시점 | 공개 자료 | 로컬 전용 자료 |
| --- | --- | --- | --- |
| `15.01` | 1024 partial 판정 직후 | GPU telemetry GIF·대표 PNG | H.264 MP4 |
| `15.02` | 1024/2048 final 판정 직후 | 두 rung 비교 GIF·대표 PNG | H.264 MP4 |

throughput 단계는 카메라로 로봇 동작을 평가하는 실험이 아니다. 모든 프레임에 `PPO THROUGHPUT SMOKE / NOT POLICY QUALIFICATION / NOT RECOVERY EVIDENCE`를 표시한다. 실제 off-screen camera를 녹화하지 않았다면 robot camera footage라고 쓰지 않는다.

Garden은 이번 준비 상태만으로 새 글을 발행하지 않는다. 최종 runtime 결과와 `15.01/15.02`가 생긴 뒤 공개 가치가 있는지 다시 판단한다. 실패가 나면 내부 기록으로 먼저 보존하고, 후속 성공이나 원인 규명과 함께 설명할 수 있을 때 공개 글에 넣는다.

## throughput 뒤에 이어질 작업

1. rev24 final synthesis에서 stable maximum을 확정한다.
2. normal-force matrix를 policy observation에 연결하는 Gate01을 별도 사전등록한다.
3. matrix Gate01을 fresh scratch로 실행해 safety termination과 observation provenance를 확인한다.
4. Gate01 PASS 뒤에만 R0 scratch qualification을 실행한다.
5. 네 전복 자세별 성공률 `≥80%`, 중앙 복구시간 `≤4s`, safety termination `0`을 통과한 checkpoint만 복구 정책으로 승인한다.
6. 승인 뒤 `S1-low` 횡경사 WALK, 외란, residual height, 발별·공간 마찰, 낮은/높은 경사 RECOVER, link-mass 순으로 연다.

rev24 throughput은 이 긴 계보의 실행 용량을 고르는 단계다. 경사 보행, 마찰 적응, 링크 질량 변화, 전복 복구 성능은 아직 이 체크포인트의 주장 범위가 아니다.
