# G009 R0 rev24 GPU throughput 실행 인계

- 체크포인트 날짜: 2026-09-01
- 증거 ID: `G009-5-E017`
- 구현·실행 커밋: `0437e3766e6ff50a6b05a788a6cc7872ee582b89`
- 실행 상태: 첫 `1024 env`는 source-bundle 정렬 불일치로 기각했고, 수정 후 fresh `1024/2048 env`를 모두 PASS
- 확정 stable maximum: 이번 사전등록 ladder의 상한인 `2048 env`

## 지금 멈춘 위치

교차언어 ordinal 정렬을 수정한 clean commit에서 rev24의 `1024 env`와 `2048 env` headless scratch PPO throughput rung을 새 실행 ID와 새 checkpoint로 다시 실행했다. 두 rung 모두 wrapper run-health와 canonical source binding, VRAM·NaN·exit gate를 통과했고 최종 synthesis는 `stable_max_envs=2048`로 PASS했다. 이는 처리량 smoke이며 복구 정책 qualification이나 복구 성공 증거가 아니다. Matrix Gate01은 별도 rev25 계약·코드·검증기 구현과 사전등록을 완료했지만 actual GPU Gate01은 아직 실행하지 않았고, 정식 policy qualification도 미실행이다. `15.01/15.02` 미디어도 아직 만들지 않았으므로 Garden 발행 판단은 보류한다.

| 항목 | 현재 상태 |
| --- | --- |
| active R0 계약 | `g009_r0_recover_rev24` |
| active 계약 SHA-256 | `64eb108bb736d9ba8c1727c3a56ddc3fefaafaba25a98f93c1f8505704c5dd91` |
| solver | position `8`, velocity `0` |
| max depenetration velocity | `1.0m/s` |
| action | scale `0.70`, EMA alpha `0.2` |
| PPO 초기 noise | `0.5` |
| throughput protocol | `1024 env` PASS 뒤에만 `2048 env` |
| 실제 throughput 결과 | fresh 1024·2048 canonical PASS, stable maximum `2048 env` |
| policy qualification | `not_run` |
| recovery success claim | `false` |

## 2026-09-04 첫 1024 실행과 기각 판정

원격 `main`과 일치하는 clean source commit `1c0b35c2f13b77620d57ad62175ddb87f68bf828`에서 `1024 env × 24 steps × 5 iterations`, seed `42`, headless scratch PPO를 실제 실행했다. Isaac Lab commit과 공식 benchmark SHA-256은 고정 계약과 일치했고 프로세스 exit code는 `0`이었다.

| 항목 | 실제 측정값 |
| --- | ---: |
| `steps/s` | `7,355`, `11,977`, `11,829`, `12,220`, `11,614` |
| 평균 / 중앙값 | `10,999 / 11,829 steps/s` |
| peak VRAM | `4,397 / 12,288 MiB` (`35.78%`) |
| peak / mean GPU utilization | `64% / 9.63%` |
| peak temperature / power | `55°C / 61.8W` |
| numeric invalid maximum | `0` |
| checkpoint | `model_4.pt`, SHA-256 `0d745dca05a97dd1849b584a0dae100642990afaf07432f416910507c41b67be` |

하지만 canonical synthesis는 `source_bundle_matches_commit=false`로 fail-closed했다. `run_training.ps1`은 `Sort-Object`의 Windows 문화권 순서로 source path를 배열했고 Python synthesis는 Unicode ordinal 순서로 배열했다. 이 때문에 동일한 12개 파일과 동일한 개별 SHA-256을 사용했지만 aggregate SHA-256의 바이트 순서가 달라졌다. runtime report의 `source_bundle.matches_repository_commit=true`만으로 이 불일치를 우회하지 않는다.

- rejected raw report: `reports/runs/g009_r0_rev24_gpu_throughput_1024env_5iter_s42_rejected_ordinal_sort.json`, SHA-256 `fd798eb3afd6e79e123f1d85971b6a9b24599f4a4724aa94ca6c06cbfe57c828`
- rejected synthesis: `reports/runs/g009_r0_rev24_gpu_throughput_synthesis_s42_rejected_ordinal_sort.json`, SHA-256 `6782044cbd46dadc4f2becba4eb5d5efb27ebcbc182447630980d75f0dbef596`
- stdout: `%USERPROFILE%\IsaacLab\logs\harness\g009_r0_rev24_throughput_1024_s42.stdout.log`, SHA-256 `af8b9555de934318b3d4f555968f4e8aee3c6c588c5234cfb6669a951732d5e9`
- stderr: 같은 harness 경로의 `.stderr.log`, 빈 파일 SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

따라서 이 실행은 PPO 처리량 참고 수치일 뿐 rev24 1024 PASS가 아니다. `2048 env`는 실행하지 않았고 stable maximum도 확정하지 않았다. ordinal source 정렬 수정과 회귀 테스트가 clean commit으로 반영된 뒤 1024부터 새 report·checkpoint로 다시 실행한다. 기각된 단계에는 `15.01` 성공 미디어를 만들지 않으며, 재실행 PASS 뒤에만 `PPO THROUGHPUT SMOKE / NOT POLICY QUALIFICATION / NOT RECOVERY EVIDENCE` 표기를 붙여 생성한다.

## 2026-09-04 수정 후 canonical 재실행

source path를 repository-relative 실제 casing으로 정규화하고 ordinal 정렬하는 수정이 반영된 clean commit `0437e3766e6ff50a6b05a788a6cc7872ee582b89`에서 두 rung을 새 실행 ID로 수행했다. source bundle SHA-256은 두 실행 모두 `5b244f87797754926c30dbc64a30fad9d7220859f160cfef2571c60cc25aaec0`으로 일치했다.

| 항목 | `1024 env` | `2048 env` |
| --- | ---: | ---: |
| run name | `g009_r0_rev24_throughput_1024_retry01_s42` | `g009_r0_rev24_throughput_2048_retry01_s42` |
| `steps/s` | `8062, 12549, 12941, 13128, 12670` | `14874, 22947, 23020, 23168, 22576` |
| 평균 / 중앙값 | `11,870 / 12,670` | `21,317 / 22,947` |
| peak VRAM | `4,397 MiB` | `4,859 MiB` |
| peak / mean utilization | `55% / 14.96%` | `67% / 15.58%` |
| peak temperature / power | `54°C / 63.74W` | `55°C / 68.54W` |
| numeric invalid maximum | `0` | `0` |
| checkpoint SHA-256 | `0d745dca05a97dd1849b584a0dae100642990afaf07432f416910507c41b67be` | `fce57b96b0c3b0ff50e85cae273b8bc11d91c6f63a6417b69f67e91609a40e41` |
| canonical decision | PASS | PASS |

- 1024 report SHA-256: `5f39c701bbfd889c2a44f470abcf1d4e4632398e34fd9d9e70406cc7da51fb50`
- 2048 report SHA-256: `27da732b114dc2c6432926814777d4335e5df4309acaf760af45288abb1ca8e9`
- final synthesis SHA-256: `15281e134159974525fc53050e186e8e16d79108f9a068fe717d8ea26b805358`

따라서 이번 ladder 안의 안정 최대 환경 수는 `2048`이다. 더 큰 수를 시험하지 않았으므로 GPU의 절대 최대치라는 뜻은 아니다. 학습 중 hard-joint-limit 통계는 1024에서 최대 `0.25`, 2048에서 최대 `0.2083333`이었지만 rev24 throughput smoke는 이를 qualification gate로 사용하지 않는다. rev25 Matrix Gate01과 이후 정식 qualification은 hard-joint-limit maximum `0`을 별도로 요구한다.

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

위 명령은 완료된 rev24 실행을 재현하기 위한 역사적 절차다. 실제 실행에서는 staging에 남은 partial synthesis를 1024 단독 판정 원본으로 보존했고, 최종 synthesis가 두 입력의 순서와 시간 경계를 검증했다. 1024 종료 시각이 2048 시작 시각보다 늦으면 fail-closed하는 계약도 유지했다. rev24는 canonical 1024·2048 PASS와 `stable_max_envs=2048` 확정까지 완료됐다.

## 미디어와 공개 기록

rev24의 실제 rung과 final synthesis는 완료됐지만 `15.01/15.02` 미디어는 만들지 않았다. 아래 표는 검증된 report를 바탕으로 회고 미디어를 만들 경우 적용할 계약이며, 미디어 부재는 rev24 throughput PASS를 복구 정책 성공 영상으로 대체하지 않았다는 뜻이다.

| 번호 | 시점 | 공개 자료 | 로컬 전용 자료 |
| --- | --- | --- | --- |
| `15.01` | 1024 partial 판정 직후 | GPU telemetry GIF·대표 PNG | H.264 MP4 |
| `15.02` | 1024/2048 final 판정 직후 | 두 rung 비교 GIF·대표 PNG | H.264 MP4 |

throughput 단계는 카메라로 로봇 동작을 평가하는 실험이 아니다. 모든 프레임에 `PPO THROUGHPUT SMOKE / NOT POLICY QUALIFICATION / NOT RECOVERY EVIDENCE`를 표시한다. 실제 off-screen camera를 녹화하지 않았다면 robot camera footage라고 쓰지 않는다.

원본 MP4는 30fps로 보존한다. 공개 GIF는 15fps를 목표로 만들며 12fps 아래로 낮추지 않는다. `15.01/15.02`처럼 그래프를 직접 그리는 텔레메트리는 몇 장의 정지 화면을 오래 표시하지 않고, 재생 구간 전체의 중간 프레임을 렌더링한다. GIF가 10 MiB를 넘으면 길이, 해상도, 팔레트 순서로 줄이고 프레임레이트는 유지한다. sidecar에는 source FPS, target·actual GIF FPS, frame count, duration, 가장 긴 frame 표시 시간과 temporal strategy를 남긴다. 고정 압축 우선순위와 실제 적용 단계는 각각 `compression_policy_order`, `compression_steps_applied`에 구분해 기록한다. `15.01/15.02` builder는 MP4를 ffprobe로 검사하고 실제 GIF를 `inspect_gif_encoding()`으로 측정한 뒤 계약 검사를 통과해야 한다.

Garden 새 글은 아직 발행하지 않는다. rev24 canonical runtime 결과는 확보했지만 `15.01/15.02` 미디어가 없고 rev25 Matrix Gate01 actual run도 미실행이므로, 후속 실행과 미디어가 갖춰진 뒤 공개 가치를 다시 판단한다. 실패가 나면 내부 기록으로 먼저 보존하고, 후속 성공이나 원인 규명과 함께 설명할 수 있을 때 공개 글에 넣는다.

## 현재 상태와 rev24 이후 작업

1. `[완료]` rev24 final synthesis에서 canonical `1024 env`와 `2048 env` PASS를 확인하고 stable maximum을 `2048 env`로 확정했다.
2. `[준비 완료·미실행]` rev25 whole-body terrain-contact Matrix Gate01의 계약·코드·검증기를 사전등록했다.
3. `[다음 실행]` Matrix Gate01을 fresh scratch `1024 env × 24 steps × 1 iteration`으로 실행해 connectivity, safety termination, observation provenance를 확인한다.
4. Gate01 PASS 뒤에만 R0 scratch qualification을 실행한다.
5. 네 전복 자세별 성공률 `≥80%`, 중앙 복구시간 `≤4s`, safety termination `0`을 통과한 checkpoint만 복구 정책으로 승인한다.
6. 승인 뒤 `S1-low` 횡경사 WALK, 외란, residual height, 발별·공간 마찰, 낮은/높은 경사 RECOVER, link-mass 순으로 연다.

rev24 throughput은 이 긴 계보의 실행 용량을 고르는 단계다. 경사 보행, 마찰 적응, 링크 질량 변화, 전복 복구 성능은 아직 이 체크포인트의 주장 범위가 아니다.
