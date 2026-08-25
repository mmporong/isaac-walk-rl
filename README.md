# Isaac Walk RL

Windows 네이티브 환경에서 Isaac Sim 4.5와 Isaac Lab 2.1.1을 사용해 사족보행 PPO 실험을 재현하고, 보상·지형·외란 회복을 한 축씩 확장하는 프로젝트입니다.

## 고정 스택

| 구성 | 고정값 | 상태 |
| --- | --- | --- |
| 운영체제 | Windows 11 네이티브 | 확정 |
| Isaac Sim | 4.5.0 binary | 로컬 설치·Python smoke 확인 |
| Isaac Lab | v2.1.1 / `90b79bb2d44feb8d833f260f2bf37da3487180ba` | 소스 설치·태그 확인 |
| Python | Isaac Sim 번들 3.10.x | 로컬 3.10.15 확인 |
| RL | RSL-RL PPO / `rsl-rl-lib==2.3.3` | 설치·CUDA import 확인 |
| 관측 | 상태 기반, headless | 확정 |

이 프로젝트는 `Ubuntu 24.04/Jazzy` 기본 로보틱스 환경의 예외입니다. RL 학습 자체에는 ROS 2나 WSL2가 필요하지 않으므로 설치하지 않습니다.

## 검토 후 보정한 전제

- Isaac Lab 2.1.1은 `pip install isaaclab==2.1.1` 대상이 아닙니다. 공식 태그 소스를 저장소 밖에 clone해 설치합니다.
- Isaac Lab 2.2는 Sim 4.5에서 무조건 사용할 수 없는 버전이 아닙니다. 이 프로젝트는 재현성을 위해 2.1.1을 고정합니다.
- RTX 3060 12GB에서 2048/4096 environments가 된다는 보장은 없습니다. 64부터 단계적으로 올리며 peak VRAM과 steps/s를 실측합니다.
- Go2를 “가장 많이 쓰이는 모델”이라고 단정하지 않습니다. ANYmal-C 공식 baseline을 관문으로 삼고, Isaac Lab 내장 Go2 태스크를 심화 대상으로 사용합니다.
- RBQ v1.20.0에는 공개 URDF·STL 경로가 있습니다. 공식 Isaac Lab v2.1.1·v2.3.2·조사 시점 main에는 대상 구현이 없으므로, 마지막 단계는 외부 커스텀 자산의 라이선스와 호환성을 fail-closed로 판정합니다.

## 실행 순서

1. Isaac Lab v2.1.1 소스 설치 및 등록 태스크 검증
2. ANYmal-C flat 50-iteration smoke와 300-iteration baseline
3. Go2 flat 환경 수별 VRAM·steps/s 측정
4. 세 보상 항목의 one-factor ablation
5. official Go2 rough baseline에서 terrain curriculum과 official domain randomization 공통 조건 고정
6. 동일 rough·공통 official DR 조건에서 `events.push_robot`만 변경한 push curriculum의 고정 프로토콜 비교
7. RBQ 외부 자산 라이선스·호환성 사전조사

구체적인 명령과 단계별 완료 조건은 `PROMPT_WINDOWS.md`, 측정 상태는 `docs/VALIDATION_MATRIX.md`, 모든 실행 기록은 `RUN_NOTES.md`에서 관리합니다.

## 저장소 경계

- 이 저장소: 커스텀 코드, 설정, 재현 스크립트, 문서, 정량 결과표, 검증용 소형 GIF·스크린샷
- 저장소 밖: `%USERPROFILE%\IsaacLab`, `E:\IsaacSim\isaac-sim-4.5.0`, 학습 로그, 체크포인트, 원본 영상, 중간 생성 자산

## 환경 매니페스트와 저장소 검증

이 프로젝트의 명령은 PowerShell 7.x(`pwsh`)에서 실행합니다. 현재 검증 버전은 7.6.5이며, Windows PowerShell 5.1은 지원 검증 대상이 아닙니다.

다음 명령을 실행하면 사용자명이나 인증정보를 기록하지 않고 Git, GPU, Isaac 설치 상태와 현재 commit 상태를 `reports/environment_manifest.json`에 갱신합니다.

```powershell
cd "$HOME\isaac-walk-rl"
.\scripts\collect_environment.ps1
.\scripts\verify_isaaclab.ps1
.\scripts\validate_repository.ps1
```

Isaac Lab을 다른 위치에 설치한 경우 `-IsaacLabPath` 인자로 경로를 지정합니다. 사용자 홈 아래 경로는 보고서에서 `%USERPROFILE%`로 치환됩니다. Isaac Lab v2.1.1의 Windows `isaaclab.bat -p`는 nested batch를 `call` 없이 실행해 성공 후에도 exit 1을 전달할 수 있으므로, runtime 검증은 동일한 공식 bundled `python.bat`를 직접 사용하고 wrapper 문제는 경고로 기록합니다.

## ANYmal-C 학습 하네스

`scripts/run_training.ps1`은 Isaac Sim bundled `python.bat`로 headless RSL-RL 학습을 실행하고, 원시 로그는 저장소 밖 `%USERPROFILE%\IsaacLab\logs\harness`에 두며 휴대 가능한 JSON 요약만 `reports/runs`에 기록합니다.

```powershell
cd "$HOME\isaac-walk-rl"
.\scripts\run_training.ps1 -Task Isaac-Velocity-Flat-Anymal-C-v0 -NumEnvs 64 -MaxIterations 50 -Seed 42 -RunName "anymalc_flat_smoke_s42_YYYYMMDD-HHmm"
```

G003에서는 1-iteration probe, 50-iteration smoke, 300-iteration flat baseline을 순서대로 통과했습니다. 측정값과 체크포인트 해시는 `reports/runs/g003_anymal_summary.json`을 기준으로 봅니다.

## Go2 환경 수 scale ladder

`scripts/run_scale_ladder.ps1`은 Go2 flat 태스크를 seed 42, 각 10 iterations로 64→256→512→1024→2048 environments 순서로 실행합니다. rung 실패나 GPU 측정·회복 실패가 있으면 상위를 중단하며, 4096은 2048이 PASS하고 peak VRAM이 총 12,288 MiB의 80%(9,830.4 MiB) 이하일 때만 실행합니다.

```powershell
cd "$HOME\isaac-walk-rl"
.\scripts\run_scale_ladder.ps1
```

현재 호스트에서는 4096까지 짧은 10-iteration 실행이 통과해 `highest_operational=4096`, `highest_safe=4096`으로 측정됐습니다. 이는 장기 안정성 판정이 아닙니다. 전체 표, checkpoint hash와 TensorBoard 경로는 `reports/runs/g004_go2_scale_summary.json`과 `RUN_NOTES.md`에 있습니다. 사용자 제공 MuJoCo 51k steps/s는 동일 조건 벤치마크가 아니므로 직접 비교하지 않습니다.

## Go2 flat 보상 ablation

G005에서는 4096 environments × 300 iterations 조건에서 baseline과 `dof_torques_l2`, `action_rate_l2`, `feet_air_time` 단일 제거 variants를 seeds 42/43/44로 학습했습니다. 12/12 runs가 실패 없이 완료됐고, 총 학습 시간은 105.4분, 실행별 평균 처리량의 평균은 60,238.2 steps/s, 최대 peak VRAM은 4,822 MiB였습니다.

```powershell
cd "$HOME\isaac-walk-rl"
.\scripts\run_reward_ablation.ps1

# 중단된 queue 재개
.\scripts\run_reward_ablation.ps1 -Resume

# 기존 증거를 엄격 검증하고 summary 재생성
& "$HOME\IsaacLab\_isaac_sim\python.bat" .\scripts\summarize_reward_ablation.py `
  --manifest .\configs\g005_reward_ablation.json `
  --queue .\reports\runs\g005_reward_ablation_state.json `
  --output .\reports\runs\g005_reward_ablation_summary.json
```

해석과 한계는 [`docs/G005_REWARD_ABLATION.md`](docs/G005_REWARD_ABLATION.md), 정량 summary는 [`reports/runs/g005_reward_ablation_summary.json`](reports/runs/g005_reward_ablation_summary.json), job별 checkpoint·TensorBoard·해시는 [`reports/runs/g005_reward_ablation_state.json`](reports/runs/g005_reward_ablation_state.json)에 있습니다. 학습 reward는 variant마다 정의가 달라 직접 비교하지 않습니다. Isaac Lab 본체는 계속 저장소 밖 `$HOME\IsaacLab`에 둡니다.

## Rough·DR·외란 회복 결과

G006은 4096 env × 1500 iterations × seeds 42/43/44로 baseline과 push curriculum을 비교했습니다. pooled 회복률은 `99.5370%` 대 `99.5988%`로 push curriculum이 `+0.0617%p`였지만, paired bootstrap 95% CI가 `-0.7716%p ~ +0.9568%p`이므로 유의한 개선을 주장하지 않습니다. 두 variant의 guardrail 생존률은 모두 `100%`였습니다.

해석·seed별 결과·추적 및 에너지 proxy는 [`docs/G006_ROUGH_PUSH_RECOVERY.md`](docs/G006_ROUGH_PUSH_RECOVERY.md), 정량 summary는 [`reports/runs/g006_summary.json`](reports/runs/g006_summary.json), job별 checkpoint·평가 보고서 해시는 [`reports/runs/g006_queue_state.json`](reports/runs/g006_queue_state.json)에 있습니다. 이 durable JSON의 경로와 평가 command는 저장소 상대경로 또는 `%USERPROFILE%`, `%REPO_ROOT%`, `%ISAACLAB_ROOT%` 표현으로만 기록하며, 실제 evaluator 실행 시점에만 허용된 root 안의 절대경로로 resolve합니다.

seed 42의 실제 정책 재생 비교 GIF와 스크린샷, 로컬 전용 원본 영상의 무결성 정보는 [`docs/G006_VISUAL_EVIDENCE.md`](docs/G006_VISUAL_EVIDENCE.md)에 있습니다. 시각 자료는 정성 작동 증거이며 정량 평가를 대체하지 않습니다.

```powershell
cd "$HOME\isaac-walk-rl"
python .\scripts\summarize_g006.py `
  --manifest .\configs\g006_rough_push.json `
  --queue-state .\reports\runs\g006_queue_state.json `
  --isaaclab-root "$HOME\IsaacLab" `
  --output .\reports\runs\g006_summary.json
```

## RBQ 외부 자산 호환성 게이트

G007은 RBQ v1.20.0의 8개 자산 경로와 Git 객체를 고정했습니다. `rbq_description/package.xml`의 Apache-2.0 선언이 URDF·STL blob에 적용되는 범위와 로컬 처리 권한은 확인되지 않아 `license_scope_unresolved`로 차단합니다. 자산 다운로드·변환·smoke는 실행하지 않았으며, 이 blocker는 G007의 재현 가능한 완료 경로이지 G006이나 전체 프로젝트의 중단 사유가 아닙니다.

```powershell
cd "$HOME\isaac-walk-rl"
python .\scripts\validate_rbq_assets.py --manifest .\configs\g007_rbq_asset_manifest.json --expect-blocked --report .\reports\g007_rbq_compatibility_spike.json
```

상세 판정, 공식 출처, 해시와 blocker 해제 조건은 [`docs/G007_RBQ_COMPATIBILITY_SPIKE.md`](docs/G007_RBQ_COMPATIBILITY_SPIKE.md)에 있습니다.
