# G005 Go2 flat 보상 ablation

## 결론

Go2 flat에서 `dof_torques_l2`, `action_rate_l2`, `feet_air_time`을 한 번에 하나씩 제거한 4 variants × 3 seeds 실험을 완료했다. 각 정책은 4,096 environments, 300 iterations로 학습했고, 고정된 26개 명령 조합에서 명령당 10 environments를 20초 동안 평가했다.

- `action_rate_l2` 제거는 선형 속도 추종, mechanical power proxy, action 변화량을 함께 악화시켰다. 세 seed 모두 같은 방향으로 action 변화량이 증가했으므로 세 항목 중 유지 근거가 가장 분명하다.
- `dof_torques_l2` 제거는 torque L2, mechanical power proxy, action 변화량을 증가시켰다. 넘어짐 2건도 이 variant의 한 seed·한 측면 명령에서만 관측됐다.
- `feet_air_time` 제거는 yaw RMSE와 두 에너지 proxy를 낮췄지만, first-contact 횟수와 raw feet-air-time 지표가 크게 달라졌다. flat 20초 평가만으로 제거가 우월하다고 결론 내릴 수 없으며, rough terrain과 외란 회복에서 다시 확인해야 한다.
- baseline은 780 trials에서 넘어짐 없이 모든 평가를 마쳤다. 현재 단계에서는 세 보상 모두를 유지한 baseline을 G006 비교 기준으로 사용한다.

이 순위는 고정된 flat 평가에서 관측한 탐색적 결과다. 보상 제거가 지표 변화를 직접 일으켰다는 강한 인과 주장이나 실물 성능 일반화는 하지 않는다.

## 실험 계약

| 항목 | 값 |
| --- | --- |
| 학습 task | `Isaac-Velocity-Flat-Unitree-Go2-v0` |
| 학습 규모 | 4,096 environments × 300 iterations |
| variants | `baseline`, `no_torque`, `no_action_rate`, `no_feet_air_time` |
| training seeds | 42, 43, 44 |
| 총 실행 | 12/12 complete, failed 0 |
| 총 학습 wall time | 105.4분 |
| 실행별 평균 처리량의 평균 | 60,238.2 steps/s |
| 최대 peak VRAM | 4,822 MiB |
| 평가 task | `Isaac-Velocity-Flat-Unitree-Go2-Play-v0` |
| 평가 seed | 20260824 |
| 평가 격자 | 26 command conditions × 10 environments |
| 평가 길이 | 1,000 steps × 0.02초 = 20초 |

baseline weight는 torque `-0.0002`, action rate `-0.01`, feet air time `0.25`다. 각 ablation은 해당 항목 하나만 `0.0`으로 바꾸고 나머지 두 값을 고정했다.

## 3-seed 평균

괄호 안은 seed-paired baseline 대비 변화율이다. 낮을수록 좋은 RMSE·proxy 지표에서 `+`는 악화, `-`는 개선을 뜻한다. fall rate와 survival은 3-seed 평균이다.

| variant | linear RMSE (m/s) | yaw RMSE (rad/s) | torque L2 | mechanical power proxy (W) | action-rate L2 | fall rate | survival |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.11253 | 0.16024 | 386.32 | 123.67 | 2.2523 | 0.0000% | 100.0000% |
| `no_torque` | 0.11365 (+1.00%) | 0.15883 (-0.88%) | 432.35 (+11.92%) | 130.44 (+5.48%) | 3.1757 (+41.00%) | 0.2564% | 99.7436% |
| `no_action_rate` | 0.12419 (+10.35%) | 0.16680 (+4.09%) | 403.58 (+4.47%) | 138.70 (+12.15%) | 5.0308 (+123.36%) | 0.0000% | 100.0000% |
| `no_feet_air_time` | 0.11408 (+1.38%) | 0.14964 (-6.62%) | 365.80 (-5.31%) | 111.89 (-9.53%) | 2.1846 (-3.01%) | 0.0000% | 100.0000% |

학습 중 기록된 reward는 제거한 항목에 따라 정의와 스케일이 달라지므로 variant 간 직접 비교에 사용하지 않는다. 표의 값은 모두 동일한 평가 프로토콜에서 수집한 물리량 또는 물리량 proxy다.

## paired 해석

### `no_torque`

- torque L2 `+11.92%`, mechanical power proxy `+5.48%`, action-rate L2 `+41.00%`로 실용 임계값 5%를 넘었다.
- 세 지표는 seed 42·43·44에서 모두 증가했다.
- seed 43의 `vx=0.0`, `vy=-0.5`, `yaw=0.0` 명령에서만 10 trials 중 2건이 넘어졌다. 전체 780 trials 기준 2건이며, 3-seed 평균 fall rate는 0.2564%다.
- fall rate 변화는 사전 정의한 절대 2% 임계값을 넘지 않았고 한 seed·한 명령에 국한됐다. 따라서 일반적인 안정성 저하로 단정하지 않는다.

### `no_action_rate`

- linear RMSE `+10.35%`, mechanical power proxy `+12.15%`, action-rate L2 `+123.36%`로 실용 임계값을 넘었다.
- 세 값 모두 seed 42·43·44에서 같은 악화 방향이었다.
- yaw RMSE `+4.09%`와 torque L2 `+4.47%`는 각각의 5% 임계값에는 미달했다.

### `no_feet_air_time`

- yaw RMSE `-6.62%`, torque L2 `-5.31%`, mechanical power proxy `-9.53%`로 평균은 개선 방향이었다.
- mechanical power proxy와 yaw RMSE는 세 seed 모두 개선했지만, torque L2는 2/3 seeds만 개선했다.
- first-contact count는 `+77.58%`, raw feet-air-time 평균은 `-75.94%`로 크게 변했다. 보행 접촉 양상이 바뀌었다는 신호이므로, flat 효율만으로 이 보상 제거를 채택하지 않는다.

## 실용 임계값과 한계

사전에 고정한 실용 임계값은 tracking RMSE 상대 변화 5%, torque/action/power proxy 상대 변화 5%, fall rate 절대 변화 2%다. 임계값은 통계적 유의성을 뜻하지 않으며 다음 한계를 함께 적용한다.

- variant당 `n=3`이므로 표본 표준편차와 paired delta는 탐색적 근거다.
- flat terrain, 20초 horizon, 26 commands × 10 trials 조건만 평가했다.
- 평가 seed는 모든 정책에 동일한 `20260824` 하나다.
- mechanical power는 `sum(abs(applied_torque * joint_velocity))`의 시뮬레이션 proxy이며 전기 에너지 측정값이 아니다.
- 넘어짐이 일어나면 해당 첫 episode 이후 환경은 집계에서 제외된다. 따라서 `no_torque` seed 43은 260,000이 아니라 258,135 active-state samples이며, early fall이 있는 variant의 연속 지표에는 조건부 표본 편향이 생길 수 있다.
- 실물, rough terrain, domain randomization, 외란 회복 성능은 이 단계에서 검증하지 않았다.

## 실행·재개·요약

Isaac Lab 본체는 저장소 밖 `$HOME\IsaacLab`에 그대로 두고 다음 명령을 실행한다.

```powershell
cd "$HOME\isaac-walk-rl"
.\scripts\run_reward_ablation.ps1
```

중단 후에는 기존 state와 산출물의 설정·명령·스크립트·checkpoint 해시를 다시 검증한 뒤 완료된 job을 건너뛴다.

```powershell
cd "$HOME\isaac-walk-rl"
.\scripts\run_reward_ablation.ps1 -Resume
```

엄격 검증을 거쳐 summary를 다시 만든다.

```powershell
cd "$HOME\isaac-walk-rl"
& "$HOME\IsaacLab\_isaac_sim\python.bat" .\scripts\summarize_reward_ablation.py `
  --manifest .\configs\g005_reward_ablation.json `
  --queue .\reports\runs\g005_reward_ablation_state.json `
  --output .\reports\runs\g005_reward_ablation_summary.json
```

## 증거와 해시

- 엄격 summary: [`../reports/runs/g005_reward_ablation_summary.json`](../reports/runs/g005_reward_ablation_summary.json), file SHA256 `14ea2c0243c43d5b12a3788a69a3205497a81ee1084c364c22c2cc30d1dd58f4`
- queue state: [`../reports/runs/g005_reward_ablation_state.json`](../reports/runs/g005_reward_ablation_state.json), file SHA256 `f55bb26e1c52afb75082ebb30ae9cbe2c50b9bea5f03dc7c54acbb53310d7a42`
- canonical config SHA256 `3e8455a9efba77f67b2ac436d5eef41421dfeac10f9e67ab9620c6775b6c2576`
- config file SHA256 `5f5cf8127424460c4b2555d28969e85d9664589337ba4edf71dd9ed72112cdde`
- evaluation protocol SHA256 `4ff2f271ed7e217966ed7e09a1f0de5bfacc056020721623af6211d264835d9c`
- evaluation script SHA256 `60b22beaf6189ae0f3bc0aeaa98f264a7ffe853f4de2c5b49e266a2716bd7965`
- variant SHA256: baseline `21c1caf3439e21a0433271bc1f44407354a63f8f617236f02a156f07812c9dd1`, no_torque `9507afe4889c4d6e7aac82c05f4c0e30acc7ccaea300535e86d9dc90fe29d652`, no_action_rate `02dfa8cb9b8cc3a0400433d59efe855459a2fdcd0a3dcb886fb03c9db9277065`, no_feet_air_time `225dcd61df79a936e52aa79bfef6b59f027e47727b6da36ef590bfd7a7113a8e`

12개 checkpoint SHA256, TensorBoard 디렉터리, 학습·평가 명령과 각 보고서 경로는 queue state에 job별로 고정돼 있다. 다음 단계 G006은 이 baseline을 유지한 채 rough terrain, domain randomization, 고정 push protocol의 회복률을 비교한다.
