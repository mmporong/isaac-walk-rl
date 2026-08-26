# G008 혼합 마찰 바닥과 링크 질량 한계 시험

## 결론부터 말하면

마찰 학습과 링크 질량 학습은 둘 다 실제 PPO 학습으로 수행했다. 다만 학습을 돌렸다는 사실과 정책이 좋아졌다는 판정은 분리해야 한다.

friction S1 정책은 기준 바닥 `μ_s/μ_d=0.8/0.6`에서 전진·후진·좌회전·우회전을 모두 통과했다. 고마찰 `0.8/0.6`과 저마찰 띠가 0.5m마다 바뀌는 바닥에서는 전진, 후진, 좌회전을 완주된 최저 조건 `0.2/0.1`까지 연속 통과했다. 우회전은 `0.7/0.5`에서 먼저 실패하고 `0.6/0.4`에서 한 번 다시 통과했다. 결과가 단조롭지 않으므로 우회전이나 네 방향 전체에 대해서는 보수적인 혼합 마찰 하한을 확정하지 않았다.

`0.1/0.05`도 시험했지만 결과로 쓰지 않았다. 32환경 실행이 네 번 모두 100~200 control step 뒤 Isaac Sim native process 종료로 끝났고 atomic JSON이 만들어지지 않았다. 이 값은 정책 PASS도 FAIL도 아닌 미확정이다. 현재 증거가 닿는 최저 계수는 `0.2/0.1`이다.

leg-mass S1은 16개 다리 body의 질량을 환경마다 `0.95~1.05`배 바꾸고 inertia tensor도 같은 비율로 다시 계산하며 학습했다. 300-iteration 학습은 완료됐지만 기준 질량의 우회전부터 gate를 통과하지 못했다. hip·thigh·calf·foot를 한 그룹씩 `0.8~1.2`배 바꾼 후속 시험에서도 두 정책의 전진·후진은 25개 조건을 모두 통과했으나, leg-mass S1은 회전 성능을 회복하지 못했다. 그래서 leg-mass S2는 열지 않았다.

## 단계별 동작 영상

혼합 마찰과 링크 그룹 질량 단계는 정량 평가 뒤 같은 checkpoint와 물리 조건으로 다시 촬영했다. MP4는 로컬에만 보관하고 GIF와 네 방향 스크린샷만 Git에 넣었다.

![혼합 마찰 바닥 네 방향 재생](media/g008/g008_stage_periodic_friction.gif)

![링크 그룹별 1.2배 질량 비교](media/g008/g008_stage_link_mass_groups.gif)

혼합 마찰 영상의 파란 띠는 `0.2/0.1`, 갈색 띠는 `0.8/0.6`이다. 링크 질량 영상은 hip·thigh·calf·foot를 각각 `1.2배`로 바꾼 네 실행을 동기화했다. 원본 경로, 실제 mass readback, 해시와 스크린샷은 `docs/G008_VISUAL_EVIDENCE.md`에 있다. 영상 한 번의 동작으로 아래 다중 환경 gate 결과를 바꾸지는 않는다.

## 학습 스택과 PPO 규모

실행 환경은 Windows 11, RTX 3060 12GB, Isaac Sim 4.5, Isaac Lab 2.1.1, RSL-RL 2.3.3, PyTorch 2.7.0+cu128이다. 학습과 평가는 모두 GUI가 없는 headless 모드로 돌렸다. 정책 입력은 235차원, action은 Go2 관절 12개이며 actor MLP는 `512-256-128` 은닉층을 쓴다.

friction S1과 leg-mass S1은 같은 command checkpoint `model_1798.pt`에서 따로 갈라졌다. 서로의 checkpoint를 이어받지 않았기 때문에 마찰 효과와 질량 효과를 한 학습에 섞지 않았다.

| 항목 | friction S1 | leg-mass S1 |
| --- | ---: | ---: |
| 병렬 환경 | 1,024 | 1,024 |
| PPO iteration | 300 | 300 |
| rollout step/env/iteration | 24 | 24 |
| iteration당 rollout batch | 24,576 | 24,576 |
| epoch | 5 | 5 |
| mini-batch/epoch | 4 | 4 |
| mini-batch 크기 | 6,144 | 6,144 |
| optimizer mini-batch step | 6,000 | 6,000 |
| 총 transition | 7,372,800 | 7,372,800 |
| seed | 42 | 42 |

학습기는 PPO다. friction S1은 발 collision material만 `μ_s=0.72~0.88`, `μ_d=0.52~0.68` 범위로 바꿨다. leg-mass S1은 hip·thigh·calf·foot 16개 body를 각각 `0.95~1.05`배로 독립 표본화하고 `recompute_inertia=True`를 적용했다. 두 실험 모두 command, reward, observation, actuator 설정은 기준 정책과 동일하게 유지했다.

checkpoint와 SHA-256은 다음과 같다.

| 정책 | checkpoint | SHA-256 |
| --- | --- | --- |
| command | `model_1798.pt` | `53cc09043088bcd53618d2ae1f90c7f2e91d01eab7090cc63922486942b2ed47` |
| friction S1 | `model_2097.pt` | `40af0a0f80489d705e1e8fdeedd2f765177d3d67bf757709b9195cc2bbeaaee0` |
| leg-mass S1 | `model_2097.pt` | `8976cfff6eee6d1a998c7aa554b23d98b01d3d64da02b43ac3133a9186ae97fa` |

## Part A. 공간적으로 바뀌는 마찰 바닥

### 바닥을 만든 방식

이번 시험은 시간에 따라 모든 바닥의 마찰이 동시에 바뀌는 randomization이 아니다. x축을 따라 고마찰과 저마찰이 공간적으로 반복된다. 로봇이 전진하거나 후진하면 발마다 서로 다른 재질을 밟고, 제자리 회전 중에도 발 위치에 따라 마찰이 갈린다.

- 띠 폭: `0.5m`
- 한 주기: 저마찰 0.5m + 고마찰 0.5m = `1.0m`
- 모든 환경에 보장한 최소 범위: 길이 `24m`, 폭 `4m`
- 표면 높이: `0.002m`
- 환경 원점 간격: `16m`
- 32환경 장면의 전역 띠 수: 240개, x 범위 `-60~60m`
- 고마찰 재질: `0.8/0.6`
- 저마찰 재질: case별 `0.7/0.5`부터 `0.1/0.05`

240개 띠를 240개 collider로 만든 것은 아니다. 하나의 정적 triangle mesh를 만들고, face material subset을 번갈아 배정했다. PhysX가 이를 convex hull로 근사하지 않도록 `MeshCollisionAPI.approximation=none`을 적용했다. 이 방식은 Isaac Sim의 triangle-mesh multi-material 예제와 같은 구조다.

기본 ground plane은 생성하지 않았다. 마찰 메시 아래에 다른 collider가 남으면 contact offset 때문에 발이 두 재질을 동시에 밟을 수 있기 때문이다. height scan에는 같은 높이의 별도 non-collision mesh를 사용했다. 모든 완료 보고서에서 다음 조건을 런타임 readback으로 확인했다.

- 기본 ground collision prim 존재: `false`
- height-scan mesh의 CollisionAPI: `false`
- 환경 원점의 1m 마찰 주기 위상 오차: `0`
- 로봇 material: `1.0/1.0`, 바닥과의 결합: `multiply`

따라서 설정한 바닥 계수가 유효 접촉 계수가 된다. 재질은 시뮬레이션 시작 전에 고정되며 접촉 중에 바꾸지 않는다.

### 평가 프로토콜

각 계수 case는 독립 headless Isaac Sim 프로세스에서 실행했다. 32환경을 두 정책에 16개씩 나누고, 정책별 전진·후진·좌회전·우회전에 4환경씩 배정했다. 두 정책은 같은 simulation clock 안에서 동시에 추론했다.

| 항목 | 값 |
| --- | ---: |
| control frequency | 50Hz |
| PhysX step | 0.005s |
| substep/control step | 4 |
| horizon | 500 control step, 10초 |
| warmup 제외 | 처음 50 step |
| 평가 seed | 20260826 |
| observation corruption | 끔 |
| push/base mass/leg mass randomization | 끔 |
| 명령 | 전진 `0.6m/s`, 후진 `-0.4m/s`, 좌·우 yaw `±0.5rad/s` |

64환경도 시도했지만 전역 multi-material mesh와 scene replication을 결합하는 단계에서 Isaac Sim native 종료가 재현됐다. 그래서 완료율을 우선해 32환경, 정책·방향당 4반복으로 고정했다. 이는 단일 seed와 함께 이번 결과의 통계적 한계다.

contact sensor는 이 multi-material triangle mesh에서 발 접촉 force sample을 반환하지 않았다. 그러므로 slip 평균을 0으로 채우지 않고 `null`로 저장했다. 생존 판정은 기존 `base_contact` termination에 더해 다음 독립 조건을 사용했다.

- base 높이가 마찰 표면보다 `0.18m` 아래로 내려감
- 몸체 up-axis의 world-z 성분이 `0.5` 아래로 내려감

둘 중 하나라도 발생하면 kinematic fall로 센다. `0.2/0.1`의 friction S1 우회전에서 이 판정으로 1회 낙상을 잡았다.

### 시험 결과

마찰 특화 정책의 방향별 결과는 다음과 같다.

| 방향 | nominal `0.8/0.6` | 높은 값부터 연속 통과한 최저 완료 case | 첫 실패 | 더 낮은 개별 PASS |
| --- | --- | --- | --- | --- |
| 전진 | PASS | `0.2/0.1` | 없음 | `0.2/0.1` |
| 후진 | PASS | `0.2/0.1` | 없음 | `0.2/0.1` |
| 좌회전 | PASS | `0.2/0.1` | 없음 | `0.2/0.1` |
| 우회전 | PASS | 없음 | `0.7/0.5` | `0.6/0.4` |
| 네 방향 전체 | PASS | 없음 | `0.7/0.5`, 우회전 | `0.6/0.4` |

`0.2/0.1`에서 friction S1의 수치는 다음과 같다.

| 방향 | 선속도 RMSE | yaw RMSE | 최대 roll | 최대 pitch | 낙상 | 판정 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 전진 | `0.040m/s` | `0.052rad/s` | `0.023rad` | `0.028rad` | 0 | PASS |
| 후진 | `0.062m/s` | `0.058rad/s` | `0.197rad` | `0.331rad` | 0 | PASS |
| 좌회전 | `0.061m/s` | `0.110rad/s` | `0.079rad` | `0.126rad` | 0 | PASS |
| 우회전 | `0.076m/s` | `0.204rad/s` | `0.966rad` | `0.393rad` | 1 | FAIL |

전진·후진·좌회전의 발 저마찰 영역 노출 비율은 각각 약 `48.8%`, `46.7%`, `62.6%`였다. 띠 전환 수도 164회, 112회, 19회로 기록됐다. 시작 위치의 한 재질에만 머문 결과가 아니다.

command 정책은 nominal에서 좌회전 yaw RMSE가 `0.262rad/s`로 gate `0.25rad/s`를 넘었다. 이 baseline이 먼저 실패했으므로 command 정책에는 네 방향 마찰 하한을 붙이지 않았다. 전진과 후진만 보면 두 방향 모두 완료된 `0.2/0.1`까지 연속 통과했다.

### 마찰은 실제로 영향을 줬나

영향은 있었다. 다만 "계수가 낮아질수록 매 단계가 일정하게 나빠진다"는 모양은 아니었다. 접촉 재질이 바뀌면 보행 위상, 띠 경계를 넘는 순간, 지지발 조합이 함께 달라진다. 우회전은 `0.7/0.5`에서 실패한 뒤 `0.6/0.4`에서 다시 통과했고, 이후 더 낮은 값에서는 다시 실패했다. 4반복의 최대 자세 gate는 이런 위상 차이에 민감하다.

그래서 다음 세 값을 섞지 않았다.

1. 연속 통과 하한: 높은 계수부터 처음 실패하기 전까지 이어진 구간
2. 최저 개별 PASS: 중간 실패 뒤 더 낮은 계수에서 다시 통과한 결과
3. 미확정: 시뮬레이터가 끝까지 실행되지 않아 PASS/FAIL을 낼 수 없는 결과

현재 가장 강하게 말할 수 있는 결론은 friction S1이 전진·후진·좌회전을 혼합 `0.2/0.1`까지 수행했다는 것이다. 우회전과 네 방향 전체의 보수적 혼합 하한은 아직 없다.

### `0.1/0.05`를 제외한 이유

최저 case는 네 번 실행했다. 처음 두 번은 `completed_steps=200`, kinematic fall 판정을 넣은 뒤의 두 실행은 `completed_steps=100`에서 Kit native process가 종료됐다. 네 실행 모두 shell에는 exit code 0처럼 보였지만 atomic JSON이 없었다. 실행 harness는 보고서 존재 여부를 별도로 확인해 이를 실패로 처리했다.

이 현상은 정책 낙상과 구분한다. 재현 가능한 것은 "Isaac Sim 4.5가 이 multi-material 조건을 끝까지 처리하지 못했다"는 사실뿐이다. CPU PhysX 재현, 더 작은 env batch, 다른 material discretization 또는 Isaac Sim 업그레이드로 원인을 분리하기 전에는 `0.1/0.05` 성능을 주장하지 않는다.

## Part B. 다리 링크 그룹 질량 변화

### 학습 때 무엇을 바꿨나

leg-mass S1은 `.*_(hip|thigh|calf|foot)` 정규식으로 16개 다리 body를 선택했다. 환경마다 body별 uniform scale `0.95~1.05`를 독립적으로 뽑았다. 같은 환경에서도 앞왼쪽 thigh와 뒤오른쪽 calf의 scale이 다를 수 있다.

1,024환경 runtime probe에서 실제 scale은 `0.950001~1.049993`, 총 다리 질량은 `7.8296~8.3781kg`이었다. nominal inertia tensor에는 해당 body의 mass ratio를 곱했다. 기존 probe의 최대 inertia 오차는 약 `1.86e-9`였다. 링크 형상과 center of mass 위치는 고정했으므로, 장착물로 COM이 이동하는 경우까지 재현한 것은 아니다.

### 그룹별 held-out 시험

어떤 링크 그룹이 민감한지 보기 위해 한 번에 한 그룹만 바꿨다.

- 조건: nominal 1개 + hip·thigh·calf·foot별 `0.8, 0.9, 0.95, 1.05, 1.1, 1.2` = 25개
- 정책: command, leg-mass S1
- 방향: 전진, 후진, 좌회전, 우회전
- 배치: 800환경, 정책·조건·방향당 4반복
- 길이: 300 control step, warmup 50 step
- 바닥: 평면 `0.8/0.6`
- seed: 20260826

| 그룹 | nominal group mass | 0.8배일 때 총 다리 질량 | 1.2배일 때 총 다리 질량 |
| --- | ---: | ---: | ---: |
| hip | `2.712kg` | `7.55kg` | `8.64kg` |
| thigh | `4.608kg` | `7.17kg` | `9.02kg` |
| calf | `0.616kg` | `7.97kg` | `8.22kg` |
| foot | `0.160kg` | `8.06kg` | `8.13kg` |

두 정책은 전진·후진 25개 조건을 모두 통과했고 기존 contact 기반 낙상은 0건이었다. 회전에서 차이가 났다.

command 정책은 nominal 네 방향을 통과했다. 그룹별 전 방향 PASS factor는 hip `0.8, 1.05, 1.1, 1.2`, thigh `1.05, 1.1, 1.2`, calf `0.95`, foot `0.9, 0.95, 1.05, 1.2`였다. 결과가 단조롭지 않고 조건당 반복이 4개이므로 정밀 허용 공차표로 보지는 않는다.

leg-mass S1은 nominal 우회전 yaw RMSE가 `0.44rad/s`로 실패했다. 25개 조건 모두 좌회전 또는 우회전 중 하나 이상이 gate를 넘었다. 질량 randomization이 코드에 없었던 것이 아니라, `7,372,800` transition 학습 결과가 회전 강건성을 얻지 못한 것이다.

역학적으로 링크 질량이 바뀌면 관절 공간 관성행렬 `M(q)`, Coriolis·원심 항, 중력항이 함께 바뀐다. 같은 action에서도 관절 가속과 swing timing이 달라진다. 회전은 네 발의 접촉 타이밍과 접선력 비대칭으로 yaw moment를 만들어야 해서 직선 속도보다 민감할 수 있다.

다만 calf가 가장 위험하다고 단정할 수는 없다. 같은 scale factor라도 thigh와 foot의 총질량 변화가 다르고, 이번 gate는 단일 seed의 peak 자세에 민감하다. 다음 비교는 scale뿐 아니라 그룹마다 같은 kg을 더하는 조건도 필요하다.

## 재현 명령

```powershell
cd "$HOME\isaac-walk-rl"

& "$HOME\IsaacLab\_isaac_sim\python.bat" .\scripts\evaluate_g008_periodic_friction.py `
  --command-checkpoint "$HOME\IsaacLab\logs\rsl_rl\unitree_go2_rough\2026-08-26_11-11-12_g008_command_finetune_g006_s42_e1024_i300\model_1798.pt" `
  --friction-checkpoint "$HOME\IsaacLab\logs\rsl_rl\unitree_go2_rough\2026-08-26_11-37-54_g008_friction_s1_finetune_command_s42_e1024_i300\model_2097.pt" `
  --case-id mixed_020_010 --num-envs 32 --horizon-steps 500 --warmup-steps 50 `
  --output .\reports\runs\g008_periodic_friction_case_mixed_020_010_e32_h500_s20260826.json `
  --headless --hard-exit-after-report

& "$HOME\IsaacLab\_isaac_sim\python.bat" .\scripts\evaluate_g008_link_mass_sensitivity.py `
  --command-checkpoint "$HOME\IsaacLab\logs\rsl_rl\unitree_go2_rough\2026-08-26_11-11-12_g008_command_finetune_g006_s42_e1024_i300\model_1798.pt" `
  --leg-mass-checkpoint "$HOME\IsaacLab\logs\rsl_rl\unitree_go2_rough\2026-08-26_12-06-51_g008_leg_mass_s1_finetune_command_s42_e1024_i300\model_2097.pt" `
  --num-envs 800 --horizon-steps 300 --warmup-steps 50 `
  --output .\reports\runs\g008_link_mass_sensitivity_command_vs_leg_mass_s1_e800_h300_s20260826.json `
  --headless --hard-exit-after-report
```

`--hard-exit-after-report`는 Windows headless Kit가 JSON 저장 뒤 teardown에서 오래 멈추는 경우를 피한다. 임시 파일을 쓴 뒤 atomic replace하고 종료하므로, 완료 판정은 exit code만 보지 않고 JSON 존재·status·source hash를 함께 확인한다.

## 증거 파일

- 완료 case 집계: `reports/runs/g008_periodic_friction_sweep_command_vs_friction_s1_e32_h500_s20260826.json`
- 완료 원본 7개: `reports/runs/g008_periodic_friction_case_*_e32_h500_s20260826.json`
- `0.1/0.05` 실패 기록: `reports/runs/g008_periodic_friction_case_mixed_010_005_e32_h500_s20260826_failure.json`
- 링크 질량 민감도: `reports/runs/g008_link_mass_sensitivity_command_vs_leg_mass_s1_e800_h300_s20260826.json`
- 혼합 마찰 촬영·파생물: `reports/runs/g008_stage_periodic_friction_capture.json`, `reports/runs/g008_stage_periodic_friction_visual_evidence.json`
- 링크 질량 촬영·파생물: `reports/runs/g008_stage_link_mass_*_capture.json`, `reports/runs/g008_stage_link_mass_visual_evidence.json`
- 공개 GIF·스크린샷과 로컬 MP4 해시: `docs/G008_VISUAL_EVIDENCE.md`
- 평가기: `scripts/evaluate_g008_periodic_friction.py`, `scripts/aggregate_g008_periodic_friction.py`, `scripts/evaluate_g008_link_mass_sensitivity.py`
- 회귀 검증: `tests/test_g008_dynamics_stress_reports.py`

## 다음에 자세히 할 일

1. `0.1/0.05` native 종료를 최소 재현으로 분리한다. 8·16·32환경, CPU/GPU PhysX, 단일 재질·두 재질, face subset 수를 조합해 정책과 무관한 simulator 문제인지 확인한다.
2. 완료된 `0.2/0.1~0.7/0.5`를 seed `20260827`, `20260828`로 반복한다. 세 seed가 모두 통과한 방향·계수만 최종 하한으로 채택한다.
3. 띠 폭을 `0.25, 0.5, 1.0m`로 바꾼다. 보폭과 띠 폭의 비가 접촉 전환과 yaw 안정성에 미치는 영향을 본다.
4. x축 띠 다음에는 2차원 checkerboard와 좌우 비대칭 patch를 별도 Part로 만든다. 현재 결과를 임의의 혼합 지형 전체로 일반화하지 않는다.
5. contact sensor가 multi-material mesh에서 힘을 반환하지 않는 원인을 해결한다. 해결 전에는 slip·contact impulse 수치를 만들지 않는다. 해결 뒤 foot slip distance, tangential impulse, yaw moment를 시계열로 추가한다.
6. friction S1은 우회전 표본과 자세 penalty를 보강해 command checkpoint에서 다시 학습한다. nominal과 혼합 `0.7/0.5` 우회전을 모두 통과해야 S2로 넓힌다.
7. leg-mass S1도 command checkpoint에서 다시 시작한다. 좌·우 yaw 명령을 균형 있게 넣고 checkpoint 선택을 평균 reward가 아니라 네 방향 최솟값으로 바꾼다.
8. 질량은 scale 비교와 같은 추가 질량 kg 비교를 분리한다. 센서·보호대처럼 COM을 옮기는 장착물은 별도 asset variant로 검증한다.
9. 마찰과 질량을 동시에 섞는 학습은 두 단일축이 다중 seed gate를 통과한 뒤에만 진행한다.

이 수치는 Isaac Sim 안의 stress-test 범위다. 실물 Go2의 발 패드 마찰, 바닥 재질, 장착물 질량과 COM을 측정하기 전에는 안전 운용 한계로 쓰지 않는다.
