# Isaac Lab 사족보행 Physical AI 학습 시작점

이 문서는 프로젝트에 이미 있는 강화학습·Isaac 자료를 새로 대체하지 않고, 하나의 사족보행 제어·학습 폐루프 안에서 다시 찾을 수 있게 만든 학습 지도예요. 강화학습 이론을 앞에서부터 모두 공부하지 않고, 현재 Go2 실험에서 관찰한 현상을 설명하거나 다음 통제실험을 설계하는 데 필요한 개념만 꺼내 적용해요.

프로젝트의 실행 기준은 재현성을 위해 Isaac Sim 4.5.0, Isaac Lab 2.1.1, RSL-RL 2.3.3으로 고정돼 있어요. 2026-08-31 현재 Isaac Lab 최신 공개판은 3.0 Beta 2 Patch 1이며 multi-backend physics, backend-neutral actuator, kit-less workflow를 확장하고 있지만 beta 단계예요. 따라서 최신판으로 즉시 이식하지 않고, 최신 문서의 개념과 API 변화는 비교 학습에만 쓰고 실험은 고정 버전에서 재현해요.

공식 방향도 이 구분과 맞아요. NVIDIA의 최신 Physical AI 학습 경로는 `perceive → reason → act`를 시뮬레이션, 정책 학습, 평가, 실물 배포로 연결하고, Isaac Lab은 병렬 환경에서 robot policy를 학습하는 층으로 둬요. 이 프로젝트는 그중 상태 기반 사족보행 policy training과 sim-to-real readiness 검증에 집중해요.

공식 기준:

- [NVIDIA Physical AI Learning](https://docs.nvidia.com/learning/physical-ai/index.html)
- [Getting Started With Isaac Lab](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/index.html)
- [Isaac Lab manager-based environment](https://isaac-sim.github.io/IsaacLab/develop/source/tutorials/03_envs/create_manager_base_env.html)
- [Isaac Lab actuators](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/actuators.html)
- [Isaac Lab releases](https://github.com/isaac-sim/IsaacLab/releases)
- [RSL-RL PPO configuration](https://github.com/leggedrobotics/rsl_rl/blob/main/docs/guide/configuration.rst)

## 이 프로젝트에서 항상 먼저 그릴 폐루프

```text
외란·reset·물성 randomization
          |
          v
로봇 상태/센서
  world·base pose, IMU에 대응하는 각속도·중력방향,
  관절각·관절속도, 접촉력, 지형·거리
          |
          v
Observation Manager
  단위·좌표계 확인 -> noise/clip -> actor/critic 관측 분리
          |
          v
Policy actor πθ(a|o) ------------ Critic Vφ(o 또는 privileged state)
          |                                      |
          v                                      |
정규화 Action a_t                               |
          |                                      |
          v                                      |
Action Manager: scale·offset·clip·EMA            |
          |                                      |
          v                                      |
Joint target q_des [rad]                         |
          |                                      |
          v                                      |
PD/Actuator: target -> effort limit 포함 torque τ [N·m]
          |                                      |
          v                                      |
Physics: 동역학·중력·관성·마찰·외란·접촉력 λ      |
          |                                      |
          v                                      |
다음 로봇 상태 s_(t+1)·ContactSensor -----------+
          |
          v
Reward·Termination: 추적·자세·토크·접촉·성공을 수치화
          |
          v
Rollout: (o_t, a_t, r_t, done, V_t)
          |
          v
Advantage·Return -> PPO clipped update -> 새 checkpoint
          |
          `---- 같은 평가 grid에서 baseline과 수치 비교 ----> 다음 실험
```

외란, IMU, 관절 상태, 토크, 접촉력, 보상은 서로 떨어진 과목이 아니에요. 외란은 physics에 들어가 다음 상태를 바꾸고, IMU와 encoder에 대응하는 값은 그 상태를 observation으로 바꾸며, policy action은 actuator torque를 거쳐 다시 physics에 들어가요. 접촉력은 physics의 결과이면서 다음 observation, reward, termination의 입력이고, reward는 그 한 바퀴가 목표에 얼마나 가까웠는지를 PPO에 전달해요.

## 좌표계와 단위 계약

수식이나 텐서를 읽기 전에 아래 네 공간 중 어디인지 먼저 말해요. 좌표계가 빠진 속도·힘·회전 설명은 완료로 보지 않아요.

| 공간 | 이 프로젝트에서 읽는 방향 | 대표 값 |
|---|---|---|
| world frame `W` | `+Z`가 위쪽인 시뮬레이션 전역 좌표 | base 위치 `[m]`, 지형 normal, `net_forces_w [N]` |
| base/body frame `B` | Go2 몸통 기준 `+X` 전진, `+Y` 왼쪽, `+Z` 위 | base 선속도 `[m/s]`, 각속도 `[rad/s]`, projected gravity |
| joint frame | 각 관절축의 양의 회전은 URDF/USD joint axis가 결정 | 관절각 `q [rad]`, 관절속도 `q_dot [rad/s]`, 토크 `tau [N·m]` |
| action space | 신경망이 출력하는 무차원 정규화 값 | `a_t` 12차원, 보행 기본 scale `0.25`, RECOVER scale `0.70` |

회전은 오른손 법칙을 사용해요. base frame의 `+omega_z`는 위에서 내려다볼 때 반시계 방향인 좌회전, `-omega_z`는 우회전이에요. hip, thigh, calf의 굽힘·폄 방향은 다리와 joint axis마다 부호가 다를 수 있으므로 “양수는 항상 굽힘”처럼 일반화하지 않고 asset의 joint axis와 실제 readback을 확인해요.

projected gravity는 가속도계 원시값 자체가 아니라 자세를 반영해 중력 방향을 body frame으로 투영한 값이에요. 현재 RECOVER actor의 `base_angular_velocity`와 `projected_gravity`는 각각 실기체 IMU gyroscope와 attitude estimator에 대응하고, 관절 위치·속도는 encoder와 미분·필터 결과에 대응해요. 정확한 sim/hardware 대응표는 [`recover_contracts.py`](../../src/isaac_walk_g009/recover_contracts.py)에 있어요.

## 1. 로봇 상태와 Observation

상태 `s_t`는 시뮬레이터가 알고 있는 전체 물리 상태이고, observation `o_t`는 policy에 실제로 보여 주기로 한 값이에요. 둘을 같다고 두면 실물에서 측정할 수 없는 값을 actor가 몰래 사용하는 문제가 생겨요.

현재 보행 정책의 `P-WALK-235`는 다음 235개 값을 사용해요.

| 항목 | 차원 | 단위·좌표계 | 폐루프에서 하는 일 |
|---|---:|---|---|
| base linear velocity | 3 | `[m/s]`, base frame | 전진·측면 속도 추종 오차를 policy가 보게 해요 |
| base angular velocity | 3 | `[rad/s]`, base frame | roll·pitch 흔들림과 yaw 회전을 보여 줘요 |
| projected gravity | 3 | 무차원 방향, base frame | 몸통 기울기와 위쪽 방향을 알려 줘요 |
| velocity command | 3 | `[m/s, m/s, rad/s]`, base frame | 전진·후진·좌·우회전 목표예요 |
| relative joint position | 12 | `[rad]`, joint frame | 각 다리가 기본 자세에서 얼마나 벗어났는지 보여 줘요 |
| joint velocity | 12 | `[rad/s]`, joint frame | 관절 운동 방향과 빠르기를 보여 줘요 |
| previous action | 12 | 무차원 | action 변화와 진동을 판단할 기억 한 칸이에요 |
| height scan | 187 | 지형 높이 정규화 값, base 주변 | 발 앞 지형의 높낮이를 policy에 제공해요 |

코드와 증거는 [`G006_PORTFOLIO.md`](../G006_PORTFOLIO.md)와 [`G008_COMMAND_FRICTION_LINK_MASS.md`](../G008_COMMAND_FRICTION_LINK_MASS.md)에서 확인해요.

RECOVER는 actor가 실물에서 얻을 수 있는 `P-RECOVER-83`만 받고, critic은 학습 중에만 terrain normal, 실제 마찰, CoM, 전체 질량, fall class 같은 privileged suffix를 더 받아 `C-RECOVER-107`을 구성해요. actor/critic 경계와 noise는 [`recover_env_cfg.py`](../../src/isaac_walk_g009/recover_env_cfg.py), 항목 차원과 실기체 대응은 [`recover_contracts.py`](../../src/isaac_walk_g009/recover_contracts.py)에 있어요.

관측을 바꿀 때는 다음을 같은 줄에 적어요.

1. 원시 source가 무엇인지 적어요. 예: `ContactSensor.data.net_forces_w`.
2. 좌표계와 단위를 적어요. 예: world-frame force `[N]`.
3. actor가 실물에서도 얻을 수 있는지 적어요.
4. noise, normalization, clip, missing-value 처리를 적어요.
5. tensor shape와 joint/body 순서를 runtime에서 다시 읽어요.

## 2. Policy, Action, Joint target

actor policy는 observation에서 12개 관절 action의 분포를 만들어요.

\[
a_t \sim \pi_\theta(a_t\mid o_t)
\]

여기서 `o_t`는 현재 observation, `a_t`는 무차원 action, `theta`는 actor network 파라미터예요. G006·G008 actor는 `235 → 512 → 256 → 128 → 12`, critic은 `235 → 512 → 256 → 128 → 1`인 ELU MLP예요. actor 출력 12개는 torque가 아니라 관절 위치 목표를 만들기 전의 정규화 action이에요.

기본 joint-position action은 다음처럼 읽어요.

\[
q_{des,t}=q_{default}+s_a a_t
\]

| 기호 | 뜻 | 단위 | 실제 움직임 |
|---|---|---|---|
| `q_default` | 서 있는 기본 관절각 | `[rad]` | hip·thigh·calf의 기준 자세 |
| `s_a` | action scale | `[rad/action]` | action 1.0이 목표각을 얼마나 움직이는지 결정 |
| `a_t` | actor의 정규화 출력 | 무차원 | 양·음 부호가 각 joint axis 방향으로 목표를 이동 |
| `q_des,t` | actuator가 추종할 목표각 | `[rad]` | PD가 실제로 따라갈 관절 목표 |

G006·G008 보행 기본 scale은 `0.25`예요. G009 RECOVER는 큰 자세 변화와 joint limit 안전을 함께 다루기 위해 `EMAJointPositionToLimitsAction`, scale `0.70`, EMA `0.2`, soft-limit factor `0.9`를 사용해요. RECOVER 경로는 [`recover_env_cfg.py`](../../src/isaac_walk_g009/recover_env_cfg.py)와 [`recover_contracts.py`](../../src/isaac_walk_g009/recover_contracts.py)에서 확인해요.

따라서 “policy가 토크를 출력한다”라고 설명하면 현재 프로젝트에는 틀려요. 이 프로젝트의 WALK·RECOVER actor는 joint-position action을 출력하고, Action Manager가 scale·offset·clip·EMA를 적용해 joint target을 만든 뒤 actuator가 torque를 만들어요.

## 3. PD/Actuator와 Torque

회전 관절의 이상적인 explicit PD는 다음처럼 읽어요.

\[
\tau_j=\operatorname{clip}\left(
k_p(q_{des}-q)+k_d(\dot q_{des}-\dot q)+\tau_{ff},
-\tau_{max},\tau_{max}
\right)
\]

| 기호 | 뜻 | 단위 | Go2에서 보이는 현상 |
|---|---|---|---|
| `q_des-q` | 목표각과 현재각의 오차 | `[rad]` | 다리를 원하는 방향으로 움직이게 해요 |
| `k_p` | stiffness | `[N·m/rad]` | 크면 목표를 강하게 따라가지만 충격·진동이 커질 수 있어요 |
| `q_dot_des-q_dot` | 목표·현재 각속도 오차 | `[rad/s]` | 현재 회전을 감속하거나 목표 속도를 따라가게 해요 |
| `k_d` | damping | `[N·m·s/rad]` | 작으면 출렁이고 너무 크면 둔해질 수 있어요 |
| `tau_ff` | feed-forward torque | `[N·m]` | 모델이 미리 요구하는 토크예요. 현재 기본 위치 action 설명에서는 중심 항이 아니에요 |
| `tau_max` | 모터 effort limit | `[N·m]` | 요구 토크가 모터 한계를 넘으면 포화돼요 |

현재 Go2는 explicit `DCMotor` actuator를 사용하고, readback의 stiffness는 `25`, damping은 `0.5`, motor effort limit은 `23.5 N·m`, velocity limit은 `30 rad/s`예요. `DCMotor`는 PD 요구 torque를 모터의 속도 의존 torque 한계로 제한해요. 최신 Isaac Lab은 actuator 명령과 telemetry를 backend-neutral `ActuatorCollection`으로 분리하지만, 이 프로젝트는 2.1.1의 고정 API와 실제 runtime readback을 근거로 설명해요. API 이름이 바뀌어도 `action → joint target → actuator → applied torque`라는 물리 의미는 유지돼요.

implicit actuator는 solver 안에서 PD를 적용하고, explicit actuator는 Isaac Lab 모델이 torque를 계산·제한해 solver에 넘겨요. 어느 쪽인지 확인하지 않고 stiffness 숫자만 비교하지 않아요. torque reward와 energy proxy는 `computed torque`가 아니라 실제 평가 코드가 읽은 `applied_torque` 계약을 확인해요.

## 4. Physics, 외란, 마찰, 접촉력

관절 torque가 들어오면 physics가 다음 상태를 계산해요. 접촉을 포함한 축약된 관절 동역학은 다음 구조로 읽어요.

\[
M(q)\ddot q+C(q,\dot q)\dot q+g(q)
=S^T\tau+J_c(q)^T\lambda+\tau_{ext}
\]

| 항 | 뜻 | 단위·좌표 | 사족보행에서의 의미 |
|---|---|---|---|
| `M(q)q_ddot` | 질량·관성이 만드는 가속 저항 | actuated joint 행은 `[N·m]` | thigh·calf 질량이 커지면 같은 torque로 덜 가속돼요 |
| `C(q,q_dot)q_dot` | 속도에 따른 Coriolis·원심 항 | actuated joint 행은 `[N·m]` | 다리를 빠르게 휘두를수록 커져요 |
| `g(q)` | 중력 일반화 힘 | actuated joint 행은 `[N·m]` | 몸통·다리를 버티는 torque를 요구해요 |
| `S^T tau` | actuator가 넣은 일반화 torque | `[N·m]` | hip·thigh·calf 모터의 실제 구동이에요 |
| `J_c^T lambda` | 발·몸통 접촉력이 관절에 만든 효과 | `lambda [N]`, 변환 뒤 `[N·m]` | stance 발의 지지력, 미끄럼, 충격이 관절로 전달돼요 |
| `tau_ext` | push 같은 외력이 만든 일반화 힘 | `[N·m]` 또는 floating-base `[N, N·m]` | 몸통을 밀어 자세와 속도를 흐트러뜨려요 |

floating-base 전체 식의 앞 6개 행은 몸통 병진력 `[N]`과 회전 모멘트 `[N·m]`가 섞이므로 모든 행을 단순히 torque 단위라고 부르지 않아요. 현재 프로젝트의 링크 질량 randomization은 `M`, `C`, `g`를 바꾸고, push event는 `tau_ext`를 바꾸며, 발 마찰은 가능한 `lambda`의 범위를 바꿔요.

발 접선력은 단순화한 Coulomb 마찰원뿔로 다음처럼 읽어요.

\[
\sqrt{F_x^2+F_y^2}\leq \mu F_z
\]

`F_x, F_y [N]`는 지면 접평면 방향의 힘, `F_z [N]`는 지면 normal 방향 지지력, `mu`는 무차원 마찰계수예요. 마찰이 낮아지면 전후 가속뿐 아니라 네 발 힘의 모멘트로 만드는 yaw 회전도 제한돼요. G008이 전진·후진과 좌·우회전을 따로 평가하는 이유예요.

외란은 [`rough_env_cfg.py`](../../src/isaac_walk_g006/rough_env_cfg.py)의 `events.push_robot`에서 baseline과 push curriculum의 유일한 normalized diff로 만들었어요. 마찰·링크 질량은 [`env_cfg.py`](../../src/isaac_walk_g008/env_cfg.py)에서 별도 분기로 유지해 서로 섞지 않았어요. 접촉력·마찰·질량 readback과 한계는 [`G008_COMMAND_FRICTION_LINK_MASS.md`](../G008_COMMAND_FRICTION_LINK_MASS.md)와 [`G008_PERIODIC_FRICTION_AND_LINK_MASS_LIMITS.md`](../G008_PERIODIC_FRICTION_AND_LINK_MASS_LIMITS.md)에서 확인해요.

## 5. 다음 상태, Reward, 성공 조건

reward는 다음 상태를 보고 그 한 control step이 목표에 얼마나 가까웠는지 수치화해요.

\[
r_t=\sum_i \Delta t\,w_i\,\rho_i(s_t,a_t,s_{t+1})
\]

`Delta t=0.02 s`는 현재 50 Hz control step, `rho_i`는 각 raw reward term, `w_i`는 term weight예요. raw term은 속도오차 제곱, torque 제곱, action 변화량처럼 서로 다른 단위를 가질 수 있고, weight가 이를 학습용 scalar로 바꿔요. 합산 reward는 물리적 에너지 단위가 아니라 설계된 목적함수 값이에요.

현재 G008의 대표 항은 다음과 같이 폐루프 위치가 달라요.

| reward | 읽는 값 | 폐루프에서 바꾸려는 행동 |
|---|---|---|
| `track_lin_vel_xy_exp` | 다음 base 선속도와 command 차이 | 전진·후진·측면 속도를 맞춰요 |
| `track_ang_vel_z_exp` | 다음 yaw rate와 command 차이 | 좌·우회전 속도를 맞춰요 |
| `lin_vel_z_l2`, `ang_vel_xy_l2` | 수직속도와 roll/pitch 각속도 | 몸통 튐과 흔들림을 줄여요 |
| `dof_torques_l2` | actuator torque | 같은 추종에서 과도한 torque 사용을 줄여요 |
| `action_rate_l2` | `a_t-a_(t-1)` | 급격한 관절 목표 변화와 진동을 줄여요 |
| `feet_air_time` | contact transition과 공중시간 | 발을 끌기보다 보행 주기를 만들도록 유도해요 |

reward는 성공 조건과 같지 않아요. mean reward는 학습 명령·환경 전체 평균이고, 실제 채택은 held-out 방향별 성공률, tracking RMSE, 자세 peak, fall, torque·power를 봐야 해요. G008 G0는 final mean reward `33.64`, episode length `1000`이었지만 정식 우회전 gate에 실패했어요. G009 rev8도 action smoothing으로 덜 움직여 손실이 줄었지만 복구 성공은 `0`이었어요.

보상 함수와 weight의 기계 판독 원본은 [`g008_reward_contract_s20260826.json`](../../reports/runs/g008_reward_contract_s20260826.json), RECOVER 보상·termination은 [`recover_env_cfg.py`](../../src/isaac_walk_g009/recover_env_cfg.py)에서 확인해요.

## 6. PPO에서 reward가 checkpoint로 바뀌는 과정

critic은 현재 상태에서 앞으로 받을 discounted reward의 기대값을 추정해요.

\[
V_\phi(o_t)\approx \mathbb{E}\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k}\right]
\]

actor는 “이 action이 현재 policy의 평균적인 선택보다 얼마나 나았는가”를 advantage로 배워요. 현재 설정의 GAE는 다음처럼 연결돼요.

\[
\delta_t=r_t+\gamma V_\phi(o_{t+1})-V_\phi(o_t)
\]

\[
\hat A_t=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}
\]

`delta_t`는 한 step TD error, `gamma=0.99`는 미래 reward 할인, `lambda=0.95`는 advantage의 긴 구간 정보와 추정 분산을 조절하는 GAE 계수예요. 이 값들은 물리 단위가 없고 reward scale을 따라가요.

PPO는 새 policy가 rollout을 만든 old policy에서 한 번에 너무 멀리 움직이지 않게 확률비를 잘라요.

\[
R_t(\theta)=\frac{\pi_\theta(a_t\mid o_t)}{\pi_{\theta_{old}}(a_t\mid o_t)}
\]

\[
L^{clip}=\mathbb{E}_t\left[
\min\left(R_t\hat A_t,\operatorname{clip}(R_t,1-\epsilon,1+\epsilon)\hat A_t\right)
\right]
\]

`R_t`와 `epsilon=0.2`는 무차원이에요. advantage가 양수면 그 action 확률을 높이고, 음수면 낮추되 update가 old policy에서 지나치게 벗어나지 않게 해요. clipping은 나쁜 reward를 좋은 reward로 바꾸는 기능이 아니라 parameter update의 보폭 제한이에요.

현재 보행 설정은 환경당 24 control step을 모은 뒤 PPO update를 해요. `num_envs=N`이면 rollout 하나는 `24N` transitions이고, 이를 4 mini-batch로 나눠 5 epoch 학습하므로 iteration당 optimizer mini-batch update는 20회예요. actor·critic 구조, clip, gamma, lambda의 실제 runtime 계약은 [`G006_PORTFOLIO.md`](../G006_PORTFOLIO.md), [`G008_REWARD_AND_ROAD_CURRICULUM.md`](../G008_REWARD_AND_ROAD_CURRICULUM.md), [`g008_reward_contract_s20260826.json`](../../reports/runs/g008_reward_contract_s20260826.json)에서 확인해요.

## PPO와 실제 결과를 같이 읽는 법

| 실험 | 폐루프에서 바꾼 한 축 | 실제 결과 | 배워야 할 관계 |
|---|---|---|---|
| G005 no-torque | reward의 torque penalty만 제거 | torque L2 `+11.92%`, power proxy `+5.48%`, action-rate L2 `+41.00%` | reward가 action 변화와 actuator 사용량을 실제로 바꿔요 |
| G006 push curriculum | physics 외란 분포만 변경 | 회복률 `99.5370% → 99.5988%`, paired CI가 0 포함 | checkpoint 두 개의 작은 평균 차이만으로 개선을 주장하지 않아요 |
| G008 G0/T1 | 지형 분리 후 yaw air-time gate만 변경 | 둘 다 우회전 held-out gate 실패 | PPO가 reward를 최적화해도 평가 success를 자동 보장하지 않아요 |
| G009 rev8 | action smoothing 강화 | 손실 감소처럼 보였지만 strict success `0` | 덜 움직이는 지역해를 reward 개선으로 오인하면 안 돼요 |
| G009 rev12~22 | actuator 뒤 physics/contact 계측을 진단 | Gate01·Gate10·PPO 차단, learned policy 미자격 | 학습 전에 sensor·contact authority와 안전 계약부터 검증할 수 있어요 |

G005 수치는 [`G005_REWARD_ABLATION.md`](../G005_REWARD_ABLATION.md), G006 결과는 [`G006_ROUGH_PUSH_RECOVERY.md`](../G006_ROUGH_PUSH_RECOVERY.md), G008 결과는 [`G008_REWARD_AND_ROAD_CURRICULUM.md`](../G008_REWARD_AND_ROAD_CURRICULUM.md), G009의 현재 한계는 [`G009_MOUNTAIN_SLOPE_RECOVERY.md`](../G009_MOUNTAIN_SLOPE_RECOVERY.md)에서 확인해요.

## 기존 자료를 새 구조에서 찾는 법

기존 문서·코드·보고서는 삭제하거나 이름을 바꾸지 않아요. 아래 표가 재분류 색인이에요.

| 학습 질문 | 기존 자료 | 폐루프 위치 | 현재 판정 |
|---|---|---|---|
| 환경과 학습 하네스가 실제로 도는가 | 루트 [`README.md`](../../README.md), [`VALIDATION_MATRIX.md`](../VALIDATION_MATRIX.md) | 전체 루프의 재현 기반 | G003·G004 완료 |
| reward 하나가 행동을 어떻게 바꾸는가 | [`G005_REWARD_ABLATION.md`](../G005_REWARD_ABLATION.md) | Reward → PPO → 새 policy | 4 variants × 3 seeds 완료 |
| 외란이 physics와 회복에 어떤 영향을 주는가 | [`G006_ROUGH_PUSH_RECOVERY.md`](../G006_ROUGH_PUSH_RECOVERY.md), [`rough_env_cfg.py`](../../src/isaac_walk_g006/rough_env_cfg.py) | Event/외란 → Physics → 다음 상태 | 작은 회복률 차이, 우월성 미주장 |
| observation·network·PPO batch가 어떻게 연결되는가 | [`G006_PORTFOLIO.md`](../G006_PORTFOLIO.md) | Observation → actor/critic → PPO | runtime 계약 기록 완료 |
| 방향 명령·마찰·질량을 왜 분리하는가 | [`G008_COMMAND_FRICTION_LINK_MASS.md`](../G008_COMMAND_FRICTION_LINK_MASS.md), [`env_cfg.py`](../../src/isaac_walk_g008/env_cfg.py) | Command/Event → Physics/Contact | command·friction·mass 별도 분기 |
| 평균 reward와 실제 성공이 왜 다른가 | [`G008_REWARD_AND_ROAD_CURRICULUM.md`](../G008_REWARD_AND_ROAD_CURRICULUM.md) | Reward/PPO → checkpoint → held-out 평가 | 두 후보 checkpoint 기각 |
| actor와 critic의 관측 경계는 무엇인가 | [`recover_env_cfg.py`](../../src/isaac_walk_g009/recover_env_cfg.py), [`recover_contracts.py`](../../src/isaac_walk_g009/recover_contracts.py) | State/Sensor → Observation | P83/C107 계약 고정 |
| 접촉력·solver 실패를 학습 실패와 어떻게 구분하는가 | [`G009_MOUNTAIN_SLOPE_RECOVERY.md`](../G009_MOUNTAIN_SLOPE_RECOVERY.md) | Actuator → Physics/Contact → 계측 | rev22까지 진단, 성공 policy 미자격 |
| MPC/WBC는 현재 PPO와 어떻게 다른가 | [`mpc_wbc/README.md`](mpc_wbc/README.md) | 별도 model-based 제어 비교축 | 학습 자료이며 현재 구현 아님 |

## 권장 학습 순서

이론 목차가 아니라 현재 실험의 한 바퀴를 따라가요.

1. `Python tensor와 shape`: `num_envs × observation_dim`, broadcasting, indexing을 실제 report로 확인해요.
2. `상태와 관측`: base/world/joint frame, IMU·encoder·contact source를 `recover_env_cfg.py`에서 찾아요.
3. `policy와 action`: 12차원 actor 출력, scale·offset·EMA가 joint target이 되는 과정을 계산해요.
4. `actuator와 동역학`: PD, effort limit, mass/inertia, external force가 다음 상태를 어떻게 바꾸는지 설명해요.
5. `접촉과 reward`: contact force가 observation·reward·termination에 각각 어떻게 재사용되는지 추적해요.
6. `PPO update`: reward, value, advantage, clipping이 한 checkpoint update로 이어지는 순서를 설명해요.
7. `통제실험`: G005·G006·G008 중 하나를 골라 한 조건만 바꾸고 baseline과 수치를 비교해요.
8. `실패 귀속`: observation, action, actuator, physics/contact, reward, PPO, evaluation 중 어디서 실패했는지 판정해요.
9. `강건성`: 외란·마찰·질량·지형을 한 축씩 넓히고 nominal guardrail을 함께 확인해요.
10. `비교 제어`: 필요할 때만 MPC/WBC 자료로 넘어가 PPO와 model-based controller의 입력·출력·가정을 비교해요.

## 한 번의 Isaac Lab 통제실험

한 실험에서 변경 요인은 하나만 둬요. “마찰도 바꾸고 reward도 바꾸고 network도 키운 실험”은 결과가 좋아져도 원인을 설명할 수 없어요.

### 1. 사전등록

| 항목 | 기록할 내용 |
|---|---|
| 질문 | 폐루프 어느 지점의 무엇을 확인하는가 |
| baseline | task, source commit, config hash, 시작 checkpoint hash |
| 한 변수 | 예: `dof_torques_l2 weight -0.0002 → 0.0` |
| 고정 조건 | seed 집합, env 수, iterations, command grid, terrain, 평가시간 |
| 주지표 | success rate, 방향별 tracking RMSE, 자세 안정성 |
| 부지표 | torque L2, power proxy, action-rate, contact/load, fall rate |
| 채택 gate | 최악 방향과 nominal guardrail을 포함한 수치 기준 |

### 2. 실행과 checkpoint

smoke는 wiring 검증이고 성능 실험이 아니에요. `64 env × 1 iteration`이 통과하면 observation/action/reward/loss/checkpoint 경로가 연결됐다는 뜻일 뿐이에요. 성능 비교는 같은 budget과 seed 집합으로 학습하고, checkpoint·TensorBoard·stdout·GPU 회수·source/config hash를 함께 남겨요.

### 3. 평가

학습 mean reward로 checkpoint를 채택하지 않아요. 같은 held-out evaluator에서 baseline과 variant를 다음 순서로 비교해요.

1. 성공률과 분자/분모를 적어요.
2. 전진·후진·좌회전·우회전별 RMSE를 적어요.
3. roll/pitch peak 또는 support-normal 기준 자세 오차를 적어요.
4. fall·termination 원인과 생존률을 적어요.
5. torque·power·action-rate·contact 지표를 적어요.
6. 3 seeds 미만이면 기술통계라고 표시하고 일반화하지 않아요.
7. 영상은 동작 증거로만 쓰고 정량 JSON을 대신하지 않아요.

### 4. 실패 원인 귀속

| 증상 | 먼저 의심할 폐루프 위치 | 다음 한 변수 실험 |
|---|---|---|
| tensor shape·순서가 예상과 다름 | Sensor/Observation | joint/body order readback만 고정해 재실행 |
| action이 작거나 관절 한계에 붙음 | Action/Joint target | action scale 또는 reset-target 불일치 하나만 검사 |
| 목표각은 정상인데 충격·진동이 큼 | PD/Actuator | stiffness, damping, effort limit, solver 중 하나만 검사 |
| CPU/GPU에서 접촉력이 갈림 | Physics/Contact/Measurement | 같은 pose·action·step의 backend readback 비교 |
| mean reward는 오르지만 성공률이 안 오름 | Reward/Evaluation | 방향별 reward 분해 또는 success gate를 먼저 검사 |
| 학습이 불안정하고 KL·loss가 튐 | PPO update | learning rate, clip, batch 중 하나만 변경 |
| nominal은 통과하고 특정 조건만 실패 | Distribution | 마찰·질량·지형 중 실패 축 하나만 분리 |

## 학습 완료 판정

이 프로젝트에서 “이론을 읽었다”는 완료가 아니에요. 아래 여섯 단계를 같은 주제에서 모두 통과해야 해요.

| 단계 | 통과 증거 |
|---:|---|
| 1. 자료 없이 제어 루프 설명 | 상태에서 PPO update까지 끊기지 않고 말하며 외란·IMU·토크·접촉력을 제자리에 놓아요 |
| 2. 코드/설정 위치 확인 | observation, action, actuator/readback, reward, PPO config 경로를 직접 찾아요 |
| 3. 직접 실험 | 재현 명령, source/config hash, checkpoint 또는 report가 있어요 |
| 4. 조건 변경 | 한 변수만 바꾼 diff가 있어요 |
| 5. 수치 비교 | baseline/variant의 success·reward·자세·torque 중 관련 지표를 같은 표로 비교해요 |
| 6. 실패 원인 설명 | 실패를 폐루프의 한 위치에 귀속하고 반증 가능한 다음 실험을 제안해요 |

하나라도 빠지면 `부분`, 실행·측정이 없으면 `미검증`으로 기록해요. 성공률은 반드시 분자/분모, 조건 범위, seed 또는 반복 횟수와 함께 적어요.

## 면접 답변 최종 확인

각 실험을 마치면 아래 네 질문에 파일과 수치를 근거로 답해요.

1. `무엇을 했나`: 바꾼 한 변수, 사용한 baseline/checkpoint, 평가 grid, 핵심 수치를 말해요.
2. `왜 이 방법인가`: 폐루프에서 원인을 격리하기 위해 왜 이 한 축과 지표를 골랐는지 말해요.
3. `어디까지가 내 구현인가`: 공식 Isaac Lab Go2 task·RSL-RL PPO·PhysX와, 이 저장소의 event normalization·reward variant·평가 protocol·진단 코드·테스트를 구분해요.
4. `다시 하면 무엇을 바꾸나`: 결과가 약했던 지점 하나와 다음 통제실험 하나를 말해요. 막연히 “데이터를 더 모은다”라고 답하지 않아요.

현재 구현 경계는 다음과 같아요.

| 구분 | 내용 |
|---|---|
| upstream | Isaac Sim/PhysX, Isaac Lab manager와 내장 Go2 task, 기본 actuator·reward term, RSL-RL PPO |
| 이 저장소 구현 | G006 push 비교, G008 command·마찰·질량·도로·reward variant, G009 terrain·RECOVER 계약·접촉 진단, 실행/평가/요약/검증 스크립트 |
| 아직 구현 아님 | 실물 Go2 배포, 검증된 sim-to-real 성능, 정식 MPC/WBC controller, 자격을 통과한 G009 RECOVER policy |

이 구분을 지키면 기존 결과를 과장하지 않으면서도 Python, 동역학, PPO, reward, 실험 설계를 하나의 Physical AI 제어 문제로 설명할 수 있어요.
