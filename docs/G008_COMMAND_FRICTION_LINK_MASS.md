# G008 논문 기반 방향 명령·마찰·다리 링크 질량 실험

## 현재 결론

G008은 서로 다른 원인을 한 번에 섞지 않는다. 먼저 전진·후진·제자리 좌회전·제자리 우회전을 학습 분포에서 충분히 자주 만나게 만들었다. 다음으로 발바닥 마찰과 다리 링크 질량을 별도 태스크로 나누고, 각 축을 좁은 범위에서 시작해 세 단계로 넓히도록 구성했다.

2026-08-26에 다음 항목을 실제로 확인했다.

- 명령, 마찰 S1, 링크 질량 S1 태스크가 각각 `64 env × 1 iteration × seed 42` headless 학습을 마쳤다.
- 세 실행 모두 프로세스 종료 코드, checkpoint, TensorBoard, 치명 로그 검사와 GPU 회수 조건을 통과했다.
- 처음부터 학습한 command run은 실행 자체는 통과했지만 300 iterations 뒤 평면에서 정지에 가까운 지역해로 수렴했다.
- G006 `model_1499.pt`에서 300 iterations를 이어 학습한 `model_1798.pt`는 평면 네 방향 gate를 모두 통과했다. 64개 환경이 5초 동안 모두 생존했고, 네 방향 모두 목표와 같은 속도 부호를 냈다.
- 같은 checkpoint를 rough terrain에서 평가하면 좌·우 회전은 통과하지만 전진과 후진의 순간 자세가 기준을 넘는다. 방향 명령 기능과 거친 지형 자세 강건성을 구분해 기록한 이유다.
- command, friction S1, leg-mass S1 checkpoint를 평면·seed 42·같은 명령 시퀀스로 다시 재생해 동기화 비교 영상을 만들었다. friction 패널에는 실제 표본 `μ_s=0.8152`, `μ_d=0.5799`가 들어갔고, leg-mass 패널의 16개 다리 body scale은 `0.9575~1.0452`였다.

friction S1은 장기 학습과 randomized-domain/nominal-domain 평면 평가를 통과했다. leg-mass S1은 학습 실행은 완료했지만 두 평면 평가에서 우회전 yaw gate를 잃었다. 두 축의 rough guardrail도 승인되지 않았다. S2·S3는 구현돼 있어도 이 상태에서는 열지 않는다.

## 실험을 네 파트로 나눈 이유

### Part 1. 논문과 공식 구현 조사

문헌에서 범위를 찾더라도 현재 로봇과 학습기에 그대로 대입하지 않았다. 논문마다 로봇, 접촉 모델, 적응 모듈, 구동기 모델이 다르기 때문이다. G008에서는 문헌 수치를 상한 후보로 사용하고, 첫 단계는 nominal 부근으로 줄였다.

### Part 2. 네 방향 명령

기존 uniform sampler도 음의 전진 속도와 양·음 yaw rate를 만들 수 있다. 다만 연속분포만 사용하면 순수 후진 `(v_x<0, v_y=0, ω_z=0)`과 순수 회전 `(v_x=0, v_y=0, ω_z≠0)`은 측도상 거의 나타나지 않는다. "범위 안에 있다"와 "충분히 학습한다"는 다른 말이다.

그래서 명령 sampler의 80%를 정확한 primitive에 배정하고, 나머지 20%는 연속 SE(2) 명령으로 남겼다. 이 분리 덕분에 네 방향을 반복 학습하면서도 옆걸음과 곡선 보행을 버리지 않는다.

### Part 3. 발바닥 마찰

마찰은 접촉력 한계를 바꾼다. 낮은 마찰에서는 같은 발 디딤과 같은 토크 명령을 내도 필요한 접선력을 만들지 못할 수 있다. 이 축은 몸체 관성 변화와 원인이 다르므로 별도 태스크로 둔다.

### Part 4. 다리 링크 질량

질량 변화는 관성행렬, 중력항, 관절 가속에 필요한 토크를 바꾼다. 특히 원위부 링크의 질량 증가는 같은 총질량 증가라도 몸통 payload와 다르게 swing cost를 키운다. 발 질량 변화와 몸통 질량 변화를 같은 `base mass` randomization으로 대신할 수 없다.

## 실행 스택과 headless 구조

| 항목 | 고정값 |
| --- | --- |
| OS | Windows 11 네이티브 |
| Isaac Sim | 4.5.0 |
| Isaac Lab | v2.1.1, commit `90b79bb2d44feb8d833f260f2bf37da3487180ba` |
| 강화학습 | RSL-RL `2.3.3`의 PPO |
| 로봇 | Isaac Lab 내장 Unitree Go2 |
| 물리 timestep | `0.005 s`, 200 Hz |
| action decimation | `4` |
| 정책 제어주기 | `0.02 s`, 50 Hz |
| episode horizon | `20 s`, 최대 1,000 control steps |
| 학습 표시 | `--headless` |
| 학습 장치 | `cuda:0`, RTX 3060 12 GB |

headless는 물리 시뮬레이션을 생략한다는 뜻이 아니다. PhysX 접촉 계산, 센서 갱신, 정책 추론과 PPO update는 그대로 수행하고 화면 창과 실시간 카메라 렌더링만 띄우지 않는다. 학습 증거는 checkpoint, TensorBoard, stdout, GPU 사용량으로 남긴다. 영상은 카메라를 켠 별도 headless rendering 실행에서 만들며 정량 평가를 대신하지 않는다.

## PPO에서 실제로 한 번의 iteration이 뜻하는 것

Go2 rough runner 설정은 환경당 24 control step을 모은 뒤 PPO를 한 번 갱신한다.

| 항목 | 값 |
| --- | ---: |
| rollout steps per environment | `24` |
| PPO epochs per iteration | `5` |
| mini-batches per epoch | `4` |
| clip parameter | `0.2` |
| value loss coefficient | `1.0` |
| entropy coefficient | `0.01` |
| initial learning rate | `1e-3` |
| schedule | adaptive, desired KL `0.01` |
| discount `γ` | `0.99` |
| GAE `λ` | `0.95` |
| max gradient norm | `1.0` |
| empirical observation normalization | 사용하지 않음 |

예를 들어 command qualification의 `1,024 env × 300 iterations`는 다음 규모다.

- 수집 transition: `1,024 × 24 × 300 = 7,372,800`
- iteration당 rollout batch: `24,576 samples`
- mini-batch 크기: `24,576 / 4 = 6,144 samples`
- iteration당 optimizer mini-batch step: `5 epochs × 4 = 20`
- 전체 optimizer mini-batch step: `300 × 20 = 6,000`

`64 env × 1 iteration` 스모크는 알고리즘 성능 실험이 아니다. 이때 rollout은 1,536 transition이고 mini-batch는 384 samples다. 스모크가 증명하는 것은 태스크 등록, 환경 생성, observation/action 연결, loss 계산, checkpoint 저장과 자원 회수 경로다.

정책 observation은 235차원이다. base 선속도 3, base 각속도 3, projected gravity 3, 명령 3, 관절 위치 12, 관절 속도 12, 직전 action 12, height scan 187을 사용한다. actor와 critic은 각각 `512 → 256 → 128` ELU MLP이며 actor 출력은 12개 관절 action이다. 관절 위치 목표 scale은 `0.25`, Go2 actuator의 stiffness는 `25.0`, damping은 `0.5`다.

## 실제로 사용한 보상함수

보상은 문서에 적은 의도만으로 정하지 않았다. Isaac Lab 런타임에서 각 태스크의 최종 설정을 다시 읽고, 함수 원본과 가중치, 매개변수, PPO 설정의 SHA-256까지 `reports/runs/g008_reward_contract_s20260826.json`에 고정했다. command, 불규칙 도로 G0, 혼합 마찰 도로 S1은 아래 보상 계약이 서로 같다.

control step (t)에서 RewardManager가 더하는 값은 다음과 같다.

\[
r_t = \Delta t \sum_i w_i\,\rho_i(t), \qquad \Delta t = 0.02\;\mathrm{s}
\]

가중치에 `0.02s`가 곱해지므로, 아래 표의 숫자는 step마다 그대로 더하는 상수가 아니다. 물리는 `0.005s`마다 계산하고 action은 네 번의 물리 step마다 갱신한다.

| 이름 | 가중치 (w_i) | 원시 항 (\rho_i) | 역할 |
| --- | ---: | --- | --- |
| `track_lin_vel_xy_exp` | `+1.5` | `exp(-||v_cmd,xy-v_base,xy||² / 0.5²)` | 전·후·측방 선속도 추종 |
| `track_ang_vel_z_exp` | `+0.75` | `exp(-(ω_cmd,z-ω_base,z)² / 0.5²)` | 좌·우 yaw-rate 추종 |
| `lin_vel_z_l2` | `-2.0` | `v_base,z²` | 불필요한 수직 속도 억제 |
| `ang_vel_xy_l2` | `-0.05` | `ω_base,x² + ω_base,y²` | roll·pitch 각속도 억제 |
| `dof_torques_l2` | `-0.0002` | `Στ_j²` | 큰 관절 토크 억제 |
| `dof_acc_l2` | `-2.5e-7` | `Σq̈_j²` | 급한 관절 가속 억제 |
| `action_rate_l2` | `-0.01` | `Σ(a_t,j-a_t-1,j)²` | action 진동 억제 |
| `feet_air_time` | `+0.01` | `Σ((last_air_time-0.5s)·first_contact)·I(||v_cmd,xy||>0.1m/s)` | 움직일 때 발을 들어 다음 접촉까지 유지 |
| `flat_orientation_l2` | `0.0` | projected gravity의 body x/y 성분 제곱합 | 설정에는 있지만 비활성 |
| `dof_pos_limits` | `0.0` | soft joint limit 밖 거리의 합 | 설정에는 있지만 비활성 |

`undesired_contacts`도 Go2 설정에서 `None`이라 계산하지 않는다. 종료 조건은 20초 timeout과 base 접촉 force `>1N`이다. base 접촉은 episode를 끝내지만 별도의 scalar termination penalty는 없다. 따라서 현재 목적함수는 평가에서 사용하는 `|roll|`, `|pitch|`, 접촉 중 발 미끄럼을 직접 벌점으로 주지 않는다. `ang_vel_xy_l2`가 자세 변화 속도를 간접 억제할 뿐, 기울어진 채 정지한 자세 자체를 직접 벌점으로 계산하지는 않는다.

여기서 회전과 맞지 않는 지점이 하나 확인됐다. 제자리 좌·우회전 명령은 `[0, 0, ±0.5]`라서 `||v_cmd,xy||=0`이다. 기존 `feet_air_time`은 yaw 명령을 보지 않으므로 이 두 명령에서 항상 꺼진다. 반면 제자리 회전도 네 발이 몸체 중심 둘레에서 접선 속도를 만들어야 한다. G0 후속 실험은 다른 가중치와 지형, PPO 설정을 그대로 두고 활성 조건만 다음처럼 바꾼다.

\[
I\left(\lVert v_{cmd,xy}\rVert>0.1\;\mathrm{m/s}\;\lor\;|\omega_{cmd,z}|>0.1\;\mathrm{rad/s}\right)
\]

이 변형은 `feet_air_time` 가중치를 키운 실험이 아니다. 순수 회전에서 기존 항이 0이던 문제만 고친 단일 축 ablation이다. `tests/test_g008_config_diff.py`는 태스크 설정 차이가 이 함수와 yaw threshold뿐인지 확인하고, 순수 yaw 명령에서는 보상이 켜지며 정지 명령에서는 0인지 수치로 검사한다.

## Part 2: 방향 명령 설계

### 좌표계와 부호

명령 벡터는 로봇 base frame의 다음 순서다.

\[
\mathbf{c} = [v_x, v_y, \omega_z]
\]

- `+v_x`: 전진
- `-v_x`: 후진
- `+ω_z`: 위에서 봤을 때 반시계 방향, 좌회전
- `-ω_z`: 위에서 봤을 때 시계 방향, 우회전

좌·우 회전은 heading target이 아니라 yaw-rate target이다. `heading_command=False`로 두었기 때문에 절대 방위를 향해 도는 controller가 중간에 개입하지 않는다.

### 실제 sampler

| branch | 확률 | 범위 또는 primitive |
| --- | ---: | --- |
| exact primitive | `0.80` | 아래 다섯 명령 중 가중 추출 |
| continuous SE(2) | `0.20` | `v_x∈[-0.8,0.8]`, `v_y∈[-0.5,0.5]`, `ω_z∈[-0.8,0.8]` |

| primitive | 명령 `[v_x, v_y, ω_z]` | primitive branch 내부 가중치 | 전체 sampler에서의 질량 |
| --- | --- | ---: | ---: |
| forward | `[+0.60, 0, 0]` | `0.225` | `0.18` |
| backward | `[-0.40, 0, 0]` | `0.225` | `0.18` |
| left turn | `[0, 0, +0.50]` | `0.225` | `0.18` |
| right turn | `[0, 0, -0.50]` | `0.225` | `0.18` |
| stand | `[0, 0, 0]` | `0.10` | `0.08` |

명령은 4~6초마다 다시 뽑는다. continuous branch에는 기존 `rel_standing_envs=0.02`도 적용되므로 전체 정지 비율은 표의 8%보다 약간 높다.

### 회전에서 발이 해야 하는 일

몸체의 평면 twist가 \((\mathbf{v}_{COM}, \omega_z)\)일 때 몸체 기준 위치가 \(\mathbf{r}_i\)인 발의 목표 지면 상대 속도 구조는 다음과 같다.

\[
\mathbf{v}_i = \mathbf{v}_{COM} + \boldsymbol{\omega} \times \mathbf{r}_i
\]

제자리 회전에서는 \(\mathbf{v}_{COM}=0\)이어도 각 발의 항은 0이 아니다. 앞·뒤, 좌·우 발이 몸체 중심에서 서로 다른 위치에 있으므로 접선 방향도 달라진다. 지면 반력으로 만드는 yaw moment는 다음 합으로 볼 수 있다.

\[
\tau_z = \sum_i (r_{i,x}F_{i,y} - r_{i,y}F_{i,x})
\]

좌우 다리 속도를 반대로 주는 규칙만으로는 접촉 시점, 발 위치, 마찰 한계를 함께 만족하기 어렵다. 이번 구현은 네 primitive를 별도 controller로 하드코딩하지 않고, 같은 12관절 PPO 정책이 명령 observation을 보고 보행 패턴을 선택하게 한다.

### 방향 평가 프로토콜

평가는 64개 환경을 네 방향에 16개씩 고정 배정한다. horizon은 250 control step, 즉 5초다. 시작 50 step, 1초는 가속 transient로 보고 RMSE에서 제외한다. 첫 episode만 세고 base contact 또는 timeout 이후의 자동 reset sample은 제외한다. command 자체의 자세 gate는 지형 경사와 원인을 섞지 않도록 plane에서 판단하고, 원래 rough terrain은 별도 stress 결과로 남긴다.

기록 항목은 다음과 같다.

- 선속도 vector RMSE와 yaw-rate RMSE
- 달성한 평균 `v_x`, `v_y`, `ω_z`와 명령 부호 일치 여부
- base contact 생존률
- roll/pitch 절댓값의 최댓값
- 관절 applied torque 벡터의 L2 norm 평균
- `Σ|τ_j q̇_j|` 기계적 파워 proxy

gate는 네 방향 각각 생존률 100%, 선속도 RMSE `≤0.25 m/s`, yaw RMSE `≤0.25 rad/s`, `|roll|≤0.35 rad`, `|pitch|≤0.35 rad`, 명령 부호 일치를 모두 요구한다. 파워 값은 시뮬레이션 proxy이지 배터리 전력 측정값이 아니다.

### 실제 command qualification 결과

같은 `1,024 env × 300 iterations × seed 42`라도 초기화에 따라 결과가 달랐다. 처음부터 학습한 run은 `model_299.pt`까지 정상 완료했고 최종 mean episode length도 `980.99`였지만, 평면 고정 평가에서는 전진 평균 `v_x`와 좌·우 평균 yaw rate가 거의 0이었다. 학습 파이프라인 통과와 명령 추종 성공은 같은 판정이 아니다.

G006 rough baseline의 `model_1499.pt`에서 이어 학습한 run은 loaded iteration을 포함해 `1499~1798`의 300회 update를 수행했다. 이 범위는 RSL-RL 로그의 `1798/1799` 표기와 `model_1798.pt`를 함께 확인했다. resume 검증식의 off-by-one 오류도 이 실행에서 발견해 회귀 테스트를 추가했다.

| 항목 | scratch command | G006 warm-start command |
| --- | ---: | ---: |
| env × iterations | `1,024 × 300` | `1,024 × 300` |
| transitions | `7,372,800` | `7,372,800` |
| wall time | `1,201.052 s` | `1,077.001 s` |
| mean throughput | `6,299.93 steps/s` | `7,039.55 steps/s` |
| final mean reward | `22.26` | `35.41` |
| final mean episode length | `980.99` | `1,000.00` |
| peak VRAM | `5,907 MiB` | `5,892 MiB` |
| 평면 네 방향 gate | FAIL | PASS |

warm-start checkpoint의 SHA-256은 `53cc09043088bcd53618d2ae1f90c7f2e91d01eab7090cc63922486942b2ed47`이다. 평면 결과는 다음과 같다.

| 방향 | 생존률 | 선속도 RMSE | yaw RMSE | max roll | max pitch | 판정 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 전진 | `100%` | `0.0466 m/s` | `0.0741 rad/s` | `0.0736 rad` | `0.0769 rad` | PASS |
| 후진 | `100%` | `0.0794 m/s` | `0.0786 rad/s` | `0.1460 rad` | `0.2088 rad` | PASS |
| 좌회전 | `100%` | `0.0566 m/s` | `0.1154 rad/s` | `0.1246 rad` | `0.2155 rad` | PASS |
| 우회전 | `100%` | `0.0505 m/s` | `0.1152 rad/s` | `0.2159 rad` | `0.1816 rad` | PASS |

rough terrain에서도 네 방향 생존률과 추적 오차는 기준 안이었다. 다만 전진은 max roll/pitch가 `0.3713/0.4788 rad`, 후진은 max pitch가 `0.3505 rad`여서 자세 gate를 넘었다. 좌·우 회전은 rough에서도 통과했다. 따라서 command 기능은 평면 gate로 승인하고, S2·S3 범위 확대 전에는 rough 전진·후진 자세를 다시 확인한다.

## Part 3: 바닥 마찰 단계

### 접촉역학

Coulomb 마찰의 접선력 한계는 다음과 같다.

\[
\sqrt{F_x^2 + F_y^2} \le \mu F_z
\]

평면에서 수직항력이 \(mg\)에 가깝다고 놓으면 이상적인 가속도 상한은 \(|a_{xy}|≤\mu g\)다. 실제 사족보행에서는 한 시점의 stance 발 수, 수직력 분배, 충격과 발 미끄럼 때문에 이 값보다 먼저 한계가 나타날 수 있다.

마찰이 낮아지면 전진 가속만 나빠지는 것이 아니다. yaw moment도 각 발의 접선력에서 나오기 때문에 제자리 회전이 먼저 흔들릴 수 있다. 반대로 마찰이 매우 높으면 slip이 줄지만 접촉 충격과 관절 torque peak가 커질 수 있다. 그래서 속도 RMSE만 보지 않고 자세, torque, power proxy를 같이 기록한다.

### Isaac Lab에서 적용한 방식

- 복제된 terrain의 전역 USD material을 환경마다 다시 만드는 대신, robot foot material을 환경·foot shape별로 바꾼다. terrain 계수가 1.0이고 combine mode가 multiply인 현재 설정에서는 접촉쌍의 유효 계수가 sampled foot 계수와 같아진다. 물리적으로는 바닥-발 접촉계수를 바꾸지만 구현상 수정 대상은 발 material이다.
- startup event에서 로봇의 `.*_foot` collision shape만 바꾼다.
- 환경·foot shape마다 64개 material bucket 중 하나를 배정한다. 네 발에 같은 계수를 강제하지 않으므로 좌우 비대칭 접촉도 표본에 들어간다.
- restitution은 0으로 고정한다.
- `make_consistent=True`로 dynamic friction이 static friction보다 커지지 않게 한다.
- terrain material은 static/dynamic 모두 1.0이고 combine mode가 multiply다. 이 조건에서 유효 계수는 sampled foot 계수와 같다.

| stage | static friction | dynamic friction | 이상적 `μ_d g` 범위 | 의미 |
| --- | --- | --- | --- | --- |
| S1 narrow | `0.72~0.88` | `0.52~0.68` | `5.10~6.67 m/s²` | nominal `0.8/0.6` 주변 배선·표면 오차 |
| S2 moderate | `0.62~1.00` | `0.42~0.78` | `4.12~7.65 m/s²` | 눈에 띄는 미끄럼 차이 |
| S3 research envelope | `0.50~1.25` | `0.30~1.00` | `2.94~9.81 m/s²` | 문헌 상한을 포함한 stress stage |

S3의 static 상한 `1.25`는 Tan 등의 contact friction 범위에 닿지만, dynamic 하한 `0.30`과 결합한 전체 stage를 그 논문과 동일한 실험이라고 부르지는 않는다. RMA의 friction `0.05~4.5`는 적응 모듈까지 포함한 설정이다. 현재 plain PPO에 그대로 넣으면 지나치게 보수적인 정책을 만들 가능성이 있어 채택하지 않았다.

## Part 4: 다리 링크 질량 단계

### 어떤 몸체를 바꾸는가

런타임에서 확인한 Go2 body 중 다음 정규식에 맞는 16개만 대상이다.

```text
.*_(hip|thigh|calf|foot)
```

| link group | 개수 | link당 nominal mass | group mass |
| --- | ---: | ---: | ---: |
| hip | 4 | `0.678 kg` | `2.712 kg` |
| thigh | 4 | `1.152 kg` | `4.608 kg` |
| calf | 4 | `0.154 kg` | `0.616 kg` |
| foot | 4 | `0.040 kg` | `0.160 kg` |
| 합계 | 16 | - | `8.096 kg` |

base `6.921 kg`과 head dummy 두 개는 이 이벤트에서 바꾸지 않는다. 각 환경·각 body가 독립적으로 uniform scale을 뽑는다.

| stage | body별 mass scale | 16개 링크가 모두 경계값일 때의 합계 |
| --- | --- | --- |
| S1 narrow | `0.95~1.05` | `7.6912~8.5008 kg` |
| S2 moderate | `0.90~1.10` | `7.2864~8.9056 kg` |
| S3 literature envelope | `0.80~1.20` | `6.4768~9.7152 kg` |

표의 합계 경계는 모든 링크가 동시에 같은 끝값을 뽑는 이론적 corner다. 실제 sampler는 링크마다 독립이므로 총질량은 중앙에 더 모인다. 반면 한쪽 calf나 foot만 무거운 비대칭도 나타날 수 있어 swing과 yaw에 영향을 줄 수 있다.

### 왜 inertia를 같이 바꾸는가

다관절 로봇의 운동방정식은 다음 꼴이다.

\[
M(q)\ddot q + C(q,\dot q)\dot q + g(q)
= S^T\tau + J(q)^T\lambda
\]

링크 질량을 바꾸면 관성행렬 \(M\), Coriolis/centrifugal 항 \(C\), 중력항 \(g\)가 함께 달라진다. 질량만 바꾸고 inertia tensor를 그대로 두면 물리적으로 서로 맞지 않는 body를 만들 수 있다. G008은 `recompute_inertia=True`를 사용해 nominal inertia를 mass ratio만큼 scale한다.

이 근사는 형상이 같고 밀도가 균일하게 변한다는 가정이다. center of mass와 collision geometry는 바뀌지 않는다. 실제로 발끝에 센서나 보호대를 붙이면 질량뿐 아니라 COM과 inertia 분포가 바뀌므로, 이 실험만으로 해당 하드웨어 변경을 완전히 재현했다고 볼 수 없다.

원위부 질량의 효과는 관절축에서 본 관성으로 설명할 수 있다. 단순화하면 필요한 swing torque는 \(\tau≈I\alpha\)이고, 점질량 항은 \(I≈mr^2\)로 커진다. 몸통 가까운 hip 질량 100 g과 발끝의 100 g은 총질량은 같아도 hip/knee가 느끼는 관성 효과가 다르다. body별 독립 sampler를 쓴 이유다.

## S1 물성 runtime probe

설정 객체만 비교하면 PhysX에 들어간 실제 값을 놓칠 수 있다. 그래서 seed 42, 1,024개 환경을 생성한 직후 robot view에서 material, mass, inertia를 다시 읽었다.

| 확인 항목 | friction S1 task | leg-mass S1 task |
| --- | ---: | ---: |
| foot static friction | `0.722592~0.876980` | `0.800000` 고정 |
| foot dynamic friction | `0.529500~0.672877` | `0.600000` 고정 |
| restitution | `0.0` | `0.0` |
| leg mass scale | `1.0` 고정 | `0.950001~1.049993` |
| sampled total leg mass | `8.0960 kg` | `7.8296~8.3781 kg` |
| inertia 재계산 최대 오차 | 약 `9.31e-10` | 약 `1.86e-9` |

friction task에서 링크 질량이 1.0으로 유지되고, leg-mass task에서 foot friction이 nominal `0.8/0.6`으로 유지되는 것도 함께 확인했다. 두 축이 설정 diff뿐 아니라 런타임 물성에서도 분리됐다는 증거다. inertia 오차는 `default_inertia × mass_ratio`와 runtime inertia의 절댓값 차이이며, 부동소수점 수준이었다.

## Friction S1 학습과 평가

command `model_1798.pt`에서 friction S1을 `1,024 env × 300 iterations × seed 42`로 이어 학습했다. `model_2097.pt`까지 완료됐고 SHA-256은 `40af0a0f80489d705e1e8fdeedd2f765177d3d67bf757709b9195cc2bbeaaee0`이다.

| 항목 | 결과 |
| --- | ---: |
| wall time | `1,234.808 s` |
| mean throughput | `6,213.25 steps/s` |
| final mean reward | `35.19` |
| final mean episode length | `1,000.00` |
| peak VRAM | `5,936 MiB` |
| final terrain level mean | `2.2684` |

물성이 변한 평면과 nominal `0.8/0.6` 평면에서 같은 64환경 평가를 수행했다. 두 조건 모두 생존률 100%와 네 방향 gate를 통과했다.

| 평가 domain | 선속도 RMSE 범위 | yaw RMSE 범위 | max roll 범위 | max pitch 범위 | 판정 |
| --- | ---: | ---: | ---: | ---: | --- |
| randomized friction S1 | `0.05~0.06 m/s` | `0.08~0.16 rad/s` | `0.08~0.12 rad` | `0.06~0.13 rad` | PASS |
| nominal friction | `0.05~0.06 m/s` | `0.07~0.15 rad/s` | `0.06~0.11 rad` | `0.08~0.13 rad` | PASS |

평면 S1 gate는 통과했지만 rough 학습 중 terrain level mean은 초반 약 3.45에서 최종 2.27로 내려갔다. 낮은 난이도로 이동한 결과라 rough 강건성이 좋아졌다는 근거로 쓰지 않는다. command checkpoint의 rough 전진·후진 자세 실패도 남아 있어 friction S2는 이 상태에서 열지 않는다.

## Leg-mass S1 학습과 평가

friction checkpoint를 이어받지 않고 같은 command `model_1798.pt`에서 leg-mass S1을 따로 학습했다. `1,024 env × 300 iterations × seed 42`가 `model_2097.pt`까지 완료됐고 SHA-256은 `8976cfff6eee6d1a998c7aa554b23d98b01d3d64da02b43ac3133a9186ae97fa`다.

| 항목 | 결과 |
| --- | ---: |
| wall time | `1,373.046 s` |
| mean throughput | `5,668.77 steps/s` |
| final mean reward | `35.25` |
| final mean episode length | `993.71` |
| peak VRAM | `5,908 MiB` |
| final terrain level mean | `2.2945` |

randomized mass 평면과 nominal mass 평면에서 전진·후진·좌회전은 통과했다. 우회전은 생존과 자세 기준을 지켰지만 yaw 추종이 느려졌다.

| 우회전 조건 | 평균 달성 yaw rate | yaw RMSE | torque L2 mean | power proxy mean | 판정 |
| --- | ---: | ---: | ---: | ---: | --- |
| command checkpoint, nominal | `-0.4533 rad/s` | `0.1152 rad/s` | `12.60 Nm` | `12.19 W` | PASS |
| leg-mass S1, randomized | `-0.2348 rad/s` | `0.2956 rad/s` | `15.71 Nm` | `10.65 W` | FAIL |
| leg-mass S1, nominal guardrail | `-0.2353 rad/s` | `0.2947 rad/s` | `15.63 Nm` | `10.56 W` | FAIL |

randomized와 nominal에서 거의 같은 퇴화가 남았으므로 특정 sampled mass 한 번의 영향으로 설명할 수 없다. 이 300-iteration checkpoint가 우회전 정책을 덜 쓰는 쪽으로 이동한 결과일 수 있지만, seed 42 한 번으로 원인을 확정하지 않는다. torque norm은 커지고 power proxy는 낮아진 점도 함께 남긴다. 우회전 각속도가 절반 수준으로 줄어든 채 관절 속도가 작아졌을 가능성을 다음 시계열 분석에서 확인한다. nominal guardrail이 실패했으므로 leg-mass S2는 중단한다.

leg-mass 학습의 terrain level mean도 초반 약 3.43에서 최종 2.29로 내려갔다. 평면 우회전 퇴화와 rough curriculum 후퇴가 함께 나타나 S1 재설계가 먼저다.

## 난이도 상승 규칙

실행 순서는 다음과 같다.

1. command suite 자체의 네 방향 gate를 통과시킨다.
2. friction S1을 command suite와 비교한다.
3. leg mass S1을 command suite와 비교한다.
4. 한 축의 S1이 randomized-domain과 nominal-domain에서 모두 기준을 만족할 때 그 축의 S2로 간다.
5. S2도 같은 조건을 통과할 때만 S3를 연다.
6. friction과 mass를 동시에 섞는 실험은 두 단일축 결과가 나온 뒤 별도 상호작용 실험으로 다룬다.

| 실행 단계 | env | iterations | seeds | 용도 |
| --- | ---: | ---: | --- | --- |
| smoke | 64 | 1 | 42 | 코드·GPU·산출물 경로 |
| S1 | 1,024 | 300 | 42 | 좁은 범위의 학습 가능성 |
| S2 | 2,048 | 600 | 42, 43 | 중간 범위와 seed 변동 |
| S3 | 4,096 | 1,500 | 42, 43, 44 | 문헌 범위 stress test |

한 seed의 S1은 성능 결론이 아니라 다음 단계 실행 가능성 판단이다. S2부터 두 seed, S3에서 세 seed를 쓰고, nominal guardrail이 나빠지면 randomized-domain 성능이 좋아도 범위를 넓히지 않는다.

## 논문에서 실제 설계로 옮긴 부분

| 근거 | 확인한 내용 | G008 적용 | 그대로 복사하지 않은 부분 |
| --- | --- | --- | --- |
| [Rudin et al., 2022](https://proceedings.mlr.press/v164/rudin22a.html) | massively parallel PPO와 terrain curriculum | 1,024→2,048→4,096 env ladder, official rough curriculum 유지 | 논문의 전체 reward/curriculum을 재현했다고 주장하지 않음 |
| [Margolis & Agrawal, 2023](https://proceedings.mlr.press/v205/margolis23a/margolis23a.pdf) | parameterized velocity command와 행동 다양성 | `v_x,v_y,ω_z` 명령, primitive와 continuous branch 병행 | 별도 behavior parameter나 gait family는 추가하지 않음 |
| [Tan et al., 2018](https://arxiv.org/html/1804.10332) | mass `80~120%`, contact friction `0.5~1.25` 등 dynamics randomization | mass S3와 static friction S3의 상한 envelope | 첫 학습부터 전체 범위를 사용하지 않음 |
| [Kumar et al., 2021, RMA](https://arxiv.org/html/2107.04034) | friction, payload, motor strength의 linear curriculum과 adaptation | 쉬운 범위부터 넓히는 원칙, payload/마찰 평가 관점 | RMA의 `0.05~4.5` friction과 adaptation network는 현재 PPO에 넣지 않음 |
| [Xie et al., 2021](https://arxiv.org/html/2011.02404) | 불필요한 randomization이 보수적 정책을 만들 수 있음 | friction/mass 단일축 분리, nominal guardrail | 모든 dynamics parameter를 한 번에 섞지 않음 |
| [Isaac Lab 2.1.1 events](https://isaac-sim.github.io/IsaacLab/v2.1.1/_modules/isaaclab/envs/mdp/events.html) | material bucket, body별 mass, inertia 재계산의 실제 동작 | 공식 helper를 그대로 호출하고 설정 diff를 테스트 | private fork나 별도 PhysX patch를 만들지 않음 |

## sim-to-real로 넘어가기 전에 필요한 측정

이번 구현은 sim-to-real을 고려한 학습 설계이지, 실물 전이를 끝냈다는 증거가 아니다. 다음 항목을 실물에서 확인해야 범위를 물리량과 연결할 수 있다.

### 마찰

- 실제 발 패드와 시험 표면의 static/dynamic friction을 경사면 또는 pull test로 측정한다.
- 마른 바닥, 먼지, 타일, 매트처럼 표면별 반복 분포를 남긴다.
- PhysX combine mode와 실물 접촉이 일대일로 대응한다고 가정하지 않는다.
- 미끄럼이 시작되는 힘, 정상력, slip velocity를 같이 기록한다.

### 링크 질량

- CAD/제조사 값과 실제 분해 계측값을 구분한다.
- 케이블, 보호대, 센서, 나사까지 장착 상태의 질량을 잰다.
- 질량만 변하는지 COM 위치도 변하는지 확인한다.
- 관절별 current/torque와 swing 가속 응답을 비교한다.

### 아직 randomize하지 않은 축

motor strength, actuator delay, control latency, encoder/IMU bias, battery voltage는 G008의 현재 단일축 실험에서 고정이다. 실물 gap이 남으면 이 항목을 식별한 뒤 별도 파트로 추가해야 한다. friction과 mass 결과가 좋다는 이유로 actuator gap까지 해결됐다고 해석하지 않는다.

## 재현 명령

```powershell
cd "$HOME\isaac-walk-rl"

# import-light 계약 테스트
py -m pytest .\tests\test_g008_contracts.py .\tests\test_g008_direction_evaluation.py -q

# Isaac 설정 격리 테스트
& "$HOME\IsaacLab\_isaac_sim\python.bat" -m pytest .\tests\test_g008_config_diff.py -q

# 세 파트 runtime smoke
.\scripts\run_g008_stage.ps1 -Part command -Stage 0 -NumEnvs 64 -MaxIterations 1 -Seed 42 -RunName g008_command_smoke_e64_i1_s42
.\scripts\run_g008_stage.ps1 -Part friction -Stage 1 -NumEnvs 64 -MaxIterations 1 -Seed 42 -RunName g008_friction_s1_smoke_e64_i1_s42
.\scripts\run_g008_stage.ps1 -Part leg_mass -Stage 1 -NumEnvs 64 -MaxIterations 1 -Seed 42 -RunName g008_leg_mass_s1_smoke_e64_i1_s42

# G006 장기 checkpoint에서 command 분포 미세조정
.\scripts\run_g008_stage.ps1 -Part command -Stage 0 -NumEnvs 1024 -MaxIterations 300 -Seed 42 `
  -ResumeRun 2026-08-24_18-31-51_g006_production_baseline_e4096_i1500_s42 `
  -ResumeCheckpoint model_1499.pt `
  -RunName g008_command_finetune_g006_s42_e1024_i300

# S1 물성 실제값 probe
& "$HOME\IsaacLab\_isaac_sim\python.bat" .\scripts\probe_g008_dynamics.py `
  --task Isaac-G008-Velocity-Rough-Go2-Friction-S1-v0 --seed 42 --num-envs 1024 `
  --output .\reports\runs\g008_friction_s1_runtime_probe.json --headless --device cuda:0
& "$HOME\IsaacLab\_isaac_sim\python.bat" .\scripts\probe_g008_dynamics.py `
  --task Isaac-G008-Velocity-Rough-Go2-LegMass-S1-v0 --seed 42 --num-envs 1024 `
  --output .\reports\runs\g008_leg_mass_s1_runtime_probe.json --headless --device cuda:0

# 네 방향 고정 평가
& "$HOME\IsaacLab\_isaac_sim\python.bat" .\scripts\evaluate_g008_directions.py `
  --checkpoint "$HOME\IsaacLab\logs\rsl_rl\unitree_go2_rough\<run>\model_*.pt" `
  --task Isaac-G008-Velocity-Rough-Go2-CommandSuite-v0 `
  --domain-mode nominal --terrain-mode plane `
  --num-envs 64 --horizon-steps 250 --warmup-steps 50 `
  --output .\reports\runs\g008_directional_qualification.json --headless --device cuda:0

# 원본 MP4는 저장소 밖에 기록
& "$HOME\IsaacLab\_isaac_sim\python.bat" .\scripts\record_g008_directions.py `
  --checkpoint "$HOME\IsaacLab\logs\rsl_rl\unitree_go2_rough\<run>\model_*.pt" `
  --output-dir "$HOME\IsaacLab\logs\visual_evidence\g008" `
  --output-name g008_directions_s42.mp4 `
  --report .\reports\runs\g008_direction_visual_evidence.json `
  --headless --device cuda:0
```

## 증거 파일

- 설계 수치의 단일 원본: `configs/g008_locomotion_dynamics.json`
- command sampler: `src/isaac_walk_g008/commands.py`
- friction/mass 환경: `src/isaac_walk_g008/env_cfg.py`
- 7개 task 등록: `src/isaac_walk_g008/registry.py`
- 학습 wrapper: `scripts/run_g008_stage.ps1`
- 네 방향 평가기: `scripts/evaluate_g008_directions.py`
- runtime 물성 probe: `scripts/probe_g008_dynamics.py`
- 네 방향 로컬 영상 recorder: `scripts/record_g008_directions.py`
- 세 정책 격리 촬영기와 FFmpeg 합성기: `scripts/record_g008_policy_comparison.py`, `scripts/build_g008_comparison_media.py`
- resume 검증기와 회귀 테스트: `scripts/revalidate_g008_resume_report.ps1`, `tests/test_g008_resume_revalidation.py`
- 스모크 실행 보고서: `reports/runs/g008_*_smoke_e64_i1_s42.json`
- G006 checkpoint 방향 평가: `reports/runs/g008_directional_qualification_g006_s42.json`
- scratch command 학습·평가: `reports/runs/g008_command_qualification_e1024_i300_s42.json`, `reports/runs/g008_directional_qualification_command_s42_plane.json`
- warm-start command 학습: `reports/runs/g008_command_finetune_g006_s42_e1024_i300.json`
- warm-start 평면·rough 평가: `reports/runs/g008_directional_qualification_finetune_g006_s42_plane.json`, `reports/runs/g008_directional_qualification_finetune_g006_s42_rough.json`
- S1 물성 실제값: `reports/runs/g008_friction_s1_runtime_probe.json`, `reports/runs/g008_leg_mass_s1_runtime_probe.json`
- friction S1 학습: `reports/runs/g008_friction_s1_finetune_command_s42_e1024_i300.json`
- friction S1 randomized·nominal 평가: `reports/runs/g008_directional_qualification_friction_s1_s42_randomized_plane.json`, `reports/runs/g008_directional_qualification_friction_s1_s42_nominal_plane.json`
- leg-mass S1 학습: `reports/runs/g008_leg_mass_s1_finetune_command_s42_e1024_i300.json`
- leg-mass S1 randomized·nominal 평가: `reports/runs/g008_directional_qualification_leg_mass_s1_s42_randomized_plane.json`, `reports/runs/g008_directional_qualification_leg_mass_s1_s42_nominal_plane.json`
- 공개 GIF·접촉시트와 로컬 원본 영상 해시: `docs/G008_VISUAL_EVIDENCE.md`, `reports/runs/g008_direction_visual_evidence.json`
- 세 정책 비교 촬영 보고서: `reports/runs/g008_policy_command_capture.json`, `reports/runs/g008_policy_friction_s1_capture.json`, `reports/runs/g008_policy_leg_mass_s1_capture.json`
- 비교 MP4와 공개 파생물 해시: `reports/runs/g008_policy_comparison_visual_evidence.json`
- 공간 혼합 마찰·링크 그룹 질량 후속 해석: `docs/G008_PERIODIC_FRICTION_AND_LINK_MASS_LIMITS.md`
- 공간 혼합 마찰 스윕: `reports/runs/g008_periodic_friction_sweep_command_vs_friction_s1_e32_h500_s20260826.json`
- 링크 그룹 질량 민감도: `reports/runs/g008_link_mass_sensitivity_command_vs_leg_mass_s1_e800_h300_s20260826.json`
- 공간 혼합 마찰 단계 촬영: `reports/runs/g008_stage_periodic_friction_capture.json`, `reports/runs/g008_stage_periodic_friction_visual_evidence.json`
- 링크 그룹 질량 단계 촬영: `reports/runs/g008_stage_link_mass_*_capture.json`, `reports/runs/g008_stage_link_mass_visual_evidence.json`

저장소의 JSON에는 `%USERPROFILE%` 치환 경로와 SHA-256을 기록한다. 원시 checkpoint와 TensorBoard는 저장소 밖 `$HOME\IsaacLab\logs`에 둔다.

## 현재 해석의 경계

- 1-iteration smoke와 runtime probe 결과로 마찰·질량 강건성을 주장하지 않는다. probe는 설정값이 실제 물성에 들어갔는지만 확인한다.
- 기존 G006 checkpoint 방향 평가는 새 command distribution의 학습 전 기준선이다. warm-start 300-iteration 결과와 구분한다.
- scratch와 warm-start 비교는 seed 42 한 쌍이다. 초기화 효과의 통계적 결론이 아니라 다음 학습 경로를 고르는 진단이다.
- 5초 64환경 평가는 장시간 열·전력·마모를 포함하지 않는다.
- rough terrain의 순간 자세 최댓값은 지형 seed와 초기 배치에 민감하다. 다음 비교에서는 동일 seed와 지형 조건을 고정한다.
- S2·S3 태스크가 실행 가능하도록 등록돼 있다는 사실과 그 stage가 학습·검증됐다는 주장을 구분한다.

## 다음에 자세히 진행할 일

### 1. rough 자세 gate를 지형 기준으로 다시 정의한다

현재 rough 평가는 world frame의 roll/pitch 최댓값을 평면과 같은 `0.35 rad` 기준으로 자른다. 경사면을 정상적으로 따라가는 자세까지 불안정으로 셀 수 있다. 다음 평가에는 terrain type·level, 발 접촉점, base orientation 시계열을 함께 남기고, 발 접촉점으로 추정한 local support plane의 법선과 base up vector 사이 각도를 계산한다. world-frame 자세와 support-plane-relative 자세를 같이 봐야 전진·후진 실패가 실제 흔들림인지 경사를 따른 결과인지 구분할 수 있다.

자세 peak 전후 0.5초의 commanded velocity, achieved velocity, foot slip, contact impulse, torque와 power proxy도 묶어 저장한다. 한 프레임의 최댓값만으로 원인을 정하지 않는다. 경사 추종으로 판정되면 rough gate를 local plane 기준으로 고치고, 미끄럼이나 접촉 충격이면 command·마찰 조건을 유지한 채 reward와 terrain curriculum을 따로 진단한다.

### 2. S1을 seed 43·44에서 반복한다

현재 S1 결과는 seed 42 한 번이다. friction과 leg-mass를 각각 seed 43·44로 반복하고, 평면 randomized/nominal 평가와 rough local-plane 평가를 같은 프로토콜로 실행한다. checkpoint별 SHA-256, terrain level 변화, 방향별 RMSE, torque, power, 자세 peak를 seed별 표로 남긴다. 세 seed 중 하나라도 nominal guardrail을 잃으면 S2로 넘어가지 않는다.

2026-08-26에 추가한 공간 혼합 마찰과 그룹별 질량 민감도도 같은 원칙으로 seed를 늘린다. 마찰은 높은 계수부터 연속 통과한 하한, 더 낮은 계수의 개별 통과, native 종료로 판정하지 못한 case를 분리한다. 현재 friction S1은 전진·후진·좌회전만 완료 최저 `0.2/0.1`까지 연속 통과했고, 우회전 하한과 `0.1/0.05`는 확정하지 않았다. 질량은 scale 비교와 동일 kg 증분 비교를 따로 둔다. 상세 프로토콜과 첫 결과는 `G008_PERIODIC_FRICTION_AND_LINK_MASS_LIMITS.md`에 있다.

### 3. friction S2와 leg-mass S2를 계속 분리한다

S1과 rough gate가 통과하면 friction S2는 static `0.62~1.00`, dynamic `0.42~0.78`, leg-mass S2는 body별 `0.90~1.10`으로 넓힌다. 두 실험은 같은 command checkpoint 계열에서 갈라지고 서로의 checkpoint를 이어받지 않는다. `2,048 env × 600 iterations × seeds 42/43`을 사용하며, randomization 범위 이외의 normalized config diff가 0인지 다시 검사한다.

마찰에서는 slip distance와 yaw moment 부족을, 질량에서는 joint별 torque·power와 swing tracking을 더 자세히 본다. 질량 결과는 hip/thigh/calf/foot 그룹별 scale을 함께 저장해 원위부 질량과 근위부 질량의 효과를 사후 분석할 수 있게 한다.

### 4. S3 전에 실물 범위를 측정한다

S3는 논문 envelope를 그대로 실행하는 단계가 아니다. Go2 발 패드와 실제 바닥 조합의 static/dynamic friction, 장착물 포함 링크 질량과 COM, actuator 지연과 strength, IMU bias를 먼저 측정한다. 측정 범위가 S3보다 좁으면 실측 범위를 우선한다. COM 변화가 크면 현재 `recompute_inertia=True`만으로는 부족하므로 별도 asset variant나 COM randomization을 설계한다.

### 5. 마지막에 상호작용 실험을 연다

friction과 mass의 단일축 S2까지 통과한 뒤에만 두 축을 함께 바꾼다. 이때 단일축 두 조건과 결합 조건을 같은 seed·명령·지형 셀에서 비교한다. 결합 조건의 손실이 두 단일축 손실의 합보다 큰지 확인해 상호작용을 기술한다. 실물 전이 후보는 nominal 성능, randomized 성능, seed 분산, torque·power 비용을 함께 만족한 checkpoint로 제한한다.
