# MPC/WBC 수령 자료 분석과 isaac-walk-rl 반영안

- 수집·검토일: 2026-08-30
- 적용 범위: 사족보행 Centroidal MPC, 접촉 일정, foothold, swing 제어, 현재 Go2 PPO 실험과의 연결
- 상태: 원문 수집·PDF 시각 검토·프로젝트 반영 완료, MPC/WBC 코드 구현은 미실행

## 먼저 결론

수령한 자료는 사족보행 제어를 다음 두 경로로 나눠 이해하는 데 유용하다.

```text
Command
  |-> Reference + future contact schedule
  |      `-> Centroidal MPC -> stance-foot GRF -> frame transform -> J^T F -> joint torque
  |
  `-> Gait -> foothold -> swing trajectory -> Cartesian PD/feedforward -> J^T F -> joint torque
```

현재 저장소의 Go2 정책은 이 구조를 구현한 MPC/WBC가 아니다. RSL-RL PPO가 50 Hz에서 12개 관절의 위치 목표를 직접 출력한다. 따라서 이 자료를 현재 checkpoint나 G009 RECOVER에 곧바로 섞지 않는다. 이번 반영은 다음 세 가지로 제한한다.

1. MPC/WBC의 수학과 구현 경계를 프로젝트 문서에 고정한다.
2. 현재 PPO를 해석하는 read-only 진단 항목과 향후 model-based baseline 순서를 정한다.
3. 논문 사실, Notion의 QUATTRO 구현 설명, 이 저장소에 적용할 가설을 분리한다.

특히 G009 R0의 prone/supine/side self-righting은 몸통·비발 접촉을 적극 사용하는 문제다. 접촉 발의 Ground Reaction Force만으로 몸통을 제어하는 논문의 single-rigid-body MPC 가정과 직접 맞지 않는다. MPC/WBC 자료를 rev16 CPU/GPU 접촉력 차이의 즉시 수정안으로 사용하지 않는다.

## 출처와 보존 경계

### Notion 자료

Chrome에서 사용자가 연 원문을 직접 탐색했다. 최상위 페이지와 여섯 하위 페이지의 본문 끝, 제목 구조, 링크와 이미지·첨부 여부를 확인했다.

- [사족보행 로봇 MPC WBC 총정리](https://app.notion.com/p/MPC-WBC-3cace2d5e06080d197a5e1be5f2e7ffa)
- [Centroidal MPC 동역학 기초 - A·B·c 행렬, Skew, SO(3), Jacobian 연결](https://app.notion.com/p/Centroidal-MPC-A-B-c-Skew-SO-3-Jacobian-3cace2d5e06081a1b935d48be865b92a?pvs=25)
- [Centroidal MPC Prediction부터 OSQP까지 - M·S·d, Q/R, Constraint, Receding Horizon](https://app.notion.com/p/Centroidal-MPC-Prediction-OSQP-M-S-d-Q-R-Constraint-Receding-Horizon-3cace2d5e06081a1a9e6c85d98930ff3?pvs=25)
- [Centroidal MPC Reference Generator - Position·Velocity·Body Pose와 Terrain-adaptive 확장](https://app.notion.com/p/Centroidal-MPC-Reference-Generator-Position-Velocity-Body-Pose-Terrain-adaptive-3cace2d5e06081e29170c780ebfcff83?pvs=25)
- [Centroidal MPC Gait Schedule - Phase·Duty Factor·Trot Contact Prediction](https://app.notion.com/p/Centroidal-MPC-Gait-Schedule-Phase-Duty-Factor-Trot-Contact-Prediction-3cace2d5e06081429b17e225b77c7597?pvs=25)
- [Centroidal MPC Foothold Planner - Raibert Touchdown·Velocity Feedback·Terrain 확장](https://app.notion.com/p/Centroidal-MPC-Foothold-Planner-Raibert-Touchdown-Velocity-Feedback-Terrain-3cace2d5e0608179abd9c64298186a0e?pvs=25)
- [Centroidal MPC Swing Trajectory - Cubic Bézier·Apex·Velocity Continuity](https://app.notion.com/p/Centroidal-MPC-Swing-Trajectory-Cubic-B-zier-Apex-Velocity-Continuity-3cace2d5e0608165a4eaf3c4cbb88f93?pvs=25)

확인된 하위 본문 절 수는 각각 `19, 27, 21, 20, 20, 26`개다. 별도 이미지나 파일 첨부는 없었다. `gait.py`, `planner.py`, `trajectory.py`, `estimator.py` 표시는 파일 다운로드가 아니라 Notion이 `http://<이름>.py/`로 자동 링크한 잘못된 웹 링크다. 자료가 설명하는 실제 QUATTRO 소스는 수령하지 않았으므로 코드와 설명의 일치 여부는 아직 검증할 수 없다.

### PDF

로컬 수령 파일은 `%USERPROFILE%/Downloads/convex_mpc_2fix_260706_135518.pdf`다.

- 제목: *Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control*
- 저자: Jared Di Carlo, Patrick M. Wensing, Benjamin Katz, Gerardo Bledt, Sangbae Kim
- 분량: 8쪽
- 파일 크기: `5,037,596 bytes`
- SHA-256: `edbbc5415c6c7bef5c959ce290f218acdfbf2f315e2de665495f6d78fc76ab1d`

8쪽 모두 PNG로 렌더링해 식, 표, 제어 블록도, 결과 그래프와 손글씨·강조 표시를 시각 확인했다. PDF는 논문 본문 위에 강조와 메모가 합쳐진 수령본이다. 원본 바이너리는 저작권·재배포 범위를 확인하지 않았고 저장소에 중복 보관하지 않는다. 저장소에는 경로를 `%USERPROFILE%`로 치환한 출처 정보, 해시, 분석과 채택 판단만 남긴다.

기계 판독 가능한 수집 기록은 [`reports/research/mpc_wbc_material_intake_20260830.json`](../reports/research/mpc_wbc_material_intake_20260830.json)에 있다.

논문의 식 (1)~(33), Table I, Figure 2~10, 페이지별 근거와 Go2 이전 금지선은 [`MIT_CHEETAH3_CONVEX_MPC_PAPER_REVIEW_20260830.md`](MIT_CHEETAH3_CONVEX_MPC_PAPER_REVIEW_20260830.md)에 별도로 정리했다.

## Notion 여섯 문서의 핵심

### 1. 동역학 A·B·c, Skew, SO(3), Jacobian

Notion은 상태와 입력을 다음처럼 둔다.

\[
x=[p,\operatorname{rpy},v,\omega]\in\mathbb{R}^{12},
\qquad
u=[F_{FL},F_{FR},F_{BL},F_{BR}]\in\mathbb{R}^{12}
\]

한 스텝 모델은 다음 affine system이다.

\[
x_{k+1}=Ax_k+Bu_k+c
\]

- `A`: 위치에 선속도, 자세에 각속도가 전달되는 자연 진행
- `B`: 각 발의 힘이 위치·자세·선속도·각속도에 미치는 영향
- `c`: 위치와 선속도에 누적되는 중력

발 `i`의 힘은 병진 가속도 `F_i/m`와 회전 가속도 `I^{-1}(r_i×F_i)`를 동시에 만든다. 외적을 행렬곱으로 바꾸는 `[r_i]_×`가 B 행렬의 회전 블록에 들어간다. 이후 `R^T R=I`, `R^T R_dot`의 skew 성질, `so(3)`, twist·screw, `V=J(q)q_dot`, virtual work에서의 `tau=J^T F`까지 연결한다.

학습용 장점은 12×12 B 전체를 외우지 않고 발 하나의 12×3 영향 블록을 네 개 이어 붙이는 방식으로 설명한다는 점이다.

주의할 점은 `rpy_{k+1}≈rpy_k+omega_k dt`가 원 논문의 yaw-dependent mapping보다 더 강한 근사라는 점이다. 좌표계와 Euler angle 순서를 확인하지 않고 그대로 구현하면 안 된다.

### 2. Prediction, Q/R, 제약, QP, receding horizon

한 스텝 모델을 horizon 전체로 쌓으면 다음 condensed prediction이 된다.

\[
X=Mx_0+SU+d
\]

- `M=[A; A^2; ...; A^N]`: 현재 상태의 미래 전파
- `S`: 과거 입력일수록 더 많은 A를 거치는 block-lower-triangular 행렬
- `d`: 매 스텝 들어오는 중력 affine 항의 누적

추적과 힘 사용량을 함께 벌점화한다.

\[
J=(X-X_{ref})^T\bar Q(X-X_{ref})+U^T\bar R U
\]

`e=Mx_0+d-X_ref`를 두면, OSQP 표준형의 한 관례는 다음과 같다.

\[
P=2(S^T\bar Q S+\bar R),
\qquad
q=2S^T\bar Q e
\]

제약은 수직력 범위, 선형화한 마찰 피라미드, swing 발 force 0을 포함한다. 미래 force sequence 전체를 풀되 첫 입력만 적용하고 새 상태를 읽어 다시 푸는 receding horizon, 이전 해를 한 칸 이동하는 warm start, 실패 시 stance 발들이 `mg/n`을 나눠 받는 fallback도 설명한다.

Notion의 QUATTRO 예시는 `N=12`, `dt=0.02s`, `X∈R^144`, `U∈R^144`, 약 `0.24s` horizon이다. Q 예시는 상태 순서 `[p,rpy,v,omega]`에 대해 `diag(2,2,180,120,120,25,8,8,18,5,5,3)`이다. 이 값은 현재 Go2 PPO나 원 논문의 공통 상수가 아니므로 복사하지 않는다.

### 3. Reference Generator

평지 reference는 현재 상태와 command에서 0.02~0.24초 미래를 만든다.

\[
\begin{aligned}
p_{x,ref}(t)&=p_{x,0}+v_{x,cmd}t \\
p_{y,ref}(t)&=p_{y,0}+v_{y,cmd}t \\
p_{z,ref}(t)&=h_{nominal} \\
\operatorname{roll}_{ref}&=0,
\quad \operatorname{pitch}_{ref}=0 \\
\operatorname{yaw}_{ref}(t)&=\operatorname{yaw}_0+\dot\psi_{cmd}t \\
v_{ref}&=[v_{x,cmd},v_{y,cmd},0] \\
\omega_{ref}&=[0,0,\dot\psi_{cmd}]
\end{aligned}
\]

position reference는 장기 drift를 제한하고 velocity reference는 원하는 운동 속도를 직접 정한다. `Reference=목표값`, `Q=목표 추종 중요도`로 역할을 구분한다.

지형 적응 확장은 solver 전체 교체보다 다음 입력을 바꾸는 방식으로 제시한다.

- terrain height를 반영한 CoM height
- support plane normal을 반영한 roll/pitch
- 경사·계단 phase에 따른 CoM와 body pose
- gait, foothold, Q/R, force constraint의 terrain-dependent override

이 방향은 현재 [`support_plane.py`](../src/isaac_walk_g009/support_plane.py)의 normal·접평면·COM 계산과 개념적으로 맞는다. 다만 현재 코드는 PPO reward/observation/evaluation을 위한 모듈이며 MPC reference generator는 아니다.

### 4. Gait Schedule

Notion의 QUATTRO trot 예시는 다음 값을 사용한다.

- step frequency: `1.4 Hz`
- stride period: 약 `0.714s`
- duty factor: `0.65`
- stance: 약 `0.464s`
- swing: `0.25s`
- phase offset `[FL,FR,BL,BR]=[0.5,0,0,0.5]`

각 발 phase와 contact는 다음처럼 계산한다.

\[
\phi_i=(\phi_{global}+\phi_{offset,i})\bmod 1,
\qquad
\phi_i<D\Rightarrow stance
\]

`D>0.5`라서 두 대각선 pair가 동시에 닿는 four-foot overlap 구간이 생긴다. 문서는 이를 공격적 trot보다 안정적인 초기 실기용 설정으로 해석한다.

MPC에는 현재 contact만이 아니라 horizon의 future contact schedule이 들어간다. 각 미래 timestep에서 swing 발의 GRF를 0으로 고정하므로 MPC는 다음 support 전환을 미리 고려한다. lift-off는 `1→0`, touchdown은 `0→1` transition으로 감지하며 foothold planner의 시작·종료 위치 기록에 사용한다.

이 `1.4/0.65` 값은 현재 Isaac Lab Go2 policy에서 읽은 값이 아니다. G009 설정이나 PPO gait가 이 주기를 사용한다고 기록하지 않는다.

### 5. Foothold Planner

Notion은 touchdown xy를 다음 구조로 설명한다.

\[
p_{td,xy}=p_{nominal,xy}
+\frac{1}{2}T_{stance}v_{ref,xy}
+k_v(v_{xy}-v_{ref,xy})
\]

- nominal: yaw로 회전한 hip 기준 기본 workspace
- feedforward: stance 중 몸통 이동의 절반만큼 미리 디디기
- feedback: 실제 속도 오차를 다음 foothold에 반영

예를 들어 `v_ref=0.2m/s`, `T_stance=0.4s`면 feedforward는 `4cm`다. stance→swing에서 실제 lift-off 위치를 저장하고 swing→stance에서 실제 touchdown 위치를 다시 저장해 계획 오차가 다음 cycle에 누적되지 않게 한다.

평지 planner의 z 고정은 계단·rough terrain에 부족하다. 문서는 다음 확장을 제안한다.

\[
p_{td}=p_{Raibert}+\Delta p_{terrain}
\]

또는 model-based nominal을 유지하고 RL이 `Delta p_RL`만 출력하는 residual foothold 구조를 제안한다. 이는 direct torque RL보다 action dimension과 해석 위험을 줄일 수 있는 향후 후보지만, terrain map·reachability·안전 영역 정의가 먼저 필요하다.

### 6. Swing Trajectory

Notion 구현은 실제 lift-off와 계획 touchdown 사이에 apex를 둔다.

\[
p_{apex,xy}=\frac{p_{lo,xy}+p_{td,xy}}{2},
\qquad
z_{apex}=\max(z_{lo},z_{td})+h_{clearance}
\]

설명된 QUATTRO 기본 clearance는 `0.035m`다. 전체 swing을 두 개의 cubic Bézier로 나눈다.

- 전반: lift-off → apex, `P0=P1=p_lo`
- 후반: apex → touchdown, `P2=P3=p_td`

3차 Bézier의 endpoint derivative에서 `P0=P1`이면 lift-off 목표 속도가 0이고, `P2=P3`이면 touchdown 목표 속도가 0이다. apex에서는 두 segment의 z derivative가 0이고 xy derivative가 이어지도록 control point를 둔다.

전체 swing progress `s`를 각 segment의 `u`로 바꾸면 `du/dt=2/T_swing`이고 다음 속도 reference가 나온다.

\[
v_{des}=\frac{dp}{du}\frac{2}{T_{swing}}
\]

이 위치·속도는 Cartesian PD `F_task=K_p(p_des-p)+K_d(v_des-v)`를 거쳐 `tau=J^T F_task`로 변환된다.

## PDF 원 논문에서 확인한 기준

### 제어 구조와 주기

논문의 Figure 2와 Results는 계층별 주기를 분리한다.

- MPC/reference 상위 블록: 논문 초록 `20~30 Hz`, Figure 2 `30 Hz`, 실제 gait별 prediction `25~50 Hz`
- state estimation, swing planning, leg impedance: `1 kHz`
- 가장 낮은 torque/force control 블록: Figure 2 `4.5 kHz`
- prediction horizon: `0.33~0.5s`, gait에 따라 `10~16` timestep
- QP solver: `qpOASES`, typical solve가 `1ms` 미만

단일 숫자를 보편값으로 고정하기보다 블록과 gait별 실행 주기를 함께 기록해야 한다.

### 모델과 근사

논문은 로봇을 contact patch force를 받는 single rigid body로 모델링한다. 다리 질량은 Cheetah 3 전체 질량의 약 10%라서 다물체 효과를 생략할 수 있다고 설명한다.

원 논문의 상태는 중력 상태를 추가한 13차원 형태이며, 설명 순서는 대략 `[Theta, p, omega, p_dot, g]`다. 동역학은 yaw와 미래 foothold에 의존하는 linear time-varying system이다.

주요 근사는 다음과 같다.

- Z-Y-X Euler angle 사용
- 작은 roll/pitch에서 `Theta_dot≈R_z(yaw)^T omega`
- `omega×(I omega)` gyroscopic 항 생략
- world inertia를 yaw 회전만으로 근사
- 미래 yaw와 foothold를 reference/footstep planner에서 받아 `B_c[n]` 계산
- 현재 첫 timestep은 실제 상태로 다시 계산해 즉시 근사를 정확하게 유지

논문의 결론은 horizon 전체의 매우 정밀한 모델보다 자주 갱신되는 순간 동역학의 정확도가 더 중요할 수 있다는 것이다. 이는 큰 외란 뒤에도 최대 약 40 ms 안에 새 reference와 동역학을 계산하는 receding-horizon 구조에 의존한다.

### force constraint와 QP

stance 발 하나에 대해 논문이 명시한 부등식은 다음 여섯 개다.

\[
f_{min}\le f_z\le f_{max},
\quad
-\mu f_z\le f_x\le\mu f_z,
\quad
-\mu f_z\le f_y\le\mu f_z
\]

공중 발 force는 별도 equality `D_i u_i=0`으로 고정한다. 논문 Table I의 예시는 `mu=0.6`, `f_min=10N`, `f_max=666N`, force weight `1e-6`이다. 이 값은 43 kg으로 모델링한 Cheetah 3의 실험값이며 15.019 kg Go2에 직접 적용하지 않는다.

논문은 상태 변수를 QP에서 제거한 condensed formulation을 사용한다.

\[
X=A_{qp}x_0+B_{qp}U
\]

\[
J(U)=\lVert A_{qp}x_0+B_{qp}U-x_{ref}\rVert_L
+\lVert U\rVert_K
\]

\[
H=2(B_{qp}^T L B_{qp}+K),
\qquad
g=2B_{qp}^T L(A_{qp}x_0-y)
\]

첫 `3n`개 force만 실제 joint torque 계산에 사용한다.

### stance와 swing

stance에서는 world GRF를 body frame으로 회전하고 Jacobian transpose로 joint torque를 만든다.

\[
\tau_i=J_i^T R_i^T f_i
\]

swing에서는 위치·속도 feedback과 operational-space feedforward를 합친다. 논문은 apparent mass에 따라 `K_p`를 조정해 폐루프 natural frequency가 다리 자세에 따라 크게 바뀌지 않도록 한다.

논문에 나온 foothold 휴리스틱은 다음과 같다.

\[
p^{des}=p^{ref}+v^{CoM}\Delta t/2
\]

Notion의 `k_v(v-v_ref)` feedback 항과 3.5 cm의 two-Bézier swing은 이 식에 명시되지 않은 QUATTRO 구현 확장이다.

### 실험 결과와 한계

논문이 보고한 Cheetah 3 결과는 다음 범위다.

- trot: 약 `1.2m/s`
- flying trot: `1.7m/s`
- bound: 시험 `2.5m/s`
- gallop: 최대 `3m/s`
- lateral: 최대 `1m/s`
- yaw: 최대 `180deg/s`
- pronk: 약 `15cm`, `150ms` stance와 `350ms` flight
- 계단: swing 높이만 늘리고 지형 사전 정보 없이 debris가 있는 네 계단을 flying trot으로 시도

고속 한계는 MPC solve보다 다리 swing 속도와 actuator limit에서 나타났다. 3 m/s 이상에서는 발을 충분히 앞으로 회수하지 못해 자세·yaw 제어가 나빠졌다. 모델 기반 상위 제어 성능과 actuator·swing tracking 한계를 분리해 봐야 한다.

## Notion과 논문 사이에서 보정한 항목

| 주제 | Notion/QUATTRO 설명 | 원 논문 | 이 저장소의 처리 |
| --- | --- | --- | --- |
| 상태 | 12차원 `[p,rpy,v,omega]` + affine gravity `c` | 13차원, gravity state 포함 | 두 표현을 동치로 단정하지 않고 별도 수식 테스트 |
| 자세 진행 | `rpy_next≈rpy+omega dt` | 작은 roll/pitch에서 yaw-dependent `R_z^T omega` | yaw가 있을 때 오차를 수치 검증하기 전 채택 금지 |
| solver | OSQP | qpOASES | solver 선택은 별도 dependency/성능 결정으로 분리 |
| horizon | `12×0.02=0.24s` | `0.33~0.5s`, 10~16 step | robot·gait·solve budget별 측정값으로 결정 |
| force 제약 | 발당 7 row로 설명 | stance 발당 6 inequality + swing equality | 실제 constraint builder 행 수를 테스트로 증명 |
| gait | `1.4Hz`, duty `0.65` | gait별 25~50 Hz MPC, gait period도 다름 | QUATTRO 값은 Go2 설정으로 복사하지 않음 |
| foothold | Raibert + velocity-error feedback | `p_ref+v_CoM Delta t/2` | feedback 부호·frame을 별도 A/B 시험으로 검증 |
| swing | 2 cubic Bézier, 3.5 cm | 구체 control point/3.5 cm 미명시 | QUATTRO 구현 설명으로만 보존 |
| WBC | `J^T F`와 swing Cartesian PD 중심 | 별도 full-body hierarchical QP는 아님 | 이 자료만으로 full WBC 구현을 완료했다고 부르지 않음 |

제목에 WBC가 들어가지만 수령 자료의 중심은 Centroidal MPC, stance GRF→torque, swing Cartesian control이다. floating-base dynamics와 모든 joint/contact task를 한 QP에서 우선순위·제약과 함께 푸는 완전한 WBC 정식화는 별도 자료가 필요하다.

## 현재 isaac-walk-rl에 반영하는 방법

### 지금 채택

1. **용어와 좌표계 계약**
   - world/body/support-plane frame을 결과 JSON에 명시한다.
   - GRF 또는 접선력은 어느 frame 값인지 기록한다.
   - 발 순서 `FL/FR/BL/BR`를 코드·리포트에서 고정한다.

2. **read-only 보행 진단 후보**
   - 미래 contact schedule과 실제 contact transition 차이
   - 발별 `|F_x|/(mu F_z)`, `|F_y|/(mu F_z)` 또는 원형 cone ratio
   - stance/swing별 force leakage
   - nominal/Raibert foothold와 실제 touchdown 오차
   - swing clearance, early/late contact, touchdown 속도
   - `J^T F` torque proxy와 PPO action/실제 torque의 상관

3. **경사 WALK 설계에 쓰는 개념**
   - world-horizontal reference 대신 support-plane 기준 body reference
   - terrain normal을 critic/evaluation에만 두는 현재 privilege 경계 유지
   - terrain-dependent reference/foothold residual은 한 축씩 분리
   - contact 2점 trot을 정적 polygon 실패로 보지 않는 기존 규칙 유지

### 지금 채택하지 않음

- 현재 PPO checkpoint에 MPC force를 섞는 변경
- G009 RECOVER reward·action·solver calibration을 MPC 수치로 교체
- Cheetah 3의 `mu`, force limit, mass, horizon, gait frequency를 Go2에 복사
- 새 QP solver dependency 추가
- blind flat foothold planner를 산 비탈·계단 성능으로 주장
- Notion의 QUATTRO 설명을 실제 소스 검증 없이 재현 완료로 표기

## 단계별 학습·구현 로드맵

### M0. 자료 intake와 차이 고정 - 완료

- Notion 여섯 페이지 끝까지 수집
- PDF 8쪽 텍스트·시각 검토
- 출처 URL, PDF hash, 첨부 상태 기록
- 논문 사실과 QUATTRO 설명 차이 문서화

### M1. import-light 수학 검증 - 다음 후보

Isaac Sim 없이 NumPy만 사용하는 최소 모듈과 테스트를 만든다.

1. `[r]_x F == r×F`
2. 12-state affine와 13-state gravity-augmented 표현의 한 스텝 비교
3. yaw-dependent orientation mapping과 단순 `rpy+omega dt` 오차
4. horizon 3의 M/S/d를 직접 전개한 값과 builder 비교
5. QP Hessian 대칭·positive semidefinite와 dimension
6. stance/swing constraint의 행 수와 force feasibility

DoD는 수식별 실패 테스트, dimension table, 고정 seed 수치 결과다. 이 단계에서는 solver를 호출하지 않는다.

### M2. PPO read-only telemetry baseline

현재 G008/G009 WALK 정책의 action을 바꾸지 않고 같은 replay에서 다음을 측정한다.

- contact phase와 transition
- 발별 GRF와 friction utilization
- support-plane reference 오차
- 실제 touchdown과 Raibert nominal의 차이
- swing clearance·touchdown 속도
- joint torque/velocity와 `J^T F` proxy

동일 checkpoint·seed·terrain·command로 telemetry on/off의 trajectory hash 또는 핵심 state가 같아야 한다. 관찰 코드가 정책 동작을 바꾸면 실패다.

### M3. 별도 flat Centroidal MPC baseline

현재 PPO를 교체하지 않고 별도 task/registry로 만든다. solver 선택과 dependency는 공식 문서·Isaac 번들 가용성·license·Windows 성능을 조사한 뒤 결정한다.

게이트 순서:

1. standing force balance
2. four-foot stance의 small velocity command
3. 고정 trot contact schedule
4. receding horizon과 warm start
5. fail-safe force distribution
6. 1환경 시각 증거 뒤 64환경 정량 평가

PPO와 MPC는 같은 command·terrain·seed·평가 지표를 사용하되 action semantics와 제어 주기가 다르므로 학습 reward를 직접 비교하지 않는다.

### M4. terrain reference/foothold 확장

M3 flat baseline이 통과한 뒤 한 번에 한 축만 연다.

1. support-plane body pose reference
2. terrain height 기반 touchdown z
3. reachable/safe foothold xy
4. early/late contact adaptation
5. 경사별 force constraint와 friction margin

각 단계는 flat nominal guardrail을 통과해야 다음 단계로 간다.

### M5. residual RL 후보

model-based baseline이 안정된 뒤에만 다음 중 한 action space를 선택한다.

- `Delta p_foothold`
- body height/roll/pitch reference residual
- Q/R 또는 gait parameter residual

direct torque residual과 동시에 열지 않는다. residual clip, rate limit, fallback, privileged observation 경계를 먼저 고정한다. 현재 RSL-RL PPO의 joint-position action과 같은 checkpoint lineage를 공유하지 않는다.

## 사용자의 복습 순서

1. 종이에 horizon 3의 `x1,x2,x3`를 전개해 `M,S,d`를 복원한다.
2. 같은 상태를 12-state affine와 13-state augmented 형태로 써서 gravity가 어디에 들어가는지 비교한다.
3. yaw가 0이 아닐 때 `Theta_dot=R_z^T omega`와 `Theta_dot=omega`의 차이를 수치로 계산한다.
4. 발 하나의 `r×F`, `[r]_x F`, `J^T F`를 임의 벡터로 검산한다.
5. `mu=0.6`에서 주어진 `Fz`로 허용되는 `Fx,Fy` 범위를 손으로 계산한다.
6. `f=1.4Hz`, `D=0.65`의 contact schedule을 horizon 표로 직접 만든다.
7. `0.5 T_stance v` foothold가 속도와 stance time에 따라 몇 cm 변하는지 계산한다.
8. two-Bézier control point를 놓고 lift-off/apex/touchdown의 derivative를 확인한다.
9. 마지막에만 현재 저장소의 `support_plane.py`, contact telemetry, reward/evaluation 계약과 연결한다.

## 완료 주장과 남은 위험

이번 작업으로 자료 수집과 프로젝트 반영은 완료했다. 다음은 아직 완료가 아니다.

- QUATTRO 원본 코드 확보·대조
- full WBC 정식화 조사
- QP solver 선택·설치
- MPC/WBC 코드 구현과 Isaac runtime 실행
- Go2 force/torque limit 식별
- PPO 대비 정량 비교
- 실기체·sim-to-real 검증

따라서 이 문서는 `MPC/WBC 구현 완료`가 아니라 `수령 자료를 검증 가능한 후속 작업으로 변환한 기준 문서`다.
