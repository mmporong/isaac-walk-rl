# MIT Cheetah 3 Convex MPC 논문 검토와 Go2 적용 경계

- 검토일: 2026-08-30
- 대상: *Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control*
- 저자: Jared Di Carlo, Patrick M. Wensing, Benjamin Katz, Gerardo Bledt, Sangbae Kim
- 수령본: 8쪽, 강조·메모가 병합된 PDF
- 목적: 논문의 주장을 재현 가능한 수학·제어 계약으로 분해하고 현재 `isaac-walk-rl`에 채택할 부분과 보류할 부분을 구분한다.

## 결론부터

이 논문은 완전한 다물체 WBC를 제안한 논문이 아니다. 작은 roll/pitch를 전제로 한 single-rigid-body 모델로 미래 접촉 발의 Ground Reaction Force(GRF)를 푼 뒤 stance 발에는 `J^T R^T f`, swing 발에는 별도의 Cartesian impedance와 operational-space feedforward를 적용한 계층형 제어기다.

현재 저장소의 Go2 정책은 50 Hz에서 12개 관절 위치 목표를 출력하는 RSL-RL PPO다. 논문의 전제인 torque-controlled joint, 1 kHz state/swing/impedance loop, 25~50 Hz MPC, 정해진 미래 contact schedule과 직접 연결되지 않는다. 따라서 논문 수치나 force 출력을 현재 PPO action에 섞지 않는다.

이 저장소에서 논문을 사용하는 순서는 다음과 같다.

1. Isaac Sim과 무관한 수학 테스트로 13-state LTV, condensed prediction, constraint를 검증한다.
2. 현재 PPO를 바꾸지 않는 read-only contact/GRF/foothold telemetry를 만든다.
3. torque-control semantics와 solver 성능을 별도 결정한 뒤 flat Centroidal MPC baseline을 만든다.
4. flat baseline이 통과한 뒤 support-plane reference, terrain foothold와 residual RL을 한 축씩 연다.

## 출처 무결성과 해석 규칙

수령 PDF의 식별값은 다음과 같다.

- 페이지: `8`
- 파일 크기: `5,037,596 bytes`
- SHA-256: `edbbc5415c6c7bef5c959ce290f218acdfbf2f315e2de665495f6d78fc76ab1d`
- 암호화: 없음
- 폼·JavaScript: 없음

PDF의 강조와 필기 표시는 학습자의 주석으로 취급한다. 저자의 본문·식·표·그림 캡션과 같은 권위를 부여하지 않는다. 원본 바이너리는 재배포 범위를 확인하지 않았으므로 저장소에 복사하지 않고 portable path와 hash만 [`mpc_wbc_material_intake_20260830.json`](../reports/research/mpc_wbc_material_intake_20260830.json)에 둔다.

## 페이지별 검토 지도

| PDF 쪽 | 주요 내용 | 저장소에서 확인할 계약 |
| --- | --- | --- |
| 1 | 초록, 동적 보행 문제, convex MPC의 기여와 보고 속도 | `20~30 Hz`, `<1 ms`, 최대 `0.5 s`는 Cheetah 3 결과이며 Go2 보장값이 아님 |
| 2 | Figure 2 제어 계층, Figure 3 좌표계, 로봇·state machine, swing feedback 식 (1) | world/body frame, MPC 30 Hz·state/swing 1 kHz·하위 torque/force 4.5 kHz 분리 |
| 3 | swing feedforward 식 (2), apparent-mass gain 식 (3), stance torque 식 (4), rigid-body 식 (5)~(13) | `J^T R^T f`, 작은 roll/pitch, Euler 순서, gyroscopic 항 생략 |
| 4 | yaw-only inertia 식 (14)~(15), 13-state LTV 식 (16)~(17), MPC 식 (18)~(21), force constraint 식 (22)~(24) | yaw·foothold 의존 `B`, 발당 6 inequality, swing force equality |
| 5 | ZOH 식 (25)~(26), condensed QP 식 (27)~(32), Table I, 실험 설정, foothold 식 (33) | `H,g` 차원, model mass 43 kg, horizon 10~16, `0.33~0.5 s`, 25~50 Hz |
| 6 | qpOASES 구현, solve-time Figure 4, trot·kick·stairs·pronk 결과 | solver와 하드웨어 결과 분리, 계단 3회 결과의 실패도 보존 |
| 7 | Figure 7~10, bound·gallop, 고속 한계, 결론 시작 | 최대 3 m/s의 제한이 solver보다 swing/actuator 속도에 있었음 |
| 8 | 결론·향후 연구와 참고문헌 | fixed manual gait를 higher-level contact planner로 대체하는 것은 미래 과제 |

## 제어 계층과 주기

Figure 2는 모든 블록이 같은 속도로 실행되는 제어기가 아님을 보여준다.

```text
operator command + contact sequence
                |
                v
reference trajectory ----> convex MPC ----> world GRF
       |                                      |
       |                                      v
       |                               body-frame force
       |                                      |
       v                                      v
swing trajectory ----------------------> J^T mapping
                \                           /
                 \                         /
                  joint torque/force control
```

- MPC와 reference 상위 블록: Figure 2에서 `30 Hz`
- 초록의 표현: `20~30 Hz`
- 실제 gait별 prediction: `25~50 Hz`
- state estimation, swing planning, leg impedance: `1 kHz`
- 가장 낮은 torque/force control: Figure 2에서 `4.5 kHz`

서로 다른 문맥의 주기를 하나의 고정 숫자로 합치지 않는다. 논문의 MPC 출력을 필터 없이 joint torque 식에 사용했다는 사실도 1 kHz·4.5 kHz 하위 계층과 함께 읽어야 한다.

## 모델의 정확한 범위

### Single rigid body 전제

로봇 몸통을 접촉점에서 힘을 받는 하나의 강체로 본다. 논문은 Cheetah 3 다리 질량이 전체의 약 10%라 이 근사가 가능하다고 설명한다. 이 비율과 타당성은 Go2에 자동으로 이전되지 않는다.

각 접촉 발 `i`에 대해 CoM에서 접촉점까지의 벡터를 `r_i`, GRF를 `f_i`라 두면 다음 관계를 사용한다.

\[
\ddot p=\frac{1}{m}\sum_i f_i-g
\]

\[
\frac{d}{dt}(I\omega)=\sum_i r_i\times f_i
\]

\[
\dot R=[\omega]_\times R
\]

이 모델은 몸통·무릎·허벅지 접촉을 사용하는 self-righting과 맞지 않는다. 따라서 G009 RECOVER의 비발 접촉력 및 CPU/GPU force divergence를 직접 해결하는 모델로 사용하지 않는다.

### 자세와 각운동량 근사

Z-Y-X Euler angle `Theta=[roll,pitch,yaw]`를 쓰고 작은 roll/pitch에서 다음처럼 근사한다.

\[
\dot\Theta\approx R_z(\psi)^T\omega
\]

단순한 `rpy_next=rpy+omega*dt`와 다르다. yaw가 0이 아닐 때 두 식의 오차를 테스트하기 전에는 Notion의 단순식을 논문 구현으로 부르지 않는다.

또한 다음 gyroscopic 항을 버린다.

\[
\frac{d}{dt}(I\omega)=I\dot\omega+\omega\times(I\omega)
\approx I\dot\omega
\]

world inertia도 roll/pitch를 버리고 yaw 회전만 사용한다.

\[
\hat I=R_z(\psi)\,{}^B I\,R_z(\psi)^T
\]

작은 roll/pitch 근사는 정상 보행 근처의 MPC에는 쓸 수 있지만 prone·supine·side 자세의 self-righting에는 적용할 수 없다.

### 13-state LTV

중력을 상태에 포함해 다음 형태로 만든다.

\[
\dot x=A_c(\psi)x+B_c(r_1,\ldots,r_n,\psi)u
\]

- 상태 차원: `13`
- 발 하나당 입력: `3`
- `A_c`: yaw에 의존
- `B_c`: yaw와 미래 foothold에 의존

미래 horizon의 `B_c[n]`는 reference yaw와 footstep planner의 `r_i`로 계산한다. 첫 timestep은 실제 현재 상태로 다시 계산한다. 논문이 큰 외란 뒤에도 작동하는 핵심은 먼 미래 모델의 정밀함보다 현재 timestep을 자주 다시 맞추는 receding-horizon 구조에 있다.

## MPC와 condensed QP

### 원래 finite-horizon 문제

논문은 tracking error와 force effort를 최소화하면서 dynamics와 force constraint를 만족시킨다.

\[
\min_{x,u}\sum_{i=0}^{k-1}
\lVert x_{i+1}-x_{i+1,ref}\rVert_{Q_i}
+\lVert u_i\rVert_{R_i}
\]

subject to:

\[
x_{i+1}=A_i x_i+B_i u_i
\]

\[
c_i\le C_i u_i\le \bar c_i
\]

\[
D_i u_i=0
\]

`D_i u_i=0`은 미래 contact schedule에서 swing으로 지정된 발의 force를 0으로 만든다. 즉 contact sequence 자체를 MPC가 최적화하는 논문은 아니다.

### Force constraint

stance 발 한 개에 명시된 inequality는 여섯 개다.

\[
f_{min}\le f_z\le f_{max}
\]

\[
-\mu f_z\le f_x\le\mu f_z
\]

\[
-\mu f_z\le f_y\le\mu f_z
\]

이는 원형 friction cone이 아니라 square pyramid 근사다. swing 발은 위 여섯 부등식에 포함시키는 대신 별도 equality로 0 force를 강제한다. 따라서 Notion의 “발당 7 row” 설명과 동일하다고 기록하지 않는다.

### Condensing

상태 trajectory를 최적화 변수에서 제거한다.

\[
X=A_{qp}x_0+B_{qp}U
\]

\[
J(U)=\lVert A_{qp}x_0+B_{qp}U-x_{ref}\rVert_L
+\lVert U\rVert_K
\]

QP 표준형은 다음과 같다.

\[
\min_U \frac{1}{2}U^T H U+U^T g
\]

\[
H=2(B_{qp}^T L B_{qp}+K)
\]

\[
g=2B_{qp}^T L(A_{qp}x_0-y)
\]

`H`와 `g`의 크기는 상태 수가 아니라 발 수와 horizon에 따라 `3nk`로 정해진다. 구현 테스트에서는 다음을 증명해야 한다.

1. horizon을 직접 전개한 `X`와 condensed matrix 결과가 같다.
2. `H`가 대칭이고 수치 오차 범위에서 positive semidefinite다.
3. stance/swing schedule에 따라 variable·constraint 수가 정확히 바뀐다.
4. 최적 force sequence의 첫 `3n`개만 실제 제어에 전달된다.

## Swing과 stance 제어

### Stance

MPC가 world frame에서 계산한 GRF를 body frame으로 바꾸고 Jacobian transpose로 torque를 만든다.

\[
\tau_i=J_i^T R_i^T f_i
\]

frame convention, 발 순서, Jacobian 정의가 다르면 부호가 맞아 보여도 torque 방향이 틀릴 수 있다. 수치 테스트에서 virtual work와 finite difference를 함께 확인해야 한다.

### Swing

논문의 swing torque는 위치·속도 feedback과 operational-space feedforward의 합이다.

\[
\tau_i=J_i^T[K_p(p_{ref}-p)+K_d(v_{ref}-v)]+\tau_{i,ff}
\]

\[
\tau_{i,ff}=J_i^T\Lambda_i(a_{ref}-\dot J_i\dot q_i)+C_i\dot q_i+G_i
\]

leg configuration에 따라 apparent mass가 바뀌므로 일정한 natural frequency를 유지하도록 `K_p`를 조정한다.

\[
K_{p,i}=\omega_i^2\Lambda_{i,i}
\]

논문은 Notion의 two-cubic-Bézier control point나 `3.5 cm` clearance를 명시하지 않는다. 그 값은 QUATTRO 구현 설명으로 분리한다.

### Foothold

논문이 제시한 xy 휴리스틱은 다음과 같다.

\[
p^{des}=p^{ref}+v^{CoM}\Delta t/2
\]

`p_ref`는 hip 아래의 지면 위치이고 `Delta t`는 다음 stance 시간이다. Notion의 `k_v(v-v_ref)` 항은 논문 식 (33)에 없다. 해당 feedback을 추가한다면 부호·frame·clip·reachability를 별도 실험으로 검증한다.

## 논문 수치와 이전 금지선

### Table I

| 항목 | 논문 값 | 해석 |
| --- | ---: | --- |
| 모델 질량 | `43 kg` | 본문 로봇 설명의 45 kg과 구분되는 controller model 값 |
| `Ixx` | `0.41 kg m^2` | Cheetah 3 모델 값 |
| `Iyy`, `Izz` | `2.1 kg m^2` | Cheetah 3 모델 값 |
| 마찰계수 `mu` | `0.6` | 시험 환경 가정값 |
| `f_min` | `10 N` | stance 수직력 하한 |
| `f_max` | `666 N` | stance 수직력 상한 |
| force weight | `1e-6` | QP effort regularization |
| orientation weight | `1` | 표에 제시된 Cheetah 튜닝값 |
| z weight | `50` | 표에 제시된 Cheetah 튜닝값 |
| yaw-rate weight | `1` | 표에 제시된 Cheetah 튜닝값 |
| velocity weight | `1` | 표에 제시된 Cheetah 튜닝값 |

현재 Go2 mass, inertia, actuator limit, contact patch와 Isaac PhysX friction combine mode가 다르다. 이 표의 값을 config default로 복사하지 않는다.

### Horizon과 solver

- horizon: gait 한 주기, `0.33~0.5 s`
- timestep 수: `10~16`
- prediction frequency: gait에 따라 `25~50 Hz`
- solver: `qpOASES`
- 구현: C++와 Eigen3
- 보고된 typical solve: `1 ms` 미만

Notion의 `N=12`, `dt=0.02 s`, `0.24 s`, OSQP는 논문의 구현값이 아니다. solver 선택은 Isaac Sim 4.5 번들 가용성, Windows 설치, license, warm-start, worst-case latency와 failure semantics를 조사한 후 결정한다.

## 실험 결과를 읽는 법

### 보고된 동작 범위

- trot: `1.2 m/s`
- flying trot: `1.7 m/s`
- bound: `2.5 m/s`까지 시험
- gallop: 최대 `3 m/s`
- lateral: 최대 `1 m/s`
- yaw: 최대 `180 deg/s`
- pronk: 약 `15 cm`, `150 ms` stance와 `350 ms` flight

같은 gain과 weight로 여러 gait를 제어했다는 결과는 Cheetah 3 하드웨어와 해당 state machine에서의 결과다. Go2 PPO나 다른 torque-control stack의 보편적 보장이 아니다.

### 외란과 계단

kick recovery는 몸통 roll/pitch가 약 10도 이내에 남았다는 시계열 예시다. 현재 저장소의 다중 seed·고정 protocol 성공률과 같은 통계 결과가 아니다.

계단은 debris 약 15개를 둔 네 단 계단에서 세 번 시도했다.

1. 첫 시도: 잘못 입력된 `120 deg/s` yaw command 뒤 중단
2. 두 번째: 오르내리기 성공
3. 세 번째: 발 이탈·무릎 inversion 뒤 회복해 다시 올랐지만 내려오다 전원 스위치를 차서 종료

논문은 지형 map 없이 swing 높이를 늘리고 early/late contact detection을 사용했다. 이를 “terrain-aware 계단 보행 성공률 100%”로 해석하지 않는다.

### 고속 한계

3 m/s 이상에서 주된 한계는 QP solve time보다 swing leg와 actuator velocity였다. Cheetah 3 시험 다리는 약 `15 rad/s`, Cheetah 2는 `24 rad/s`였고, simulation에서는 leg velocity limit을 `30 rad/s`로 높였을 때 `6 m/s`를 보고했다.

따라서 model-based 상위 제어기를 개선하기 전에 다음을 분리 측정해야 한다.

- desired/actual swing foot tracking
- joint velocity saturation
- torque saturation
- touchdown timing error
- yaw drift와 lateral workspace margin

## Notion 자료와의 차이

| 항목 | 논문 | Notion QUATTRO 설명 | 처리 |
| --- | --- | --- | --- |
| 상태 | gravity 포함 13-state LTV | 12-state affine `A,B,c` | 동치 여부를 수치 테스트로만 판단 |
| 자세 | `R_z(yaw)^T omega` | `rpy+omega*dt` | yaw 비영점 오차 검증 전 채택 금지 |
| solver | qpOASES | OSQP | 별도 dependency 결정 |
| horizon | `0.33~0.5 s`, 10~16 | `0.24 s`, 12 step | robot·gait별 측정 |
| force 제약 | stance 6 inequality, swing equality | 발당 7 row 설명 | builder 행 수 테스트 |
| foothold | `p_ref+v_CoM Delta t/2` | velocity-error feedback 추가 | 확장 가설로 분리 |
| swing | trajectory 형상·3.5 cm 미명시 | two-Bézier, 3.5 cm | QUATTRO 구현값으로만 보존 |
| WBC 범위 | stance `J^T R^T f`, swing task control | `J^T F` 중심 | full-body hierarchical QP로 부르지 않음 |

## 현재 저장소 적용 체크리스트

### 바로 적용할 문서·진단 규칙

- GRF의 world/body/support-plane frame을 결과 JSON에 명시한다.
- 발 순서를 `FL/FR/BL/BR`로 고정한다.
- contact schedule과 실제 touchdown/liftoff transition을 함께 기록한다.
- stance force utilization과 swing force leakage를 분리한다.
- solver 평균 시간뿐 아니라 p95/p99, timeout, infeasible, fallback을 기록한다.
- gait 결과에는 actuator·joint-velocity·swing tracking 한계를 함께 기록한다.

### 코드 구현 전 게이트

1. `skew(r)F == r cross F`
2. 13-state continuous model dimension과 frame test
3. ZOH discrete model 검산
4. horizon 3 직접 전개와 `A_qp/B_qp` 비교
5. `H/g` dimension·symmetry·PSD test
6. stance 6 inequality와 swing equality test
7. `J^T R^T f` virtual-work test
8. warm-start·infeasible·timeout fallback 계약

### 현재 범위 밖

- 기존 PPO checkpoint 변경
- G009 RECOVER를 small-angle stance MPC로 대체
- 새 solver 설치
- torque-control task 구현
- full-body WBC 또는 hierarchical QP 구현
- 실기체·sim-to-real 주장

## 완료 상태

논문 8쪽의 본문, 식 (1)~(33), Table I, Figure 2의 제어 주기, Figure 4~10의 실험·한계를 저장소 적용 관점에서 분리했다. 논문 분석과 Notion 비교는 완료했다.

아직 완료하지 않은 것은 논문 알고리즘의 Go2 구현과 재현 실험이다. 후속 구현은 [`MPC_WBC_SOURCE_AND_INTEGRATION_20260830.md`](MPC_WBC_SOURCE_AND_INTEGRATION_20260830.md)의 M1~M5 게이트를 따른다.
