# 06. Swing Trajectory: 발을 부드럽게 들어 옮기기

- 대응 원문: [Centroidal MPC Swing Trajectory - Cubic Bézier·Apex·Velocity Continuity](https://app.notion.com/p/Centroidal-MPC-Swing-Trajectory-Cubic-B-zier-Apex-Velocity-Continuity-3cace2d5e0608165a4eaf3c4cbb88f93?pvs=25)
- 원문 대응 범위: 1~26절
- 선수 지식: [`05_foothold_planner.md`](05_foothold_planner.md)의 lift-off 위치와 touchdown 목표
- 학습 목표: apex와 두 개의 cubic Bézier로 swing 발의 위치·속도 목표를 만드는 이유를 설명한다.

## 먼저 한 문장

Foothold Planner가 착지점을 정하면 Swing Trajectory는 실제 lift-off 위치에서 발을 들어 올리고, 장애물 여유를 확보한 뒤, 착지점에 부드럽게 내려놓는 시간별 경로를 만든다.

## 1. 직선으로 보내면 왜 안 되는가

lift-off와 touchdown을 직선으로 연결하면 발이 지면을 스치거나 작은 돌·계단 모서리에 걸릴 수 있다. 중간에 가장 높은 점 apex를 둬 clearance를 만든다.

$$
p_{apex,xy}=\frac{p_{lo,xy}+p_{td,xy}}{2}
$$

$$
z_{apex}=\max(z_{lo},z_{td})+h_{clearance}
$$

- `p_lo`: 실제 lift-off 위치
- `p_td`: 계획 touchdown 위치
- `h_clearance`: 지면과 장애물을 피하기 위한 추가 높이

Notion의 QUATTRO 설명은 기본 clearance로 `0.035m`를 제시하지만 모든 로봇과 지형의 공통값이 아니다.

## 2. Cubic Bézier 기본식

네 control point `P_0,P_1,P_2,P_3`로 3차 Bézier 곡선을 만든다.

$$
B(u)=(1-u)^3P_0+3(1-u)^2uP_1+3(1-u)u^2P_2+u^3P_3
$$

여기서 `0≤u≤1`이다.

- `u=0`이면 `B(0)=P_0`
- `u=1`이면 `B(1)=P_3`
- 중간 control point가 곡선의 진행 방향과 속도 형상을 정한다.

미분은 다음과 같다.

$$
\frac{dB}{du}=3(1-u)^2(P_1-P_0)
+6(1-u)u(P_2-P_1)
+3u^2(P_3-P_2)
$$

## 3. 전체 swing을 두 구간으로 나누기

```text
전반부: lift-off -> apex
후반부: apex -> touchdown
```

하나의 곡선보다 두 구간으로 나누면 apex의 높이와 출발·도착 속도를 제어하기 쉽다.

## 4. Lift-off에서 부드럽게 출발하기

첫 Bézier에서 `P_0=P_1=p_lo`로 둔다. 시작 미분은:

$$
\left.\frac{dB}{du}\right|_{u=0}=3(P_1-P_0)=0
$$

따라서 위치 reference가 lift-off 순간 갑자기 튀지 않고 목표 출발 속도도 0이 된다.

## 5. Touchdown에서 부드럽게 멈추기

두 번째 Bézier에서 `P_2=P_3=p_td`로 둔다. 끝 미분은:

$$
\left.\frac{dB}{du}\right|_{u=1}=3(P_3-P_2)=0
$$

따라서 touchdown 목표 속도가 0이 된다. 실제 발이 반드시 0 속도로 닿는다는 뜻은 아니며, tracking 오차와 조기 접촉을 별도로 측정해야 한다.

## 6. Apex에서 이어지기

두 곡선은 apex에서 위치가 같아야 한다. 또한 z 방향 미분을 0으로 만들면 최고점에서 발이 위로 올라가다 아래로 내려오는 방향을 자연스럽게 바꾼다.

xy 방향 미분도 두 segment 사이에서 맞추면 apex 통과 순간 수평 속도가 갑자기 변하지 않는다. 이를 velocity continuity라고 한다.

위치만 이어지고 속도가 끊기면 Cartesian PD가 큰 순간 오차를 만들어 토크가 튈 수 있다.

## 7. 전체 progress를 각 구간의 `u`로 변환하기

Gait Schedule이 주는 전체 swing progress를 `s∈[0,1]`라 한다.

$$
u=
\begin{cases}
2s, & 0\le s<0.5\\
2s-1, & 0.5\le s\le1
\end{cases}
$$

- `s=0~0.5`: 첫 Bézier
- `s=0.5~1`: 두 번째 Bézier

각 segment가 전체 swing 시간의 절반을 사용하므로:

$$
\frac{du}{dt}=\frac{2}{T_{swing}}
$$

## 8. 위치뿐 아니라 실제 시간 속도도 계산하기

Bézier 미분 `dB/du`는 parameter `u`에 대한 변화율이다. 실제 시간에 대한 발 속도는 chain rule을 사용한다.

$$
v_{des}=\frac{dp}{dt}
=\frac{dp}{du}\frac{du}{dt}
=\frac{dp}{du}\frac{2}{T_{swing}}
$$

같은 곡선이라도 swing 시간이 짧으면 더 빠르게 움직여야 한다. 그래서 위치 reference만 있고 속도 reference가 없으면 damping과 feedforward가 정확히 작동하기 어렵다.

## 9. Cartesian PD와 관절 토크

목표 발 위치·속도와 실제 발 상태의 차이로 task-space force를 만든다.

$$
F_{task}=K_p(p_{des}-p)+K_d(v_{des}-v)
$$

그 힘을 Jacobian transpose로 관절 토크에 연결한다.

$$
\tau=J^TF_{task}
$$

stance와 swing의 목적은 다르다.

| 구간 | 목표 | 대표 계산 |
|---|---|---|
| Stance | 몸통을 지지하고 가속 | MPC GRF `F` -> `J^T F` |
| Swing | 발을 착지점까지 추종 | Bézier `p_des,v_des` -> Cartesian PD -> `J^T F_task` |

## 10. Stance penetration curve를 혼동하지 않기

stance 발은 이미 바닥과 접촉하고 있으므로 공중 발처럼 가상의 아래쪽 곡선을 따라가게 만들 필요가 없다. 접촉 force와 지면 constraint가 stance를 담당한다.

가상의 penetration reference를 사용하면 contact solver·발 강성·Cartesian PD가 서로 싸울 수 있다. 실제 구현에서 제거 이유와 force continuity를 실험으로 확인해야 한다.

## 11. 전체 swing 흐름

```text
gait에서 lift-off 감지
  -> 실제 lift-off 위치 저장
  -> foothold planner가 touchdown 계산
  -> apex 계산
  -> 두 Bézier에서 p_des와 v_des 계산
  -> Cartesian PD/feedforward
  -> J^T F_task
  -> touchdown 감지 및 실제 위치 기록
```

## 원문 26절 대응표

| 이 문서 | Notion 원문 절 |
|---|---|
| swing 역할과 apex | 1~3 |
| Bézier 식과 미분 | 4~5 |
| 두 segment와 control point | 6~14 |
| progress 변환과 속도 | 15~19 |
| Cartesian PD와 stance/swing 분리 | 20~21 |
| stance penetration과 전체 흐름 | 22~24 |
| 핵심 수식과 다음 학습 | 25~26 |

## 논문과 구분할 것

원 논문은 Notion의 two-cubic-Bézier control point와 `3.5cm` clearance를 명시하지 않는다. 이는 QUATTRO 구현 설명으로 보존한다. Cheetah 3의 고속 한계도 QP 계산보다 swing 속도와 actuator limit에서 나타났으므로 원하는 곡선만 만들고 끝내지 않는다.

측정할 항목:

- 실제/목표 swing 발 위치 오차
- 실제 touchdown 속도
- early/late contact
- 최소 clearance
- 관절 속도와 토크 saturation
- yaw drift와 lateral workspace margin

## 확인 문제

1. lift-off와 touchdown을 직선으로 연결하지 않는 이유를 설명한다.
2. apex의 xy와 z를 각각 어떻게 정하는지 적는다.
3. `P_0=P_1`이면 시작 속도가 0이 되는 이유를 Bézier 미분으로 설명한다.
4. `P_2=P_3`의 목적은 무엇인가?
5. `dB/du`에 `2/T_swing`을 곱해야 하는 이유를 설명한다.
6. stance와 swing 발이 서로 다른 제어를 사용하는 이유를 설명한다.

### 최소 통과 기준

다음을 자료 없이 말할 수 있으면 된다.

> Swing Trajectory는 실제 lift-off와 계획 touchdown 사이에 apex를 두고, 두 Bézier로 위치와 속도를 부드럽게 연결한 뒤 Cartesian 제어와 `J^T`로 관절 토크를 만든다.
