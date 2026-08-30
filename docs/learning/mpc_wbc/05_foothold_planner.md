# 05. Foothold Planner: 다음 발을 어디에 디딜 것인가

- 대응 원문: [Centroidal MPC Foothold Planner - Raibert Touchdown·Velocity Feedback·Terrain 확장](https://app.notion.com/p/Centroidal-MPC-Foothold-Planner-Raibert-Touchdown-Velocity-Feedback-Terrain-3cace2d5e0608179abd9c64298186a0e?pvs=25)
- 원문 대응 범위: 1~20절
- 선수 지식: [`04_gait_schedule.md`](04_gait_schedule.md)의 lift-off와 touchdown
- 학습 목표: nominal 위치, feedforward, velocity feedback, terrain correction으로 touchdown 목표를 만드는 이유를 설명한다.

## 먼저 한 문장

Foothold Planner는 공중으로 들릴 발이 다음에 어디에 닿아야 몸통의 원하는 움직임을 만들 수 있는지 계산한다.

Gait Schedule이 **언제**를 정한다면 Foothold Planner는 **어디에**를 정한다.

## 1. 기본 touchdown 식

Notion의 QUATTRO 설명은 평지 touchdown의 xy를 다음 세 항으로 나눈다.

$$
p_{td,xy}=p_{nominal,xy}
+\frac12T_{stance}v_{ref,xy}
+k_v(v_{xy}-v_{ref,xy})
$$

```text
기본 발 위치
+ 앞으로 움직일 만큼 미리 놓기
+ 실제 속도 오차 보정
= 다음 착지 목표
```

## 2. Nominal foot position

`p_nominal`은 몸통 또는 hip 기준으로 발이 편안하게 지지할 수 있는 기본 위치다.

- 너무 몸 안쪽: 지지 폭이 좁고 다리끼리 충돌할 수 있음
- 너무 바깥쪽: 관절 가동범위와 토크 부담 증가
- 너무 앞이나 뒤: 다음 swing 여유가 줄어듦

로봇이 yaw 방향을 바꾸면 nominal 위치도 원하는 기준 frame에 맞춰 회전해야 한다. world·body·hip frame을 섞지 않는다.

## 3. Feedforward: 몸이 이동할 곳을 미리 예상하기

stance 동안 몸통은 계속 움직인다. 발을 현재 hip 바로 아래에만 놓으면 몸이 지나간 뒤 발이 뒤쪽으로 밀릴 수 있다. 그래서 stance 시간의 절반 동안 몸이 이동할 거리를 미리 더한다.

$$
\Delta p_{ff}=\frac12T_{stance}v_{ref}
$$

예를 들어:

$$
v_{ref}=0.2m/s,
\qquad
T_{stance}=0.4s
$$

이면:

$$
\Delta p_{ff}=0.5\times0.4\times0.2=0.04m
$$

즉 nominal보다 4 cm 앞에 디딘다.

## 4. Feedback: 실제 속도 오차 보정

명령보다 실제 몸통이 너무 빠르거나 느릴 수 있다. Notion은 다음 항을 추가한다.

$$
\Delta p_{fb}=k_v(v-v_{ref})
$$

이 항의 부호와 frame은 구현마다 반드시 확인해야 한다. 예를 들어 너무 빠르게 전진할 때 발을 더 앞에 놓는 것이 감속에 도움이 되는지는 stance dynamics와 controller 정의에 따라 해석해야 한다.

따라서 `k_v`를 복사하기 전에 다음을 시험한다.

1. x 방향만 있는 작은 속도 오차를 넣는다.
2. touchdown 목표가 어느 방향으로 움직이는지 본다.
3. 그 변화가 다음 stance impulse를 통해 속도를 줄이는지 확인한다.
4. 좌우와 yaw 조합에서도 frame이 맞는지 검증한다.

## 5. Lift-off 위치를 저장하는 이유

계획한 출발점과 실제 발 위치가 다를 수 있다. `stance→swing` 순간의 실제 발 위치를 저장하면 swing trajectory가 현실의 출발점에서 시작한다.

저장하지 않으면 목표 궤적 시작점이 실제 발과 달라 첫 순간에 큰 위치 오차와 토크 명령이 생길 수 있다.

## 6. Touchdown 실제 위치도 다시 저장하기

`swing→stance` 순간 실제 접촉 위치는 계획 touchdown과 다를 수 있다.

- 지면 높이 오차
- 조기 접촉 또는 늦은 접촉
- 관절 추종 오차
- 발 미끄러짐

실제 touchdown을 다음 cycle의 기준으로 사용하면 계획 오차가 계속 누적되는 것을 줄일 수 있다. 계획값과 실제값은 둘 다 기록해 오차를 측정한다.

## 7. 발 하나의 상태 전환

```text
stance
  -> lift-off 감지
  -> 실제 lift-off 위치 저장
  -> 다음 touchdown 목표 계산
  -> swing trajectory 생성
  -> touchdown 감지
  -> 실제 touchdown 위치와 계획 오차 저장
  -> stance
```

## 8. 평지 planner의 한계

xy 목표만 계산하고 z를 고정하면 다음 상황에서 부족하다.

- 계단 윗면이 현재 지면보다 높음
- 돌이나 구멍 때문에 nominal 위치가 위험함
- 경사면에서 발바닥이 닿을 지면의 법선이 다름
- 목표가 관절 reachability 밖에 있음
- swing 경로가 장애물과 충돌함

terrain-aware planner는 단순히 z만 바꾸는 것이 아니라 안전성·도달 가능성·접촉 품질을 함께 판단해야 한다.

## 9. Terrain correction

평지 Raibert 목표에 지형 보정량을 추가하는 형태로 나눌 수 있다.

$$
p_{td}=p_{Raibert}+\Delta p_{terrain}
$$

`Delta p_terrain`이 고려할 수 있는 정보:

- 후보점의 높이와 법선
- 경사도
- edge와 hole까지의 거리
- 마찰 추정
- 관절 도달 가능 영역
- 다른 발과의 최소 간격

먼저 model-based nominal을 만들고 지형에 의한 수정량만 별도 모듈로 두면 실패 원인을 나누기 쉽다.

## 10. Residual RL과 연결

향후 RL을 붙인다면 전체 관절 토크보다 작은 보정량만 출력하게 할 수 있다.

$$
p_{td}=p_{model}+\Delta p_{RL}
$$

장점은 action dimension과 해석 범위가 작아진다는 것이다. 하지만 안전 영역, clip, terrain observation, 실패 fallback이 정의되기 전에는 안전하다고 볼 수 없다.

현재 PPO checkpoint의 action을 이 구조로 곧바로 바꾸지 않는다. 먼저 flat model-based baseline과 read-only touchdown telemetry가 필요하다.

## 11. 모듈 역할 구분

| 모듈 | 정하는 것 |
|---|---|
| Reference | 몸통의 원하는 미래 상태 |
| Gait | 발이 stance/swing인 시간 |
| Foothold | swing 발의 다음 착지 위치 |
| Swing Trajectory | 출발점부터 착지점까지의 시간별 경로 |
| MPC | stance 발의 지면반력 |

## 원문 20절 대응표

| 이 문서 | Notion 원문 절 |
|---|---|
| touchdown 식과 nominal | 1~3 |
| feedforward와 수치 예시 | 4~6 |
| velocity feedback과 전체 식 | 7~8 |
| lift-off/touchdown 실제 위치 | 9~11 |
| 평지 한계와 terrain correction | 12~14 |
| residual RL과 skill 구조 | 15~16 |
| 역할 구분과 전체 흐름 | 17~20 |

## 논문과 구분할 것

Cheetah 3 논문에 명시된 foothold 휴리스틱은 대략 `p_ref+v_CoM Delta t/2`다. Notion의 `k_v(v-v_ref)` feedback은 QUATTRO 구현 설명에 추가된 항이다. 논문 식과 같다고 쓰지 않으며 부호·frame·clip을 별도로 검증한다.

## 확인 문제

1. Gait Schedule과 Foothold Planner의 차이를 설명한다.
2. `v_ref=0.3m/s`, `T_stance=0.5s`일 때 feedforward 거리를 계산한다.
3. nominal foot position이 너무 멀면 어떤 문제가 생길 수 있는가?
4. lift-off 실제 위치를 저장하지 않았을 때 생길 수 있는 문제를 설명한다.
5. terrain correction이 z 높이 하나만의 문제가 아닌 이유를 세 가지 적는다.
6. residual foothold RL을 현재 PPO에 바로 적용하지 않는 이유를 설명한다.

### 최소 통과 기준

다음을 자료 없이 말할 수 있으면 된다.

> Foothold Planner는 nominal 위치에 몸의 예상 이동과 속도 오차 보정을 더해 다음 touchdown을 정하며, 실제 lift-off/touchdown과 지형·도달 가능성을 함께 확인해야 한다.
