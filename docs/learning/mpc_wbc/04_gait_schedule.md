# 04. Gait Schedule: 어느 발이 언제 땅에 닿는가

- 대응 원문: [Centroidal MPC Gait Schedule - Phase·Duty Factor·Trot Contact Prediction](https://app.notion.com/p/Centroidal-MPC-Gait-Schedule-Phase-Duty-Factor-Trot-Contact-Prediction-3cace2d5e06081429b17e225b77c7597?pvs=25)
- 원문 대응 범위: 1~20절
- 학습 목표: phase, frequency, duty factor, phase offset으로 현재와 미래의 contact schedule을 만든다.

## 먼저 한 문장

Gait Schedule은 발을 어디에 놓을지 정하지 않는다. 각 발이 언제 바닥을 지지하고 언제 공중에서 움직일지만 정한다.

```text
stance = 바닥을 밀어 몸을 지지하는 구간
swing  = 발을 들어 다음 착지점으로 옮기는 구간
```

## 1. 주기를 0부터 1까지의 phase로 표현하기

한 보행 주기를 `0≤phi<1`로 표현한다. step frequency가 `f` Hz라면 한 주기 시간은:

```text
한 주기 시간 T = 1 / 주파수 f
```

시간이 흐르면 global phase를 다음처럼 전진시킬 수 있다.

```text
global phase = (이전 global phase + f·dt)의 소수 부분
```

`mod 1`은 phase가 1을 넘으면 다시 0부터 시작하게 한다.

## 2. Phase offset으로 발 순서 만들기

각 발의 phase는 global phase에 발별 offset을 더한다.

```text
발 i의 phase = (global phase + 발 i의 offset)의 소수 부분
```

Trot에서는 대각선 발이 같은 시기에 움직인다.

- 한 쌍: FL + BR
- 다른 쌍: FR + BL

Notion의 예시 offset은 발 순서 `[FL,FR,BL,BR]`에서 `[0.5,0,0,0.5]`다. 이 배열은 QUATTRO 예시이며 현재 Go2 PPO의 실제 phase를 읽은 값이 아니다.

## 3. Duty factor: 한 주기 중 땅에 있는 비율

Duty factor `D`는 한 발이 주기 중 stance에 머무는 비율이다.

```text
발 i의 phase < duty factor D  -> stance
```

```text
발 i의 phase ≥ duty factor D  -> swing
```

예를 들어 `D=0.65`면 한 발은 주기의 65% 동안 땅을 지지하고 35% 동안 공중에 있다.

```text
stance 시간 = D · T
swing 시간  = (1 - D) · T
```

## 4. 수치 예시

Notion의 예시 `f=1.4Hz`, `D=0.65`를 계산하면:

```text
T = 1 / 1.4 ≈ 0.714초
```

```text
stance 시간 ≈ 0.464초
swing 시간  ≈ 0.250초
```

`D>0.5`이므로 두 대각선 pair의 stance가 일부 겹친다. 그 순간에는 네 발이 모두 닿을 수 있어 초기 실기에서 안정 여유가 커질 수 있다. 하지만 이 수치를 다른 로봇에 그대로 복사하면 안 된다.

## 5. Swing progress

swing 구간 안에서 발이 얼마나 진행했는지를 0에서 1로 정규화한다.

```text
swing 진행률 s = (발 phase - D) / (1 - D),    0 ≤ s < 1
```

- `s=0`: lift-off 직후
- `s=0.5`: swing 중간
- `s→1`: touchdown 직전

이 값은 [`06_swing_trajectory.md`](06_swing_trajectory.md)가 발의 현재 목표 위치와 속도를 계산할 때 사용한다.

## 6. Contact transition

매 제어 주기에서 이전 contact와 현재 contact를 비교하면 사건을 감지할 수 있다.

| 변화 | 사건 | 기록할 것 |
|---|---|---|
| `1→0` | lift-off | 실제 발 출발 위치 |
| `0→1` | touchdown | 실제 착지 위치와 계획 오차 |

이 사건은 foothold planner와 swing trajectory의 시작·종료 조건이 된다.

## 7. MPC에는 미래 contact가 필요하다

현재 contact만 알려주면 MPC는 horizon 중간에 어느 발이 떠날지 알 수 없다. 그래서 미래 각 timestep의 contact를 미리 계산한다.

```text
시간       k    k+1  k+2  k+3  ...
FL         1     1    0    0
FR         0     0    1    1
BL         0     0    1    1
BR         1     1    0    0
```

- `1`: stance, GRF를 최적화할 수 있음
- `0`: swing, GRF를 0으로 고정

MPC는 곧 support를 잃을 발에 계속 큰 힘을 배정하지 않고 다음 contact 전환을 미리 고려할 수 있다.

## 8. Gait가 하지 않는 일

Gait Schedule은 시간표다. 다음 항목은 다른 모듈이 담당한다.

| 질문 | 담당 모듈 |
|---|---|
| 몸통이 미래에 어디에 있어야 하는가? | Reference Generator |
| 어느 발이 언제 stance/swing인가? | Gait Schedule |
| swing 발이 어디에 착지할 것인가? | Foothold Planner |
| 그 지점까지 어떤 곡선으로 움직일 것인가? | Swing Trajectory |
| stance 발이 얼마의 힘을 낼 것인가? | Centroidal MPC |

## 9. 전체 보행 흐름

```text
phase와 offset
    -> 현재/future contact schedule
        -> stance 발: MPC GRF -> J^T F
        -> swing 발: foothold -> swing trajectory -> Cartesian 제어
```

## 원문 20절 대응표

| 이 문서 | Notion 원문 절 |
|---|---|
| frequency, global phase, offset | 1~4 |
| duty factor와 접촉 예시 | 5~8 |
| swing progress와 advance | 9~10 |
| future contact와 MPC 제약 | 11~14 |
| reference·foothold·swing과 역할 구분 | 15~17 |
| 핵심 수식과 QUATTRO 의미 | 18~20 |

## 확인 문제

1. `f=2Hz`이면 stride period는 몇 초인가?
2. duty factor가 0.6이면 stance와 swing은 각각 주기의 몇 %인가?
3. trot에서 대각선 발을 같은 phase로 두는 이유를 설명한다.
4. 현재 contact뿐 아니라 future contact schedule이 필요한 이유는 무엇인가?
5. `1→0`과 `0→1` transition이 각각 무엇인지 적는다.
6. Gait Schedule과 Foothold Planner의 역할 차이를 설명한다.

### 최소 통과 기준

다음을 자료 없이 설명하면 된다.

> Gait Schedule은 frequency, phase offset, duty factor로 각 발의 stance/swing 시간표를 만들며, 미래 contact는 MPC에서 swing 발 힘을 0으로 제한하는 데 사용된다.
