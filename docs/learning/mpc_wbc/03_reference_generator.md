# 03. Reference Generator: 원하는 미래 몸 상태 만들기

- 대응 원문: [Centroidal MPC Reference Generator - Position·Velocity·Body Pose와 Terrain-adaptive 확장](https://app.notion.com/p/Centroidal-MPC-Reference-Generator-Position-Velocity-Body-Pose-Terrain-adaptive-3cace2d5e06081e29170c780ebfcff83?pvs=25)
- 원문 대응 범위: 1~21절
- 학습 목표: 사용자 명령을 horizon 전체의 몸통 목표 `X_ref`로 바꾸는 과정을 설명한다.

## 먼저 한 문장

MPC는 미래를 예측할 수 있어도 원하는 미래가 무엇인지 스스로 알지 못한다. Reference Generator는 속도 명령을 미래 각 시점의 위치·자세·속도 목표로 번역한다.

```text
전진 0.2 m/s, 회전 0.1 rad/s
        |
        v
0.02초 뒤 목표, 0.04초 뒤 목표, ... , horizon 끝의 목표
```

## 1. Reference와 현재 상태의 차이

- 현재 상태 `x_0`: 센서와 estimator가 추정한 실제 로봇 상태
- reference `x_ref`: 앞으로 로봇이 되었으면 하는 목표 상태
- MPC: 예측 상태와 reference의 차이가 작아지도록 발 힘을 고른다.

$$
X_{ref}=
\begin{bmatrix}
x_{ref,1}\\x_{ref,2}\\\vdots\\x_{ref,N}
\end{bmatrix}
$$

현재 목표 하나만 주는 것이 아니라 prediction horizon과 같은 길이의 목표 궤적을 만든다.

## 2. 평지에서 위치 목표 만들기

현재 위치가 `p_0`이고 명령 속도가 일정하다고 가정한다.

$$
p_{x,ref}(t)=p_{x,0}+v_{x,cmd}t
$$

$$
p_{y,ref}(t)=p_{y,0}+v_{y,cmd}t
$$

예를 들어 `v_x=0.2 m/s`이면 0.1초 뒤 목표 x 위치는 현재보다 2 cm 앞이다.

높이는 평지에서 nominal body height를 유지하도록 둔다.

$$
p_{z,ref}(t)=h_{nominal}
$$

## 3. 평지에서 자세 목표 만들기

평지에서는 몸통이 옆이나 앞뒤로 기울지 않는 것을 기본 목표로 둘 수 있다.

$$
roll_{ref}=0,
\qquad
pitch_{ref}=0
$$

yaw는 명령한 회전 속도를 시간에 따라 적분한다.

$$
yaw_{ref}(t)=yaw_0+\dot\psi_{cmd}t
$$

`yaw_ref=0`으로 고정하면 로봇이 회전 명령을 받아도 항상 처음 방향으로 돌아가려 할 수 있다. 그래서 현재 yaw를 기준으로 미래 yaw 목표를 쌓는다.

## 4. 속도 목표 만들기

평지의 선속도 reference는 다음처럼 둘 수 있다.

$$
v_{ref}=[v_{x,cmd},v_{y,cmd},0]
$$

각속도 reference는 yaw 회전 명령만 남긴다.

$$
\omega_{ref}=[0,0,\dot\psi_{cmd}]
$$

전체 평지 reference는 상태 순서 `[p,rpy,v,omega]`에 맞춰 조립한다.

## 5. 위치와 속도 목표를 둘 다 쓰는 이유

속도 목표만 있으면 원하는 속도는 따라가더라도 작은 오차가 계속 누적되어 위치가 멀리 흘러갈 수 있다. 위치 목표는 장기 drift를 붙잡아준다.

반대로 위치 목표만 있으면 목표점을 따라가기 위해 순간 속도를 과도하게 바꿀 수 있다. 속도 목표는 움직임의 방향과 크기를 직접 알려준다.

```text
position reference -> 장기적으로 어디에 있어야 하는가
velocity reference -> 지금 어떤 속도로 움직여야 하는가
```

## 6. Reference와 `Q`는 다른 역할이다

- Reference: 목표값 자체
- `Q`: 각 목표 오차를 얼마나 중요하게 볼지 정하는 가중치

예를 들어 `z_ref=0.30m`는 원하는 높이다. z 가중치를 크게 하는 것은 그 높이 오차를 강하게 벌점화하겠다는 뜻이다. 목표값을 바꾸는 것과 가중치를 바꾸는 것은 같은 조작이 아니다.

## 7. 경사면에서는 평지 reference가 틀릴 수 있다

경사면에서 world 기준 `roll=0`, `pitch=0`만 고집하면 몸통이 지면과 어긋날 수 있다. terrain normal 또는 support plane을 이용해 지면 기준 자세 목표를 만들 수 있다.

예:

- terrain height에 맞춘 몸통 높이
- support-plane normal에 맞춘 roll/pitch
- 경사 진행 방향에 맞춘 body heading
- 발 디딜 높이에 맞춘 CoM trajectory

현재 저장소의 `support_plane.py`는 normal·접평면·CoM 진단을 제공하지만 MPC reference generator는 아니다. 개념이 연결된다는 이유만으로 현재 PPO 제어 경로에 MPC reference를 삽입하지 않는다.

## 8. 계단과 거친 지형에서 추가로 필요한 것

단순히 `z_ref`만 올리면 계단 보행이 완성되지 않는다. 다음 요소가 서로 맞아야 한다.

1. 지형 높이와 법선 추정
2. 몸통의 높이·roll·pitch reference
3. 미래 contact schedule
4. 각 swing 발의 touchdown 높이
5. 발 reachability와 충돌 여유
6. terrain-dependent force constraint

Reference Generator는 전체 solver를 교체하기보다 MPC가 따라갈 목표를 지형에 맞게 바꾸는 모듈이다.

## 9. Skill 또는 동작 모드와 연결

보행 모드마다 reference 규칙을 나눌 수 있다.

| 모드 | reference의 핵심 |
|---|---|
| Stand | 현재 xy 근처, nominal height, 수평 자세, 0 속도 |
| Flat Walk | 명령 xy 속도와 yaw rate, nominal height |
| Slope | support-plane 기준 높이와 자세 |
| Stair Up | 계단 phase에 맞춘 높이·pitch·속도 프로파일 |
| Recovery | 큰 자세 오차와 비발 접촉을 다루는 별도 모델 필요 |

특히 recovery는 small-angle stance Centroidal MPC reference만 바꿔 해결할 수 있는 문제가 아니다.

## 원문 21절 대응표

| 이 문서 | Notion 원문 절 |
|---|---|
| Reference와 horizon | 1~2 |
| x/y/z 위치 목표 | 3~5 |
| roll/pitch/yaw 목표 | 6~8 |
| 선속도·각속도 목표 | 9~11 |
| 위치·속도와 `Q`의 역할 | 12~14 |
| 계단·경사·terrain adaptation | 15~17 |
| skill과 코드 연결 | 18~19 |
| 복습과 다음 학습 | 20~21 |

## 확인 문제

1. 현재 상태와 reference의 차이를 설명한다.
2. `v_x=0.3m/s`일 때 0.2초 뒤 x 위치 목표는 현재보다 몇 m 앞인가?
3. 위치 reference와 속도 reference를 함께 쓰는 이유를 설명한다.
4. Reference와 `Q`의 역할 차이를 예로 든다.
5. 경사면에서 world-horizontal 자세 목표가 불리할 수 있는 이유를 설명한다.

### 최소 통과 기준

다음을 자료 없이 말할 수 있으면 된다.

> Reference Generator는 사용자 명령과 지형 정보를 horizon 전체의 위치·자세·속도 목표로 바꾸고, MPC는 그 목표 오차를 Q 가중치에 따라 줄이는 힘을 계산한다.
