# MPC/WBC 초보자 학습 안내

이 폴더는 수령한 Notion의 여섯 하위 문서를 동역학을 오래 쉬었던 학습자도 다시 따라갈 수 있도록 한 페이지씩 풀어 쓴 자료다. 원문을 복사한 것이 아니라, 각 페이지의 개념 순서와 수식을 유지하면서 쉬운 말·숫자 예시·오해 방지·자가 점검을 추가했다.

## 자료의 경계

- **Notion 원문**: QUATTRO 구현을 설명한 수령 자료다.
- **논문 기준**: *Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control*과 대조해 차이를 표시한다.
- **현재 저장소**: Go2의 50 Hz joint-position PPO 실험 저장소다. 아래 자료를 읽었다고 MPC/WBC가 구현된 것은 아니다.
- **WBC라는 이름**: 수령 자료의 중심은 Centroidal MPC, stance GRF, swing Cartesian 제어와 `J^T F`다. full-body hierarchical QP 전체를 다루는 완전한 WBC 교재는 아니다.

전체 출처와 기술적 차이는 [`../../MPC_WBC_SOURCE_AND_INTEGRATION_20260830.md`](../../MPC_WBC_SOURCE_AND_INTEGRATION_20260830.md), 논문 검토는 [`../../MIT_CHEETAH3_CONVEX_MPC_PAPER_REVIEW_20260830.md`](../../MIT_CHEETAH3_CONVEX_MPC_PAPER_REVIEW_20260830.md)에 있다.

## 여섯 문서

| 순서 | 쉬운 질문 | 학습 문서 | 대응 Notion |
|---:|---|---|---|
| 1 | 네 발의 힘으로 몸통의 다음 상태를 어떻게 계산하는가? | [`01_centroidal_dynamics.md`](01_centroidal_dynamics.md) | [A·B·c, Skew, SO(3), Jacobian](https://app.notion.com/p/Centroidal-MPC-A-B-c-Skew-SO-3-Jacobian-3cace2d5e06081a1b935d48be865b92a?pvs=25) |
| 2 | 여러 미래 중 가장 나은 힘 계획을 어떻게 고르는가? | [`02_prediction_qp.md`](02_prediction_qp.md) | [Prediction부터 OSQP까지](https://app.notion.com/p/Centroidal-MPC-Prediction-OSQP-M-S-d-Q-R-Constraint-Receding-Horizon-3cace2d5e06081a1a9e6c85d98930ff3?pvs=25) |
| 3 | MPC가 따라가야 할 미래 목표는 누가 만드는가? | [`03_reference_generator.md`](03_reference_generator.md) | [Reference Generator](https://app.notion.com/p/Centroidal-MPC-Reference-Generator-Position-Velocity-Body-Pose-Terrain-adaptive-3cace2d5e06081e29170c780ebfcff83?pvs=25) |
| 4 | 어느 발이 언제 땅에 닿는지는 어떻게 정하는가? | [`04_gait_schedule.md`](04_gait_schedule.md) | [Gait Schedule](https://app.notion.com/p/Centroidal-MPC-Gait-Schedule-Phase-Duty-Factor-Trot-Contact-Prediction-3cace2d5e06081429b17e225b77c7597?pvs=25) |
| 5 | 공중의 발을 어디에 내려놓을 것인가? | [`05_foothold_planner.md`](05_foothold_planner.md) | [Foothold Planner](https://app.notion.com/p/Centroidal-MPC-Foothold-Planner-Raibert-Touchdown-Velocity-Feedback-Terrain-3cace2d5e0608179abd9c64298186a0e?pvs=25) |
| 6 | 그 착지점까지 발을 어떻게 부드럽게 이동시키는가? | [`06_swing_trajectory.md`](06_swing_trajectory.md) | [Swing Trajectory](https://app.notion.com/p/Centroidal-MPC-Swing-Trajectory-Cubic-B-zier-Apex-Velocity-Continuity-3cace2d5e0608165a4eaf3c4cbb88f93?pvs=25) |

## 전체 연결

```text
사용자 속도 명령
  |
  +-> Reference Generator: 몸통이 미래에 어디에 있어야 하는가
  |
  +-> Gait Schedule: 미래에 어느 발이 stance/swing인가
  |      |
  |      +-> Foothold Planner: swing 발이 어디에 착지할 것인가
  |             |
  |             `-> Swing Trajectory: 착지점까지 어떤 경로로 움직일 것인가
  |
  `-> Centroidal MPC
         동역학 A/B/c + 미래 예측 M/S/d + 목표 Q/R + 접촉 제약
              |
              `-> stance 발의 지면반력 GRF
                       |
                       `-> tau = J^T F -> 관절 토크
```

## 한 문서를 공부하는 방법

1. **첫 읽기**: 수식을 계산하지 말고 굵은 질문과 그림 같은 설명만 읽는다.
2. **두 번째 읽기**: 기호의 차원과 단위를 옆에 쓴다.
3. **세 번째 읽기**: 문서 끝의 확인 문제를 자료 없이 답한다.
4. 답이 막히면 정답 전체를 외우지 말고 해당 절 하나만 다시 읽는다.
5. Notion의 특정 수치와 논문의 수치를 현재 Go2 설정으로 복사하지 않는다.

권장 순서는 `01 → 03 → 04 → 05 → 06 → 02`다. 원문 번호는 Prediction이 두 번째지만, 처음 배우는 경우 목표·보행·발 움직임을 먼저 본 뒤 QP를 공부하는 편이 전체 그림을 잡기 쉽다.
