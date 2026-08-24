# 기술 노트

## v2.1.1 locomotion 기준선

- 알고리즘: RSL-RL PPO
- flat 기본 학습 budget: 300 iterations
- rough 기본 학습 budget: 1500 iterations
- 공통 보상 항목:
  - `dof_torques_l2`: 기본 weight `-1e-5`
  - `action_rate_l2`: 기본 weight `-0.01`
  - `feet_air_time`: 기본 weight `0.125`
- ANYmal-C flat override:
  - `dof_torques_l2`: `-2.5e-5`
  - `feet_air_time`: `0.5`
- 공통 velocity 환경에는 10~15초 간격의 x/y velocity push event와 rough terrain curriculum이 이미 있다.

위 값은 Isaac Lab v2.1.1 소스의 기준값이다. Go2에서 실제로 상속·override되는 최종값은 설치된 고정 commit에서 다시 추출해 기록한다.

## 설계 원칙

- baseline을 보존하고 한 번에 한 축만 바꾼다.
- “자연스러움” 같은 주관 평가만 사용하지 않는다. 추적 오차, 에너지 proxy, 넘어짐률, 회복률을 함께 본다.
- 환경 수와 처리량은 다른 GPU 벤치마크에서 추정하지 않고 현재 RTX 3060 12GB에서 측정한다.
- 영상은 보조 증거다. 정책 checkpoint, 설정 diff, seed, 지표가 주 증거다.

## RBQ 호환성 메모

- 공식 URDF: `rbq_sdk/ros2/src/rbq_description/urdf/rbq.urdf`
- mesh: `rbq_sdk/ros2/src/rbq_description/meshes/stl/`
- `rbq_description/package.xml`: Apache License 2.0 선언
- 저장소 루트에는 전체 자산 범위를 명확히 하는 root LICENSE가 없으므로 재배포 범위를 확대 해석하지 않는다.
- 현행 공식 Isaac Lab 예제는 Python 3.11 / Isaac Sim 5.1 / Isaac Lab 2.3.2 대상이다. 이 프로젝트에서는 참조 구현으로만 pin한다.
