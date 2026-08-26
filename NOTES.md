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
- 공통 velocity 환경에는 10~15초 간격의 x/y velocity push event가 있고 rough 환경에는 terrain curriculum이 이미 있다. G006은 기존 구성을 먼저 확인한 뒤 `events.push_robot`만 비교 축으로 정규화한다.

위 값은 Isaac Lab v2.1.1 소스의 기준값이다. Go2에서 실제로 상속·override되는 최종값은 설치된 고정 commit에서 다시 추출해 기록한다.

## 설계 원칙

- baseline을 보존하고 한 번에 한 축만 바꾼다.
- "자연스러움" 같은 주관 평가만 사용하지 않는다. 추적 오차, 에너지 proxy, 넘어짐률, 회복률을 함께 본다.
- 환경 수와 처리량은 다른 GPU 벤치마크에서 추정하지 않고 현재 RTX 3060 12GB에서 측정한다.
- 영상은 보조 증거다. 정책 checkpoint, 설정 diff, seed, 지표가 주 증거다.

## G006 rough 비교 해석 원칙

- official `UnitreeGo2RoughEnvCfg`를 rough baseline으로 사용한다. baseline과 push curriculum은 같은 terrain curriculum과 official domain randomization을 공통으로 유지하며, normalized config diff는 `events.push_robot`뿐이다.
- flat 결과는 맥락 자료로만 사용한다. flat→rough 또는 domain randomization 단독 인과효과를 주장하지 않는다.
- baseline 회복률이 높은 ceiling에 가까우면 작은 절대 차이는 기술통계로만 해석하며 정책 우월성으로 주장하지 않는다.
- 정책별 training seed는 `n=3`이므로 결과는 descriptive-only이며 통계적 유의성을 주장하지 않는다.
- mechanical power는 `sum(abs(torque * joint_velocity))`로 계산한 시뮬레이션 proxy이며 전기 에너지 소비량이 아니다.

## RBQ 호환성 메모

- 2026-08-24 G007은 `external_custom_compatibility_spike`로 구현했고 targeted 46 tests가 PASS, 코드 검토가 APPROVE였다.
- RBQ v1.20.0 tag object는 `741ce5733dcd7c0babec663bb7e1afbc02a776ca`, source commit은 `68bc33b77719d357b4323fb88549efd905caf721`이다.
- 고정 대상은 `rbq_sdk/ros2/src/rbq_description/urdf/rbq.urdf`, `package.xml`, STL 6개로 모두 8개 blob이다.
- GitHub repository API의 detected license `null`은 저장소 전체 라이선스를 감지하지 못했다는 뜻이며, 무허가 또는 금지의 증명이 아니다.
- `package.xml`은 Apache-2.0을 선언하지만 asset blob 적용 범위와 로컬 처리·재배포 권한은 미확정이다.
- 공식 Isaac Lab v2.1.1, v2.3.2, 조사 시점 main 고정 소스에는 대상 match가 없다. 따라서 상위 버전의 공식 구현 이식을 전제하지 않는다.
- `license_scope_unresolved`가 해제되기 전에는 자산 다운로드·변환·topology 검증·smoke를 실행하지 않는다.
- 상세 근거와 재현 명령은 `docs/G007_RBQ_COMPATIBILITY_SPIKE.md`에 있다. G006 production과 전체 ultragoal 완료 여부는 별도로 판정한다.

## G008 역학 메모

- 명령은 base frame `[v_x,v_y,ω_z]`다. `+ω_z`는 좌회전, `-ω_z`는 우회전으로 평가한다.
- 연속 uniform 범위에 순수 후진·순수 yaw가 포함된다는 사실만으로 학습 빈도가 보장되지 않는다. exact primitive를 별도 확률 질량으로 둔다.
- 발 접선력은 `sqrt(Fx²+Fy²)≤μFz`를 넘을 수 없다. 낮은 마찰은 선가속뿐 아니라 `Σ(r_xF_y-r_yF_x)`로 만드는 yaw moment도 제한한다.
- 링크 질량은 `M(q)q̈+C(q,q̇)q̇+g(q)=Sᵀτ+Jᵀλ`의 관성·Coriolis/centrifugal·중력항을 바꾼다. mass를 바꿀 때 inertia를 함께 scale하되 COM과 geometry가 고정된 근사임을 명시한다.
- 마찰과 링크 질량을 한 번에 randomize하지 않는다. 단일축 결과와 nominal guardrail이 나온 뒤에만 상호작용 실험을 연다.
- 상세 수치와 문헌 근거는 `docs/G008_COMMAND_FRICTION_LINK_MASS.md`에 있다.
- 공간 혼합 마찰 결과는 직선 보행 한계와 전 방향 회전 gate를 따로 읽는다. 단일 seed에서 중간 계수 실패 뒤 더 낮은 계수가 다시 통과하면 연속 통과 하한만 보수적 한계로 사용한다.
- 그룹별 질량 screen에서 scale이 같아도 thigh와 foot의 총질량 변화는 다르다. scale 기준 민감도와 동일 kg 추가 질량 민감도를 같은 결과로 취급하지 않는다.
- 실행 동작이 바뀌는 stage는 정량 JSON만 남기지 않는다. 로컬 MP4, 공개 GIF, 네 방향 PNG, checkpoint·물리 readback·파일 해시 JSON을 같은 단계의 완료 증거로 묶는다.
