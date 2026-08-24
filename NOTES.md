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

- 2026-08-24 G007은 `external_custom_compatibility_spike`로 구현했고 targeted 46 tests가 PASS, 코드 검토가 APPROVE였다.
- RBQ v1.20.0 tag object는 `741ce5733dcd7c0babec663bb7e1afbc02a776ca`, source commit은 `68bc33b77719d357b4323fb88549efd905caf721`이다.
- 고정 대상은 `rbq_sdk/ros2/src/rbq_description/urdf/rbq.urdf`, `package.xml`, STL 6개로 모두 8개 blob이다.
- GitHub repository API의 detected license `null`은 저장소 전체 라이선스를 감지하지 못했다는 뜻이며, 무허가 또는 금지의 증명이 아니다.
- `package.xml`은 Apache-2.0을 선언하지만 asset blob 적용 범위와 로컬 처리·재배포 권한은 미확정이다.
- 공식 Isaac Lab v2.1.1, v2.3.2, 조사 시점 main 고정 소스에는 대상 match가 없다. 따라서 상위 버전의 공식 구현 이식을 전제하지 않는다.
- `license_scope_unresolved`가 해제되기 전에는 자산 다운로드·변환·topology 검증·smoke를 실행하지 않는다.
- 상세 근거와 재현 명령은 `docs/G007_RBQ_COMPATIBILITY_SPIKE.md`에 있다. G006 production과 전체 ultragoal 완료 여부는 별도로 판정한다.
