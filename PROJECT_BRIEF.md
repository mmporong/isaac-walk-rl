# 프로젝트 브리프: Isaac Sim 4.5 사족보행 RL

## 목표 결과

Isaac Sim 4.5 / Isaac Lab 2.1.1 / RSL-RL PPO의 Windows 네이티브 재현 환경을 고정하고, ANYmal-C baseline에서 시작해 Go2의 보상·지형·외란 회복 실험을 정량 비교한다. 이어서 네 방향 명령과 마찰·다리 링크 질량의 단계별 dynamics randomization을 단일축으로 검증한다. RBQ 공개 자산은 외부 커스텀 자산으로 분류하고 라이선스·호환성 게이트를 증거 기반으로 판정한다.

## 성공 기준

1. 버전, 소스 commit, Python, PyTorch, CUDA wheel, 드라이버, GPU를 자동 수집한 환경 manifest가 있다.
2. v2.1.1 소스에서 대상 Gym task ID와 RSL-RL 설정을 직접 검증한다.
3. ANYmal-C flat 50 iterations가 오류 없이 끝나고, 이어서 300-iteration baseline의 명령·지표·체크포인트 경로를 기록한다.
4. Go2 flat에서 환경 수를 64→256→512→1024→2048 순서로 측정하고, 4096은 VRAM 여유 20% 이상과 오류 없음이 확인될 때만 실행한다.
5. `dof_torques_l2`, `action_rate_l2`, `feet_air_time`을 한 번에 하나씩 변경한 실험을 동일 budget과 3개 이상의 seed로 비교한다.
6. rough terrain과 domain randomization 단계의 추적 오차, 넘어짐률, 에너지 proxy를 baseline과 비교한다.
7. 외란 회복 성공 조건을 코드로 고정하고 baseline/개선 정책에 동일한 push grid를 적용해 회복률, 분자/분모, Wilson 95% 신뢰구간을 보고한다.
8. RBQ URDF·mesh 경로와 source commit을 고정하고, 자산 라이선스 범위와 로컬 처리 권한을 확인한다. 허가가 불명확하면 자산을 받거나 변환하지 않고 재현 가능한 blocker를 문서화한다.
9. 전진·후진·좌회전·우회전 명령을 고정 평가하고, 마찰과 16개 다리 링크 질량을 별도 S1→S2→S3 태스크로 확장한다. 각 단계는 randomized-domain과 nominal-domain guardrail을 통과해야 다음 범위로 간다.

## 검증된 고정값

- Isaac Sim `4.5.0`
- Isaac Lab tag `v2.1.1`, commit `90b79bb2d44feb8d833f260f2bf37da3487180ba`
- Python `3.10.x`
- RSL-RL `2.3.3`, PPO
- 태스크:
  - `Isaac-Velocity-Flat-Anymal-C-v0`
  - `Isaac-Velocity-Rough-Anymal-C-v0`
  - `Isaac-Velocity-Flat-Unitree-Go2-v0`
  - `Isaac-Velocity-Rough-Unitree-Go2-v0`
- RBQ URDF: `rbq_sdk/ros2/src/rbq_description/urdf/rbq.urdf`

## 실측 전 가설

- RTX 3060 12GB에서 상태 기반 Go2 2048 environments가 안정적으로 동작할 수 있다.
- 4096 environments는 peak VRAM과 처리량 게이트를 통과하면 동작할 수 있다.
- 세 보상 축의 one-factor 변경이 추적 성능·에너지·보행 자연스러움 사이의 해석 가능한 trade-off를 만든다.
- curriculum과 domain randomization 뒤 고정 push protocol의 회복률이 baseline보다 개선된다.

위 항목은 결과가 아니라 검증할 가설이며, 수치가 나오기 전에는 성공으로 기록하지 않는다.

## 비목표

- ROS 2 또는 WSL2 설치
- Ubuntu/Jazzy 이식
- 카메라·LiDAR 등 비전 관측
- 실기체 배포와 sim-to-real
- 기존 Isaac Lab 원본의 직접 수정
- 체크포인트·로그·영상·대용량 mesh의 Git 저장

## 실행 목표

### G1. 저장소와 환경 계약

- 구조, ignore 규칙, 버전 근거, 환경 manifest 스크립트를 만든다.
- DoD: 정적 검사와 로컬 환경 수집 명령이 통과하고 생성 파일이 Git 경계를 지킨다.

### G2. Isaac Lab 설치와 태스크 검증

- `%USERPROFILE%\IsaacLab`에 정확한 tag/commit을 설치하고 기존 Isaac Sim binary를 연결한다.
- DoD: 패키지 import, task ID 4개, `rsl-rl-lib==2.3.3`, CUDA 사용 가능 여부가 실제 출력으로 확인된다.

### G3. ANYmal-C 관문

- 64 environments / 50 iterations smoke 후 flat baseline을 실행한다.
- DoD: 정상 종료, TensorBoard 경로, 체크포인트 경로, peak VRAM, steps/s, 주요 보상 지표가 기록된다.

### G4. Go2 flat과 자원 측정

- 환경 수 사다리를 실행하고 안전한 최대치를 정한다.
- DoD: 각 환경 수의 VRAM·steps/s·성공/실패 표와 MuJoCo 51k 기준과의 조건 차이가 기록된다.

### G5. 보상 ablation

- 세 항목을 독립 변경하고 seed 반복을 수행한다.
- DoD: 동일 budget 비교표, TensorBoard 근거, 명령과 설정 diff가 있다.

### G6. rough·DR·외란 회복

- rough curriculum과 domain randomization을 적용하고 외란 회복을 고정 평가한다.
- DoD: baseline 대비 회복률과 신뢰구간, 추적·낙상·에너지 지표가 있다.

### G7. RBQ 외부 자산 호환성 사전조사와 최종 승인

- RBQ v1.20.0의 자산 경로·Git 객체·라이선스 근거를 고정하고, 공식 Isaac Lab 공개 소스에 RBQ 구현이 있는지 확인한다.
- DoD: 허가가 확인되면 별도 자산 검증과 smoke 증거를 만든다. 허가가 불명확하면 자산 다운로드·변환 전에 fail-closed로 멈추는 재현 가능한 blocker와 해제 조건을 기록한다. 전체 저장소는 code review와 독립 검증을 통과한다.

### G8. 방향 명령과 물성 단일축 curriculum

- 논문과 Isaac Lab 고정 소스를 조사해 명령 분포와 dynamics 범위를 정한다.
- 전진·후진·제자리 좌회전·제자리 우회전을 exact primitive로 반복 학습하되 연속 SE(2) 명령도 유지한다.
- 발바닥 접촉 마찰과 16개 다리 링크 질량을 서로 다른 태스크로 구성하고, 각 축을 S1→S2→S3으로 넓힌다.
- DoD: 설정 격리 테스트, headless smoke, runtime 물성 probe, 네 방향 평가, stage별 nominal guardrail과 문헌 수치의 채택·배제 근거가 있다. 실행하지 않은 stage는 완료로 기록하지 않는다.

## 외부 부수효과

- 새 Git 저장소와 단계별 커밋을 만든다.
- 원격 저장소는 비공개로 생성하는 것을 안전 기본값으로 하며, 로컬 검증을 통과한 커밋만 push한다.
- 기존 `physical-ai-lab`의 사용자 변경은 수정·스테이징·커밋하지 않는다.

## 후속 연구 게이트: Centroidal MPC/WBC

수령한 QUATTRO Notion 학습 자료와 MIT Cheetah 3 convex MPC 논문은 현재 PPO의 즉시 교체안이 아니라 별도 model-based baseline의 사전 자료로 관리한다. 원문 수집·차이 분석은 [`docs/MPC_WBC_SOURCE_AND_INTEGRATION_20260830.md`](docs/MPC_WBC_SOURCE_AND_INTEGRATION_20260830.md), 기계 판독 가능한 출처·해시는 [`reports/research/mpc_wbc_material_intake_20260830.json`](reports/research/mpc_wbc_material_intake_20260830.json)을 기준으로 한다.

후속 순서는 `import-light 동역학·prediction·constraint 수학 검증 → 기존 PPO의 read-only contact/GRF/foothold telemetry → 별도 flat Centroidal MPC baseline → terrain-dependent reference/foothold → residual RL`이다. 새 QP solver 의존성, torque-control task, WBC 정식화는 별도 설계·호환성·성능 게이트 없이는 추가하지 않는다. G009 R0 self-righting은 비발·몸통 접촉을 사용하는 별도 문제이므로 이 연구 게이트가 현재 RECOVER 안전 진단을 대체하거나 차단하지 않는다.

## 중단 조건

요청된 목표가 증거와 함께 완료되거나, 같은 설치/실행 blocker가 대체 경로까지 포함해 재현되고 현재 하드웨어·고정 버전 안에서 의미 있는 다음 단계가 없을 때만 중단한다.
