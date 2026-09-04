# isaac-walk-rl 작업 규칙

## 프로젝트 경계

- 이 저장소는 Windows 네이티브 Isaac Sim 4.5 / Isaac Lab 2.1.1 사족보행 RL 실험 코드와 재현 기록만 관리한다.
- Isaac Sim 설치본, Isaac Lab 원본 clone, 체크포인트, TensorBoard 원시 로그, 영상, 생성 USD 및 대용량 mesh는 저장소 밖에 둔다.
- ROS 2, WSL2, Ubuntu/Jazzy, 비전 관측, 실기체 배포와 sim-to-real은 현재 범위가 아니다.
- 외부 프로젝트나 원본 Isaac Lab 파일을 직접 수정하지 않고, 이 저장소의 설정 오버라이드와 확장 코드로 구현한다.

## 고정 버전

- Isaac Sim: `4.5.0`
- Isaac Lab: tag `v2.1.1`, commit `90b79bb2d44feb8d833f260f2bf37da3487180ba`
- Python: Isaac Sim 번들 `3.10.x`
- RL 라이브러리: `rsl-rl-lib==2.3.3`
- 알고리즘: RSL-RL PPO

버전 변경은 별도 호환성 검증과 결정 기록 없이 수행하지 않는다.

## 실험 규약

- 실행 이름은 `<robot>_<terrain>_<change>_s<seed>_<YYYYMMDD-HHmm>` 형식을 사용한다.
- baseline과 변형은 같은 학습 budget, 평가 조건 및 seed 집합을 사용한다.
- 보상 ablation은 `dof_torques_l2`, `action_rate_l2`, `feet_air_time` 중 한 축만 한 번에 바꾼다.
- 모든 실행은 `RUN_NOTES.md`에 명령, commit, 환경 수, seed, peak VRAM, steps/s, 결과와 실패 원인을 기록한다.
- 같은 원인으로 세 번 실패하면 반복하지 말고 원인·증거·다음 가설을 기록한다.
- 4096 environments는 사전 보장값이 아니다. 낮은 환경 수부터 측정하고 VRAM 여유와 안정성 게이트를 통과한 경우에만 올린다.

## 학습 대화 기본 계약

- 새 학습은 [`docs/learning/README.md`](docs/learning/README.md)에서 현재 폐루프 위치와 연결 문서를 먼저 읽은 뒤 시작한다. 채팅의 즉석 설명을 정본 학습자료로 대신하지 않는다.
- 필요한 문서가 없으면 질문부터 하지 않고 `docs/learning/`에 설명·수치 예제·점검문제 1·2·3·4·실습·완료기준이 들어간 Markdown 파일을 먼저 만든다.
- 새 학습 순서는 `정본 문서 → 예제 → 문제 1·2·3·4 → 채점·보충 → Isaac Lab 통제실험 → 마지막 무자료 폐루프 설명`이다. 자료 없는 질문으로 시작하는 것은 다음 날 또는 주말 복습에만 적용한다.
- 같은 주제의 보충은 새 채팅 답변으로만 남기지 않고 기존 학습 문서에 반영한다.
- 이 프로젝트의 학습 중심은 Isaac Lab 사족보행을 이용한 Python·동역학·PPO·reward·강화학습 실전 적용이다. 강화학습 이론을 처음부터 순서대로 강의하지 않고 현재 실험을 설명하거나 검증하는 데 필요한 개념부터 다룬다.
- 설명은 항상 `로봇 상태/센서 → Observation → Policy → Action → Joint target → PD/Actuator → Torque → Physics → 다음 로봇 상태 → Reward → PPO Update` 폐루프에 놓는다. 외란, IMU, 관절 상태, 토크, 접촉력, 보상을 독립 과목처럼 떼어 설명하지 않는다.
- 공식이 나오면 변수의 의미, 단위, 좌표계, Go2에서의 실제 힘·회전·관절 방향을 함께 설명한다. 좌표계가 빠진 속도·힘·회전 설명은 완료로 보지 않는다.
- PPO 설명은 actor/critic, value, advantage, clipping, reward를 실제 checkpoint와 held-out 평가 결과에 연결한다. mean reward 상승을 성공률이나 자세 안정성 향상과 같은 뜻으로 쓰지 않는다.
- Isaac Lab 실험은 한 번에 한 변수만 변경한다. baseline과 variant는 시작 checkpoint, seed 집합, 학습 budget, 평가 grid를 고정하고 checkpoint hash, reward, 성공률, 방향별 tracking RMSE, 자세 안정성, torque·power·contact 지표 중 관련 항목을 비교한다.
- 학습 완료는 `자료 없이 폐루프 설명 → 코드/설정 위치 확인 → 직접 실험 → 한 조건 변경 → 수치 비교 → 실패 원인 설명`을 같은 주제에서 모두 수행한 상태다.
- 학습이나 실험을 마치면 `무엇을 했나 / 왜 이 방법인가 / 어디까지가 내 구현인가 / 다시 하면 무엇을 바꾸나` 네 질문을 파일·수치 근거로 확인한다.
- 기존 G003~G009와 MPC/WBC 자료는 삭제하거나 성공 서사로 다시 쓰지 않는다. [`docs/learning/README.md`](docs/learning/README.md)의 폐루프 색인으로 재분류하고, 각 문서의 기존 한계와 미실행·미자격 판정을 보존한다.
- 실행은 고정 버전 Isaac Sim 4.5.0 / Isaac Lab 2.1.1 / RSL-RL 2.3.3에서 재현한다. 최신 Isaac Lab 문서는 개념·API 변화 비교에 사용하되 별도 migration gate 없이 현재 runtime을 올리지 않는다.

## 단계별 시각 증거

- 학습 stage, 물리 randomization 범위, 평가 지형 또는 정책 checkpoint가 바뀌면 완료 보고 전에 해당 단계의 동작을 다시 촬영한다.
- 촬영 세트는 로컬 전용 원본 MP4, Git 공개 GIF, 대표 방향 스크린샷 PNG, 촬영 조건·checkpoint·물리 readback·파일 SHA-256을 담은 JSON으로 구성한다.
- MP4는 `%USERPROFILE%\IsaacLab\logs\visual_evidence\<goal_id>`에만 두고 저장소에는 넣지 않는다. 기존 G008 증거의 `goal_id`는 `g008`이며 경로와 해시는 그대로 유지한다. GIF와 PNG는 각각 10 MiB 아래로 만든다.
- 다음 촬영부터 원본 MP4는 30fps로 보존한다. 공개 GIF는 15fps를 목표로 만들고 12fps 미만으로 낮추지 않는다.
- 카메라 GIF는 30fps 원본에서 프레임을 직접 샘플링한다. 텔레메트리 GIF는 소수의 정지 화면을 반복하지 않고 중간 프레임을 실제로 렌더링한다.
- GIF가 10 MiB를 넘으면 `길이 → 해상도 → 팔레트` 순서로 줄인다. 프레임레이트를 12fps 아래로 낮춰 용량을 맞추지 않는다.
- sidecar에는 source FPS, target·actual GIF FPS, frame count, duration, 최대 frame duration, temporal strategy, 해상도와 palette 색상 수를 기록한다. 압축 우선순위는 `compression_policy_order`, 실제 수행한 단계는 `compression_steps_applied`로 구분한다.
- 새 미디어 builder는 MP4를 ffprobe로 검사해 30fps인지 확인하고, `inspect_gif_encoding()`의 실제 측정값을 `validate_gif_encoding_metadata()`로 검증한 뒤에만 결과를 발행한다.
- 영상은 동작 증거다. 성능 판정은 같은 단계의 다중 환경 정량 평가 JSON을 기준으로 하고, 시각 증거 보고서에서 그 파일의 경로와 SHA-256을 연결한다.
- 집계 문구나 문서만 바뀌고 실행 동작·물리 조건·checkpoint가 그대로라면 기존 촬영을 재사용할 수 있다. 재사용 여부와 근거는 시각 증거 보고서에 남긴다.

## 완료와 검증

- 완료 주장은 새로 실행한 명령과 출력 또는 생성된 수치표를 근거로 한다.
- TensorBoard 화면이나 영상만으로 완료하지 않는다. 재현 명령과 정량 지표를 함께 남긴다.
- 변경 파일만 경로별로 스테이징하고, 커밋 전 `git diff --cached --stat`을 확인한다.
- 커밋 메시지는 `<type>(scope): 한글 설명` 형식을 사용한다.
