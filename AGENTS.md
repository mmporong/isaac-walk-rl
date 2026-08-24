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

## 완료와 검증

- 완료 주장은 새로 실행한 명령과 출력 또는 생성된 수치표를 근거로 한다.
- TensorBoard 화면이나 영상만으로 완료하지 않는다. 재현 명령과 정량 지표를 함께 남긴다.
- 변경 파일만 경로별로 스테이징하고, 커밋 전 `git diff --cached --stat`을 확인한다.
- 커밋 메시지는 `<type>(scope): 한글 설명` 형식을 사용한다.
