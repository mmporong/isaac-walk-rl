# 검증 매트릭스

| 주장 또는 단계 | 증거 기준 | 현재 상태 |
| --- | --- | --- |
| Sim 4.5 ↔ Lab 2.1.1 호환 | v2.1.1 릴리스와 설치 문서 | 검증 완료 |
| Python 3.10 | v2.1.1 설치 문서와 로컬 번들 Python | 검증 완료 |
| RSL-RL 2.3.3 | v2.1.1 `isaaclab_rl/setup.py`와 로컬 package metadata | 설치·버전 검증 완료 |
| 대상 task ID 4개 | headless AppLauncher 이후 Gym registry | 로컬 등록 검증 완료 |
| headless 프로세스 정상 종료 | bundled `python.bat` 직접 실행, AppLauncher 종료 코드 0 | 검증 완료; `isaaclab.bat -p` exit 1은 wrapper false-negative 경고로 분리 |
| RTX 3060 12GB에서 2048 envs | 현재 호스트 짧은 학습 | 10 iterations PASS; peak 4,058 MiB(33.02%), GPU 회복 확인 |
| RTX 3060 12GB에서 4096 envs | 2048 PASS·peak 80% 이하·GPU 회복 뒤 조건부 실행 | 게이트 충족 후 10 iterations PASS; peak 4,822 MiB(39.24%), GPU 회복 확인 |
| ANYmal-C 50 iterations | 정상 종료와 checkpoint/log | 64 env, seed 42, exit 0, model_49.pt 확인으로 검증 완료 |
| ANYmal-C flat 300 iterations | 정상 종료, TensorBoard scalar, checkpoint hash | 64 env, seed 42, exit 0, 299/300, model_299.pt 확인으로 검증 완료 |
| Go2 flat scale ladder | 64→2048 순차 실행과 조건부 4096, 각 checkpoint·TensorBoard·GPU 회복 | seed 42, 각 10 iterations, 64/256/512/1024/2048/4096 모두 PASS; 장기 baseline은 별도 |
| 보상 ablation | 동일 budget·3 seeds 이상 비교 | 실행 대기 |
| rough·DR | flat baseline 대비 지표 | 실행 대기 |
| 외란 회복 개선 | 고정 protocol과 Wilson 95% CI | 실행 대기 |
| RBQ 2.1.1 backport | API 매핑과 smoke 또는 blocker | 조사 대기 |

## 1차 근거

- Isaac Lab v2.1.1 release: https://github.com/isaac-sim/IsaacLab/releases/tag/v2.1.1
- v2.1.1 installation source: https://github.com/isaac-sim/IsaacLab/tree/90b79bb2d44feb8d833f260f2bf37da3487180ba/docs/source/setup/installation
- v2.1.1 ANYmal-C task registration: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/__init__.py
- v2.1.1 Go2 task registration: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/__init__.py
- Isaac Sim 4.5 requirements: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html
- RBQ URDF: https://github.com/RainbowRobotics/RBQ/blob/main/rbq_sdk/ros2/src/rbq_description/urdf/rbq.urdf
- RBQ current simulator dependencies: https://github.com/RainbowRobotics/RBQ/blob/main/rbq_simulator/rbq_lab/dependencies.yaml
