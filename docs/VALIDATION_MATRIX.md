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
| 보상 ablation | 동일 budget·3 seeds 이상 비교 | PASS; Go2 flat 4096 env × 300 iterations, 4 variants × seeds 42/43/44, 12/12 complete·failed 0, 고정 26×10×20초 평가와 strict hash 검증 완료 |
| rough·DR | official `UnitreeGo2RoughEnvCfg` baseline과 공통 terrain curriculum·official DR 고정, normalized diff는 `events.push_robot`만 허용, official rough baseline 대비 추적 오차·낙상률·에너지 proxy 기술통계 비교. flat→rough 또는 DR 단독 인과효과는 주장하지 않음 | PASS; 4096 env × 1500 iterations × seeds 42/43/44, 6/6 jobs complete. push curriculum은 tracking error sq `-9.1290%`, yaw error sq `-9.7386%`, torque L2 `+4.4411%`, mechanical power proxy `+3.6121%` |
| 외란 회복 비교 | 동일 rough·공통 official DR 조건의 고정 protocol과 Wilson 95% CI | PASS; 6480 push trials와 540 guardrail trials. 회복률 `99.5370%`→`99.5988%` (`+0.0617%p`), paired bootstrap 95% CI `-0.7716%p ~ +0.9568%p`; guardrail 양쪽 `100%`, 유의한 개선 주장 없음 |
| G006 durable JSON 이식성 | 실제 실행은 허용 root 안의 resolved 절대경로, 저장 command와 그 hash는 `%USERPROFILE%`·`%REPO_ROOT%`·`%ISAACLAB_ROOT%` 뒤에 문자열 끝 또는 separator만 허용하고 선택한 token root containment를 통과한 경로에 바인딩; summary path는 repo/queue-relative 또는 `%USERPROFILE%`로 기록하고 Isaac root는 CLI로 명시; legacy complete state는 전체 artifact/hash 검증 후에만 무재실행 마이그레이션 | PASS; 6/6 legacy commands portable migration, production training/evaluation artifact hash 변경 0, 시스템/Isaac Python strict summary SHA-256 `c0ef40715ce09915d3789249168228090049d7edc5a2cea82231c4ebddbfe76a`로 byte-identical |
| G008 네 방향 명령 경로 | exact forward/backward/left-yaw/right-yaw primitive를 포함한 custom command term, direct yaw-rate, 태스크 등록·학습 스모크·고정 명령 평가 | warm-start `1,024 env × 300 iterations`와 평면 64환경 평가 PASS; 생존 64/64, 선속도 RMSE `0.0466~0.0794`, yaw RMSE `0.0741~0.1154`. rough는 좌·우 PASS, 전진·후진 자세 gate FAIL |
| G008 마찰 단일축 S1 | command 설정 대비 material event만 변경, `.*_foot`, static `0.72~0.88`, dynamic `0.52~0.68`, 64 buckets, dynamic≤static | config diff·smoke·1,024환경 runtime probe PASS. `1,024 env × 300 iterations` 뒤 randomized·nominal 평면 네 방향 gate PASS. terrain level mean이 약 3.45→2.27로 하락해 rough 개선과 S2 진입은 미승인 |
| G008 다리 링크 질량 단일축 S1 | command 설정에 독립적인 16-body mass event만 추가, body별 `0.95~1.05`, inertia 재계산 | config diff·smoke·1,024환경 runtime probe PASS. `1,024 env × 300 iterations` 학습은 완료했으나 randomized·nominal 평면 우회전 yaw RMSE `0.2956/0.2947`로 FAIL. nominal guardrail 실패, S2 미승인 |
| G008 공간 혼합 마찰 스트레스 시험 | 폭 `0.5 m`의 face별 고·저마찰을 단일 triangle mesh에 배치, ground collider 없음, 완료 case당 32환경·500 step, command/friction S1 동시 배치, 띠 노출·전환·방향 gate·독립 kinematic fall | friction S1 전진·후진·좌회전은 완료 최저 `0.2/0.1`까지 연속 PASS. 우회전은 `0.7/0.5` 첫 FAIL 뒤 `0.6/0.4` 개별 PASS라 전 방향 하한 미확정. `0.1/0.05`는 4회 native 종료로 평가 미확정. contact force 누락으로 slip은 `null`, 완료 case에서 friction S1 우회전 kinematic fall 1건 |
| G008 링크 그룹 질량 민감도 | hip·thigh·calf·foot 중 한 그룹씩 `0.8~1.2`배, inertia 동비율 재계산, 25조건×2정책×4방향×4반복, 총 800환경·300 step | 두 정책 모두 전진·후진 25/25 조건 PASS, 전체 0 falls. command nominal 네 방향 PASS, leg-mass S1 nominal 우회전 yaw RMSE `0.44 rad/s`로 FAIL. leg-mass S2 미승인 유지 |
| G008 세 정책 시각 비교 | command·friction S1·leg-mass S1을 평면·seed 42·같은 900-step 명령으로 별도 프로세스에서 추론하고 runtime 물성과 checkpoint hash를 촬영 보고서에 고정 | 3/3 촬영 exit 0, 정책당 H.264 899 frames. 1280×380 로컬 비교 MP4와 720×214 공개 GIF, 네 방향 접촉시트의 SHA-256·ffprobe·10 MiB 제한 검증 PASS |
| G008 단계 변경 시각 증거 | 혼합 `0.8/0.6 ↔ 0.2/0.1` 마찰 한 편과 hip·thigh·calf·foot `1.2배` 질량 네 편을 seed 20260826·같은 900-step 명령으로 촬영, 원본 MP4 로컬 전용, GIF·PNG·물리 readback JSON 공개 | 5/5 원본 H.264 1280×720·50fps·899 frames·17.98초. 자막 MP4 2개는 로컬에 보관하고 공개 GIF `5.18/9.05 MB`, 네 방향 PNG `0.51/1.15 MB`의 파일 해시·checkpoint·friction/mass/inertia readback 검증 PASS |
| G009 산 비탈 C0/S0 | 24-cell analytic slope·normal·material gate, Isaac `5/15/25°` USD geometry·friction·reset readback, 동일 checkpoint·seed·명령·카메라의 headless off-screen 캡처, source commit과 artifact SHA-256 결합 | C0/S0 PASS. analytic `24/24`, G009 순수 Python `68` tests, Isaac G009/G008 config `7/7·8/8`, 세 캡처·GIF·PNG·sidecar 검증 완료. 25°는 최대 tilt `84.7832°`, 하방 이동 `2.3925 m`로 stress 실패 경계이며 G009 PPO 학습·WALK 성공은 미주장 |
| RBQ 외부 자산 호환성 사전조사 | source·8 blob·라이선스 근거 고정, fail-closed blocker 재현 | G007 gate 구현 완료; `license_scope_unresolved`, expect-blocked exit 0·require-ready exit 3, targeted 46 tests PASS·code review APPROVE. 자산·파생물 다운로드/변환/smoke 미실행 |

## 1차 근거

- Isaac Lab v2.1.1 release: https://github.com/isaac-sim/IsaacLab/releases/tag/v2.1.1
- v2.1.1 installation source: https://github.com/isaac-sim/IsaacLab/tree/90b79bb2d44feb8d833f260f2bf37da3487180ba/docs/source/setup/installation
- v2.1.1 ANYmal-C task registration: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/__init__.py
- v2.1.1 Go2 task registration: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/__init__.py
- Isaac Sim 4.5 requirements: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html
- G008 설계 문서: [`G008_COMMAND_FRICTION_LINK_MASS.md`](G008_COMMAND_FRICTION_LINK_MASS.md)
- G008 혼합 마찰·링크 그룹 질량 한계: [`G008_PERIODIC_FRICTION_AND_LINK_MASS_LIMITS.md`](G008_PERIODIC_FRICTION_AND_LINK_MASS_LIMITS.md)
- G008 비교 영상·해시: [`G008_VISUAL_EVIDENCE.md`](G008_VISUAL_EVIDENCE.md)
- G008 dynamics event 구현: https://isaac-sim.github.io/IsaacLab/v2.1.1/_modules/isaaclab/envs/mdp/events.html
- G008 병렬 PPO·terrain curriculum 근거: https://proceedings.mlr.press/v164/rudin22a.html
- G008 velocity command 근거: https://proceedings.mlr.press/v205/margolis23a/margolis23a.pdf
- G008 dynamics randomization 범위·비판 근거: https://arxiv.org/html/1804.10332, https://arxiv.org/html/2107.04034, https://arxiv.org/html/2011.02404
- G009 산 비탈 설계·실행 기록: [`G009_MOUNTAIN_SLOPE_RECOVERY.md`](G009_MOUNTAIN_SLOPE_RECOVERY.md)
- G009 S0 시각·물리 증거: [`../reports/runs/g009_s0_visual_evidence.json`](../reports/runs/g009_s0_visual_evidence.json)
- RBQ v1.20.0 tag object API: https://api.github.com/repos/RainbowRobotics/RBQ/git/tags/741ce5733dcd7c0babec663bb7e1afbc02a776ca
- RBQ 고정 commit URDF: https://raw.githubusercontent.com/RainbowRobotics/RBQ/68bc33b77719d357b4323fb88549efd905caf721/rbq_sdk/ros2/src/rbq_description/urdf/rbq.urdf
- RBQ 고정 commit package.xml: https://raw.githubusercontent.com/RainbowRobotics/RBQ/68bc33b77719d357b4323fb88549efd905caf721/rbq_sdk/ros2/src/rbq_description/package.xml
- G006 정량 판정: [`G006_ROUGH_PUSH_RECOVERY.md`](G006_ROUGH_PUSH_RECOVERY.md)
- G007 상세 판정: [`G007_RBQ_COMPATIBILITY_SPIKE.md`](G007_RBQ_COMPATIBILITY_SPIKE.md)
