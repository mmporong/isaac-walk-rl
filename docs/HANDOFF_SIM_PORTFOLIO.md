# 시뮬레이션 포트폴리오 작업 인계

기준일: 2026-09-27, Asia/Seoul. 이 문서는 다른 Codex 세션 또는 다른 Windows PC에서 이어가기 위한 내부 인계다. 문서 작성 중 새 GPU 학습은 시작하지 않았다.

## 1. 목표와 현재 결론

목표는 **Isaac Sim / Isaac Lab / PPO를 의미 있게 활용한 시뮬레이션 전용 사족보행 포트폴리오**다. 실물 로봇 제작은 하지 않는다. 현재 작업에 ROS, WSL, 실기체 배포, sim-to-real 검증을 추가하지 않는다.

현재 자료로 포트폴리오 사례를 정리할 수 있다. 다만 “산비탈에서 뒤집힌 로봇의 복구까지 성공했다”는 상태는 아니다.

- 주 사례: G006의 rough terrain 보행 및 외란 회복 통제실험. 다중 seed, 같은 budget, 정량 평가와 통계적 한계를 보여 준다.
- 보조 사례: G005 보상 ablation, G008 이동 명령·마찰·링크 질량 변화의 검증 범위와 실패 경계.
- 후속 연구: G009 뒤집힘 복구. 최신 rev32 / rev33은 **안전 smoke 실패**, 복구 정책 자격 없음. 진단 성과와 복구 성과를 분리한다.
- 최신 G009 실패 자료는 연구 기록으로 보존한다. 이번 인계는 Garden/포트폴리오 사이트 발행이 아니며, 최신 실패 실험을 새로운 성공 글로 게시하지 않았다.

추천 재개 순서는 **기존 검증 사례의 포트폴리오 구성 → 필요할 때만 제한된 G009 진단**이다. 실패 학습을 오래 돌리는 일을 포트폴리오 완성의 필수 조건으로 만들지 않는다.

## 2. 저장소와 기준 상태

| 항목 | 값 |
| --- | --- |
| 소스 원격 | https://github.com/mmporong/isaac-walk-rl |
| 이 PC의 작업 경로 | `C:\Users\LIMMM\worktrees\isaac-walk-rl-g009-r0` |
| 이 PC의 작업 branch | `g009-r0-qualification` |
| 인계 문서 추가 전 검증된 결과 commit | `074e6b6ca74492782330079725380984236db296` |
| 결과 commit의 원격 위치 | 작성 시작 시 `origin/main`과 동일 |
| 전체 실행 기록 | [RUN_NOTES.md](../RUN_NOTES.md) |
| 최신 실험 상세 | [G009_MOUNTAIN_SLOPE_RECOVERY.md](G009_MOUNTAIN_SLOPE_RECOVERY.md) |
| 프로젝트 규칙 | [AGENTS.md](../AGENTS.md) |

이 문서의 commit은 결과 commit보다 뒤에 추가된다. 다른 세션은 `git log -1`, `git status --short --branch`, `git remote -v`로 실제 현재 상태를 확인한다. 다른 worktree나 기존 변경을 덮어쓰지 않는다. 이 PC의 폴더 이름을 다른 PC의 설치 위치로 가정하지 않는다.

관련 저장소는 이 PC에서 `C:\Users\LIMMM\physical-ai-lab`, `C:\Users\LIMMM\robotics-garden`에 있다. 포트폴리오 대상은 `mmporong/portfolio-v2`이며 `/p/`는 배포 경로다. 새 호스트에서는 각 디렉터리의 실제 Git origin과 branch를 확인한 뒤 사용한다. 이번 작업은 이 세 저장소의 콘텐츠나 배포 상태를 변경하지 않았다.

## 3. 검증된 성과와 말하면 안 되는 주장

| 파트 | 수행한 내용 | 판정과 제한 |
| --- | --- | --- |
| G005 | 4 variants × seeds 42/43/44 = 12 runs, 4096 env × 300 iterations. baseline 대비 torque / action-rate / feet-air-time 보상 한 축씩 제거 | 12/12 완료. 서로 다른 보상 정의의 raw training reward를 동일 척도로 비교하지 않는다 |
| G006 | baseline / push curriculum × seeds 42/43/44 = 6 runs, 4096 env × 1500 iterations. 6480 push trials + 540 no-push guardrail trials | 회복률 99.5370% 대 99.5988%, 차이 +0.0617%p. paired bootstrap 95% CI [-0.7716, +0.9568]%p이므로 유의한 개선을 주장하지 않는다. 이는 외란 후 보행 회복이지 뒤집힌 자세에서 일어서기가 아니다 |
| G008 명령 | G006 checkpoint에서 1024 env × 300 iterations warm-start, 전진·후진·좌회전·우회전 평가 | flat 방향 gate 통과. 생존 64/64, 선속도 RMSE 0.0466~0.0794 m/s, yaw-rate RMSE 0.0741~0.1154 rad/s |
| G008 friction S1 | 1024 env × 300 iterations, nominal / randomized flat 방향 평가 | 해당 flat 범위 통과. 모든 혼합 마찰 도로·지형을 해결한 것이 아니다 |
| G008 질량·도로 | 링크 질량 민감도, 주기적 마찰, 불규칙 도로·curriculum 비교 | leg-mass S1 우회전 gate 실패. 도로 후보도 보편적 강건성 입증 아님. 조건별 결과와 거부 판정을 유지 |
| G009 경사 | S0 기하 분석 0/5/10/15/20/25도 × 4 headings = 24/24, 기존 정책 재생 일부 조건 | 기하 검증과 정책 학습은 별개. 25도 실패는 stress-cell이지 로봇의 최대 등판각 측정값이 아니다 |
| G009 복구 | contact 관측 검증, 학습·관절 제한 telemetry, rev32/33 damping smoke 및 HD 촬영 | 계측과 증거 연결은 진행됐으나 최신 정책 모두 안전 gate 실패. 네 초기 자세의 정량 복구 성공률은 미측정 |

근거:

- [G005 보상 실험](G005_REWARD_ABLATION.md), [12-run 집계](../reports/runs/g005_reward_ablation_summary.json)
- [G006 포트폴리오 원고](G006_PORTFOLIO.md), [실험 설계·결과](G006_ROUGH_PUSH_RECOVERY.md), [집계](../reports/runs/g006_summary.json), [시각 증거](G006_VISUAL_EVIDENCE.md)
- [G008 명령·마찰·질량](G008_COMMAND_FRICTION_LINK_MASS.md), [주기적 마찰·질량 경계](G008_PERIODIC_FRICTION_AND_LINK_MASS_LIMITS.md), [불규칙 도로](G008_IRREGULAR_ROAD.md), [보상·도로 curriculum](G008_REWARD_AND_ROAD_CURRICULUM.md), [시각 증거](G008_VISUAL_EVIDENCE.md)

## 4. 실제 학습 도구·budget·물리 설정

고정 runtime은 Windows 네이티브 Isaac Sim **4.5.0**, Isaac Lab **2.1.1**, Isaac Lab commit `90b79bb2d44feb8d833f260f2bf37da3487180ba`, Isaac 번들 Python **3.10.x**, `rsl-rl-lib==2.3.3`, **PPO**다. 로봇은 Isaac Lab의 Unitree Go2 자산이다. 원본 Isaac Lab은 수정하지 않고 이 저장소의 확장 코드와 override를 사용한다.

headless는 GUI 없이 물리 시뮬레이션과 학습을 실행한다는 뜻이다. GPU 학습을 하지 않는다는 뜻이 아니다. 최신 학습은 `headless=true`, `cuda:0`, scratch seed 42였다. 기존 정책이 걷는 영상은 checkpoint를 추론 재생한 것이며, 영상 촬영 순간에 PPO를 재학습한 것이 아니다.

- 물리: `dt=0.005 s` = 200 Hz, control decimation 4 = 50 Hz.
- G006: rollout 24 steps/env, PPO epochs 5, mini-batches 4. 자세한 관측·보상·curriculum 구성은 G006 원고를 정본으로 사용한다.
- 최신 G009 smoke 각각: 1024 env × rollout 24 × 50 iterations = **1,228,800 transitions**, epochs 5 × mini-batches 4 × 50 = **1000 optimizer updates**.
- iteration, PPO epoch, mini-batch, 병렬 env는 서로 다른 단위다. “50 epoch 학습”으로 줄여 쓰지 않는다.
- 최신 G009 공통 고정: Kp 25 N·m/rad, effort limit 23.5 N·m, action scale 0.60, calf reset -2.37 rad, entropy coefficient 0, initial noise 0.5, EMA 0.2, solver iterations [8, 0]. 보상·관측·관절 제한을 임의로 확대하지 않는다.
- 보상함수는 코드/설정과 해당 실험 문서에 연결한다. reward가 좋아 보인다는 이유로 복구 성공이나 안전성을 판정하지 않는다.

### 최신 두 실험

| 지표 | rev32 / G009-5-E025 | rev33 / G009-5-E026 |
| --- | --- | --- |
| 바꾼 축 | rear RL/RR calf Kd 0.5 → 1.0 N·m·s/rad | front FL/FR calf Kd 0.5 → 1.0; rear 1.0 유지 |
| wall time | 248.860 s | 233.997 s |
| 평균 처리량 | 6662.58 steps/s | 6447.66 steps/s |
| peak VRAM | 4133 MiB | 4259 MiB |
| 관찰된 peak GPU / 온도 | 57% / 54°C | 55% / 55°C |
| 관절 제한 nonzero scalar samples | 2/50 | 1/50 |
| attribution 실제 사건 | FR calf lower 2건 | prone RL calf lower 1건 |
| 최종 mean reward | -6.68 | -11.67 |
| numeric invalid | 0 | 0 |
| GPU 보호·회복 | PASS | PASS |
| training safety gate | FAIL | FAIL |
| prone HD 재생 | 8 s timeout, stable_success=false | 8 s timeout, stable_success=false |
| 네 자세 복구 정량 자격 | not_measured / 미자격 | not_measured / 미자격 |

`4 → 2 → 1` 사건 수를 학습 성능의 인과적 개선으로 쓰지 않는다. 단일 seed이고 정책 궤적이 달라지며 두 후보 모두 zero-event 안전 기준을 만족하지 않았다.

정본 설정: [rev32](../configs/g009_r0_rev32_damping.json), [rev33](../configs/g009_r0_rev33_damping.json).

정본 실행 보고서:

- [rev32 학습](../reports/runs/go2_flat_g009_r0_rev32_rear_damping_s42_20260927_0910.json), [attribution](../reports/runs/go2_flat_g009_r0_rev32_rear_damping_s42_20260927_0910_attribution.json), [intervention](../reports/runs/go2_flat_g009_r0_rev32_rear_damping_s42_20260927_0910_intervention.json)
- [rev33 학습](../reports/runs/go2_flat_g009_r0_rev33_front_damping_s42_20260927_0931.json), [attribution](../reports/runs/go2_flat_g009_r0_rev33_front_damping_s42_20260927_0931_attribution.json), [intervention](../reports/runs/go2_flat_g009_r0_rev33_front_damping_s42_20260927_0931_intervention.json)

## 5. 무엇이 문제였고 다음에 무엇을 진단하나

확인된 현상: rev33의 prone / env 109 / episode 202 / iteration 40에서 RL calf가 lower limit를 `0.0107293129 rad` 초과했다. q=-2.7334294319 rad, lower=-2.7227001190 rad, target=-1.6769111156 rad, qdot=+0.3309272528 rad/s, torque=+23.5 N·m이었다. foot contact 3.308870861 BW, non-foot contact 0.743777 BW가 동반됐다. BW는 body weight로 정규화한 힘이다.

이 종료 snapshot에서는 target과 torque가 제한 안쪽을 향하고 있었다. 따라서 “정책이 제한 밖 목표를 명령했다”만으로 설명되지 않는다. 접촉에 의한 backdrive와 solver 영향은 후보 해석이지, 원인이 분리 검증된 상태가 아니다. 종료 순간 qdot만으로 제한에 접근하던 속도를 알 수 없다.

이미 거부된 재시도도 보존한다. rev13 solver velocity 0→1은 CPU 반복에서 contact peak 15.9716 BW > 15, rev15 position 8→16은 GPU force 16.788 BW > 15로 거부됐다. rev26 full300도 관절 제한 nonzero 57/300으로 거부됐다. rev28/29/30 smoke는 각각 4/50, 1/50, 4/50이었다. reset 변경 조건은 열리지 않았으므로 -2.28 rad 후보를 이미 적용된 것으로 쓰지 않는다.

**다음 G009 작업은 아직 구현·실행하지 않았다.** 기존 attribution은 종료 snapshot이며 physics-substep의 사건 직전 이력은 없다.

1. 새 진단을 사전 등록한다. 짧고 bounded인 ring buffer로 사건 직전 **200 Hz physics-substep**의 q, qdot, joint target, computed/applied effort, 같은 다리 접촉력을 동일 시간축에 기록한다. 시간·환경·관절·좌표계·단위를 명시하고 정책 update와 분리한다.
2. 동일 사건 재현과 계측의 동기화·메모리 상한·원본 불변성을 검증한다. 기존 50 Hz 종료 기록을 200 Hz 이력으로 이름만 바꾸지 않는다.
3. 결과가 가르는 역학/학습 가설 **한 축만** 선택한다. damping·torque·관절 tolerance·solver를 동시에 바꾸지 않는다. 더 높은 Kd를 반복하는 단계는 중단된 상태다.
4. 새 candidate의 사전 등록된 50-iteration smoke를 실행한다. hard-limit와 numeric invalid 최대 count가 모두 0, runtime readback 안정, GPU 보호 PASS가 다음 단계의 필요조건이다. smoke PASS는 복구 성공이 아니다.
5. 통과했을 때만 별도 full300 contract/wrapper 및 네 초기 자세 정량 qualification을 준비한다. 기존 rev31/32/33 bootstrap은 50-iteration 진단 경로이므로 숫자만 300으로 바꾸거나 legacy rev26 eligibility를 재사용하지 않는다.
6. 정량 qualification 후 다중 seed, held-out 조건, 5/10도부터 경사·횡단·상이한 발 마찰을 순차 확장한다. 초기 자세, 성공 지속 시간, timeout, auto-reset 제외, 관절/접촉 안전 기준을 먼저 고정한다. 아직 이 후속 학습을 했다고 보고하지 않는다.

## 6. 영상·checkpoint: Git에 있는 것과 옮겨야 할 것

Git에는 코드, 설정, 정량 JSON, GIF, PNG가 있다. **checkpoint, TensorBoard 원시 로그, 원본 MP4, Isaac 설치본은 없다.** 다른 PC에서 문서 편집은 clone만으로 가능하지만 기존 정책 재생에는 아래 로컬 자산이 필요하다.

### 공개 크기 미디어와 정본 sidecar

- [rev32 GIF](media/g009/R0/diagnostic/g009_5_r0_diag_rev32_01_prone_hd_s42.gif), [PNG](media/g009/R0/diagnostic/g009_5_r0_diag_rev32_01_prone_hd_s42_still.png), [capture](../reports/runs/g009_5_r0_diag_rev32_01_prone_hd_s42.json), [media sidecar](../reports/runs/g009_5_r0_diag_rev32_01_prone_hd_s42_media.json)
- [rev33 GIF](media/g009/R0/diagnostic/g009_5_r0_diag_rev33_01_prone_hd_s42.gif), [PNG](media/g009/R0/diagnostic/g009_5_r0_diag_rev33_01_prone_hd_s42_still.png), [capture](../reports/runs/g009_5_r0_diag_rev33_01_prone_hd_s42.json), [media sidecar](../reports/runs/g009_5_r0_diag_rev33_01_prone_hd_s42_media.json)

원본 MP4는 두 실험 모두 native 1920×1080, H.264 CRF18, 30 fps, 240 frames / 8 s다. 실제 50 Hz control state를 30 Hz로 샘플링했고 frame 복제·보간으로 fps를 꾸미지 않았다. GIF는 960×540, 15 fps, 60 frames / 4 s, 최대 frame duration 70 ms이며 각각 10 MiB 미만이다. `DIAGNOSTIC / NOT QUALIFIED` 표시를 유지한다. 이 영상은 실패 진단이지 성공 showcase가 아니다.

이 PC의 원본 경로:

```text
C:\Users\LIMMM\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\g009_5_r0_diag_rev32_01_prone_hd_s42.mp4
C:\Users\LIMMM\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\g009_5_r0_diag_rev33_01_prone_hd_s42.mp4
```

SHA-256:

```text
rev32 MP4: 9afa1d98cc7a996c20fba901d94a3e104ac4fd9f03ebbd5806c175806af79237
rev33 MP4: b42f36066bb5362b1caae3e0fd240a319afd3438bbe60100c01bba99547ab07e
```

진단 checkpoint 경로:

```text
C:\Users\LIMMM\IsaacLab\logs\rsl_rl\g009_recover_r0\2026-09-27_09-09-11_go2_flat_g009_r0_rev32_rear_damping_s42_20260927_0910\model_49.pt
C:\Users\LIMMM\IsaacLab\logs\rsl_rl\g009_recover_r0\2026-09-27_09-27-12_go2_flat_g009_r0_rev33_front_damping_s42_20260927_0931\model_49.pt
```

```text
rev32 checkpoint: a5edc67a2369f0005c6c89dea4473cc64691f0c0fce912d868bad832d480d34d
rev33 checkpoint: 129dcf0e99c3b7a2037405be318a3a9e7c42e7474aaf2dbebf0583262c6cc57a
```

다른 PC로 옮길 때는 checkpoint만이 아니라 해당 run 디렉터리의 params / events / 로그도 보존한다. 경로의 사용자 홈 부분은 새 PC에 맞추되 정본 evidence JSON을 새 경로로 덮어쓰지 않는다. `Get-FileHash -Algorithm SHA256`로 수신 파일을 검증한다. 이 인계 작업에서 외부로 모델이나 영상을 전송하지는 않았다.

G006 재생 자산은 [queue state](../reports/runs/g006_queue_state.json)와 [visual evidence](G006_VISUAL_EVIDENCE.md)의 run / checkpoint 경로·hash를 확인한다. 실패한 rev32/33을 다시 학습해서 자산을 대체하지 않는다.

## 7. 다른 PC / 다른 세션의 시작 절차

### 7.1 공통: 읽기 전용 확인부터

1. 해당 호스트의 AGENTS.md를 읽고 실제 repository / origin / branch / dirty 상태를 확인한다. 기존 폴더가 있으면 새 clone으로 덮어쓰지 않는다. 새 PC에서는 `mmporong/isaac-walk-rl`의 `main`을 별도 빈 경로에 clone하고 작업 branch를 만든다.
2. 이 문서, README, 최신 RUN_NOTES, G006 원고, G009 최신 판정을 읽는다. 포트폴리오 편집이 요청되면 해당 호스트의 portfolio 스킬과 대상 저장소 계약을 따른다. tmux 지원이나 OMX 권한 증명을 추정하지 않는다.
3. 문서 편집과 GPU 실험을 구분한다. 문서 정리만 한다면 Isaac 재설치나 재학습은 필요 없다. GPU 실험을 한다면 고정 버전과 [환경 매니페스트](../reports/environment_manifest.json), [환경 수집 도구](../scripts/collect_environment.ps1)를 확인한다. 새 호스트 실측은 별도 기록으로 보존하고 과거 실측을 대체하지 않는다.
4. checkpoint·원본 MP4가 없는 경우 그 사실을 보고한다. Git clone만으로 모든 자산이 옮겨졌다고 주장하지 않는다.

### 7.2 두 세션 GPU 공유

이 PC의 공용 경로는 `C:\Users\LIMMM\CodexSignals`다. 관련 도구는 source repo 밖에 있다. 다른 PC에서는 `%USERPROFILE%\CodexSignals`에 **README.md, Use-GpuLease.ps1, Get-GpuLeaseStatus.ps1**을 함께 확보한다. 이전 PC의 live lock / JSON 세션 상태는 복사하지 않는다.

- G009와 RL 세션의 모든 Isaac / Kit / CUDA 실행을 `Use-GpuLease.ps1`로 감싼다. OS 독점 FileStream 핸들이 실제 lock이며 JSON은 관찰 정보다.
- 시작·종료에 GPU/CPU 및 lease 상태를 다시 확인한다. `gpu-ready.json`의 존재만으로 완료를 판단하지 말고 leaseId / completedUtc / state와 현재 프로세스를 확인한다.
- 중첩 lease, 래퍼 밖 실행, 다른 세션 프로세스 강제 종료를 하지 않는다. exit 73은 bounded 대기 timeout, 74는 launch 실패, 75는 GPU probe 실패다.
- 도구가 없는 새 PC에서는 GPU 공유가 준비됐다고 가정하지 않는다. 검증된 래퍼를 확보하기 전 concurrent GPU 작업을 시작하지 않는다.
- GPU 사용률 100% 자체가 목표가 아니다. 처리량·VRAM·온도·OOM/device-lost와 종료 후 회복을 검증한다. 보호 기준을 낮추지 않는다.

CPU-only 자체 검증과 상태 조회 예시:

```powershell
& "$HOME\CodexSignals\Use-GpuLease.ps1" -SelfTest
& "$HOME\CodexSignals\Get-GpuLeaseStatus.ps1" -AsJson
```

### 7.3 최소 로컬 검증

이 PC의 worktree를 재개하는 예시다. 다른 PC에서는 확인한 clone 경로로 이동한다.

```powershell
cd "$HOME\worktrees\isaac-walk-rl-g009-r0"
git status --short --branch
git log -1 --oneline
git remote -v
git diff --check
pwsh -NoProfile -File .\scripts\validate_repository.ps1
python -m pytest -q tests/test_g009_r0_rev32_damping.py tests/test_g009_r0_rev33_damping.py tests/test_g009_r0_rev32_hd.py tests/test_g009_r0_rev31_training_attribution.py
```

네 파일의 대상 테스트는 직전 구현 검증에서 39 PASS였다. 새 호스트의 실행 결과는 별도로 확인한다. CPU 테스트용 Python과 Isaac GPU 실행용 bundled Python을 혼동하지 않는다. 의존성이 없다면 준비된 테스트 환경을 사용하고, pytest 때문에 Isaac runtime을 임의로 업데이트하지 않는다.

### 7.4 증거 파일의 변경 방지

`.gitattributes`에는 실측 source와 JSON의 byte provenance를 보존하는 규칙이 있다. rev32/33 JSON은 raw Windows CRLF를 유지하고 관련 source도 실행 시 hash와 정합한다. formatter, 일괄 개행 정규화, 속성 삭제로 hash 연결을 훼손하지 않는다. 이전 후 sidecar의 hash와 실제 파일을 대조한다. 향후 변경은 새 실행 기록에 연결하고 과거 증거를 현재 source로 재서명하지 않는다.

## 8. 포트폴리오로서 다음에 완성할 것

1. G006을 주 사례로, 문제 설정 → 폐루프 구조 → PPO·관측·reward·curriculum → 동일 조건 비교 → 정량 결과 → 개선을 주장할 수 없는 이유 → 다음 가설 순서로 구성한다.
2. G005와 G008을 보조 증거로 붙인다. 성공 조건뿐 아니라 질량·혼합 마찰·불규칙 지형의 미달 조건도 짧게 명시한다.
3. 영상/GIF/그림은 성능 JSON과 checkpoint로 연결한다. 걷는 영상만으로 RL 구현이나 강건성을 증명했다고 하지 않는다. 원본 MP4는 로컬 전용, 공개는 GIF/PNG로 한다.
4. G009를 포함한다면 복구 동작 성공이 아니라 안전 기준으로 후보를 기각하고 역학 진단을 설계한 연구로 구분한다. 주 사례를 실패 영상으로 대체하지 않는다.
5. 소스 측 구성만 한다면 로컬 문서·링크·수치를 검증한다. 사이트 편집·공개를 재개한다면 실제 portfolio-v2 / Garden의 규칙, origin, branch, build, preview를 확인하고 해당 작업의 공개 권한을 따른다. 세션 이전 자체는 공개 지시가 아니다.

완료 조건은 채용 담당자가 무엇을 바꿨고, 무엇을 고정했으며, 어떤 조건에서 무엇을 측정했고, 무엇이 미달인지 영상과 수치로 추적할 수 있는 상태다. 실물이 없는 점을 보완하려고 sim-to-real 실증을 만들어 내지 않는다. G009의 완료 조건은 별개이며, 안전 gate와 다중 자세 정량 복구 검증을 통과했을 때만 복구 성공으로 다룬다.

## 9. 새 세션에 전달할 프롬프트

```text
Isaac Lab 사족보행 작업의 후속 세션이다. 실물 로봇은 만들지 않으며 시뮬레이션 전용 Physical AI 포트폴리오가 목적이다.
소스: https://github.com/mmporong/isaac-walk-rl
먼저 실제 AGENTS.md, git origin/branch/status와 docs/HANDOFF_SIM_PORTFOLIO.md를 확인하고 기존 변경·증거를 보존해.
검증된 결과의 기준 commit은 074e6b6ca74492782330079725380984236db296이다. 이후 인계 문서 commit도 확보해.
G006 주 사례, G005 reward ablation과 G008 방향/마찰/질량 보조 사례를 정리하는 것이 우선이다.
G006은 6480 push trials에서 99.5370% 대 99.5988%이며 CI가 0을 포함해 개선 우위를 주장할 수 없다. 뒤집힌 자세에서 일어서기 실험이 아니다.
G009 rev32/rev33은 50-iteration smoke 안전 FAIL이고 prone HD 재생도 timeout이다. 복구 성공이나 자격 취득으로 보고하지 마.
G009를 재개한다면 사건 직전 200Hz physics-substep 이력을 bounded하게 측정하는 새 진단을 사전 등록하고, 한 축 가설→50 smoke→안전 PASS 때만 별도 full300 계약/네 자세 평가로 진행해.
GPU 명령은 검증된 CodexSignals lease로 RL 세션과 배타적으로 실행해. 다른 세션을 kill하거나 보호 기준을 낮추지 마.
고정 runtime은 Windows Isaac Sim4.5.0/Isaac Lab2.1.1/RSL-RL2.3.3 PPO다. checkpoint와 원본 MP4는 Git 밖에 있으므로 존재/hash를 확인해.
처음에는 현재 상태·수행 범위·부족한 자산을 한글로 짧게 보고하고 요청 범위의 작업과 검증을 진행해.
새 공개·배포나 goal 등록을 이 인계만으로 추정하지 말고 복구 미성공 판정을 유지해.
```
