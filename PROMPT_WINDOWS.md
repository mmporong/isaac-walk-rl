# Windows 실행 계약

## 고정 전제

- Windows 11 네이티브
- Isaac Sim `4.5.0`: `E:\IsaacSim\isaac-sim-4.5.0`
- Isaac Lab `v2.1.1`: `%USERPROFILE%\IsaacLab`
- Isaac Lab commit: `90b79bb2d44feb8d833f260f2bf37da3487180ba`
- Python: Isaac Sim 번들 `3.10.x`
- `rsl-rl-lib==2.3.3`, PPO만 사용
- 상태 기반 headless 학습
- 프로젝트 저장소: `%USERPROFILE%\isaac-walk-rl`
- ROS 2, WSL2, 비전 관측은 설치하거나 추가하지 않음

Isaac Lab 원본은 프로젝트 저장소 밖에 두며 직접 수정하지 않는다. 커스텀 변경은 이 저장소에서 오버라이드 또는 확장 패키지로 관리한다.

## 0단계 — 설치와 등록 검증

PowerShell 7.x(`pwsh`)에서 실행한다. 현재 검증 버전은 7.6.5이며, Windows PowerShell 5.1은 지원 검증 대상이 아니다.

```powershell
git clone --branch v2.1.1 https://github.com/isaac-sim/IsaacLab.git "$HOME\IsaacLab"
git -C "$HOME\IsaacLab" rev-parse HEAD
New-Item -ItemType Junction -Path "$HOME\IsaacLab\_isaac_sim" -Target "E:\IsaacSim\isaac-sim-4.5.0"
cd "$HOME\IsaacLab"
& .\isaaclab.bat -i rsl_rl
```

DoD:

- HEAD가 고정 commit과 일치한다.
- Isaac Lab 패키지 import가 성공한다.
- `rsl-rl-lib` 버전이 2.3.3이다.
- CUDA 사용 가능 여부와 GPU 이름을 기록한다.
- 등록 목록에서 대상 task ID 4개를 확인한다.

## 1단계 — ANYmal-C 관문

설치 smoke:

```powershell
cd "$HOME\IsaacLab"
& .\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Velocity-Flat-Anymal-C-v0 --num_envs 64 --max_iterations 50 --headless
```

smoke가 끝난 후 같은 태스크의 300-iteration baseline을 수행한다. 환경 수는 자원 측정 결과에 따라 고정한다.

DoD: 정상 종료, 실행 명령, seed, peak VRAM, steps/s, TensorBoard·체크포인트 경로, 핵심 지표 기록.

## 2단계 — Go2 flat 자원 사다리

태스크는 `Isaac-Velocity-Flat-Unitree-Go2-v0`를 사용한다. 64→256→512→1024→2048 environments 순서로 같은 seed와 짧은 budget을 실행한다. 4096은 직전 단계 peak VRAM이 전체의 80% 이하이고 OOM·PhysX fallback·비정상 종료가 없을 때만 실행한다.

DoD: 환경 수별 peak VRAM, 평균 steps/s, wall time, 성공/실패 원인을 표로 기록. MuJoCo 51k 수치와 비교할 때 simulator·hardware·측정 정의가 다름을 명시.

## 3단계 — 보상 ablation

baseline을 유지한 채 다음 항목을 한 번에 하나씩 변경한다.

- `dof_torques_l2`
- `action_rate_l2`
- `feet_air_time`

각 변형은 같은 학습 budget과 3개 이상의 seed를 사용한다. 추적 오차, 에너지 proxy, 넘어짐률을 함께 보고한다.

DoD: 설정 diff, 명령, seed별 결과, 평균·분산, TensorBoard 근거, 해석 가능한 비교표.

## 4단계 — official rough baseline 고정

`Isaac-Velocity-Rough-Unitree-Go2-v0`의 official `UnitreeGo2RoughEnvCfg`를 baseline으로 사용한다. baseline과 이후 변형은 동일한 rough terrain curriculum과 official domain randomization을 공통으로 유지한다. 기본 환경에 이미 존재하는 terrain curriculum과 event를 먼저 분석하고 중복 구현하지 않는다.

DoD: official rough baseline의 고정 설정, 적용된 공통 randomization 범위, terrain curriculum 진행 증거, official rough baseline 대비 추적 오차·낙상률·에너지 proxy의 기술통계 비교. flat→rough 또는 domain randomization 단독 인과효과를 주장하지 않는다.

## 5단계 — 외란 회복

학습 전 평가 프로토콜을 먼저 고정한다.

- push 방향·크기·시점 grid
- 회복 제한 시간 `T`
- 정상 base height, roll/pitch, velocity error 범위
- 회복 뒤 무낙상 유지 시간 `H`
- seed와 episode 수

official rough baseline과 동일 rough·공통 official DR 조건에서 `events.push_robot`만 변경한 push curriculum 변형에 같은 protocol을 적용한다. G006의 비교 축은 이 한 가지이며, flat 결과는 맥락 자료로만 사용한다.

DoD: 회복 성공 분자/분모, 회복률, Wilson 95% 신뢰구간, 조건별 표와 실패 사례.

## 6단계 — RBQ 외부 자산 호환성 사전조사

RBQ v1.20.0의 source commit과 `rbq_sdk/ros2/src/rbq_description/` 아래 8개 자산 blob을 manifest에 고정한다. 공식 Isaac Lab v2.1.1·v2.3.2·조사 시점 main에는 대상 구현이 없으므로 기존 구현의 이식을 전제하지 않는다.

`rbq_description/package.xml`의 Apache-2.0 선언이 URDF·STL에 적용되는 범위와 로컬 처리 권한이 확인되지 않으면 자산을 다운로드·변환하지 않고 차단 보고서를 재현한다.

```powershell
cd "$HOME\isaac-walk-rl"
python .\scripts\validate_rbq_assets.py --manifest .\configs\g007_rbq_asset_manifest.json --expect-blocked --report .\reports\g007_rbq_compatibility_spike.json
```

DoD: source·blob inventory·라이선스 근거와 검증기 해시를 고정한다. 허가가 없으면 `license_scope_unresolved` blocker, 재현 명령, 해제에 필요한 권한 범위를 기록한다. 허가가 확인된 뒤에만 byte hash → URI/topology → converter → fixed-base smoke를 별도 수행한다.

## 실행 기록과 Git

- 이름: `<robot>_<terrain>_<change>_s<seed>_<YYYYMMDD-HHmm>`
- 모든 실행을 `%USERPROFILE%\isaac-walk-rl\RUN_NOTES.md`에 기록한다.
- 로그·체크포인트·영상은 Git에 넣지 않고 경로·해시·요약 수치만 기록한다.
- 단계별 검증 PASS 후 변경 파일만 스테이징해 한글 Conventional Commit으로 커밋한다.
- 검증된 커밋만 지정된 원격 branch에 push한다.
