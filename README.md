# Isaac Walk RL

Windows 네이티브 환경에서 Isaac Sim 4.5와 Isaac Lab 2.1.1을 사용해 사족보행 PPO 실험을 재현하고, 보상·지형·외란 회복을 한 축씩 확장하는 프로젝트입니다.

## 고정 스택

| 구성 | 고정값 | 상태 |
| --- | --- | --- |
| 운영체제 | Windows 11 네이티브 | 확정 |
| Isaac Sim | 4.5.0 binary | 로컬 설치·Python smoke 확인 |
| Isaac Lab | v2.1.1 / `90b79bb2d44feb8d833f260f2bf37da3487180ba` | 소스 설치 대기 |
| Python | Isaac Sim 번들 3.10.x | 로컬 3.10.15 확인 |
| RL | RSL-RL PPO / `rsl-rl-lib==2.3.3` | 설치 대기 |
| 관측 | 상태 기반, headless | 확정 |

이 프로젝트는 `Ubuntu 24.04/Jazzy` 기본 로보틱스 환경의 예외입니다. RL 학습 자체에는 ROS 2나 WSL2가 필요하지 않으므로 설치하지 않습니다.

## 검토 후 보정한 전제

- Isaac Lab 2.1.1은 `pip install isaaclab==2.1.1` 대상이 아닙니다. 공식 태그 소스를 저장소 밖에 clone해 설치합니다.
- Isaac Lab 2.2는 Sim 4.5에서 무조건 사용할 수 없는 버전이 아닙니다. 이 프로젝트는 재현성을 위해 2.1.1을 고정합니다.
- RTX 3060 12GB에서 2048/4096 environments가 된다는 보장은 없습니다. 64부터 단계적으로 올리며 peak VRAM과 steps/s를 실측합니다.
- Go2를 “가장 많이 쓰이는 모델”이라고 단정하지 않습니다. ANYmal-C 공식 baseline을 관문으로 삼고, Isaac Lab 내장 Go2 태스크를 심화 대상으로 사용합니다.
- RBQ는 공개 URDF와 현행 Isaac Lab 구현이 있지만, 현행 코드는 Sim 5.1 / Lab 2.3.2용입니다. 마지막 단계는 4.5 / 2.1.1 backport 가능성 검증입니다.

## 실행 순서

1. Isaac Lab v2.1.1 소스 설치 및 등록 태스크 검증
2. ANYmal-C flat 50-iteration smoke와 300-iteration baseline
3. Go2 flat 환경 수별 VRAM·steps/s 측정
4. 세 보상 항목의 one-factor ablation
5. Go2 rough, terrain curriculum, domain randomization
6. 고정 프로토콜 기반 외란 회복률 비교
7. RBQ URDF 및 공식 2.3.2 구현의 2.1.1 backport spike

구체적인 명령과 단계별 완료 조건은 `PROMPT_WINDOWS.md`, 측정 상태는 `docs/VALIDATION_MATRIX.md`, 모든 실행 기록은 `RUN_NOTES.md`에서 관리합니다.

## 저장소 경계

- 이 저장소: 커스텀 코드, 설정, 재현 스크립트, 문서, 정량 결과표
- 저장소 밖: `%USERPROFILE%\IsaacLab`, `E:\IsaacSim\isaac-sim-4.5.0`, 학습 로그, 체크포인트, 영상, 생성 자산

## 환경 매니페스트와 저장소 검증

PowerShell에서 다음 명령을 실행하면 사용자명이나 인증정보를 기록하지 않고 Git, GPU, Isaac 설치 상태와 현재 commit 상태를 `reports/environment_manifest.json`에 갱신합니다.

```powershell
cd "$HOME\isaac-walk-rl"
.\scripts\collect_environment.ps1
.\scripts\validate_repository.ps1
```

Isaac Lab을 다른 위치에 설치한 경우 `-IsaacLabPath` 인자로 경로를 지정합니다. 사용자 홈 아래 경로는 매니페스트에서 `%USERPROFILE%`로 치환됩니다.
