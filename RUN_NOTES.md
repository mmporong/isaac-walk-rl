# 실행 기록

## 기록 규칙

각 실행마다 아래 항목을 채운다. 실패 실행도 삭제하지 않는다.

- 실행 이름
- 날짜·시간과 Git commit
- task ID, robot, terrain
- 명령과 설정 diff
- seed, environment 수, iteration 수
- GPU, peak VRAM, 평균 steps/s, wall time
- checkpoint·TensorBoard·영상의 로컬 경로와 필요한 경우 해시
- 추적 오차, 에너지 proxy, 넘어짐률, 회복률 등 단계별 지표
- 판정, 실패 원인, 다음 가설

## 호스트 사전 점검

| 항목 | 확인값 | 판정 |
| --- | --- | --- |
| Isaac Sim | `E:\IsaacSim\isaac-sim-4.5.0` | 설치 확인 |
| 번들 Python | 3.10.15 | 확인 |
| Isaac Sim Python smoke | bundled `python.bat` 직접 실행으로 headless AppLauncher와 Gym registry 초기화 | PASS |
| Isaac Lab v2.1.1 | `v2.1.1` / `90b79bb2d44feb8d833f260f2bf37da3487180ba` | 설치·commit 확인 |
| RSL-RL 2.3.3 | `rsl-rl-lib==2.3.3` | 설치·버전 확인 |
| CUDA PyTorch | `torch==2.7.0+cu128`, CUDA available | RTX 3060 인식 확인 |
| GPU | RTX 3060 / 12288 MiB / driver 610.62 | manifest 수집 확인 |

환경 매니페스트는 `scripts/collect_environment.ps1`로 갱신하고 `reports/environment_manifest.json`에 보관한다. 저장소 경계 검증은 `scripts/validate_repository.ps1`을 실행하며, 실패 시 학습이나 커밋 단계로 진행하지 않는다.

## 실행 목록

### G002 Isaac Lab 설치·등록 검증

- 공식 `v2.1.1` 단일 태그를 `%USERPROFILE%\IsaacLab`에 clone했다.
- `_isaac_sim` Junction을 `E:\IsaacSim\isaac-sim-4.5.0`으로 연결했다.
- 첫 `isaaclab.bat -i rsl_rl` 실행은 `flatdict==4.0.1`의 isolated build 환경에서 `pkg_resources`를 찾지 못했다. 번들 `setuptools==70.3.0`은 변경하지 않고 `flatdict==4.0.1 --no-build-isolation`만 선설치한 후 공식 설치 명령을 재실행해 종료 코드 0을 확인했다.
- Python 3.10.15, PyTorch 2.7.0+cu128, CUDA available, RTX 3060, RSL-RL 2.3.3, Isaac Lab package 0.41.3을 확인했다.
- 실제 headless AppLauncher 초기화 뒤 ANYmal-C flat/rough와 Unitree Go2 flat/rough 네 태스크가 Gym registry에 등록됨을 확인했다.
- `isaaclab.bat -p` wrapper는 정상 결과 출력 뒤 종료 코드 1을 반환했지만, 동일 명령을 bundled `python.bat`로 직접 실행하면 정상 shutdown과 종료 코드 0을 반환했다. v2.1.1 `isaaclab.bat`가 nested batch를 `call` 없이 실행하는 Windows wrapper false-negative로 판정하고 보고서 경고에 보존했다.
- `pip check`는 Isaac Sim extension별 `pip_prebundle` metadata와 Isaac Lab의 정확한 `starlette==0.45.3` 고정 때문에 종료 코드 1을 반환한다. 핵심 RL imports, headless AppLauncher, task registry는 PASS이므로 패키지를 수정하지 않고 비차단 경고로 보고서에 보존한다.
- 재현 명령: `cd "$HOME\isaac-walk-rl"` 후 `.\scripts\verify_isaaclab.ps1`.

### G003 ANYmal-C flat 재현

- v2.1.1 소스에서 지원 인자 `--task`, `--num_envs`, `--max_iterations`, `--seed`, `--run_name`, `--headless`를 확인했다.
- flat runner 기본값은 300 iterations, 24 steps/environment, save interval 50이며 로그는 `%USERPROFILE%\IsaacLab\logs\rsl_rl\anymal_c_flat\<timestamp>_<run_name>`에 생성된다.
- 64 environments, seed 42로 1-iteration probe → 50-iteration smoke → 300-iteration baseline을 순서대로 실행했고 모두 direct Python exit 0, 요청 iteration, TensorBoard, 최종 checkpoint, 오류 부재 조건을 통과했다.
- 1차 probe 학습은 exit 0이었지만 Sim warning이 stdout 앞에 섞여 로그 경로 정규식이 실패했고 WDDM의 per-process memory가 `N/A`라 하네스 판정만 false였다. 로그 경로 패턴을 완화하고 GPU 전체 `memory.used` 샘플로 전환한 뒤 새 run name으로 probe를 재검증했다.
- 성공한 세 실행 모두 VRAM 889 MiB baseline에서 3,970 MiB peak를 기록하고 종료 후 baseline으로 회수됐다.
- 최종 scalar는 TensorBoard event accumulator를 bundled `python.bat` 환경에서 교차 확인했다. 상세 수치와 체크포인트 SHA256는 `reports/runs/g003_anymal_summary.json`에 있다.
