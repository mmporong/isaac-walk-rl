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

### G004 Go2 flat scale ladder

- `Isaac-Velocity-Flat-Unitree-Go2-v0`, seed 42, rung당 10 iterations로 64→256→512→1024→2048 environments를 순차 실행했다. 각 rung은 direct bundled Python, headless, 고유 run name을 사용했다.
- 2048 결과가 PASS, GPU 측정 complete, 종료 후 baseline 회복, peak 4,058 MiB로 총 12,288 MiB의 80% 기준(9,830.4 MiB) 이하인 것을 확인한 뒤에만 4096을 실행했다.
- 모든 rung이 exit 0, 9/10 iteration, TensorBoard event, `model_9.pt`, SHA256, 오류 부재, GPU 회복 조건을 통과했다.

| envs | wall(s) | peak VRAM MiB (%) | 전체 mean / median steps/s | 첫 iteration 제외 median steps/s | final reward / length | 판정 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 51.150 | 3,316 (26.99%) | 725.8 / 760 | 761 | -2.95 / 118.05 | PASS·safe |
| 256 | 37.844 | 3,334 (27.13%) | 4,589.9 / 4,707 | 4,709 | -3.33 / 144.80 | PASS·safe |
| 512 | 29.012 | 3,498 (28.47%) | 9,325.3 / 9,834 | 10,008 | -3.53 / 158.87 | PASS·safe |
| 1024 | 35.378 | 3,678 (29.93%) | 13,313.9 / 14,083 | 14,592 | -4.16 / 196.56 | PASS·safe |
| 2048 | 35.141 | 4,058 (33.02%) | 30,881.5 / 32,581 | 32,861 | -5.34 / 219.48 | PASS·safe |
| 4096 | 43.421 | 4,822 (39.24%) | 48,103.6 / 48,852 | 50,680 | -3.93 / 229.02 | PASS·safe |

- `highest_operational=4096`, `highest_safe=4096`이다. 이는 10-iteration 상태 기반 headless 실행의 결과이며 장기 학습 안정성이나 최적 environment 수를 뜻하지 않는다.
- 사용자 제공 MuJoCo 51k steps/s는 환경, 물리 설정, rollout 길이와 측정법이 통제된 동일 조건 벤치마크가 아니므로 직접 비교할 수 없다. summary의 비율은 참고 계산일 뿐 우열의 근거가 아니다.
- 재현 명령: `cd "$HOME\isaac-walk-rl"` 후 `.\scripts\run_scale_ladder.ps1`. 상세 run JSON과 체크포인트 SHA256는 `reports/runs/g004_go2_scale_summary.json`에 있다.
