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

### G005 Go2 flat 보상 ablation

- baseline과 `no_torque`, `no_action_rate`, `no_feet_air_time`을 4096 environments, 300 iterations, seeds 42/43/44로 실행했다. 한 variant에서 reward 하나만 0으로 바꿨다.
- 학습 12/12와 고정 평가 12/12를 완료했으며 failed job은 0이다. 총 학습 wall time은 105.4분, 실행별 평균 처리량의 평균은 60,238.2 steps/s, 최대 peak VRAM은 4,822 MiB였다.
- 모든 학습은 exit 0, 299/300 iteration, `model_299.pt`, TensorBoard event, GPU 측정·회복을 확인했다. 모든 평가는 exit 0, 정상 App close, fatal log 0, GPU·프로세스 회복을 확인했다.
- checkpoint SHA256, TensorBoard 디렉터리, 학습·평가 명령은 `reports/runs/g005_reward_ablation_state.json`의 12개 job에 보존했다. strict summary는 `reports/runs/g005_reward_ablation_summary.json`이다.
- strict 결합 해시는 canonical config `3e8455a9efba77f67b2ac436d5eef41421dfeac10f9e67ab9620c6775b6c2576`, config file `5f5cf8127424460c4b2555d28969e85d9664589337ba4edf71dd9ed72112cdde`, protocol `4ff2f271ed7e217966ed7e09a1f0de5bfacc056020721623af6211d264835d9c`, evaluation script `60b22beaf6189ae0f3bc0aeaa98f264a7ffe853f4de2c5b49e266a2716bd7965`다.
- checkpoint SHA256(seeds 42/43/44): baseline `31a9ed90…c84bd` / `9aed9000…aad9` / `9bf1963d…ce45`, no_torque `4d792707…d745` / `7c87cb35…0871` / `561e87fe…5ec`, no_action_rate `5bc9581e…1c95` / `f2fbf79a…00ba` / `853bf64b…753`, no_feet_air_time `1bcc0c53…099` / `baa0ee9f…9ae0` / `ba84a2b5…7822`다. 전체 해시는 state JSON을 기준으로 한다.
- 평가 계약은 seed 20260824, 26 commands × 10 environments, 20초다. training reward는 variant별 정의가 달라 직접 비교하지 않고, 고정 평가의 추종 RMSE·torque·power proxy·action 변화량·넘어짐률만 비교했다.
- 핵심 paired 결과: no_torque는 torque `+11.92%`, power `+5.48%`, action-rate `+41.00%`; no_action_rate는 linear RMSE `+10.35%`, power `+12.15%`, action-rate `+123.36%`; no_feet_air_time은 yaw RMSE `-6.62%`, torque `-5.31%`, power `-9.53%`, first-contact count `+77.58%`, raw feet-air-time `-75.94%`였다.
- no_torque 넘어짐 2건은 seed 43의 단일 측면 명령에서만 발생했다. 전체 780 trials 중 2건, 3-seed 평균 fall rate 0.2564%이며 절대 2% 임계값에는 미달했다.
- 표본은 variant당 `n=3`이고 flat·20초 평가에 한정된다. power는 전기 에너지가 아닌 시뮬레이션 proxy이며, early fall 이후 상태를 제외하므로 연속 지표에 조건부 표본 편향이 있을 수 있다.
- 전체 평균표, seed 방향 일관성, 실용 임계값과 한계는 `docs/G005_REWARD_ABLATION.md`에 기록했다.
- G006 실행 계약: official `UnitreeGo2RoughEnvCfg` baseline과 동일 rough terrain curriculum·공통 official DR을 유지한 채 `events.push_robot`만 변경한 push curriculum을 고정 protocol로 비교한다. 추적 오차·낙상률·에너지 proxy·회복률을 함께 보며, final production과 strict summary는 아직 대기 중이다.

### G007 RBQ 외부 자산 호환성 사전조사

- 2026-08-24에 RBQ v1.20.0 tag object `741ce5733dcd7c0babec663bb7e1afbc02a776ca`와 source commit `68bc33b77719d357b4323fb88549efd905caf721`을 고정했다.
- `rbq_sdk/ros2/src/rbq_description/` 아래 URDF 1개, `package.xml` 1개, STL 6개 등 8개 blob의 경로·크기·Git blob SHA-1 inventory를 manifest에 기록했다.
- GitHub repository API의 detected license는 `null`이지만 이는 저장소 전체 라이선스 미감지를 뜻할 뿐 무허가 또는 이용 금지를 증명하지 않는다. `package.xml`의 Apache-2.0 선언이 asset blob에 적용되는 범위와 로컬 처리·재배포 권한은 확인되지 않았다.
- 공식 Isaac Lab v2.1.1, v2.3.2, 조사 시점 main 고정 소스에서 대상 match가 없었다. G007을 공식 구현 이식이 아닌 `external_custom_compatibility_spike`로 보정했다.
- 검증기는 `license_scope_unresolved` blocker를 fail-closed로 보고한다. `--expect-blocked`는 exit 0, `--require-ready`는 exit 3을 재현했다.
- 자산 byte 다운로드·해시 검증·URI/topology 분석·converter·fixed-base smoke는 실행하지 않았다. topology의 link/joint/mesh count도 미확정이다.
- `python -m pytest tests/test_g007_rbq_gate.py -q` 결과 46 tests PASS, 코드 검토 판정은 APPROVE였다.
- 보고서 file SHA-256 `8cace17b61c944c1395bd42bff81c0cdbd8c39e8b041b0b2039f382983d8927d`, manifest canonical SHA-256 `93ec6cfa7f06d7f2c8b43ac5f057aa2e5b09767a11c515ef333b1dcac799edbf`, validator SHA-256 `28040254c014e6de99ab99dac578eee9a0ad55e94353cb6fad5d14fe75bfc36b`이다.
- 이 blocker는 프로젝트 브리프가 허용한 G007 완료 경로다. G006 production과 전체 ultragoal 완료를 뜻하지 않는다.
- 상세 판정과 해제 조건은 `docs/G007_RBQ_COMPATIBILITY_SPIKE.md`에 기록했다.
