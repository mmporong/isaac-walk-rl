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

### G008 방향 명령·마찰·링크 질량

- 의존성 번호는 `G008-1` command smoke에서 `G008-2` command PPO로 이어진 뒤 두 갈래로 나뉜다. 마찰·도로 분기는 `G008-3` friction S1, `G008-5` 주기 혼합 마찰, `G008-7` 비주기 도로, `G008-8` G0/T1 curriculum 순서다. 링크 질량 분기는 `G008-4` leg-mass S1, `G008-6` 링크 그룹 질량 민감도 순서다. 같은 묶음에서 수행한 두 분기의 시각 선후는 번호로 단정하지 않는다. 각 단계 직후 만든 자료는 `G008-9` 증거 index에서 모아 본다. `S1`, `G0`, `T1`은 별도의 protocol stage ID다.
- 2026-08-26에 G008을 command, friction, leg-mass 세 태스크 축으로 나눴다. friction과 leg-mass는 같은 환경에 동시에 넣지 않았다.
- command sampler는 80% exact primitive와 20% continuous SE(2) 명령을 섞는다. exact primitive는 전진 `[0.6,0,0]`, 후진 `[-0.4,0,0]`, 좌회전 `[0,0,0.5]`, 우회전 `[0,0,-0.5]`, 정지 `[0,0,0]`다. command resampling은 4~6초, heading controller는 비활성화했다.
- friction은 발 collision shape만 대상으로 S1 `μ_s=0.72~0.88`, `μ_d=0.52~0.68`; S2 `0.62~1.00`, `0.42~0.78`; S3 `0.50~1.25`, `0.30~1.00`으로 등록했다.
- leg-mass는 hip/thigh/calf/foot 16개 body를 환경·body별 독립 uniform scale로 바꾸며 inertia를 mass ratio로 다시 계산한다. S1 `0.95~1.05`, S2 `0.90~1.10`, S3 `0.80~1.20`이다.
- command, friction S1, leg-mass S1을 각각 64 environments, 1 iteration, seed 42로 실행했다. 세 run 모두 exit 0, model_0.pt, TensorBoard, fatal pattern 0, GPU 측정·회복을 통과했다. peak VRAM은 5,259/5,259/5,260 MiB, wall time은 20.649/18.602/18.577초였다.
- G006 baseline seed 42의 `model_1499.pt`를 G008 command task에 넣고 64 environments, 방향당 16개, 250 steps, warmup 50 steps, seed 20260826으로 nominal 평가했다. 64개 모두 생존했고 네 방향의 평균 속도 부호가 명령과 일치했다.
- 좌회전은 linear/yaw RMSE `0.0921 m/s`/`0.1412 rad/s`, 우회전은 `0.0459`/`0.1300`으로 gate를 통과했다. 전진은 `0.2151`/`0.1550`, 후진은 `0.1297`/`0.1138`로 속도 기준을 만족했지만 pitch max가 각각 `0.6937`, `0.6046 rad`라 자세 gate에서 실패했다.
- 새 command distribution을 처음부터 `1,024 env × 300 iterations × seed 42`로 학습한 run은 exit 0, `model_299.pt`, TensorBoard, GPU 회수까지 통과했다. wall time `1,201.052 s`, 평균 `6,299.93 steps/s`, peak VRAM `5,907 MiB`였지만 평면 평가에서는 전진과 좌·우 회전 응답이 거의 0인 지역해라 네 방향 gate에 실패했다.
- G006 baseline `model_1499.pt`에서 같은 budget을 이어 학습한 run은 `model_1798.pt`, SHA-256 `53cc09043088bcd53618d2ae1f90c7f2e91d01eab7090cc63922486942b2ed47`을 만들었다. RSL-RL이 loaded iteration을 포함해 `1499~1798`을 실행한다는 점을 보고서 재검증과 회귀 테스트로 고정했다. wall time `1,077.001 s`, 평균 `7,039.55 steps/s`, final mean reward `35.41`, peak VRAM `5,892 MiB`였다.
- warm-start 평면 평가에서는 64/64 생존, 네 방향 부호 일치, 선속도 RMSE `0.0466~0.0794 m/s`, yaw RMSE `0.0741~0.1154 rad/s`로 네 방향 모두 gate를 통과했다. rough에서는 좌·우 회전이 통과했고, 전진 max roll/pitch `0.3713/0.4788 rad`, 후진 max pitch `0.3505 rad` 때문에 자세 gate에 실패했다.
- 1,024환경 runtime probe에서 friction S1은 static `0.7226~0.8770`, dynamic `0.5295~0.6729`, leg mass scale은 고정 1.0이었다. leg-mass S1은 scale `0.9500~1.0500`, 총 다리 질량 `7.8296~8.3781 kg`, inertia 재계산 최대 오차 약 `1.86e-9`였고 foot static friction은 0.8로 고정됐다.
- command checkpoint에서 friction S1을 `1,024 env × 300 iterations × seed 42`로 이어 학습했다. `model_2097.pt`, SHA-256 `40af0a0f80489d705e1e8fdeedd2f765177d3d67bf757709b9195cc2bbeaaee0`, wall time `1,234.808 s`, 평균 `6,213.25 steps/s`, final mean reward `35.19`, peak VRAM `5,936 MiB`로 학습 보고서는 PASS였다.
- friction S1 checkpoint는 randomized·nominal 평면에서 모두 64/64 생존과 네 방향 gate를 통과했다. randomized 조건의 선속도 RMSE는 `0.05~0.06 m/s`, yaw RMSE는 `0.08~0.16 rad/s`였다. rough 학습의 terrain level mean은 약 3.45에서 2.27로 내려가 rough 강건성 개선이나 S2 진입을 주장하지 않는다.
- command checkpoint의 원본 MP4는 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g008\g008_directions_s42.mp4`에만 보관한다. Git에는 GIF와 네 방향 접촉시트를 넣고, 원본과 파생물의 SHA-256은 `reports/runs/g008_direction_visual_evidence.json`에 기록했다.
- command checkpoint에서 갈라진 leg-mass S1도 `1,024 env × 300 iterations × seed 42`로 완료했다. `model_2097.pt`, SHA-256 `8976cfff6eee6d1a998c7aa554b23d98b01d3d64da02b43ac3133a9186ae97fa`, wall time `1,373.046 s`, 평균 `5,668.77 steps/s`, final mean reward `35.25`, peak VRAM `5,908 MiB`였다.
- leg-mass S1은 randomized·nominal 평면에서 전진·후진·좌회전은 통과했지만 우회전 yaw RMSE가 `0.2956/0.2947 rad/s`로 기준 0.25를 넘었다. 평균 yaw rate는 `-0.2348/-0.2353 rad/s`로 command checkpoint의 `-0.4533 rad/s`보다 느렸다. nominal guardrail이 실패해 leg-mass S2를 열지 않는다.
- leg-mass 학습의 terrain level mean도 약 3.43에서 2.29로 내려갔다. friction과 mass S1 모두 rough 난이도가 후퇴했으므로 평면 gate만으로 rough 개선을 주장하지 않는다.
- command, friction S1, leg-mass S1 정책을 평면·seed 42·같은 900-step 명령으로 별도 Isaac Sim 프로세스에서 촬영했다. friction 단일 환경의 발바닥 평균은 static/dynamic `0.8152/0.5799`, leg-mass의 16개 body scale은 `0.9575~1.0452`였다. 정책별 H.264 원본과 1280×380 비교 MP4는 로컬에만 두고, 720×214 GIF와 네 방향 접촉시트만 Git에 넣는다. 해시와 ffprobe 결과는 `reports/runs/g008_policy_*_capture.json`, `reports/runs/g008_policy_comparison_visual_evidence.json`에 기록했다.
- 상세 역학, PPO batch/epoch, 문헌 채택 범위와 sim-to-real 한계는 `docs/G008_COMMAND_FRICTION_LINK_MASS.md`에 기록했다.
- 폭 `0.5 m`의 고·저마찰 face를 단일 static triangle mesh에 교차 배치하고 command/friction S1을 case당 32환경·500 step으로 비교했다. 기본 ground collider는 생성하지 않았고 non-collision height-scan mesh를 분리했다. friction S1의 전진·후진·좌회전은 완료된 최저 저마찰 `0.2/0.1`까지 연속 통과했다. 우회전은 `0.7/0.5`에서 첫 FAIL, `0.6/0.4`에서 개별 PASS가 나와 전 방향 연속 하한은 확정하지 않았다. `0.1/0.05`는 네 번 모두 100~200 step 뒤 Isaac Sim native 종료가 재현돼 미확정으로 분리했다. multi-material mesh에서 contact force sample이 비어 slip은 `null`로 두고, base 높이·body up-axis의 독립 낙상 판정을 추가했다.
- hip·thigh·calf·foot를 한 그룹씩 `0.8~1.2`배로 바꾼 800환경·300-step 민감도 시험을 수행했다. 질량과 inertia tensor가 같은 비율로 적용됐고 두 정책의 전진·후진은 25개 조건을 모두 통과했다. command는 nominal 네 방향 PASS, leg-mass S1은 nominal 우회전 yaw RMSE `0.44 rad/s`로 FAIL했다. `docs/G008_PERIODIC_FRICTION_AND_LINK_MASS_LIMITS.md`와 두 최종 JSON에 상세 근거를 남겼다.
- 혼합 마찰 `0.8/0.6 ↔ 0.2/0.1` 한 편과 hip·thigh·calf·foot `1.2배` 네 편을 headless off-screen으로 촬영했다. 원본은 모두 H.264 1280×720, 50 fps, 899 frames, 17.98초다. FFmpeg로 혼합 마찰 자막 MP4와 링크 질량 2×2 비교 MP4를 로컬에 만들고, 공개 GIF 두 개와 네 방향 스크린샷 두 개를 `docs/media/g008`에 넣었다. 각 촬영 보고서에는 checkpoint, 실제 friction/mass readback, 원본 SHA-256을 기록했다. 이후 동작 stage가 바뀌면 MP4·GIF·PNG·JSON 촬영 세트를 완료 조건에 포함한다.
- 주기 띠 다음 단계로 x/y `-28~28m`, cell `0.25m`의 비주기 2D 마찰·높이 field를 추가했다. static/dynamic 마찰 네 구간은 `0.25/0.15`, `0.40/0.28`, `0.60/0.45`, `0.80/0.60`이며 각 25%다. 높이 범위는 `0.08103m`, 최대 국부 경사는 `2.6989°`다. 기본 ground collider를 제거하고 마찰별 collision mesh 네 개와 비충돌 height-scan mesh 하나로 구성했다.
- friction S1 `model_2097.pt`를 32환경, 방향당 8개, 500 steps, warmup 50, seed `20260826`으로 평가했다. 전진·후진·좌회전은 PASS, 우회전은 max roll `0.3739rad`가 기준 `0.35rad`를 넘어 FAIL이었다. 낙상은 없었다. 네 발이 같은 마찰인 frame과 네 발이 모두 다른 frame이 모두 나타났고 네 구간의 실제 발 접촉 표본이 기록됐다.
- 같은 checkpoint에서 불규칙 도로 태스크를 64환경 × 300 iterations로 이어 학습했다. rollout은 환경당 24 steps, iteration batch `1,536`, PPO 5 epochs × 4 mini-batches, 총 transitions `460,800`, optimizer mini-batch updates `6,000`이다. wall time `440.274s`, 평균 `1,132.32 steps/s`, peak VRAM `5,058MiB`, final mean reward `35.84`, final episode length `984`였고 프로세스·checkpoint·TensorBoard·GPU 회수 gate는 PASS했다.
- 전용 학습 최종 `model_2396.pt`, SHA-256 `1384b92107b776c6c18851abd17d47efc66b9ea42306f6ca354b0b525c7c4486`은 full 평가에서 전진·후진만 통과하고 좌·우 회전 중 5회 넘어졌다. 짧은 screening에서 전 방향을 통과한 `model_2100.pt`도 full 평가에서는 좌회전 pitch `0.3949rad`로 실패했다. 통과 방향 수→낙상 수→최악 normalized gate 순으로 비교해 기존 friction S1을 유지하고 추가 학습 checkpoint는 기각했다.
- 기존 정책과 최종 전용 학습 정책을 같은 900-step 시퀀스로 다시 촬영했다. 원본 MP4 두 개와 1280×780 비교 MP4는 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g008`에만 보관하고, 720×438 GIF와 네 방향 PNG는 `docs/media/g008`에 넣었다. 정량·시각 증거와 단계별 재학습 계획은 `docs/G008_IRREGULAR_ROAD.md`에 기록했다.
- commit `4d8842d1b7ee2d4e4a26c67d363997abf9729a81` 기준으로 불규칙 도로 G0를 추가했다. S1과 높이 vertex·cell·crown·굴곡·요철·함몰은 같고 바닥 재질만 static/dynamic `0.8/0.6` 한 구간으로 고정했다. config diff 테스트는 마찰 tuple과 색 세 경로만 달라지는지 확인한다.
- 기존 friction S1 `model_2097.pt`를 G0의 terrain seed `20260826/20260827/20260828`, 32환경, 방향당 8개, 500 steps, warmup 50으로 평가했다. 앞의 두 seed는 전 방향 PASS, `20260828` 우회전은 1회 낙상으로 FAIL이었다. 전체는 `2/3` seed, `11/12` 방향 PASS다. 균일 `0.8/0.6`에서도 실패했으므로 낮은 마찰만을 원인으로 보지 않는다.
- 같은 시작 checkpoint에서 G0를 `128 env × 300 iterations × seed 20260826`으로 실제 추가 학습했다. transitions `921,600`, optimizer mini-batch updates `6,000`, wall time `537.494s`, 평균 `2,021.73 steps/s`, peak VRAM `5,463MiB`, final mean reward `33.64`, final episode length `1,000`이다. exit 0, `2396/2397`, TensorBoard·checkpoint·GPU 회수 gate를 통과했다.
- G0 저장 checkpoint 7개를 terrain seed `20260828`, 16환경·300-step으로 선별했다. `model_2100`과 `2250`만 짧은 gate를 통과했지만 32환경·500-step·세 terrain seed 정식 평가에서는 각각 `9/12`, `8/12` 방향 PASS로 탈락했다. SHA-256은 `model_2100` `de342b599389e1c43a2b62ec2d94215f677a9f806b2aabf6e778441d8db7bc5b`, `model_2250` `fd6aefbdd124b34f2825240ed9ceaedecb8deb97eb6d239465929d4fed6e5681`이다.
- 런타임 보상 계약을 추출해 `reports/runs/g008_reward_contract_s20260826.json`에 고정했다. 총보상은 `0.02s × Σ(weight × raw term)`이며 활성 항은 선속도 추종 `+1.5`, yaw 추종 `+0.75`, 수직속도 `-2.0`, roll/pitch 각속도 `-0.05`, torque `-0.0002`, 관절가속 `-2.5e-7`, action-rate `-0.01`, feet-air-time `+0.01`이다. `flat_orientation_l2`, `dof_pos_limits`는 0, `undesired_contacts`는 비활성이고 base 접촉에는 별도 scalar termination penalty가 없다.
- 기존 `feet_air_time`은 `||v_cmd,xy||>0.1m/s`일 때만 켜져 순수 좌·우회전에서는 0이다. T1은 가중치와 `0.5s` threshold를 유지하고 `|w_cmd,z|>0.1rad/s` 조건만 추가했다. config 차이는 함수와 yaw threshold 두 경로뿐이며 8개 config/reward 테스트와 16환경 신규 스모크, 128환경 checkpoint 재개 스모크가 통과했다.
- T1도 기존 friction S1 checkpoint에서 독립적으로 `128 env × 300 iterations` 학습했다. transitions `921,600`, optimizer updates `6,000`, wall time `586.972s`, 평균 `1,847.79 steps/s`, peak VRAM `5,271MiB`, final mean reward `24.06`, final episode length `993.69`이다. 학습 중 `feet_air_time` 기여는 작은 음수였으며 `last_air_time-0.5s` 구조상 짧은 swing이 벌점이 될 수 있음을 다음 가설로 남겼다.
- T1의 7개 checkpoint 중 screening을 통과한 것은 `model_2100.pt`, SHA-256 `ff66fc36b5e5f652adfae33505d85bb7f3a5fa769967f06c034684b75a841a47`뿐이다. full 평가에서는 우회전 yaw RMSE가 seed별 `0.2609/0.2752/0.2599rad/s`로 모두 기준 `0.25`를 넘었고 seed `20260828`은 roll/pitch `0.4311/0.3767rad`도 초과했다. 전체 `0/3` seed, `9/12` 방향 PASS라 기각했다.
- 전체 선택 규칙은 통과 terrain seed 수→전체 방향 PASS 수→낙상 수→최악 normalized gate ratio다. 기존 friction S1을 유지하고 G0 전용 checkpoint와 T1을 모두 기각했다. 승인 정책이 없으므로 마찰 F1은 열지 않는다. 집계는 `reports/runs/g008_road_curriculum_summary_s20260826.json`에 있다.
- G0 기존 정책과 T1 `model_2100`을 같은 terrain seed와 900-step 명령으로 다시 촬영했다. 원본 MP4 두 개와 동기화 비교 MP4는 로컬에만 두고, 720×438 GIF와 1280×780 접촉시트를 Git에 넣었다. 파일 해시와 지형·보상·checkpoint 연결은 `reports/runs/g008_road_curriculum_visual_evidence.json`, 해석과 다음 실험은 `docs/G008_REWARD_AND_ROAD_CURRICULUM.md`에 기록했다.

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

### G009 산 비탈 S0 지형·시각 증거

- `G009-1` C0에서 로컬 원본 영상 경로를 `%USERPROFILE%\IsaacLab\logs\visual_evidence\<goal_id>`로 일반화하고 G009 24개 stage registry를 고정했다. 기존 G008 로컬 영상 참조 18개와 미디어 회귀 15개를 유지했다.
- `G009-2` S0 analytic gate는 base slope와 residual height를 분리하고 residual을 0으로 고정했다. 경사 `0/5/10/15/20/25°`, 방위 `0/90/180/270°`의 24개 cell이 모두 통과했다. 최대 경사각 오차는 `7.172749647565979e-07°`, analytic normal과 triangle normal의 최대 오차는 `2.181721622226445e-05°`다.
- `G009-3` Isaac runtime task `Isaac-G009-Velocity-Slope-Go2-S0-v0`는 하나의 static triangle collision/RayCaster mesh를 사용한다. nominal ground static/dynamic friction은 `0.8/0.6`, foot material은 `1.0/1.0`, combine mode는 `multiply`다. G008에서 상속될 수 있는 push, base mass, external wrench randomization은 S0에서 껐다.
- 녹화 source commit은 `4bad4dd8634c11aa452da41ad0c2fb852e70e607`이다. 세 capture report의 `dirty_paths`는 모두 빈 배열이며 recorder SHA-256은 `3afebbdeb6df8a8cf65366bad6c8629c7a7ef0b62ec5012d64b8723e87c88b54`다.
- 재생 정책은 G008 friction S1 `model_2097.pt`, SHA-256 `40af0a0f80489d705e1e8fdeedd2f765177d3d67bf757709b9195cc2bbeaaee0`이다. G009에서 새로 학습한 정책이 아니며 S0에서는 PPO rollout batch, mini-batch, epoch, optimizer update를 실행하지 않았다.
- `G009-4` 녹화는 `--headless` off-screen camera, 1 environment, seed `20260828`, `step_dt=0.02 s`, 총 525 step으로 수행했다. 시퀀스는 정지 75, 등고선 왼쪽 200, 정지 50, 등고선 오른쪽 200 step이며 명령은 차례로 `[0,0,0]`, `[0.4,0,0]`, `[0,0,0]`, `[-0.4,0,0]`이다.
- 5° 결과는 최대 support-normal 상대 tilt `3.6857°`, 하방 이동 final/max `0.0780/0.0780 m`다. 15°는 `13.5774°`, `0.0909/0.2274 m`다. 두 실행 모두 `termination.fall=false`였다.
- 25° stress 결과는 `termination.fall=false`였지만 최대 tilt `84.7832°`, 하방 이동 `2.3925 m`였다. termination flag만으로 통과시키지 않고 기존 G008 정책의 실패 경계로 기록했다.
- 첫 세 캡처는 commit `da2963aa1cc8df6271483b8458ca579b9b8db179`에서 만들어졌지만 PhysX material readback `0.800000011920929/0.6000000238418579`와 JSON 설정 `0.8/0.6`을 exact equality로 비교해 builder가 중단됐다. bool, NaN, Inf는 계속 거부하면서 유한 float의 절대오차 `1e-6`만 허용하도록 수정했다. 이전 캡처는 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g009\archive\pre_float_tolerance_da2963`로 옮기고 새 commit에서 세 편을 전부 다시 촬영했다.
- 최종 원본 MP4 SHA-256은 5° `7bc3ac8feceeb9ce11d0162923efca157410403fe9f3c4e5371c2409c163d709`, 15° `5b65866997994b3f4523d242e4c9d1646aa3682dfccb15ad6fe5bf79a6001f99`, 25° `c72da025733933d7103792abdb8ba96986b43f5754e7a75f891de06486c211cd`다. 원본과 1440×430 합성 MP4는 로컬에만 보관한다.
- 공개 GIF는 `docs/media/g009/S0/g009_s0_slopes.gif`, 접촉시트는 `docs/media/g009/S0/g009_s0_slopes_contact_sheet.png`다. `reports/runs/g009_s0_visual_summary.json`과 `reports/runs/g009_s0_visual_evidence.json`이 source commit, checkpoint, config, analytic report, physics readback, capture report, 미디어 SHA-256을 결합한다.
- 재현 명령은 아래와 같다. `slope_05`를 `slope_15`, `slope_25_stress`로 바꾸고 report 경로도 같은 profile 이름으로 바꿔 세 프로세스를 순차 실행한다.

```powershell
cd "$HOME\isaac-walk-rl"

& "$HOME\IsaacLab\_isaac_sim\python.bat" .\scripts\record_g009_s0.py `
  --profile slope_05 `
  --checkpoint "$HOME\IsaacLab\logs\rsl_rl\unitree_go2_rough\2026-08-26_11-37-54_g008_friction_s1_finetune_command_s42_e1024_i300\model_2097.pt" `
  --config .\configs\g009_s0.json `
  --report .\reports\runs\g009_s0_slope_05_capture.json `
  --headless

py .\scripts\build_g009_s0_media.py --capture-reports `
  .\reports\runs\g009_s0_slope_05_capture.json `
  .\reports\runs\g009_s0_slope_15_capture.json `
  .\reports\runs\g009_s0_slope_25_stress_capture.json `
  --config .\configs\g009_s0.json
```

- S0 완료 당시 다음 순서를 `G009-5` R0 flat RECOVER scratch 학습 → S0 nominal WALK·R0 RECOVER torque/power/impact calibration → `G009-6` S1-low `5/10°` WALK로 정했다. 각 stage는 별도 다중 seed 평가 JSON과 MP4·GIF·PNG·sidecar를 새로 만든다. 현재 R0의 최신 상태는 아래 rev12 gate10 실패 절에 기록한다.

### G009-5 R0 평지 전복 복구 PPO 계약·진단

#### 실행 범위와 headless 의미

- R0 task는 `Isaac-G009-Recover-Flat-Go2-R0-v0`다. 물리 timestep은 `0.005 s`, control decimation은 `4`, 정책 제어 주기는 `0.02 s`(`50 Hz`), episode 상한은 `400 step`(`8.0 s`)이다.
- 여기서 `headless`는 Isaac Sim GUI 창을 띄우지 않는 실행을 뜻한다. 학습은 화면 렌더링 없이 물리·센서·PPO를 실행하고, 영상 단계에서는 같은 `--headless` 프로세스에서 카메라 extension과 off-screen renderer만 켠다. 따라서 headless 학습과 headless off-screen 녹화는 같은 말이 아니다.
- 네 canonical reset pose와 root height는 prone `0.165 m`, supine `0.060 m`, left/right side `0.163 m`다. 각 revision은 이전에 기각한 checkpoint를 재개하지 않고 scratch에서 실행했다. 보고서의 `resume.enabled=false`와 빈 `effective_hydra_overrides`를 기준으로 확인했다.

#### rev1~rev8 진단과 기각

아래 수치는 각 JSON의 마지막 TensorBoard scalar다. `hard-limit`은 마지막 기록의 `Episode_Termination/hard_joint_limit`이므로 전체 학습 구간의 누적 횟수로 해석하지 않는다. 모든 실행은 프로세스·checkpoint 생성 관점에서는 정상 종료했지만, 엄격 복구 신호가 `0`이어서 정책 checkpoint는 모두 기각했다.

rev1~rev8 report는 당시 dirty working tree에서 생성됐고 `source_bundle.matches_repository_commit=false`다. 마지막 scalar와 report 파일 hash는 실패 진단용으로 보존하지만 당시 source snapshot을 완전히 재현하는 승인 증거는 아니다. rev9부터는 clean commit, 개별 source file SHA와 source bundle SHA를 필수 provenance로 사용한다.

| revision | scratch 실행 | final reward | `stable_support / upright_hold / stable_success_once` | final hard-limit | 판정과 다음 수정 |
| --- | --- | ---: | ---: | ---: | --- |
| rev1 | `1,024 env × 50 iter`, seed `42` | `-22.25` | `0 / 0 / 0` | `0.0416667` | 성공 경험이 없고 안전 종료도 남아 기각 |
| rev2 | `64 × 1` smoke 후 `1,024 × 50` | `-18.24` | `0 / 0 / 0` | `0` | reward가 덜 음수가 됐지만 복구 신호가 없어 기각 |
| rev3 | `64 × 1` smoke | `-2.81` | `0 / 0 / 0` | `0.75` | P83/C107 관측 경계를 고정했지만 완료 episode의 75%가 hard limit으로 끝나 기각 |
| rev4 | `64 × 1` smoke | `-2.62` | `0 / 0 / 0` | `0.625` | action scale만 `0.8`로 줄였으나 hard-limit 62.5%가 남아 기각 |
| rev5 | `64 × 1` smoke | `-1.15` | `0 / 0 / 0` | `0` | scale `0.8`에 EMA `alpha=0.2`를 더해 해당 smoke의 hard-limit을 없앴지만 성공 신호가 없어 학습 후보로 채택하지 않음 |
| rev6 | `64 × 1` smoke와 `1,024 × 1` stress | `-0.97 / -0.35` | 두 실행 모두 `0 / 0 / 0` | `0 / 0.0416667` | side root height를 `0.163 m`로 보정했지만 대규모 stress에서 안전 종료가 다시 나타나 기각 |
| rev7 | `1,024 × 1` stress 후 `1,024 × 50` | pilot `-1.10` | pilot `0 / 0 / 0` | pilot final `0` | PPO 초기 표준편차를 `0.5`로 낮춰 stress의 final hard-limit은 `0`이 됐으나 50-iteration 복구 경험은 끝내 없어서 기각 |
| rev8 | `1,024 × 50` safety pilot | `-0.19` | `0 / 0 / 0` | final `0` | EMA를 `alpha=0.1`로 더 느리게 만들자 reward는 덜 음수가 됐지만 복구 신호는 여전히 `0`; 동작량 감소를 복구 개선으로 볼 수 없어 기각하고 `alpha=0.2`로 복원 |

대표 원본은 [rev1 1024×50](reports/runs/go2_flat_recover_pilot2_s42_20260828-1105.json), [rev2 1024×50](reports/runs/go2_flat_recover_rev2_pilot_s42_20260828-1222.json), [rev3 smoke](reports/runs/go2_flat_recover_rev3_smoke_s42_20260828-1250.json), [rev4 smoke](reports/runs/go2_flat_recover_rev4_smoke_s42_20260828-1300.json), [rev5 smoke](reports/runs/go2_flat_recover_rev5_smoke_s42_20260828-1302.json), [rev6 stress](reports/runs/go2_flat_recover_rev6_stress_s42_20260828-1309.json), [rev7 1024×50](reports/runs/go2_flat_recover_rev7_pilot_s42_20260828-1312.json), [rev8 1024×50](reports/runs/go2_flat_recover_rev8_safety_pilot_s42_20260828-1318.json)이다. 단순히 final reward가 `-22.25 → -0.19`로 변한 사실만으로 정책이 개선됐다고 판단하지 않는다. 세 엄격 복구 항이 계속 `0`이었기 때문이다.

#### rev9 동결 계약

- 최종 계약 ID는 `g009_r0_recover_rev9`, 계약 SHA-256은 `4e0499699a24a272cccb9687f417d97770fcbc229186e2aedde6914e45beab66`다. 단일 원본은 [configs/g009_r0.json](configs/g009_r0.json)이다.
- actor는 `P-RECOVER-83`이다. base 선·각속도, projected gravity, 관절 위치·속도, 이전 action, 4발 contact/load, body-fixed `5×3` range와 hit mask를 사용한다. pose one-hot, true terrain normal, mass·friction·wrench 같은 simulator oracle은 actor에 주지 않는다.
- critic은 actor 83차원 prefix에 privileged 24차원을 더한 `C-RECOVER-107`이다. 실물 적용에는 IMU/base estimator, joint encoder, foot contact/load estimator, 하향 range/depth camera adapter가 필요하므로 deployability 상태는 `conditional_adapter_required`다.
- critic의 `commanded_wrench` 3차원과 `normalized_pulse_time_remaining` 1차원은 D1 확장을 위해 예약한 채널이다. 외란이 비활성인 R0에서는 각각 명시적 constant-zero 함수가 값을 만들며, 현재 측정 신호나 actor 입력으로 해석하지 않는다.
- action은 12차원 `EMAJointPositionToLimitsAction`이다. normalized clip `[-1,1]`, scale `0.8`, EMA `alpha=0.2`, soft joint limit factor `0.9`를 적용한다. hard-limit 전체 범위의 `72%`만 target envelope로 쓰며 각 끝의 margin은 `14%`다. solver tolerance `0.01 rad`는 완화하지 않는다.
- PPO는 RSL-RL `2.3.3`, rollout `24 step/env`, actor·critic MLP `512-256-128`, ELU, 초기 noise std `0.5`, `5 epochs × 4 mini-batches`, iteration당 optimizer update `20`, clip `0.2`, entropy `0.01`, gamma `0.99`, GAE lambda `0.95`, adaptive learning rate 초기값 `0.001`이다.

rev9 reward는 Isaac Lab의 `control_dt` 곱을 포함해 `sum(weight × raw_rate × 0.02)`로 집계한다.

| 항 | weight | 의미 |
| --- | ---: | --- |
| `upright_progress` | `+2.0` | 할인 호환 자세 잠재차 |
| `gated_base_height_progress` | `+2.0` | 자세 gate를 통과한 높이 잠재차 |
| `soft_stand_progress` | `+2.0` | `u·z·(0.5c+0.5l)`의 할인 잠재차 |
| `stable_support` | `+0.5` | 연속 지지 상태 |
| `upright_hold` | `+5.0` | 엄격 upright 유지 |
| `stable_success_once` | `+10.0` | 성공 latch 1회 terminal impulse |
| `gated_angvel_l2` | `-0.05` | 쓰러진 상태 multiplier `0.1`, 유효 `-0.005` |
| `joint_limit` | `-2.0` | 관절 안전 |
| `torque_l2` | `-0.0002` | torque regularization |
| `joint_acc_l2` | `-2.5e-7` | 관절 가속 regularization |
| `gated_action_rate_l2` | `-0.01` | 쓰러진 상태 multiplier `0.2`, 유효 `-0.002` |
| `mechanical_power_proxy` | `-1e-5` | 기계 power proxy |
| `undesired_collision` | `-1.0` | 비발 접촉 벌점 |

잠재 보상은 `(gamma·Phi_t - Phi_t-1) / control_dt`이며 terminal transition의 `Phi_t`는 `0`으로 강제한다. episode 시작의 이전 잠재도 `0`으로 초기화해 전체 discounted shaping return이 telescope하도록 고정했다. rev7·rev8에서 자세·높이 잠재 최대치와 regularization이 복구 탐색을 막은 진단을 반영해, rev9는 soft stand 진행 항을 추가하고 쓰러진 상태의 각속도·action-rate 벌점을 낮췄다. 이 shaping 신호는 아래 엄격 성공 판정을 대체하지 않는다.

엄격 성공은 다음 조건을 `25 control step=0.5 s` 연속 만족해야 한다.

- support normal 대비 upright angle `≤20°`, base height `0.30~0.60 m`
- true support normal 방향의 양의 발 하중이 있는 발 `≥3`, 네 발 합계 `≥0.6 × 15.019 kg × 9.81 m/s²`
- force `>1 N`인 비발 접촉 `0`
- base linear speed `≤0.5 m/s`, angular speed `≤1.0 rad/s`
- `numeric_invalid=0`, URDF hard-joint-limit safety termination `0`

pose curriculum clock은 `env.common_step_counter`이며 PPO iteration당 `24` control step으로 환산한다.

| PPO iteration | prone | supine | left | right |
| ---: | ---: | ---: | ---: | ---: |
| `0~49` | `100%` | `0%` | `0%` | `0%` |
| `50~99` | `50%` | `0%` | `25%` | `25%` |
| `100~299` | `25%` | `25%` | `25%` | `25%` |

평가는 curriculum을 쓰지 않고 네 pose를 동일 개수로 stratified assignment한다. pose class는 critic-only이며 actor 관측에는 노출하지 않는다.

#### runtime calibration과 테스트

- [GPU probe](reports/runs/g009_r0_runtime_probe_gpu.json)와 [CPU probe](reports/runs/g009_r0_runtime_probe_cpu.json)는 각각 8환경, 150-step, seed `42`로 reset pose, P83/C107 차원, range/no-hit, foot load, friction·mass readback, latch, joint/torque/speed·contact 안전 경계를 검사했다. 두 report의 clean source commit은 `42647e1620907c811ab8b646732a528878b07b83`, 13개 파일 source bundle SHA-256은 `2745de1317e7d312bb18eb1ec208bfdddf5180577f9491cc825ebd09e5f96c2f`다.
- [GPU/CPU synthesis](reports/runs/g009_r0_runtime_probe_synthesis.json)은 두 probe가 같은 계약 SHA를 사용했으며 `gpu_run_health_passed=true`, `gpu_runtime_contract_passed=true`, `cpu_authoritative_separation_passed=true`, `runtime_calibration_passed=true`임을 기록한다.
- 이 PASS는 환경·센서·계약이 실행된다는 뜻일 뿐 학습 성공이 아니다. synthesis는 명시적으로 `learned_policy_qualified=false`, qualification `status=not_run`이다.
- rev9 순수 Python G009 검사 결과는 `172 passed`, Isaac 번들 Python의 `test_g009_recover_config.py`는 `6 passed`, `test_g009_config_diff.py`는 `7 passed`다. 잠재 보상 telescope, one-shot latch, actor privilege 경계, pose curriculum, source-bundle provenance, evaluation/media fail-closed, qualification 실행 조건을 포함한다.

#### 다음 실행과 qualification gate

rev9 prone pilot은 clean source에서 `1,024 env × 50 iterations × seed 42`로 scratch 실행했다. 총 rollout transition은 `1,228,800`, optimizer update는 `1,000`, wall time은 `115.616초`, 평균/중앙 처리량은 `12,895.5/13,018 steps/s`, peak VRAM은 `4,376 MiB`다. 최종 mean reward는 `0.3024127`, final episode length는 `400`이었다.

- stable support와 upright hold는 각각 50개 scalar 중 `21`개에서 nonzero였고 마지막 10개에서는 각각 `8`개가 nonzero였다.
- strict `stable_success_once`는 50개 전부 `0`이었다.
- numeric-invalid는 전 구간 `0`이지만 hard-joint-limit은 50개 중 `23`개에서 nonzero, 최대 `0.4583333`이었다.
- 마지막 rollout에서 prone `0.9791667`, left/right `0.0104167`이 기록돼 `<1,200` curriculum 경계의 one-step leak을 확인했다.
- source commit은 `030d6b4471848f538a28a8649e2d5b4e615df568`, source bundle은 `45a1b4cc9ccf73b8dedd63d69ab8e8163addb5b6cb0297daa89861a9a72abd55`, checkpoint SHA-256은 `18e87baf43351d5e36aae5cabc608666099e7460a20d2606610607bfc35b3bf1`이다.

근거는 [rev9 prone pilot report](reports/runs/go2_flat_recover_rev9_prone_pilot_s42_20260828-1421.json)다. partial recovery signal은 확인했지만 strict success와 안전 gate를 통과하지 못했으므로 rev9 checkpoint를 기각하며 300-iteration qualification으로 확장하지 않는다.

#### rev9 prone 진단 미디어

- `01 prone`을 1환경, seed 42, 400 control step(`8.0 s`, `50 Hz`)으로 headless off-screen 재생했다. renderer는 Windows D3D12를 사용했다. 첫 Vulkan 실행은 renderer 초기화 전 실패했고, 너무 넓게 촬영된 두 D3D12 시도는 로컬 `rejected_attempts`에 보존한 뒤 카메라를 다시 고정했다.
- 최종 원본은 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\g009_5_r0_diag_rev9_01_prone_s42.mp4`다. `1280×720` H.264, 400 frame, SHA-256 `acea63898220e3d355222c138b022bf77b4704705dd1c6fb84dcefd62d9a580d`, 크기 `1,114,591 bytes`이며 Git에는 넣지 않는다.
- capture source commit은 `1ba2859d6817faa49f8d49465274ca00a4377efe`, checkpoint SHA-256은 `18e87baf43351d5e36aae5cabc608666099e7460a20d2606610607bfc35b3bf1`다. terrain static friction은 `0.8`, effective foot static friction readback은 `0.8000000119`, robot total mass는 `15.0189991 kg`다.
- 재생 결과는 `strict success=0`, recovery time 없음, time-out이다. 해당 재생의 safety termination은 `0`이지만 rev9 학습 중 hard-joint-limit이 50개 기록 중 23개에서 발생했으므로 checkpoint 기각 판정은 유지한다.
- 공개 파일은 `docs/media/g009/R0/diagnostic/g009_5_r0_diag_rev9_01_prone.gif`와 `g009_5_r0_diag_rev9_01_prone_still.png`다. 오버레이에 `DIAGNOSTIC · NOT QUALIFIED`, `STRICT SUCCESS 0`, `HARD LIMIT EVENTS`를 넣었다. capture·summary·sidecar JSON은 원본 MP4, checkpoint, training report, source bundle, 공개 파생물의 SHA-256을 결합한다.

#### rev10 안전 계약

- 계약 ID는 `g009_r0_recover_rev10`, canonical SHA-256은 `b5499b4a8c111788c3c601fd983bb03907cb3779106821ce2a0be6ef447d5912`다.
- rev9에서 hard-joint-limit이 50개 기록 중 23개, 최대 `0.4583333`으로 나타났고 numeric-invalid는 전 구간 `0`이었다. rev10은 원인 분리를 위해 action scale만 `0.80 → 0.70`으로 낮췄다.
- soft joint limit factor `0.9`를 곱한 effective target range는 hard range의 `0.72 → 0.63`, 한쪽당 목표 margin은 `0.14 → 0.185`다. EMA alpha `0.2`, PPO initial noise `0.5`, 보상 항목·가중치, hard-limit tolerance `0.01rad`는 유지했다.
- pose curriculum phase end를 `(1200,2400) → (1201,2401)`로 바꿨다. 경계 판정은 control step `0/1199/1200 → phase 0`, `1201/2399/2400 → phase 1`, `2401 → phase 2`이며, `1..1200` 전 구간 prone 확률 `1.0` 회귀 검사를 추가했다.
- canonical manifest `--check`, 순수 계약 테스트 `5 passed`, Isaac 번들 구성 테스트 `7 passed`, G009 구성 차이 테스트 `7 passed`를 통과했다. 이는 구성 검증이며 학습 안전성과 복구 성능을 뜻하지 않는다.
- gate별 진단 도구는 rev10 `gate01/gate10/gate50`의 exact run name, canonical report path, 현재 HEAD, 필수 source 10개, checkpoint 경로·hash·iteration을 fail-closed로 결합한다. 같은 stem의 analysis·capture·MP4·공개 4종은 덮어쓰지 않는다.
- 진단 도구 회귀는 `49 passed`, 전체 순수 Python G009 회귀는 `222 passed`, `uvx pyright`는 `0 errors / 0 warnings / 0 informations`였고 Python compile과 `git diff --check`도 통과했다. 현재 rev10 코드에서도 실제 rev9 training bundle `45a1b4cc9ccf73b8dedd63d69ab8e8163addb5b6cb0297daa89861a9a72abd55`와 capture commit `1ba2859d6817faa49f8d49465274ca00a4377efe`의 로컬 MP4·analysis를 당시 Git blob의 LF/CRLF 후보로 재검증했다.

#### rev10 CPU 관문 실패와 rev11 reset/action 정합

- rev10 GPU runtime probe는 전체 계약을 통과했지만 CPU probe는 `left_side / reset_pose_hold / env 6`에서 physics step `131`(`0.655 s`)에 비발 접촉력 `16.066175 BW`를 기록해 상한 `15 BW`를 넘었다. numeric-invalid, hard-joint-limit, torque, speed, 누적 초과 impulse, tail settle, CPU contact separation은 통과했다.
- CPU probe를 새 프로세스로 다시 실행한 결과 JSON SHA-256까지 `4f072ca2f5bc65813bbec5f036d6ae556cf247fa60b07a639df4104528d5dbd4`로 같았다. 따라서 일회성 비결정 오차가 아니라 같은 궤적에서 재현되는 backend-sensitive contact peak로 판정했다.
- 직접 확인된 계약 불일치는 action scale `0.70`에서 기존 calf reset `-2.40 rad`를 역변환한 normalized hold action이 `-1.0`에 포화된다는 점이다. 실제 도달 target은 약 `-2.373986 rad`로 reset보다 `+0.026014 rad`(`약 1.49°`) 펴지고, EMA alpha `0.2`가 이 차이를 매 control step 반영한다. 이름과 달리 rev10 `reset_pose_hold`가 pose를 정확히 유지하지 못했다. 이 불일치와 `16.066175 BW` peak가 같은 궤적에서 반복됐지만 직접 인과는 rev11 A/B runtime 전에는 확정하지 않는다.
- rev11은 이 역학 가설을 한 변수로 검사하며 힘 상한을 완화하지 않는다. action scale `0.70`, EMA `0.2`, PPO initial noise `0.5`, reward, curriculum, hard-limit tolerance `0.01 rad`는 그대로 두고 calf reset만 `-2.40 → -2.37 rad`로 옮긴다. 모든 reset target이 normalized action 포화 없이 도달 가능해야 한다는 fail-closed runtime check와 비발 peak link attribution도 함께 기록한다.
- 기존 rev10 실패본은 `reports/runs/g009_r0_runtime_probe_rev10_cpu_attempt1_force_spike.json`과 `g009_r0_runtime_probe_rev10_cpu_attempt2_s42.json`으로 보존한다. rev11은 CPU/GPU 각각 독립 프로세스 3회가 모두 runtime contract를 통과할 때만 학습 gate를 연다. 한 번이라도 실패하면 1024환경 학습을 시작하지 않는다.
- rev11 probe는 AppLauncher 시작 전에 기존 output을 거부하고 각 프로세스에 UUID4 execution ID, UTC 시작시각, canonical `reports/runs/<file>.json` binding을 기록한다. strict synthesis는 서로 다른 여섯 execution ID와 실제 입력 경로 binding을 요구하므로 같은 JSON을 이름만 바꿔 3회 실행으로 셀 수 없다. probe와 synthesis output은 target·temporary 파일을 모두 덮어쓰지 않는다.
- clean source commit `0e43426a94acf34ca6b0346bd30729c486213d5f`, source bundle SHA-256 `22dac2899e6a709bddb9544318a8b8a3b4514c54f4c7732d7b62220a3b3f203f`에서 rev11 CPU 3회와 GPU 3회를 새 프로세스로 실행했다. 여섯 report의 execution ID와 파일 SHA-256은 모두 달랐고, 각 report의 전체 boolean check와 runtime contract가 모두 PASS였다.
- CPU 3회 worst cell은 모두 `left_side / reset_pose_hold / base / physics step 131`에서 `13.9706669 BW`, GPU 3회 worst cell은 모두 `right_side / reset_pose_hold / base / physics step 128`에서 `11.0431929 BW`였다. hold action은 전부 비포화였고 reachable target 최대 오차는 `1.1920929e-7 rad`로 `1e-6 rad` 기준 안에 들었다.
- rev10과 rev11의 통제된 한 변수 A/B에서 CPU peak는 `16.066175 → 13.970667 BW`, 약 `13.04%` 감소했고 세 번 반복됐다. 이는 reset/action 불일치 제거가 peak 감소 원인이라는 가설을 지지하지만, 한 backend·한 seed·한 짧은 probe의 결과이므로 보편적 인과로 확대하지 않는다. 이 runtime gate는 학습 환경의 안전 계약만 검증하며 `learned_policy_qualified=false`, `status=not_run`이다.

다음 revision은 rev9를 resume하지 않고 scratch로 시작한다.

1. `[완료]` rev9 checkpoint 동작을 diagnostic-only 로컬 MP4와 `NOT QUALIFIED` 오버레이가 있는 공개 GIF·PNG·JSON으로 고정했다.
2. `[완료]` rev10에서 action scale만 `0.8 → 0.70`으로 줄이고 EMA `0.2`, 초기 noise `0.5`, reward, hard tolerance를 유지했다. curriculum 경계를 `(1201,2401)`로 고쳐 50회 pilot 전 구간 prone `1.0`을 요구한다.
3. `[완료]` rev11에서 calf reset을 action envelope 안으로 옮긴 뒤 CPU/GPU runtime probe를 각각 3회 실행했다. 여섯 실행 모두 hold action 비포화, reset-target 오차 `≤1e-6 rad`, 비발 접촉력 `≤15 BW`, numeric-invalid·hard-joint-limit `0`을 통과했다.
4. `[기각]` rev11 `1,024×1` scratch gate01은 process/run-health는 PASS였지만 첫 scalar의 hard-joint-limit이 `0.0416667`이라 안전 gate를 통과하지 못했다. numeric-invalid는 `0`, prone probability는 `1.0`, curriculum phase는 `0`이었다. gate10·gate50은 실행하지 않는다.
5. `[완료]` rev12에서 articulation solver position iteration만 `4 → 8`로 올렸다. CPU/GPU runtime probe 각 3회가 모두 통과했고 raw hard-limit crossing과 non-foot contact peak가 감소했다.
6. `[완료]` rev12 `1,024×1` scratch gate01은 hard-joint-limit·numeric-invalid `0`으로 안전 관문을 통과했다. stable support·upright hold·strict success는 모두 `0`이어서 qualification은 아니다.
7. `[기각]` rev12 `1,024×10` scratch gate10은 numeric-invalid `0`이었지만 iterations `1/2/3`에서 hard-joint-limit이 각각 한 건 상당 재발해 safety gate를 통과하지 못했다. gate50은 열지 않는다.
8. `[다음]` Gate10의 정책 업데이트 경로를 유지한 pre-reset attribution으로 env·joint·action·target·torque·contact history를 먼저 귀속한다. 그 결과를 바탕으로 rev13 단일변수 A/B를 scratch로 시작한다.
9. `[대기]` 50회 안전 pilot은 stable support와 upright hold가 최소 한 번은 nonzero여야 한다. 통과한 revision만 `1,024×300`, seed 42 scratch qualification으로 연다.
10. deterministic 공식 평가에서 prone/supine/left/right 각각 성공률 `≥80%`, median recovery time `≤4.0 s`, safety termination `0`을 모두 만족해야만 learned checkpoint를 qualified로 판정한다.

rev11 gate01은 clean source commit `26fa9860470fe30ce192b342165caf2122598e8f`에서 `1,024 env × 24 step × 1 iteration`, seed `42`, headless, scratch로 실행했다. wall time은 `18.581 s`, 처리량은 `7,766 steps/s`, peak VRAM은 `4,368 MiB`, final mean reward는 `-0.52`였다. `model_0.pt` SHA-256은 `e89f92235656ef61e082333981a3045ba3582331cc1f7d6457d6806172291e4c`다. stable support, upright hold, strict success는 모두 `0`이었다.

- 분석 JSON은 hard-joint-limit nonzero와 strict success zero를 qualification block reason으로 기록했다. 학습 aggregate의 episode-summary rate는 joint·pose별 attribution을 제공하지 않으므로, 다음 revision 전에 별도 runtime instrumentation으로 위반 관절·초과량·pose/action 시점을 찾아야 한다.
- 1환경 deterministic playback은 8초 뒤 time-out됐고 stable success가 없었다. 이 단일 캡처에서는 safety termination이 재현되지 않았지만, 1,024환경 학습 aggregate의 hard-limit 실패를 상쇄하지 않는다.
- 원본 MP4는 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\g009_5_r0_diag_rev11_gate01_01_prone_s42.mp4`에만 둔다. H.264 `1280×720`, `50 fps`, 400 frame, 8초, SHA-256 `7a6ffd04430508d440625a23a8105fd06087b3d4f25390dec9ef2b64bf7c04cd`다. 공개 GIF·PNG·analysis·capture·summary·sidecar JSON에는 `DIAGNOSTIC / NOT QUALIFIED / 01 PRONE / STRICT SUCCESS 0`를 표시했다.

#### rev11 gate01 hard-limit 귀속 프로토콜

- TensorBoard의 `0.0416666679`는 `4.17%`의 환경이나 `0.0417rad` 초과량이 아니다. RSL-RL이 24-step rollout의 reset-batch 정수 count를 평균한 값이며 `0.0416666679 × 24 ≈ 1`이므로 원 gate01에서 hard-limit 종료가 한 번 기록됐다는 뜻이다.
- 원 report에는 env·joint·action·reset 직전 상태가 없고 `model_0.pt`는 해당 rollout의 PPO update 뒤 저장됐다. checkpoint에는 원 stochastic action stream과 RNG state도 없으므로 과거 사건의 bitwise identity는 복원하거나 주장하지 않는다.
- `scripts/attribute_g009_r0_gate01.py`는 동일 task, seed `42`, `1,024 env × 24 step`, prone `100%`, scratch stochastic PPO 경로를 새 프로세스에서 실행한다. PPO update 직전에 sentinel로 멈추며 checkpoint를 읽거나 만들지 않는다.
- 활성 `RecorderTerm`을 추가하면 noisy observation이 매 step 한 번 더 계산돼 Torch RNG가 달라진다. 따라서 `active_terms == []`를 유지하고 기존 `RecorderManager.record_pre_reset` 인스턴스 메서드만 감싼다. observer 앞뒤 CPU·CUDA RNG SHA 상태, policy SHA, action count `24`, rollout storage step `24`, update sentinel, diagnostic checkpoint 부재를 fail-closed로 검사한다. 장치는 원 실행 경로의 `cuda:0`으로 고정하고 Isaac Lab·RSL-RL 실행 소스 11개의 SHA-256을 기대값과 직접 비교하며, Git이 추적하는 Isaac Lab 핵심 경로 6개는 clean이어야 한다.
- 새 사건이 발생하면 reset 전에 env·pose·rollout/episode/sim step, 전체 joint position·hard limits·velocity·torque, wrapper clip 전 PPO sample, clip 후 action, EMA processed target을 기록한다. termination `(step, env)` multiset과 attribution multiset이 정확히 같고 원 hard-limit predicate를 다시 계산해 참일 때만 `attributed`다.
- `attributed`는 source/seed/protocol-matched fresh rollout에서 새 사건의 귀속이 정확하다는 뜻일 뿐이다. 원 사건과의 동일성은 `historical_event_identity_confirmed=false`, 안전 gate는 `false`, learned-policy qualification도 `false`로 남긴다. 한 번 재현되지 않으면 PASS로 바꾸지 않고 같은 고정 프로토콜을 독립 프로세스 세 번까지 실행해 GPU 비결정성을 확인한다.

clean source commit `12caebe523ae0a414630216e30d100302f693a0d`에서 GPU fresh rollout을 새 프로세스로 세 번 실행했고 모두 `attributed`였다. 세 execution ID와 report SHA-256은 서로 달랐지만 stochastic action-stream SHA-256 `46e58c33f305b168eed0b81931873c93356656398e456e1799b500c39dc22453`, policy SHA, 사건 위치와 모든 수치는 같았다.

- 사건은 세 번 모두 `rollout step 23 / env 706 / prone / FR_calf_joint / lower`였다. 실제 위치 `-2.7339249rad`, hard lower `-2.7227001rad`, raw excess `0.0112247rad`, `0.01rad` tolerance 밖 excess는 `0.0012247rad`였다.
- 같은 순간 FR calf의 PPO·clip 후 action은 `+0.1681439`, EMA target은 `-1.6222125rad`로 hard lower보다 약 `1.10049rad` 안쪽이었다. joint velocity는 `-0.1714629rad/s`, applied torque는 lower 방향과 반대인 `+23.5Nm`로 actuator 상한에 걸렸다.
- 따라서 새 rollout의 직접 관측은 사건 순간 policy target이 lower limit을 요구했다는 설명을 배제한다. actuator가 limit 안쪽으로 최대 토크를 내는데도 actual joint가 lower를 넘었으므로 이전 step의 관성, 외부 접촉력, joint/contact constraint 오차 중 하나 이상의 비명령 요인이 필요하다. event 이전 history와 body별 contact force를 아직 저장하지 않았으므로 특정 충돌 링크나 solver iteration 부족을 직접 원인으로 확정하지 않는다. rev12의 solver A/B는 이 가설을 판별하는 다음 실험이다.

#### rev12 단일변수 solver A/B

- 계약 ID는 `g009_r0_recover_rev12`, canonical SHA-256은 `d4b48d2b5fc1ea7684684a6324ba22fbfae767effeae45668c7310df382392e0`이다.
- Go2 DC motor는 `Kp=25`, `Kd=0.5`, effort/saturation limit `23.5Nm`다. 귀속 표본의 비포화 요구 토크는 `25×(-1.6222125+2.7339249)-0.5×(-0.1714629) ≈ +27.88Nm`라 실제 `+23.5Nm`는 limit 안쪽 복원 방향 포화와 일치한다.
- rev11 deterministic reset-pose-hold에서도 prone `RR_calf_joint`가 hard lower를 GPU `0.007208rad`, CPU `0.006586rad` 넘어갔다. `0.01rad` tolerance 안이라 종료되지 않았고 hold action은 비포화였으므로 stochastic policy 없이도 접촉 자세에서 calf가 hard limit 근처로 밀리는 현상이 있다.
- rev12는 articulation `solver_position_iteration_count`만 `4 → 8`로 올린다. `solver_velocity_iteration_count=0`, physics/control timestep, calf reset `-2.37rad`, action scale `0.70`, EMA `0.2`, PPO noise `0.5`, motor torque, reward, curriculum, hard-limit tolerance `0.01rad`는 유지한다.
- 먼저 runtime probe가 실제 PhysX articulation readback `position=8 / velocity=0`, numeric-invalid·hard-limit `0`, torque/contact/tail-settle 상한을 통과해야 한다. prone reset-hold raw penetration은 rev11 GPU `0.007208rad`보다 작아야 solver 가설이 지지된다. 감소하지 않으면 rev12를 기각하고 다른 변수를 겹치지 않는다.
- runtime gate를 통과한 뒤에만 새 source commit에서 seed 42, headless, `1,024 env × 24 step × 1 iteration` scratch gate01을 실행한다. hard-limit 하나라도 재발하면 gate10을 열지 않는다.

clean source commit `9da3e87e4be9142035d24e8a4a22e204f8b229d5`에서 CPU·GPU 새 프로세스를 각각 세 번 실행했다. 여섯 실행은 서로 다른 execution ID를 가졌고, source bundle SHA-256 `55e6eabbde30930b89d386b8a7533beccb903fc95934fcf6c3f2f1110ba5c0b4`와 contract SHA-256 `d4b48d2b5fc1ea7684684a6324ba22fbfae767effeae45668c7310df382392e0`은 같았다.

- 여섯 report 모두 live articulation 8개에서 solver `position=8 / velocity=0`, runtime contract PASS, run health PASS, boolean check 실패 `0`이었다. 엄격 합성도 GPU `3/3`, CPU `3/3`, CPU contact-separation `3/3` PASS다.
- prone reset-pose-hold raw hard-limit crossing은 GPU 세 번 모두 `0.0019140244rad`, CPU 세 번 모두 `0.0028049946rad`였다. rev11 대비 각각 `73.45%`, `57.41%` 감소해 solver 가설을 지지한다. tolerance `0.01rad`는 바꾸지 않았다.
- 최악의 non-foot contact는 CPU `left_side / reset_pose_hold / base`의 `9.4086094 BW`였다. `15 BW` 상한 이내이며 rev11 CPU `13.9706669 BW`보다 `32.65%` 낮다. GPU 최악값도 `9.4003544 BW`로 rev11보다 `14.88%` 낮다.
- 이 결과는 3초 deterministic runtime calibration이며 learned checkpoint 평가가 아니다. [strict 3×3 synthesis](reports/runs/g009_r0_runtime_probe_rev12_synthesis_3x3_s42.json)는 `runtime_calibration_passed=true`, `learned_policy_qualified=false`로 기록한다. 이 합성 PASS로 resume 없는 rev12 scratch gate01 실행 조건을 충족했다.

#### rev12 scratch gate01과 단계 영상

- clean source commit `61013ef8896ac2577c50c0ed15947040447c893d`에서 `go2_flat_recover_rev12_prone_gate01_s42_20260828-182222`를 resume 없이 실행했다. seed `42`, headless, `1,024 env × 24 step × 1 iteration`이고 source bundle SHA-256은 `2471c64c7fa107005c199ce8c27f42d4e9782b59452c4376e7ca981125aafffa`다.
- process/run health는 PASS, `hard_joint_limit maximum=0`, `numeric_invalid maximum=0`, prone probability `1.0`, curriculum phase `0`이다. rev11 gate01의 한 건과 달리 rev12 stochastic PPO rollout에서는 safety termination이 재발하지 않아 gate01 안전 관문을 통과했다.
- final mean reward는 `-0.51`, stable support·upright hold·strict success는 모두 `0`이다. 따라서 gate01 PASS는 solver 수정 뒤 첫 rollout이 안전했다는 뜻이며 복구 학습 성공이나 qualification을 뜻하지 않는다. 다음 단계는 동일 rev12의 resume 없는 scratch `1,024×10` gate10이다.
- checkpoint `model_0.pt` SHA-256은 `52f45ef5ae9d3c98ced51132e7fb6b5e8d78d0721a7efd9657f3fdc46ea17017`이다. 1환경 deterministic playback은 8초 time-out, safety termination `false`, strict success `0`이었다.
- 로컬 전용 MP4는 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\g009_5_r0_diag_rev12_gate01_01_prone_s42.mp4`다. H.264 `1280×720`, `50fps`, 8초, SHA-256 `4073f4b68d752a0760ed8ea31fc482ade95a150b657c658f69c5f1a2d7422982`이며 Git에는 넣지 않는다. 공개 GIF·PNG·JSON에는 `DIAGNOSTIC · NOT QUALIFIED · 01 PRONE · STRICT SUCCESS 0`를 표시한다.

#### rev12 scratch gate10 실패와 단계 영상

- Gate01 증거를 commit `281e61149574b30b524f1306eb08607467792c53`로 고정한 뒤 `go2_flat_recover_rev12_prone_gate10_s42_20260828-183416`을 resume 없이 새로 시작했다. seed `42`, headless, `1,024 env × 24 step × 10 iterations`, source bundle SHA-256 `2471c64c7fa107005c199ce8c27f42d4e9782b59452c4376e7ca981125aafffa`다.
- process/run health와 numeric-invalid는 PASS였지만 hard-joint-limit maximum이 `0.0416666679`, nonzero sample이 `3/10`이라 safety gate는 FAIL이다. iteration `1`, `2`, `3`에서 각각 `0.0416666679`였고 나머지는 `0`이다. 24-step 평균이므로 각 nonzero sample은 hard-limit 종료 한 건 상당이며 전체 세 건 상당이다.
- Gate10의 `model_0.pt` SHA-256은 Gate01 checkpoint와 같은 `52f45ef5ae9d3c98ced51132e7fb6b5e8d78d0721a7efd9657f3fdc46ea17017`이고 두 run의 source bundle SHA도 같다. 첫 rollout과 첫 PPO update가 동일하게 재현됐다는 강한 증거다. hard-limit은 logging iteration `1/2/3`, 즉 각각 이전 PPO update `1/2/3회`가 반영된 다음 rollout에서 나타났다. 다만 환경 state·episode 길이·stochastic RNG도 함께 진행되므로 policy update만을 단독 원인으로 확정하지 않는다.
- prone probability는 전 구간 `1.0`, curriculum phase는 `0`이었다. stable support·upright hold·strict success도 전 구간 `0`이므로 안전뿐 아니라 학습 신호 gate도 열리지 않았다. final mean reward는 `-5.16`, median 처리량은 `13,441.5 steps/s`, peak VRAM은 `4,356 MiB`다.
- `model_9.pt` SHA-256은 `b4bf026c446a72072ddf464aef8e5b3275b4d3f1cb1ad8980718139de2702cd2`다. 1환경 deterministic playback은 8초 time-out, safety termination `false`, strict success `0`이었다. 이 재생은 1,024환경 stochastic rollout의 세 건을 상쇄하지 않는다.
- 로컬 전용 MP4는 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\g009_5_r0_diag_rev12_gate10_01_prone_s42.mp4`다. H.264 `1280×720`, `50fps`, 8초, SHA-256 `b239460fba71c91ed36fcc83be90df696292988f8accff94e533ed5180e9997e`이며 Git에는 넣지 않는다. 공개 GIF·PNG·JSON에는 실패 진단 표식을 유지한다.
- gate50은 실행하지 않는다. 다음 작업은 gate10의 10회 PPO update 경로를 그대로 두고 pre-reset terminal state를 귀속하는 계측이며, attribution 전에는 calf reset·noise·torque·tolerance·reward를 바꾸지 않는다.

#### Gate10 full-update attribution 계약

- 새 진단 도구 commit에서도 training source binding 10개와 aggregate SHA `2471c64c7fa107005c199ce8c27f42d4e9782b59452c4376e7ca981125aafffa`는 Gate10 원본과 같아야 한다. `1,024 env × 24 step × 10 iterations`, seed `42`, `cuda:0`, headless, scratch, `init_at_random_ep_len=True`를 유지하고 공식 `OnPolicyRunner.learn()`과 원 PPO update를 10회 모두 호출한다.
- action call `240회`를 `iteration=(act_count-1)//24`, `rollout_step=(act_count-1)%24+1`로 태깅한다. active RecorderTerm은 계속 `0`으로 두고 기존 `record_pre_reset` hook만 감싸 terminal state를 reset 전에 기록한다. observer 전후 CPU/CUDA RNG state가 같아야 한다.
- 각 사건은 env·pose·iteration·rollout/episode/sim step, joint actual/lower/upper와 raw/margin excess, pre/post-clip action, EMA target, velocity, applied torque, root pose/twist를 저장한다. sensor 설정을 바꾸지 않고 기존 contact history와 별도 16-control-step ring buffer의 body별 contact force·joint/action history를 사건 env에 귀속한다.
- termination `(iteration, rollout_step, env)` multiset과 attribution multiset, 원 hard-limit predicate 재계산이 정확히 같아야 한다. hard series `[0,1/24,1/24,1/24,0,0,0,0,0,0]`, `model_0.pt` SHA `52f45ef5ae9d3c98ced51132e7fb6b5e8d78d0721a7efd9657f3fdc46ea17017`, `model_9.pt` SHA `b4bf026c446a72072ddf464aef8e5b3275b4d3f1cb1ad8980718139de2702cd2`까지 같을 때만 원 Gate10 trajectory identity를 강하게 확인한다. 하나라도 다르면 동일 조건 fresh reproduction으로만 기록한다.
- 새 프로세스 3회에서 사건 topology가 반복되는지 본다. 모든 사건이 calf lower, limit 안쪽 EMA target, 복원 방향 torque, 직전 calf/thigh/base contact 또는 lower 방향 관성, reset 과도구간과의 연결을 함께 보일 때만 rev13 calf reset `-2.37 → -2.34rad` 단일변수를 승인한다. non-calf 또는 policy target이 limit 방향인 사건이 하나라도 있으면 reset 변경은 보류한다.

#### rev13 velocity solver 단일변수 구현 gate

- rev13 계약 ID는 `g009_r0_recover_rev13`, canonical SHA-256은 `ebee855c503c77bce93c0884535d4fdf66ee5a01538fa59eef0e1b7aabba7558`이다. articulation position iteration은 `8`로 유지하고 velocity iteration만 rev12의 `0`에서 `1`로 바꿨다.
- `recover_contracts.py`의 상수와 canonical manifest, `recover_env_cfg.py`의 Isaac articulation 설정, runtime probe의 live USD readback 기대값을 같은 상수에 묶었다. calf reset `-2.37rad`, timestep `0.005/0.02s`, action scale/EMA `0.70/0.2`, PPO noise `0.5`, torque `23.5Nm`, hard-limit tolerance `0.01rad`, reward·curriculum·termination·observation noise는 변경하지 않았다.
- `py scripts/sync_g009_r0_contract.py --check`는 PASS였다. rev12 manifest와 rev13 manifest의 의미 diff는 contract ID·contract hash·변경 설명, velocity `0 → 1`, rev12 baseline velocity 메타데이터 추가뿐이었다. 이 허용 필드를 제거한 전체 계약 투영은 rev12 고정 SHA-256 `1f26f58655091a86af5a1da73be12562667f4573dfc3841b79162b3c899959f6`와 같아야 하며 회귀 테스트가 이를 직접 검사한다.
- Isaac 의존 테스트 두 파일을 제외한 전체 G009 순수 Python 검사는 `382 passed`, Isaac Sim 번들 Python의 `tests/test_g009_recover_config.py`는 `7 passed`였다. 변경 Python 파일의 `py_compile`은 PASS, import-light 변경 파일의 Pyright는 `0 errors`였다.
- rev12 Gate10 attribution 스크립트의 고정 hash는 수정하지 않았다. rev13 소스에서 과거 진단을 재실행하려 하면 config·contract·env cfg 세 경로의 mismatch를 감지해 fail-closed로 거부하는 테스트를 추가했다.
- 현재 상태는 `implemented_not_runtime_validated`이고 `learned_policy_qualified=false`다. 다음 실행은 clean source commit에서 CPU probe 3회, GPU probe 3회, strict synthesis 순서다. 여섯 runtime에서 실제 articulation `position=8 / velocity=1`, hard-limit·numeric-invalid `0`을 모두 확인하기 전에는 scratch Gate01을 열지 않는다.

#### 단계별 영상·공개 정책

- 원본 MP4는 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g009\R0` 아래에만 보관하고 Git에 넣지 않는다. 사용자가 확인할 원본과 합성 MP4도 `local_only`다.
- 공식 qualification을 통과한 checkpoint만 public media builder 입력으로 허용한다. 공개 저장소에는 GIF·PNG·정량 JSON sidecar만 둔다. 진단용 pilot 영상은 성공 증거와 분리하고 `diagnostic` 또는 `not_run/failed` 상태를 명시한다.
- G009-5 R0 pose 번호와 파일명은 `01 prone` → `02 supine` → `03 left_side` → `04 right_side` 순서로 고정한다. 로컬 파일은 `g009_5_r0_01_prone_s42.mp4`, `g009_5_r0_02_supine_s42.mp4`, `g009_5_r0_03_left_side_s42.mp4`, `g009_5_r0_04_right_side_s42.mp4` 형식이다.
- 공개 합성물 목표 경로는 `docs/media/g009/R0/g009_5_r0_four_pose_recovery.gif`와 대응 contact sheet PNG다. 평가·캡처·파생물 sidecar는 source commit, contract SHA, checkpoint SHA, pose 번호, headless/off-screen, ffprobe와 파일 SHA-256을 결합한다.
