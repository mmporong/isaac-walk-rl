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
- 25° stress 결과는 `termination.fall=false`였지만 최대 tilt `84.7832°`, 하방 이동 `2.3925 m`였다. termination flag만으로 통과시키지 않고 기존 G008 정책의 실패 경계로 기록했다. `25°`는 로봇이나 시뮬레이터의 최대 경사가 아니라 현재 protocol에서 배치한 가장 높은 stress cell이다.
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
- 구현 직후 상태는 `implemented_not_runtime_validated`, `learned_policy_qualified=false`였다. 예정 순서는 clean source commit의 CPU probe 3회, GPU probe 3회, strict synthesis였으며, 실제 articulation `position=8 / velocity=1`, hard-limit·numeric-invalid `0`을 모두 확인하기 전에는 scratch Gate01을 열지 않는 계약이었다. 아래 CPU 결과가 이 관문에서 rev13을 기각했다.

#### rev13 CPU runtime 3회 기각과 공개 진단 미디어

- clean source commit `e3734b728fcf546fea4ee05b9c8733800d6ab536`에서 seed `42`, headless, device `cpu`, `8 env × 150 control step` probe를 새 프로세스로 세 번 실행했다. execution ID는 `9e66cca532f64a7eaba06b615f38f37d`, `be56124aee694addbf0eebc7305b88d4`, `263d6b8c7430441fb4eb26c4afe1abd3`으로 서로 다르다. 세 report의 source bundle SHA-256은 `df6c6aa46181ca033791fb11ccfa76d9eab8643822da1c6cdc2e288409cabe3d`, contract SHA-256은 `ebee855c503c77bce93c0884535d4fdf66ee5a01538fa59eef0e1b7aabba7558`로 같았다.
- 세 실행 모두 live articulation 8개에서 `position=8 / velocity=1`을 읽었고 run health는 PASS였다. numeric-invalid와 hard-joint-limit termination은 `0`이었다. 그러나 세 번 모두 유일한 false check가 `nonfoot_peak_force_bounded`였고, `right_side / reset_pose_hold`의 `base`가 physics step `129`, `0.645s`에 `15.97161865234375 BW`를 기록해 고정 상한 `15 BW`를 넘었다.
- rev12 CPU rep01의 같은 cell은 `9.332860946655273 BW`였다. rev13은 force peak `+71.133147%`, root angular speed peak `+46.661288%`, joint speed peak `-32.208318%`, total excess contact delta-v `-6.887120%`, peak-step excess delta-v `-5.958619%`다. delta-v 감소와 force/root angular peak 증가는 더 시간적으로 집중되고 회전 성분이 큰 접촉 반응과 일치하지만, 이 관측만으로 해당 메커니즘의 인과를 증명하지 않는다.
- CPU 관문에서 rev13을 `rejected`로 판정했다. 따라서 GPU runtime, scratch Gate01, Gate10, PPO 학습은 실행하지 않았다. `learned_policy_qualified=false`, qualification은 `not_run`이며, 기존 strict success `0`을 성공으로 바꾸지 않는다.
- 정량 합성은 `reports/runs/g009_r0_runtime_probe_rev13_cpu_failure_synthesis_s42.json`, 공개 PNG/GIF는 `docs/media/g009/R0/diagnostic/g009_5_r0_diag_rev13_cpu_runtime_failure.{png,gif}`다. 두 공개 매체에는 `TELEMETRY ANIMATION · NOT CAMERA FOOTAGE`, `NO PPO`, `REJECTED`를 직접 표시했다.
- 로컬 전용 MP4는 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\g009_5_r0_diag_rev13_cpu_runtime_failure_s42.mp4`다. H.264 `1280×720`, `30fps`, `5.4초`, SHA-256 `2e6c38bc9ce2df3b6f50113985433d23f3f06645371e13d8cf9f0dc44940fcd0`이며 Git에는 넣지 않는다.
- 단계 번호 `04 right_side`의 실제 동작 증거도 별도로 촬영했다. clean capture commit `2c6cd014ebad03973de449ac96d16d297e74d42b`에서 원 runtime과 같은 seed `42`, CPU, `8 env`, stratified pose, env `7`, `right_side / reset_pose_hold`, physics/control `0.005/0.02s`, solver live `8/1` 조건을 headless off-screen 카메라로 `151 frames` 기록했다. 이는 조건 일치 시각 재생이며 원 report의 `15.97161865234375 BW` peak를 직접 재현했다고 주장하지 않는다.
- 실제 카메라 원본은 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\g009_5_r0_diag_rev13_04_right_side_runtime_s42.mp4`다. H.264 `1280×720`, `50fps`, `3.02초`, SHA-256 `7783b28d449874bb3a5dbb5c4d28916a0bd3e350c6e786c0c23aefc070c5eb95`이며 local-only다. 공개 GIF는 `960×540`, `30 frames`, `3.0초`, SHA-256 `ef8e57a519d3e9ce91cb0ce54bfe35b32ebd6b4418c52a71a6d34f27b6236da8`, 대표 PNG는 `1280×720`, SHA-256 `087c87ce4479962f7fe2084b3dca8d7e9f54c8cd900a565a9e90bc9b2e91457f`다. 두 파일에는 `DIAGNOSTIC`, `NOT QUALIFIED`, `NO PPO`, `RIGHT_SIDE`, `RESET_POSE_HOLD`, `REV13 REJECTED`를 표시했다.
- capture sidecar는 `reports/runs/g009_5_r0_diag_rev13_04_right_side_runtime_capture_s42.json`, 공개 파생물 sidecar는 `reports/runs/g009_5_r0_diag_rev13_04_right_side_runtime_visual_evidence.json`이다. 원 runtime commit·bundle·contract와 현재 capture commit·bundle·contract를 분리해 기록하고, MP4는 `git_policy=local_only`, GIF·PNG만 `git_public`으로 고정했다.

#### rev14 max depenetration velocity 단일변수 실제 결과

- rev14는 rev13에서 기각된 articulation solver `position=8 / velocity=1`을 유지하고 rigid-body `max_depenetration_velocity`만 `1.0 → 0.75m/s`로 낮춘 진단 후보다. calf reset, timestep, action scale/EMA, PPO noise, motor torque, reward, curriculum, termination, contact threshold는 바꾸지 않았다. source commit은 `e9c1eff15bb2679c67e325546a749dbe7f98b07c`, source bundle SHA-256은 `5c3cfa41a9c6b61a5579ed48ed17eb4f0f363eeebb9f970b61eada09fca8bacc`, contract SHA-256은 `744c53d3c8d1e608f849af405c7d0fad314b01234fc4cb9a4ab1000c69140506`이다.
- 초기 probe는 Go2 rigid body를 articulation당 13개로 잘못 가정했다. 이 preliminary report는 최종 근거로 쓰지 않았다. `root_physx_view.link_paths` 기준 실제 topology는 articulation당 19개이며, 수정한 readback은 8 articulation × 19 body = `152`개 prim 전부에서 USD/PhysX rigid-body API와 `max_depenetration_velocity=0.75m/s`를 확인했다.
- seed `42`, headless, `8 env × 150 control step`, physics/control `0.005/0.02s`, stratified pose와 zero-normalized/reset-hold action으로 CPU와 GPU를 각각 새 프로세스 세 번 실행했다. CPU 세 report와 GPU 세 report는 각각 device 안에서 의미상 동일했고 execution ID 여섯 개는 모두 달랐다. CPU runtime `3/3`, GPU runtime `3/3`, strict trade-off synthesis는 완료 단계다.
- CPU right-side/reset-hold primary peak는 세 번 모두 `8.50235366821289 BW`로 rev12 기준 `9.332860946655273 BW` 이하를 통과했다. CPU global peak는 `13.943856239318848 BW`, GPU global·primary peak는 `12.610370635986328 BW`로 모두 `15 BW` 이하였고 numeric-invalid와 hard-joint-limit termination은 CPU/GPU 모두 `0`이었다.
- CPU authoritative contact separation은 세 번 모두 `-0.010990187525749207m`였다. `-0.01m` 기준보다 `0.0009901875257492063m`, 즉 `0.9901875257492063mm` 깊다. force 감소와 더 깊은 잔류 침투가 함께 나타났으므로 strict synthesis는 rev14를 `rejected_before_gate01`로 기각했다. qualification은 `not_run`, `learned=false`다. scratch Gate01·Gate10·PPO는 실행하지 않았다. GPU runtime은 이미 완료한 단계이므로 차단 단계로 기록하지 않는다.
- 정량 합성은 `reports/runs/g009_r0_runtime_probe_rev14_tradeoff_synthesis_3x3_s42.json`이다. CPU raw report는 `g009_r0_runtime_probe_rev14_actualtopology_cpu_rep01_s42.json`부터 `rep03`, GPU raw report는 대응 `gpu_rep01`부터 `rep03`까지다.
- 단계 `04`는 env `3`, `right_side / zero_normalized` 조건의 실제 Isaac Sim headless off-screen camera footage다. 조건 일치 시각 재생이며 접촉 separation을 영상에서 직접 측정했다고 주장하지 않는다. capture commit은 `0463dc69297b6c52b546ec40670f20038a766285`, 공개 media commit은 `68fddd2`다. 로컬 전용 MP4는 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\g009_5_r0_diag_rev14_04_right_side_tradeoff_s42.mp4`, H.264 `1280×720`, `50fps`, `3.02초`, SHA-256 `0bebba8177d48357a743a9a00b93a6ed9ae403a1a53813dc71bff59c027cb865`다.
- 공개 `04` GIF·PNG는 `docs/media/g009/R0/diagnostic/g009_5_r0_diag_rev14_04_right_side_tradeoff.{gif,png}`, sidecar는 `reports/runs/g009_5_r0_diag_rev14_04_right_side_tradeoff_visual_evidence.json`이다. 단계 `05`는 force PASS와 separation FAIL을 함께 표시한 텔레메트리이며 camera footage가 아니다. 공개 파일은 `g009_5_r0_diag_rev14_05_cpu_tradeoff.{gif,png}`, sidecar는 `g009_5_r0_diag_rev14_05_cpu_tradeoff_visual_evidence.json`이다.
- Isaac Lab `RigidBodyPropertiesCfg.max_depenetration_velocity`는 solver가 접촉 침투를 해소하려고 도입할 수 있는 최대 속도다. 값을 낮추면 순간 보정 속도를 제한할 수 있지만 더 깊거나 오래 남는 침투를 만들 수 있어 접촉력 하나만으로 개선을 판정하지 않는다. `max_contact_impulse`, stabilization, contact offset, rest offset은 rev14에서 변경하지 않았다.
- 다음 rev15는 기각된 rev13·rev14 계보에서 resume하거나 누적 변경하지 않는다. 마지막 승인 runtime인 rev12의 articulation `position=8 / velocity=0`, `max_depenetration_velocity=1.0m/s`를 baseline으로 새 scratch를 만든다. 단일변수는 position iteration `8 → 16`이며 contact offset과 rest offset은 그대로 둔다. CPU separation을 progression gate에 포함해 CPU strict `3/3` 전에는 GPU와 PPO를 열지 않는다.
- 근거: [Isaac Lab rigid-body schema](https://isaac-sim.github.io/IsaacLab/v2.0.0/source/api/lab/isaaclab.sim.schemas.html), [PhysX PxRigidBody API](https://nvidia-omniverse.github.io/PhysX/physx/5.3.1/_api_build/class_px_rigid_body.html), [PhysX articulation solver API](https://nvidia-omniverse.github.io/PhysX/physx/5.3.0/_api_build/class_px_articulation_reduced_coordinate.html).

#### rev15 position iteration 단일변수와 backend force divergence

- rev15는 마지막 승인 runtime rev12 `position=8 / velocity=0`, rigid-body `max_depenetration_velocity=1.0m/s`로 돌아간 뒤 position iteration만 `8 → 16`으로 바꾼 scratch 진단이다. source commit `bc999d504e226011ff3d83e68a416b9049b406cb`, source bundle SHA-256 `218671a84f2748f7b94a426490057318b0896e2160454f6928c4277dee7435df`, canonical contract SHA-256 `5f29ba19458404b5009d3734294c57e79294efecc7fe03bf8c71c71656129832`다.
- seed `42`, headless, `8 env × 150 control step`, physics/control `0.005/0.02s`, 네 pose와 zero-normalized/reset-pose-hold action을 고정했다. CPU와 `cuda:0`에서 독립 프로세스 세 번씩 실행했고 live articulation 8개는 solver `16/0`, 152개 rigid body는 `max_depenetration_velocity=1.0m/s`였다.
- CPU는 non-foot peak `13.2482814789 BW`, authoritative separation `-0.00935308635m`, numeric-invalid `0`, hard-joint-limit `0`으로 `3/3` 통과했다. GPU는 env 7/right-side/reset-pose-hold/base, physics step `129`에서 `16.7882747650 BW`를 `3/3` 재현했다. `15 BW`보다 `11.92%` 높으므로 strict synthesis는 `rejected_before_gate01`이다.
- GPU contact separation은 authority가 없어 `unavailable`이다. 값이 비어 있다는 사실을 PASS로 해석하지 않았다. Gate01·Gate10·PPO는 실행하지 않았고, rollout batch·mini-batch·epoch·optimizer update는 모두 `0`이다. qualification은 `not_run`, `learned=false`다.
- 번호 `06`은 `cuda:0` right-side/reset-hold를 실제 Isaac Sim headless off-screen camera로 촬영한 진단 영상이고, `07`은 CPU/GPU 수치를 그린 텔레메트리다. MP4는 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\g009_5_r0_diag_rev15_06_gpu_right_side_force_fail_s42.mp4`에만 보관한다.

#### rev16 backend divergence attribution 12-run 실제 결과

- rev16 source commit은 `9ac874f48a1403e0ed838beb5e75938db5873d1c`, source bundle SHA-256은 `8b4031ad519a7487aff4eda83638c571d6494524b8872f229eba11fdb618541a`다. Arm A는 rev12 solver `8/0`, Arm B는 rev15 solver `16/0`이며 두 arm 모두 rigid-body `max_depenetration_velocity=1.0m/s`다. position iteration 외 timestep, pose/action assignment, contact threshold, reward, curriculum, termination, motor와 action 계약은 바꾸지 않았다.
- 실행 순서는 A CPU 3회 → A GPU 3회 → B CPU 3회 → B GPU 3회다. 각 그룹의 세 report를 fail-closed synthesis하고 `evidence_synthesis_valid=true`와 다음 그룹 ID를 확인한 뒤에만 다음 그룹을 열었다. 최종 입력은 12개 report, 네 그룹이며 모두 서로 다른 execution ID다.
- 각 실행은 seed `42`, headless, `8 env`, `600 physics step`, `150 control step`, physics/control timestep `0.005/0.02s`, decimation `4`다. physics row에는 env 7의 19 body force, base/non-foot force와 impulse, history slot과 clock을 기록했다. control row에는 root pose/twist, 19 link velocity, 12 joint position·velocity·torque, raw action, processed/previous EMA target을 기록했다. peak window는 physics step 기준 ±8이다.
- 실행은 GUI 창을 띄우지 않는 headless 방식이지만 PhysX와 control loop를 모두 수행했다. 진단에는 PPO runner를 만들지 않았고 rollout batch `0`, mini-batch `0`, epoch `0`, optimizer update `0`이다. RECOVER reward 식과 PPO 계약은 그대로 보존했지만 이번 결과 계산에는 사용하지 않았다.

재현 명령의 기본 형태는 다음과 같다. 실제 실행에서는 `replicate-index 1..3`과 그룹별 output 파일명을 사용하고, GPU와 다음 arm에는 직전 synthesis 파일을 `--predecessor-synthesis`로 전달했다.

```powershell
cd "$HOME\isaac-walk-rl"
$isaacPython = "$HOME\IsaacLab\_isaac_sim\python.bat"

& $isaacPython .\scripts\probe_g009_r0_rev16_backend_divergence.py `
  --arm A --replicate-index 1 --device cpu --headless `
  --output .\reports\runs\g009_r0_rev16_arm_a_cpu_rep01_retry06_s42.json

py -X utf8 .\scripts\summarize_g009_r0_rev16_backend_divergence.py `
  .\reports\runs\g009_r0_rev16_arm_a_cpu_rep01_retry06_s42.json `
  .\reports\runs\g009_r0_rev16_arm_a_cpu_rep02_retry02_s42.json `
  .\reports\runs\g009_r0_rev16_arm_a_cpu_rep03_retry02_s42.json `
  --output .\reports\runs\g009_r0_rev16_synthesis_03_a_cpu_retry02_s42.json
```

- historical reproduction은 rev12·rev15 당시의 native Torch float32 norm, mass sum, BW normalization, first-max index를 별도 projection으로 복원한다. 현재 canonical physics telemetry는 float32 source를 Python float로 옮긴 뒤 `math.fsum`과 제곱근으로 계산한다. historical fingerprint의 `abs_tol=1e-6`은 완화하지 않았다.
- 두 projection에는 별도 pair crosscheck를 적용했다. shared body/step 필드 exact, force finite/nonnegative, force delta `≤4e-6 BW`, `15 BW` 기준 양쪽 classification 동일을 모두 요구했다. 12회 모두 PASS이며 최대 delta는 B GPU `2.3343854494e-6 BW`다.

| 그룹 | right-side/reset-hold base peak | step | peak/window impulse | concentration | peak-window root/joint speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| A CPU `8/0` | `9.3328602041 BW` | `131` | `6.87535/14.06690 N·s` | `0.4887608254` | `6.58623/10.62038` |
| A GPU `8/0` | `8.7950077539 BW` | `130` | `6.47912/14.26115 N·s` | `0.4543198511` | `6.78388/7.71533` |
| B CPU `16/0` | `13.2482805877 BW` | `130` | `9.75977/14.48611 N·s` | `0.6737326952` | `6.81454/7.28135` |
| B GPU `16/0` | `16.7882770994 BW` | `129` | `12.36762/15.50992 N·s` | `0.7974004593` | `11.18898/10.78476` |

- 사전 가설의 아홉 검사는 세 replicate에 똑같이 나왔다. B GPU `>15 BW`, B CPU와 A GPU보다 한 substep 이른 peak, A GPU보다 큰 concentration, action·EMA 최대 오차 `0`, B GPU root/joint speed 상승, safety `0`은 PASS다. 유일한 FAIL은 `B GPU/B CPU concentration ≥1.20`이며 실제 비는 `1.183556126964255`다.
- 검사 하나라도 실패하면 다수결로 통과시키지 않는 계약에 따라 `hypothesis=inconclusive`, `supported_3_of_3=false`다. `18.36%`를 보고 사후 임계값을 낮추지 않았다. B GPU force도 `15 BW`를 넘으므로 position 16은 계속 기각한다.
- B CPU/GPU first-control divergence는 control step `1`의 joint velocity, first-physics divergence는 physics step `128`의 base force다. 이는 관측 시작점이지 인과 확정이 아니다. contact point·body pair·normal·separation은 CPU authority이고 GPU에서는 `unavailable`로 남겼다.
- 최종 synthesis는 `reports/runs/g009_r0_rev16_synthesis_12_full_retry01_s42.json`이다. 중간 synthesis는 `03_a_cpu`, `06_a_cpu_gpu`, `09_a_all_b_cpu` 순으로 보존한다. 최종 governance는 position16 accepted `false`, Gate01/Gate10 `forbidden`, PPO `not_run`, qualification `not_run`, `learned=false`다.
- 번호 `08`은 Arm B `cuda:0`, env 7, `right_side / reset_pose_hold`, solver `16/0`, max depenetration `1.0m/s`의 실제 Isaac Sim headless off-screen camera footage다. 조건 일치 재생이며 원 report의 force를 영상 픽셀에서 측정했다는 뜻은 아니다. 공개 GIF·PNG와 capture/visual sidecar에는 `DIAGNOSTIC`, `REJECTED`, `NO PPO`, `NOT QUALIFIED`를 고정했다.
- `08` 로컬 MP4는 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\g009_5_r0_diag_rev16_08_b_gpu_right_side_force_repro_s42.mp4`다. H.264 `1280×720`, `50fps`, `151 frames`, `3.02s`, `267,188 bytes`, SHA-256 `151146e078ce19f113e197fef931c4e32014424af2d7ce0ef20db7f6c40618b0`이며 Git 경로에는 MP4가 없다. 실행 데이터는 커밋 `9ac874f48a1403e0ed838beb5e75938db5873d1c`·bundle `8b4031ad519a7487aff4eda83638c571d6494524b8872f229eba11fdb618541a`, 카메라 recorder는 clean capture 커밋 `51f2c63eaf408525fc5ddce3249f8138b8c5baaa`·bundle `599487d4669b90472688428b2c9feb6f1d527235eec4e7017f0f2f2edd9962e1`로 분리해 검증했다.
- 번호 `09`는 A CPU/A GPU/B CPU/B GPU의 force·peak step·17-step impulse concentration·`5/10/15 BW` exposure와 `1.183556 < 1.20`을 표시한 텔레메트리다. camera footage가 아니며 GIF·PNG·JSON sidecar만 공개한다.
- 다음 rev17은 B CPU/GPU physics step `128~130`에서 peak/window impulse 분자·분모와 base·link별 하중 경로를 분리하는 진단이다. 그 결과로 물리 lever 하나를 고른 뒤 rev12 `8/0`에서 새 scratch 후보를 만든다. CPU·GPU 각 독립 `3/3`에서 force, CPU separation, numeric/hard safety를 통과하기 전에는 Gate01과 PPO를 열지 않는다.

#### 단계별 영상·공개 정책

- 원본 MP4는 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g009\R0` 아래에만 보관하고 Git에 넣지 않는다. 사용자가 확인할 원본과 합성 MP4도 `local_only`다.
- 공식 qualification을 통과한 checkpoint만 public media builder 입력으로 허용한다. 공개 저장소에는 GIF·PNG·정량 JSON sidecar만 둔다. 진단용 pilot 영상은 성공 증거와 분리하고 `diagnostic` 또는 `not_run/failed` 상태를 명시한다.
- G009-5 R0 pose 번호와 파일명은 `01 prone` → `02 supine` → `03 left_side` → `04 right_side` 순서로 고정한다. 로컬 파일은 `g009_5_r0_01_prone_s42.mp4`, `g009_5_r0_02_supine_s42.mp4`, `g009_5_r0_03_left_side_s42.mp4`, `g009_5_r0_04_right_side_s42.mp4` 형식이다.
- 공개 합성물 목표 경로는 `docs/media/g009/R0/g009_5_r0_four_pose_recovery.gif`와 대응 contact sheet PNG다. 평가·캡처·파생물 sidecar는 source commit, contract SHA, checkpoint SHA, pose 번호, headless/off-screen, ffprobe와 파일 SHA-256을 결합한다.

#### rev17 E010 offline mechanism split과 번호 10 미디어

- `G009-5-E010`은 rev16 canonical synthesis SHA-256 `d39931ad6ddf6104095a6276e9b6db3a047d044d203e034f2d38f1f172e0288d`가 묶은 12개 raw report만 다시 읽는 offline diagnostic이다. 새 Isaac Sim 실행, rollout batch, mini-batch, epoch, optimizer update, PPO update는 모두 `0`이다.
- 분석기는 12개 report의 원 SHA-256, `600` physics row, `150` control row, 19 body/12 joint 정렬, `force×0.005s` impulse, physics→control/history-slot 대응, CPU/GPU authority를 fail-closed로 다시 검증했다. 네 그룹의 physics/control/contact semantic payload는 각각 replicate `3/3` 동일했다.
- B GPU 대비 B CPU 결과는 peak base force `+26.720422%`, 17-step 전신 body impulse magnitude `+1.420270%`, base window impulse `+7.067522%`, base share `64.213292%→67.788797%`, FR+RR hip impulse `-10.833784%`다. 순간 peak 상승과 전체 window 충격량 상승을 같은 수치로 표현하지 않는다.
- focus step `128~130`에서 B CPU는 body impulse magnitude `12.273342N·s` 중 base `12.083573N·s(98.4538%)`, B GPU force aggregation은 `18.052941N·s` 중 base `14.653786N·s(81.1712%)`였다. GPU 값은 force aggregation이며 contact pair topology가 아니다.
- CPU authority 접촉 순서는 step `128: FL_hip+RL_hip`, `129: FL_hip+base`, `130: base`다. GPU callback은 `unavailable_on_gpu`이므로 CPU/GPU 접촉 topology divergence는 `step=null`이다. 최초 physics force divergence만 step `128`, `base_force_bodyweights`, delta `3.1033276173 BW`로 3/3 관측했다.
- control action·EMA trace 최대 오차는 `0`이지만 state는 control step `1`부터 달라졌다. 현재 immutable evidence만으로 solver, 초기 geometry, contact timing 중 하나를 고를 수 없어 `outcome=inconclusive`, `selected_lever=null`로 닫았다. Gate01은 `forbidden`, PPO와 qualification은 `not_run`, `learned=false`다.
- 분석 JSON은 `reports/runs/g009_r0_rev17_mechanism_split_offline_s42.json`, SHA-256 `48e596a8e61cf2b4fbcff0b1b6072d62431dc7c4c744b58db9107c398fd1cf97`, `541,762 bytes`다. `status=pass`는 분석 무결성 PASS이며 후보나 정책 qualification PASS가 아니다.
- 번호 `10`은 camera footage가 아닌 telemetry animation이다. 공개 PNG/GIF와 JSON에는 `INCONCLUSIVE`, `NO LEVER SELECTED`, `NOT PPO`, `NOT QUALIFIED`, CPU/GPU authority 한계를 직접 표시했다. PNG SHA-256은 `2e94054e25ba2a73791ab6d486884508f65bf59e571d517b798ab095dc45c925`, GIF는 `941a1956b1a91f41bd914c5188558c76ff7545bbf90deb42c50359f8c2768ff9`다.
- 개인 확인용 MP4는 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\g009_5_r0_e010_rev17_mechanism_split_s42.mp4`, H.264 `1280×720`, `30fps`, `4.8s`, `68,872 bytes`, SHA-256 `006c82cb35bda8f67c98eaa26a6a95b892e99ee84f5a584b56615c795f5f3d4c`이며 Git에 넣지 않는다.
- 전용 validator `scripts/validate_g009_r0_rev17_mechanism_media.py --check-only`는 canonical input lineage, builder/validator source hash, governance, PNG/GIF, local MP4, visual summary를 검사해 PASS했다. sidecar SHA-256은 `d3feb1f0ea67466ccfd2e03ab3338f8302b5abd23fdf9ebb1e7ed08518ed70bb`다.
- 다음 단계는 GPU authoritative constraint/contact instrumentation 가능성 조사다. 불가능하면 rev12 `8/0`에서 변경 방향과 성공·기각 기준을 먼저 고정한 단일변수 intervention probe만 연다. 원인이 선택되기 전에는 새 PPO를 실행하지 않는다.
- 최종 로컬 검증은 rev14~rev17·G009 media/probe 회귀 `452 passed in 87.56s`, 새 스크립트 Pyright `0 errors`, `py_compile` PASS, canonical resynthesis exact PASS, 전용 미디어 validator PASS, `git diff --check` PASS다. `scripts/validate_repository.ps1` 전체 검사는 이번 변경과 무관하고 status가 깨끗한 기존 rev16 CPU JSON 12개의 10 MiB 초과와 기존 `.gitattributes` 계약 1건, 총 13건 때문에 FAIL했다. 해당 baseline 파일은 이 작업에서 수정하지 않았다.

#### rev18 E011 CPU/GPU raw contact 2×2 계측 가능성 진단

- `G009-5-E011`은 rev17에서 비어 있던 GPU contact-pair authority를 실제 callback으로 확인하는 capability probe다. source commit은 `6072ac01116b8fd65f40c38d7f644ff4208a0b7e`, source bundle SHA-256은 `f6c40a35efbb380ca5644494da0e1b9812e9b66acb465aa727d63f8a9eaa2739`다. CPU 2회와 `cuda:0` 2회를 각각 새 프로세스와 고유 execution ID로 실행했다.
- 네 실행은 seed `42`, headless, camera/render off, `8 env`, source env `7`, `right_side / reset_pose_hold`, solver position/velocity iteration `16/0`, physics timestep `0.005s`를 고정했다. 실행당 정확히 `150 physics step = 0.75s`이며 action manager의 `process_action`은 step `1,5,...,149`에서 `38회` 호출했다. 매 step은 `apply_action → write_data_to_sim → sim.step(render=False) → scene.update(0.005)` 순서다.
- 일반 `env.step()`을 사용하지 않았으므로 reward manager, termination manager, curriculum manager의 post-step 계산은 호출하지 않았다. 화면에 구성된 RECOVER reward 13항은 환경 계약으로 남아 있지만 이번 JSON에는 보상값이 없다. rollout batch, mini-batch, epoch, optimizer update, PPO update는 모두 `0`이다.
- 첫 CPU rep01은 물리 루프 전에 `failed_closed`였다. `gym.make()` 뒤 env 7 articulation root에 `PhysxResidualReportingAPI.Apply()`를 늦게 적용하면서 `/World/envs/env_7/Robot/base/collisions/mesh_0`가 tensor view 사용 중 다시 파싱됐고, `CpuArticulationView::getRootTransforms`가 실패했다. 이 실행은 canonical 결과로 세지 않고 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\failed_attempts`에 `g009_5_r0_e011_rev18_cpu_rep01_preinit_failure_attempt01_s42.json`과 `g009_5_r0_e011_rev18_cpu_rep01_preinit_failure_attempt01_kit.log`로 보존했다. JSON SHA-256은 `7fc9e0dda04a1e975bbf4f6105d5cc7862df08915125e492afe46b1f16830c5e`, 로그 SHA-256은 `28335cd1df363346268f782f2df284ad58842703f58e12ee2504777bb7e06e30`이다.
- 수정 뒤 `ResidualReader`는 live USD/PhysicsContext를 바꾸지 않고 이미 사전 작성된 API만 `Get()`으로 읽는다. 새 CPU 2회 로그에서는 `was deleted while being used by a shape`, `detachShape`, `Simulation view object is invalidated`가 모두 0건이었다. residual API가 사전 작성되지 않았다는 사실은 `unavailable`로 남기고 raw contact 판정으로 승격하지 않았다.

| 슬롯 | raw callback | raw observation | probe valid | 양의 force stimulus | instrumentation bundle |
| --- | ---: | --- | --- | --- | --- |
| CPU rep01 | `150` | PASS | `true` | `true` | incomplete |
| CPU rep02 | `150` | PASS | `true` | `true` | incomplete |
| GPU rep01 | `0` | unavailable | `true` | `true` | incomplete |
| GPU rep02 | `0` | unavailable | `true` | `true` | incomplete |

- CPU 두 실행은 source env 7 기준 header `358`, contact point `767`, nonzero impulse point `414`, 최대 impulse norm `4.797629N·s`로 반복됐다. source topology와 수치가 사전 tolerance 안에서 모두 같았다. CPU force proxy 최대 norm은 약 `1951.954N`, incoming joint wrench 6D 최대 norm은 약 `396.999`다.
- GPU 두 실행은 subscription attempted/succeeded가 모두 `true`이고 malformed callback과 callback error가 없지만 callback/event가 `0`이었다. 동시에 force proxy 최대 norm 약 `2473.524N`, joint wrench 6D 최대 norm 약 `387.016`이 관측됐으므로 “접촉 자극이 없었다”는 설명은 배제한다. proxy는 raw pair topology를 대체하지 않으므로 “GPU 접촉 물리가 고장 났다”거나 “GPU에서 접촉이 없었다”고 쓰지 않는다.
- force proxy `150×19×3`과 incoming joint wrench `150×19×6`은 네 실행 모두 수집됐다. `force_matrix_w=None`, scene/source-root residual API unavailable이므로 instrumentation bundle은 `0/4 complete`, `status=unavailable`이다. 이는 raw feasibility와 독립인 보조 계측 결손이다.
- 네 raw report의 SHA-256은 CPU `f7b3f9cfc93c89d8290c645fc875605d67b8a026be9a21be5ef9ff3fa09a3c2e` / `2fd61cfe0551a4326c9d56256f32179df2ff0978a86e3b3db9aa9800dab3b994`, GPU `05bad338aa650f82d94734efdaf753e2b0cee73c4c6154772b0dfb7fdb496570` / `e7cf19001ae371fb2a0f9807a7326e7943da38abfa745dc36766a6562684622d`다. 2×2 synthesis SHA-256은 `9ca8007d88e771a5f24ca68afa46a670097e733f9e613c31fc4cc62f3fb9e01e`다.
- 합성 판정은 CPU `2/2 PASS + repeatable`, GPU `2/2 identical unavailable signature`에 따라 `outcome=unavailable_on_gpu`다. 이는 현재 조건에서 GPU raw contact-report callback 관측을 얻지 못했다는 뜻이며 일반적인 API 지원 여부나 physics ground truth 결론이 아니다. `selected_lever=null`, physics-ground-truth authority `false`, Gate01 `forbidden`, PPO·qualification `not_run`, `learned=false`다.
- 번호 `11` 자료는 카메라·보행·학습 영상이 아니라 위 2×2 결과를 재생하는 `4.8s` telemetry animation이다. 공개 PNG는 `1280×720`, `69,797 bytes`, SHA-256 `0cb74b9eab5291f91afc850771db14b9b92694475cbaf636c91c4731ea620191`, GIF는 6 frames, `37,337 bytes`, SHA-256 `208f19b30d200947490980aeb22d4b898011d13a4dce2f779aa9cc05ef8207a0`다. visual summary는 `3,790 bytes`, SHA-256 `201df8abdf380cf009a297f7be81fb360c2d7195d8eb728bfc4045d71a877f63`, 전용 validator가 확인한 sidecar는 `4,825 bytes`, SHA-256 `9c2cc14512aa271768eda68adda960bf2b5468d8b756097576a3a8264dc9180d`다.
- 개인 확인용 H.264 MP4는 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\g009_5_r0_e011_rev18_raw_contact_feasibility_s42.mp4`에만 둔다. `1280×720`, `30fps`, `4.8s`, `58,674 bytes`, SHA-256 `542a63a4a686d6d0e1eab1ba729be4aa8ea5fe6572e4ef9c1e6c9a3f341fd38e`이며 Git에는 넣지 않는다.
- canonical GPU JSON에는 subscription 성공, malformed callback `0`, callback error `null`이 기록됐다. 당시 Kit 로그에서 overflow·capacity 경고를 찾지 못했지만 그 로그는 canonical report나 sidecar에 경로·해시로 보존되지 않았으므로 공개 결론의 근거로 사용하지 않는다. GPU buffer 값을 임의로 바꾸지 않고, 다음 실행은 승인 baseline rev12 solver `8/0` 안에서 contact offset만 비교하는 동시대 Arm A/B probe다. 이 개입 전에도 PPO를 열지 않는다.

```powershell
py -m pytest -q `
  tests/test_g009_media_contract.py `
  tests/test_g009_r0_probe_synthesis.py `
  tests/test_g009_r0_diagnostic_media.py `
  tests/test_g009_r0_evaluation_media.py `
  tests/test_g009_r0_rev14_tradeoff.py `
  tests/test_g009_r0_rev14_media.py `
  tests/test_g009_r0_rev15_runtime.py `
  tests/test_g009_r0_rev15_media.py `
  tests/test_g009_r0_rev16_backend_divergence.py `
  tests/test_g009_r0_rev16_backend_divergence_summary.py `
  tests/test_g009_r0_rev16_media.py `
  tests/test_g009_r0_rev17_mechanism_split_summary.py `
  tests/test_g009_r0_rev17_mechanism_media.py `
  tests/test_g009_r0_rev18_gpu_raw_contact.py `
  tests/test_g009_r0_rev18_gpu_raw_contact_summary.py `
  tests/test_g009_r0_rev18_raw_contact_media.py
```

#### rev19 E012 contact-offset 2×2×2 A/B 사전 등록

- E012는 아직 실행 결과가 아니라 선커밋할 실험 계약이다. rev18 `16/0` 결과와 rev19 `8/0 + offset×1.5`를 직접 비교하면 solver와 offset이 함께 바뀌므로 단일변수 인과성이 깨진다. 따라서 같은 rev19 clean commit에서 Arm A와 B 모두 solver `8/0`을 쓰고 contact offset scale만 `1.0`과 `1.5`로 나눈다.
- canonical 순서는 `A CPU rep01/02 → B CPU rep01/02 → A GPU rep01/02 → B GPU rep01/02`다. CPU 네 슬롯이 raw PASS·probe-valid·manual safety available/PASS·arm별 반복성 PASS를 모두 충족하면 `reports/runs/g009_r0_rev19_contact_offset_cpu_preflight_2x2_s42.json`을 no-overwrite로 생성한다. GPU probe는 이 파일의 경로·SHA-256·source commit·probe source bundle을 AppLauncher 전에 검증해야만 실행된다. 한 arm/device가 `1/2`로 갈리면 제3회 다수결 없이 `inconclusive_nondeterministic`으로 닫는다.

| 계약 항목 | Arm A control | Arm B intervention |
| --- | --- | --- |
| solver position/velocity | `8/0` | `8/0` |
| robot contact offset | runtime baseline `×1.0` | 같은 baseline `×1.5` |
| robot rest offset | before와 동일 | before와 동일 |
| 적용 경로 | env startup의 `root_physx_view` get/set | 동일 |
| USD schema `Apply` | 금지 | 금지 |

- Go2 USD의 collision shape별 contact offset은 균일값으로 가정하거나 `0.02m`로 덮어쓰지 않는다. startup에서 `8 env × 27 collision shapes`의 실제 baseline 배열을 읽고 contact 배열의 before/after·shape·해시·최솟값·최댓값을 기록해 `after=before×scale`을 재계산한다. tensor 열은 `shape_index_00..26`으로만 식별하고 USD prim path와의 열별 대응 권위는 주장하지 않는다. collision prim path inventory는 별도 topology/count 증거다. rest setter는 호출하지 않으며 before/after 동일성이 깨지면 fail-closed다. ground는 실제 offset readback이 없으므로 unchanged를 주장하지 않고 robot root tensor setter만 호출했다는 source-bound 범위만 기록한다.
- seed `42`, `8 env`, source env `7`, `right_side / reset_pose_hold`, `150×0.005s`, action cadence 38회, headless/no-render, max depenetration `1.0m/s`, 질량·마찰·모터·reset·reward config를 고정한다. manual inner loop는 reward/termination/curriculum post-step을 계산하지 않고 PPO·Gate·qualification update는 모두 `0`이다.
- 진단 안전값은 Gate 실행과 구분한다. finite joint position/contact force, hard joint limit `+0.01rad` margin, non-foot peak force `≤15 BW`, CPU의 env별·전체·source env raw minimum separation `≥-0.01m`을 report에서 재계산한다. `robot.data.default_mass`의 `8×19` snapshot·body ordering·해시를 150 step 동안 고정하고 이 질량으로 BW 분모를 다시 계산한다. raw callback이 GPU Arm B에서만 `2/2` 생겨도 physics-ground-truth authority는 `false`, `selected_lever=null`, Gate01·PPO는 계속 닫는다.

요약기는 두 모드를 분리한다. CPU 네 report를 정확한 순서로 `--mode cpu-preflight`에 넣어 canonical preflight를 만들고, GPU 실행에는 각 probe 명령에 `--cpu-preflight reports/runs/g009_r0_rev19_contact_offset_cpu_preflight_2x2_s42.json`을 반드시 전달한다. 최종 8회 합성은 `--mode final`과 같은 preflight 입력을 사용해 `reports/runs/g009_r0_rev19_contact_offset_intervention_synthesis_2x2x2_s42.json`만 생성하며, CPU 네 report binding이 immutable artifact와 일치하고 GPU 네 report가 동일 artifact SHA를 기록했는지 검증한다. 합성의 top-level `status=complete`는 처리 완료를 뜻할 뿐 Gate·qualification PASS가 아니다.

#### rev19 E012 attempt01 collision topology 관찰 하네스 수정

- source commit `d3462069ab8dc52024730ea55287cb8382e4d50c`의 첫 CPU A1은 rollout과 정책 판정 전에 `failed_closed`로 멈췄다. 오류는 `collision topology must contain 27 unique paths per env`였으며, PhysX contact-offset setter/readback이나 학습 결과의 실패가 아니다. 이 attempt에는 PPO batch·mini-batch·epoch·optimizer update, Gate, qualification이 모두 `0/not_run`이다.
- 원인은 `stage.Traverse()`가 기본적으로 USD instance proxy 하위를 순회하지 않는다는 점을 관찰 코드가 반영하지 못한 것이다. Kit 로그에서는 Go2 articulation 8개와 articulation당 rigid body 19개가 정상 초기화됐고, 별도 read-only headless 재현에서 contact-offset tensor `[8,27]`과 env별 collision path 27개를 확인했다. 27개 collision prim은 robot root 아래 instance proxy이며 rigid body 19개와 collision shape 27개는 서로 다른 topology 수치다.
- 수정한 관찰 경로는 env별 `/World/envs/env_N/Robot` root를 검증한 뒤 `Usd.PrimRange(robot_root, Usd.TraverseInstanceProxies())`로 읽기 순회한다. env별 27개 unique path, prefix를 제거한 8환경 relative template 동일성, proxy path count를 기록한다. schema Apply·instanceable 변경·USD mutation은 하지 않으며, path inventory로 tensor column 순서를 추정하지 않는다.
- 실패 report는 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\failed_attempts\rev19\g009_5_e012_rev19_attempt01_armA_cpu_rep01_failed_report.json`, Kit 로그 사본은 같은 폴더의 `g009_5_e012_rev19_attempt01_armA_cpu_rep01_kit.log`, 두 파일의 크기·SHA-256과 실패 분류는 `g009_5_e012_rev19_attempt01_manifest.json`에 보존했다. canonical report 경로는 수정 커밋의 fresh execution ID로만 다시 사용한다.
- 근거: [OpenUSD Scenegraph Instancing](https://openusd.org/release/api/_usd__page__scenegraph_instancing.html), [OpenUSD UsdStage API](https://openusd.org/release/api/class_usd_stage.html). 기본 prim traversal은 instance proxy를 반환하지 않으며 `Usd.TraverseInstanceProxies()` predicate를 명시해야 한다.

#### rev19 E012 attempt02/03 safety observer 본체 순서 계약 수정

- topology 수정 commit `4b993a06039598ee1918f48761179bc201c4a437`의 CPU A1 attempt02는 environment 초기화와 150-step manual physics loop를 끝냈지만 report snapshot에서 `mass evidence unavailable`로 fail-closed 됐다. 이 실패도 contact-offset 물리 효과, 복구 성능, 강화학습 결과가 아니다. 실패 report와 Kit 로그는 각각 `g009_5_e012_rev19_attempt02_armA_cpu_rep01_mass_evidence_failed_report.json`, `g009_5_e012_rev19_attempt02_armA_cpu_rep01_mass_evidence_failed_kit.log`로 로컬 보존했고 `g009_5_e012_rev19_attempt02_manifest.json`에 SHA-256을 묶었다.
- canonical authority가 없는 로컬 diagnostic wrapper를 사용한 attempt03에서 내부 오류를 `ValueError: mass/contact body ordering mismatch`로 좁혔다. runtime tensor shape는 contact force `[8,19,3]`, default mass `[8,19]`로 모두 정상이었고 두 body-name 목록은 같은 19개 unique 집합이지만 순서만 달랐다. 관찰 JSON은 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\failed_attempts\rev19\g009_5_e012_rev19_attempt03_mass_observer_observation.json`, 전체 파일 binding은 `g009_5_e012_rev19_attempt03_manifest.json`에 보존했다.
- 수정 계약은 두 indexing authority를 분리한다. contact-force tensor shape는 정확히 `[8,19,3]`이어야 하며 non-foot force mask는 `sensor.body_names` 순서로 `sensor.data.net_forces_w`에 적용한다. mass snapshot과 150-step 안정성은 `robot.body_names` 순서로 `robot.data.default_mass`에 묶는다. body-weight 분모는 `sum(robot.data.default_mass, dim=1)×9.81`이므로 순서 불변이며 per-body force/mass pairing은 사용하지 않는다. 두 목록은 각각 길이 19·unique·동일 이름 집합이어야 하지만 ordered equality는 요구하지 않는다. report, CPU preflight, final synthesis는 mass ordering hash와 contact-force ordering hash를 따로 검증한다.
- observer가 한 번이라도 실패하면 이후 sample을 누적하지 않고 첫 오류를 보존한다. 누락·중복·다른 inventory, 실행 중 ordering 변화, 어느 한쪽 ordering hash drift는 모두 fail-closed다. corrected source를 별도 commit·push한 뒤 canonical CPU A1부터 새 execution ID로 다시 실행한다.

#### rev19 E012 contact-offset 2×2×2 실제 결과

- corrected source commit `723770e57a2f1e76912bbace174db64cf8571f81`에서 `A CPU 1/2 → B CPU 1/2 → CPU preflight → A GPU 1/2 → B GPU 1/2` 순서를 지켰다. 각 실행은 seed `42`, headless, 8 env, 150 physics step, solver `8/0`, max depenetration `1.0m/s`, `right_side / reset_pose_hold`이며 PPO·reward post-step·termination manager post-step·curriculum update는 실행하지 않았다.
- CPU 네 실행은 raw callback `150/150`, probe-valid, offset integrity, manual safety를 모두 통과했다. A와 B 모두 all-env non-foot peak `9.408609390258789 BW`, source env peak `9.332860946655273 BW`, all-env minimum separation `-0.009375497698783875m`, source env minimum `-0.009373113512992859m`였고 arm별 두 반복은 structure/numeric/offset/safety가 exact repeatable했다.
- GPU 네 실행은 모두 AppLauncher `cuda:0`, GPU dynamics readback PASS, manual safety PASS였지만 raw contact-report callback count가 `0`이었다. Arm A/B 모두 all-env force proxy `9.400354385375977 BW`, source env `8.79500675201416 BW`로 exact repeatable했다. callback error는 `null`이지만 GPU raw separation authority는 없으므로 “GPU 접촉이 없었다”거나 “물리가 실패했다”고 주장하지 않는다.
- Arm A는 baseline contact-offset hash와 after hash가 모두 `5beda7da5ee87b85b93a120e3e4007ea0affde47198a41d5e7966e583acdeb55`, Arm B after hash는 `f373b055b4bb9f380b651f2e7051e7f1230110d48f1bb9e3647a63b97b77f3c6`이었다. 두 arm의 baseline·rest·solver·mass·body inventory는 같고 scale만 `1.0/1.5`로 달랐지만 CPU/GPU force proxy와 callback availability 상태는 arm 간 동일했다.
- CPU preflight는 `reports/runs/g009_r0_rev19_contact_offset_cpu_preflight_2x2_s42.json`, SHA-256 `bf7218b198995bb1a6c4b53d075204174c3947726571e886623ffa933ac9fd49`이며 GPU를 정식 승인했다. 최종 합성 `reports/runs/g009_r0_rev19_contact_offset_intervention_synthesis_2x2x2_s42.json`은 SHA-256 `5d95449398b4168cc8a7d0f73d4248e77fd2257b05f5bce22e4431020f8bf576`, outcome `gpu_raw_unavailable_both_arms`, next step `stop_without_gpu_contact_absence_claim`, `selected_lever=null`이다. contact offset `×1.5`는 채택하지 않으며 Gate01·PPO·qualification은 계속 `forbidden/not_run`이다.
- 시각 증거는 로봇 카메라 영상이 아니라 canonical JSON 10개에서 만든 번호 `12` telemetry animation이다. `12.01 CPU preflight`와 `12.02 final CPU→GPU`를 분리했고 모든 프레임에 `NOT CAMERA FOOTAGE / DIAGNOSTIC ONLY / NO PPO / NOT QUALIFIED`를 표시했다. 공개 파일은 `docs/media/g009/R0/diagnostic/g009_5_r0_e012_rev19_contact_offset_intervention.gif`와 대표 PNG 2장, provenance는 `reports/runs/g009_5_r0_e012_rev19_contact_offset_intervention_visual_evidence.json`이다. H.264 MP4는 Git에 넣지 않고 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\g009_5_r0_e012_rev19_contact_offset_intervention_s42.mp4`에만 보존했다. MP4는 `1280×720`, `30fps`, 168 frames, `5.6s`, SHA-256 `baf9c0facd73b88d376d2c88a7ee5de019039b827d60870b0914e206bdbb9d03`이다.

#### rev20 다음 진단: GPU terrain-pair force matrix authority

- rev20의 단일가설은 “GPU rigid contact는 force sensor tensor에 존재하지만 전역 Python contact-report callback은 현재 clone/GPU 경로의 판정 권위가 아니며, terrain-filtered `RigidContactView.get_contact_force_matrix(dt)`는 같은 step의 terrain-pair normal force를 제공한다”이다. contact offset, friction, solver, GPU buffer, Direct GPU API, physics/control dt를 바꾸지 않는다.
- Isaac Lab 2.1.1 `ContactSensor.net_forces_w`는 sensor body별 모든 접촉의 합산 normal force이고, filter를 명시하면 `force_matrix_w`/`contact_physx_view`에서 sensor-body↔filter-body normal force를 구분할 수 있다. PhysX contact-report callback은 FOUND/PERSISTS/LOST header와 impulse/contact detail을 내는 별도 event API다. 따라서 callback count와 step별 tensor force를 같은 양으로 직접 비교하지 않는다.
- 먼저 runtime method/version guard와 reporter schema·threshold, ordered sensor paths, terrain filter paths·cardinality를 source-bound report에 기록한다. 같은 fresh reset/seed/dt에서 기존 `net_forces_w`와 terrain-filtered force matrix를 매 step 함께 읽는다. matrix-only probe이므로 detailed `get_contact_data`, `max_contact_data_count` 증설, buffer tuning은 하지 않는다.
- PASS는 view/API·path/filter 검증, finite tensor, exact ordering hash, 그리고 `net_forces_w > 1e-6N`인 step 중 terrain-pair force matrix도 `>1e-6N`인 step이 CPU와 GPU에서 각각 존재하는 것이다. callback은 결과와 무관한 diagnostic counter로만 남긴다. FAIL이면 offset/solver를 바꾸지 않고 sensor rigid-body/filter collider 선택만 dump·교정한다. PASS 뒤에야 GPU contact authority를 matrix로 전환한 새 safety gate를 별도 사전등록한다.

#### rev20 E013 terrain-pair matrix 사전등록

- machine-readable 계약은 `configs/g009_r0_rev20_terrain_contact_matrix.json`이다. 증거 ID `G009-5-E013`, 미디어 번호 `13`, 슬롯은 `cpu.rep1 → cpu.rep2 → cuda:0.rep1 → cuda:0.rep2`로 고정했다. CPU 2회가 source/baseline/view/filter/shape/finite/same-body overlap/safety/반복성을 모두 통과한 immutable preflight를 만들기 전에는 GPU AppLauncher를 열지 않는다.
- terrain filter는 rev19 CPU raw callback의 actor/collider path로 실제 확인한 `/World/ground/terrain/GroundPlane/CollisionPlane` 하나다. expected filter count `1`, sensor count `8×19=152`, direct matrix raw shape `[152,1,3]`, reshape와 ContactSensor buffer shape `[8,19,1,3]`를 요구한다. 매 step direct `RigidContactView.get_contact_force_matrix(0.005)`와 `sensor.data.force_matrix_w`가 exact equality여야 한다.
- canonical 실행 전 설치본 ABI 감사에서 `view.filter_paths/filter_names`는 논리 필터 1개를 152개 sensor별 row에 반복한 `[152,1]` 중첩 목록으로 확인됐다. raw 152개 row가 모두 exact path/name인지 검증하고, `logical_filter_paths_sha256`은 공통 논리 row 한 개, `raw_filter_paths_sha256`은 `[152,1]` 전체에 대해 각각 계산해 의미를 분리한다. `robot.root_physx_view.prim_paths`도 namespace가 아니라 `.../Robot/base` root-body path이므로 마지막 `/base`를 제거한 `.../Robot`을 body namespace로 도출한다. report에는 원본 root-body path와 도출 namespace를 함께 기록하고, 각 19개 sensor chunk의 direct-child leaf 순서가 `sensor.body_names`와 같은지 확인한다.
- threshold는 `1e-6N`이다. 같은 env·body·physics step에서 `net_forces_w`와 terrain matrix sum이 동시에 threshold를 넘는 step이 8개 각 환경에 최소 1회 있어야 한다. callback count와 CPU raw separation은 이 판정에 쓰지 않는다.
- rev19 Arm A baseline의 contact/rest/mass/body-order hash와 solver `8/0`, max depenetration `1.0m/s`를 그대로 요구한다. 여기에 ground/foot/effective 마찰 readback, action `scale=0.70`·EMA `alpha=0.2`, Go2 DCMotor의 raw config(`armature/effort_limit_sim=null`)와 resolved tensor(`0.0/1e9`)를 분리한 effort·velocity·stiffness·damping·armature/friction hash, XY/yaw range가 모두 `[0,0]`인 8환경 stratified reset root/joint state와 zero/hold target, physics/control dt `0.005/0.02s`·decimation `4`를 snapshot/hash로 묶었다. offset/rest/friction/mass/motor/reset/dt/GPU buffer/Direct GPU API/`get_contact_data` 변경은 금지한다.
- 각 raw report는 고유 lowercase UUID4 execution ID와 canonical slot/output identity를 내장하고, 네 report의 path·SHA-256·execution ID 중복을 모두 거부한다. CPU preflight는 정확히 두 CPU `{path,sha256}`를 순서대로 bind하고, GPU report는 그 immutable preflight 전체를 AppLauncher 전에 검증한다. final은 CPU 두 binding이 preflight와 exact-equal이고 GPU 두 report가 같은 preflight를 가리킬 때만 해석한다. 성공해도 바로 PPO나 Gate01을 열지 않고 matrix authority safety gate를 새로 사전등록한다.

#### rev20 E013 terrain-pair matrix 실제 결과

- source commit `fb2992965fcfb502a679065eac253a6bdcdf7086`에서 canonical 순서 `cpu.rep1 → cpu.rep2 → CPU preflight → cuda:0.rep1 → cuda:0.rep2 → final synthesis`를 완료했다. raw probe source bundle SHA-256은 `21353b2d90e43260e8446df094ef178264227a1040901d99230fb2c40b99b83c`, synthesis source bundle SHA-256은 `bf00813f9051969377d0bf7dee8092eff2df4643de3e29f2eff40d31d3be62e8`이다. 네 raw report의 path·SHA-256·execution ID는 모두 고유했고 CPU/GPU 각 두 반복은 exact field와 허용오차 대상 numeric field에서 반복 가능했다.
- 실행은 Isaac Lab/Isaac Sim의 GUI 창을 띄우지 않는 `headless=true`, `render=false` 방식이다. 각 raw run은 별도 프로세스에서 seed `42`, 8 env, 150 physics step, `physics_dt=0.005s`, control decimation `4`, environment/control step `0.02s`, 총 simulation time `0.75s`를 수행했다. 일반 `env.step()` 대신 `sim.step(render=false) → scene.update(0.005)` manual PhysX inner loop를 사용했다. 환경에 RECOVER reward manager는 구성돼 있지만 이 진단은 reward 계산·합산·최적화를 호출하지 않았고 PPO rollout batch, mini-batch, epoch, optimizer/PPO update는 모두 `0`이다.
- reset class는 env `0~7`에 `[prone, supine, left_side, right_side]`를 두 번 stratified 배치한 `[0,1,2,3,0,1,2,3]`이다. env `0~3`은 12차원 `zero_normalized`, env `4~7`은 같은 folded reset joint state를 유지하는 `reset_pose_hold` action group이다. 이것은 보행 command나 학습 action이 아니라 진단 중 초기 자세·접촉 자극을 고정하는 제어 입력이다.
- 이번 terrain class는 혼합 지형이 아니라 단일 `/World/ground/terrain/GroundPlane/CollisionPlane`이다. ground static/dynamic friction은 `0.8/0.6`, foot은 `1.0/1.0`, combine mode는 `multiply`, live effective friction은 `0.8/0.6`이다. solver `8/0`, max depenetration `1.0m/s`, contact/rest offset·질량·관성·모터·reset·timestep은 rev19 Arm A baseline과 같고 setter를 추가 호출하지 않았다. 마찰 mosaic, 요철, 경사, 링크 질량 변화는 rev20의 검증 변수가 아니다.
- contact matrix는 8 env × 19 body의 sensor 152개와 terrain filter 1개로 구성했다. direct raw `[152,1,3]`을 env-major/body-major `[8,19,1,3]`으로 reshape하고 ContactSensor buffer `[8,19,1,3]`, `net_forces_w [8,19,3]`와 같은 step에 비교했다. direct/buffer exact equality와 서로 다른 storage를 150/150 step에서 확인했다. raw `[152,1]` filter metadata SHA-256은 `f123085b9f380151dd660c449197a0a7c19e64c1c41f104a4ba5607693f86a4c`, 공통 논리 filter row SHA-256은 `0e7310394b8a9adb8b4cd6fe66f00c662855733094e5252dcd70acf1f7fcf6c0`이다. 두 해시는 같은 의미가 아니다.

| slot | raw report SHA-256 | execution ID | all/source peak | all/source integral | max/source non-foot |
| --- | --- | --- | --- | --- | --- |
| `cpu.rep1` | `d4f8a371edd77c69fb74994c56d629c3e27dd122907ade90f931eeb546c41c29` | `713be4a4945f45428336177706945a31` | `1386.23046875 / 1375.0699462890625N` | `112.00597896575924 / 64.69987430453298N·s` | `9.408608784179021 / 9.332860204105833BW` |
| `cpu.rep2` | `63d19f42e7c79cc77846f09b5245c2dc46e77630ce97020dfb29be0837375e6c` | `54ce2a52afad4f55a7b7d536579d48aa` | `1386.23046875 / 1375.0699462890625N` | `112.00597896575924 / 64.69987430453298N·s` | `9.408608784179021 / 9.332860204105833BW` |
| `cuda:0.rep1` | `363e3bfe3d3ed3b1c3bfe7c3ae0aecf6a1a4f6704cdb9c8533a73e6e3f75cd0a` | `a12591ace8f74343b8595d9ab6481af1` | `1385.014404296875 / 1295.82470703125N` | `119.45139554977413 / 65.21480977773665N·s` | `9.400355124377333 / 8.79500775388683BW` |
| `cuda:0.rep2` | `305914af13be60c869b6b3a6d103b37a616be7d1c4e6a7e99dceeb8ef53698ca` | `055f49e11d85410c8d35901407d5a9eb` | `1385.014404296875 / 1295.82470703125N` | `119.45139554977413 / 65.21480977773665N·s` | `9.400355124377333 / 8.79500775388683BW` |

- 환경별 same-body positive overlap step 수는 네 실행 모두 `[150,147,150,150,149,146,150,150]`, source env 7은 `150`이다. joint/contact finite, hard joint limit `±0.01rad` margin, default mass `[8,19]` finite·positive·unchanged, force/mass body inventory, collection error absent를 모두 통과했다. non-foot 한계는 `≤15BW`이고 관측 최대는 CPU `9.408608784179021BW`, GPU `9.400355124377333BW`였다.
- CPU preflight는 `reports/runs/g009_r0_rev20_terrain_contact_matrix_cpu_preflight_2x_s42.json`, SHA-256 `2c4996f837d0c6003d653761c53ac399e98fb80db25c7d00ee647b871ac4968c`, execution ID `e7c172f8bec647a28559f55be420929e`이며 `gpu_stage_authorized`다. final synthesis는 `reports/runs/g009_r0_rev20_terrain_contact_matrix_synthesis_2x2_s42.json`, SHA-256 `dcb8f446a212390f94f9ae5ccad97d9e770f9b8f5961f5ffb0c920f8d62580b3`, execution ID `1c8d85a4c7db4f76aee1a55ed9413ddb`이며 outcome은 `terrain_pair_matrix_authority_candidate_validated`, next step은 `preregister_matrix_authority_safety_gate`다.
- outcome은 terrain-pair aggregated normal force를 CPU와 GPU에서 판정할 관측 권위 **후보**를 검증했다는 뜻이다. contact point·separation·충격량의 physics ground truth, 보행, 방향 전환, 경사 주행, 전복 자가복구, 강화학습 또는 policy qualification 성공이 아니다. `physics_ground_truth_authority=false`, Gate01 `forbidden`, qualification `not_run`, `learned=false`를 유지한다.
- 번호 `13.01`과 `13.02`는 simulation camera footage나 로봇 운동 영상이 아니라 canonical JSON을 움직이는 그래프로 재구성한 telemetry animation이다. `13.01` 공개 GIF/PNG SHA-256은 `859ea5ec5424323367d15680964e9fec775384f0ca44c2783a70ebfd9bb534d8` / `e476af72fd62d5cec233db481c68690c36cec2c2549e86e349fe70eced64b7e6`, visual summary/sidecar는 `f15ee635c548ff17eb06f20871ffd2bf3b016df9b0576a3fc645012ce2a4d79d` / `0c2ff4f5c9a1076b6974949c3a21e91eada2d5244ca031f9cd72d26efc300e1c`다. `13.02` 공개 GIF/PNG는 `86a8f2e44d58f69212f7e95e0da9cb28b2eee6198fd7b54a740f115ae21c704d` / `c7b011bd7297ae7f173e3989cb4fb80aa9832ab8375a94e89f25f2ad05be1e59`, visual summary/sidecar는 `53f7a03d6bf524169d9593b752f044a2645486584710330301bdd3ec08c8963b` / `5520bb2ec3ca0bc7dba3ebfa5aa00fa27faa48e4110ef575e1bd438c4737892f`다. MP4는 각각 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\g009_5_r0_e013_rev20_cpu_preflight_s42.mp4` (`fa8eaa20eaadc612cbdb323d5e24c2fcda2c4007864c0fed99215adaa8462245`), `%USERPROFILE%\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\g009_5_r0_e013_rev20_final_cpu_gpu_s42.mp4` (`8eb136daf014b3cc218926f83e4bc8526742fd71a0b6e2484460bc345b9a0b9f`)에만 둔다. Git에는 GIF·대표 PNG·hash-bound JSON만 넣고 모든 프레임에 `TELEMETRY ANIMATION / NOT CAMERA FOOTAGE / DIAGNOSTIC ONLY / NO PPO / NOT QUALIFIED`를 표시한다.
- 다음 실행 순서는 matrix authority safety gate 사전등록 → matrix 기반 Gate01 fresh scratch → Gate01 safety PASS 이후에만 R0 PPO qualification이다. 그 뒤 WALK는 낮은 contour 경사 `5/10°`에서 시작해 `15/20°`로 올리고 `25°`는 stress로 유지한다. 외란 pulse와 residual height를 분리해 넣은 뒤 controlled 발별 마찰, 비주기 spatial friction mosaic, 실제 WALK 낙상 snapshot 기반 RECOVER, 낮은/높은 경사 전복 복구, spatial friction 복구, `push → fall → recover → stand → command resume` 순으로 결합한다. 링크 질량·관성 변화는 final-heldout 계약을 먼저 고정한 다음 hip/thigh/calf/foot 그룹을 한 번에 하나씩 바꾸는 별도 M1 goal로 수행한다.

#### rev21 E014 matrix authority safety gate 실제 결과

- source commit `e202ae1d514c7abfe05ce0da130c2db47e9e05f3`에서 Python 정적 bounded recursive evidence gate를 실행했다. rev21 source bundle SHA-256은 `3589f3852c4365afadb8dbdc871ffb2fe81888155b7e66c274034b5506eaf40e`다. canonical artifact는 `reports/runs/g009_r0_rev21_matrix_authority_safety_gate_s42.json`, SHA-256은 `68a60383d9d49bf009d189498f94e6fe1c03155259932dcff8965caa7d9aa250`다.
- 입력 증거는 rev20 final synthesis `dcb8f446a212390f94f9ae5ccad97d9e770f9b8f5961f5ffb0c920f8d62580b3`, CPU raw report 2개, GPU raw report 2개, CPU preflight, historical source commit `fb2992965fcfb502a679065eac253a6bdcdf7086`의 Git blob 12개다. gate는 고정 path와 SHA-256, historical blob, source bundle, canonical ordering, tensor shape·ordering·finite·storage independence·direct/buffer parity·same-body overlap·mass continuity·non-foot peak·joint margin을 저장된 증거 범위 안에서 재계산했다.
- 판정 항목 `17/17`이 PASS했고 outcome은 `matrix_authority_safety_gate_passed_for_diagnostic_preregistration`, next step은 `preregister_read_only_matrix_observation_adapter`다. canonical 생성 전 `--check-only`는 파일을 쓰지 않고 PASS했고, 생성 뒤 full artifact verification도 PASS했다. 같은 runner를 다시 실행하면 exit code `2`, reason `canonical_output_already_exists`로 종료됐으며 canonical artifact SHA-256은 바뀌지 않았다. 재실행 실패 envelope는 `C:\Users\LIMMM\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\failed_attempts\rev21\g009_5_e014_rev21_matrix_authority_safety_gate_75f18ba89aa74171bc8a26797416865d.json`에 로컬로 남겼다.
- 이 단계는 Isaac/PhysX를 다시 실행하지 않은 Python 정적 검증이다. simulator launch `0`, rollout step `0`, reward 계산 `0`, PPO/optimizer update `0`이다. 보행·앞뒤·좌우 회전·경사 주행·전복 자가복구를 qualification한 결과가 아니다. `15BW`와 hard joint limit 바깥 `-0.01rad` diagnostic margin은 저장된 rev20 증거를 해석하는 소프트웨어 판정 기준이며 실물 로봇의 안전 한계가 아니다.
- 실행 동작, 물리 조건, checkpoint가 바뀌지 않았으므로 새 동작 영상을 만들지 않았다. E013 telemetry GIF/PNG와 로컬 MP4를 같은 입력 증거의 시각 자료로 재사용하며, 정적 gate PASS를 새 로봇 동작 영상처럼 연출하지 않는다. Garden 공개는 read-only runtime adapter와 실제 시뮬레이션 결과가 나온 뒤로 보류한다.
- 다음 `preregister_read_only_matrix_observation_adapter`에서는 XYZ vector와 magnitude 중 정책 입력 표현, world/local 좌표계와 body-filter 축 처리 순서, normalization·clipping·missing contact 규칙, dtype·device·output shape, 원본 source tensor 불변성을 먼저 고정한다. 이 단계에서도 reward·policy·PPO 연결을 금지하고 rollout step `0`을 유지한다.
- rev21은 파일·JSON·Git blob을 검증하는 정적 작업이라 GPU 이점이 없으며 CPU에서 수행했다. GPU PhysX나 PPO를 실행한 것처럼 기록하지 않는다. rev22 adapter 사전등록을 통과한 뒤 실제 runtime 재관찰과 이후 학습은 `cuda:0`과 GPU dynamics를 사용한다. env·batch는 낮은 값에서 시작해 VRAM 여유, OOM 부재, physics stability를 확인하며 단계적으로 늘리고 GPU utilization, peak VRAM, steps/s를 실행별로 기록한다. 목표는 GPU 점유율을 무조건 `100%`로 만드는 것이 아니라 같은 계약을 반복해도 안정적인 최대 throughput을 찾는 것이다. CPU는 preflight, summary와 backend 재현 대조에 사용한다.

검증 명령은 Windows PowerShell에서 다음 순서로 실행한다. canonical artifact가 이미 존재하는 현재 상태에서 runner의 첫 번째 명령은 no-overwrite 규칙에 따라 exit code `2`가 정상이며, 최초 생성 전에는 read-only PASS였다.

```powershell
cd C:\Users\LIMMM\isaac-walk-rl
py C:\Users\LIMMM\isaac-walk-rl\scripts\run_g009_r0_rev21_matrix_authority_safety_gate.py --check-only
py C:\Users\LIMMM\isaac-walk-rl\scripts\summarize_g009_r0_rev21_matrix_authority_safety_gate.py --verify-artifact C:\Users\LIMMM\isaac-walk-rl\reports\runs\g009_r0_rev21_matrix_authority_safety_gate_s42.json
Get-FileHash -Algorithm SHA256 C:\Users\LIMMM\isaac-walk-rl\reports\runs\g009_r0_rev21_matrix_authority_safety_gate_s42.json
```

#### rev22 E015 read-only matrix observation adapter 사전등록 실제 결과

- source commit `0eca6b7cd4ec74e9cf28762b8e51ae80ddb73766`에서 정적 계약 gate를 실행했다. rev22 source bundle SHA-256은 `db4cd56fa83bb97bb4d335a6ca1c4181c71314b9cbaedb70eb517877f87d1489`, adapter contract SHA-256은 `05105dbb7cf8646d0c7a5bf667cc9ab78de76131819a9654e43d9465a31d5b43`다. canonical artifact는 `reports/runs/g009_r0_rev22_read_only_matrix_observation_adapter_preregistration_s42.json`, SHA-256은 `a8e536c7f5b739b983c8d1ce05c701b725b46b8926acc33b457c56ff9fad2343`, execution ID는 `9320566d739e45e285afdae75fed0db7`다.
- `18/18` 판정 항목이 PASS했고 outcome은 `read_only_matrix_observation_adapter_preregistration_passed`, next step은 `implement_and_run_read_only_matrix_observation_adapter_runtime_probe`다. canonical 생성 전 `--check-only`는 exit code `0`으로 끝났고 worktree와 output을 바꾸지 않았다. 생성 후 full verifier도 exit code `0`이었다. 같은 runner의 재실행은 exit code `2`, reason `canonical_output_already_exists`로 닫혔고 canonical SHA-256은 유지됐다. 실패 envelope는 `C:\Users\LIMMM\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\failed_attempts\rev22\g009_5_e015_rev22_read_only_matrix_observation_adapter_e585c8e237254f56892db39c9dc3d225.json`에만 보존했다.
- authoritative 후보 출력은 `terrain_pair_force_matrix_w [N,19,1,3]`의 filter 축을 먼저 합산한 world-frame XYZ `[N,19,3]`, `torch.float32`, source와 같은 device다. magnitude `[N,19]`는 합산 뒤 norm을 계산하는 진단값이며 정책 입력으로 허용하지 않는다. contact mask는 `magnitude > 1e-6N`이고, 정확히 임계값과 같은 값은 `false`다. 정상적인 zero contact는 XYZ·magnitude `0`, mask `false`로 유지하지만 source missing, shape·dtype·device 불일치, nonfinite는 zero-fill 없이 fail-closed한다. normalization, clipping, saturation, device fallback은 모두 금지했다.
- adapter는 source의 shape·dtype·device·stride·storage identity·version·exact value SHA-256을 바꾸지 않아야 하며 모든 output은 source storage와 non-alias여야 한다. 이 계약은 구현 전 oracle과 fixture를 고정한 것이며 runtime adapter 구현이나 PhysX tensor 재관찰 결과가 아니다.
- Isaac Lab 2.1.1의 `ContactSensorData.force_matrix_w` 의미를 다시 대조해 이 source를 **world-frame filtered normal contact force vector**로 제한했다. total contact force나 tangential friction force가 포함된다고 주장하지 않으며, 마찰 효과를 이 matrix만으로 직접 관측했다고도 주장하지 않는다. 이후 공간 마찰 실험은 발 미끄럼·접선 상대속도·command tracking·base drift를 별도 관측해야 한다.
- simulator launch `0`, rollout step `0`, reward 계산 `0`, policy 연결 `0`, PPO/optimizer update `0`이다. 보행·앞뒤·좌우 회전·경사 주행·전복 복구·마찰 적응·강화학습 성공 증거가 아니다. 실행 동작, 물리 조건, checkpoint가 바뀌지 않아 새 영상·GIF·PNG를 만들지 않았다. E013 telemetry는 predecessor 문맥으로만 재사용하며 rev22 성공 미디어로 표시하지 않는다.
- 검증은 rev22 대상 `56 passed`, rev20·rev21·Gate01을 포함한 통합 회귀 `245 passed`, JSON parse, `py_compile`, `git diff --check`, placeholder/skip scan PASS다. 독립 verifier 최종 판정은 CRITICAL/HIGH/MEDIUM/LOW `0/0/0/0`, `APPROVE`다. Windows 권한상 실제 symlink/reparse-point 생성 probe는 실행하지 못했고, 해당 입력을 거부하도록 작성된 코드 경로만 검토했다. `ruff`, `pyright`, `basedpyright`, `mypy`는 현재 환경에 설치되지 않아 실행하지 않았다.
- Garden 발행은 보류한다. rev22만으로는 새 시뮬레이션 동작, 새 물리·학습 결과, 직접 관련된 새 미디어가 없어 별도 공개 글의 핵심 성과로 삼기 부족하다. 다음 실제 CPU/GPU runtime adapter 결과와 직접 관련 telemetry chart·GIF/PNG가 생기면 rev20 공개 글의 후속 내용으로 묶어 발행 여부를 다시 판정한다.
- 다음은 adapter만 구현한 뒤 `8 env × 150 physics steps`를 CPU `2회`, `cuda:0` `2회` 독립 실행한다. 네 실행에서 source 불변성, XYZ·magnitude·mask oracle, shape·dtype·device, finite, repeatability를 모두 통과하기 전에는 Gate01, reward, policy, PPO를 열지 않는다. correctness PASS 뒤 별도 throughput 계약을 사전등록하고 GPU env ladder `[8,32,128,256,512,1024]`에서 median env-control-steps/s, peak VRAM, GPU utilization, OOM·numeric·physics stability를 기록한다. 목표는 GPU 점유율 `100%`가 아니라 반복 안정성을 유지하는 최대 throughput이다.

현재 canonical artifact가 이미 존재하므로 아래 runner를 `--check-only`로 다시 실행하면 no-overwrite 규칙에 따라 exit code `2`가 정상이다. full verifier와 해시 확인은 계속 read-only로 실행할 수 있다.

```powershell
cd C:\Users\LIMMM\isaac-walk-rl
py C:\Users\LIMMM\isaac-walk-rl\scripts\run_g009_r0_rev22_read_only_matrix_observation_adapter.py --check-only
py C:\Users\LIMMM\isaac-walk-rl\scripts\summarize_g009_r0_rev22_read_only_matrix_observation_adapter.py --verify-artifact C:\Users\LIMMM\isaac-walk-rl\reports\runs\g009_r0_rev22_read_only_matrix_observation_adapter_preregistration_s42.json
Get-FileHash -Algorithm SHA256 C:\Users\LIMMM\isaac-walk-rl\reports\runs\g009_r0_rev22_read_only_matrix_observation_adapter_preregistration_s42.json
```

#### rev23 E016 read-only matrix observation adapter runtime 실제 결과

- source commit `4f4fb71b5948fa909ea298ff5cce2809f66ef05f`, rev23 source bundle SHA-256 `60e77825dc1616cd32cc314343c65e4723842359fd0c6d8c93f34a34135407c4`에서 canonical 순서 `cpu.rep1 → cpu.rep2 → CPU preflight → cuda:0.rep1 → cuda:0.rep2 → final synthesis`를 완료했다. 단일 변경 축은 rev22에서 사전등록한 read-only matrix adapter의 구현과 실제 runtime 관찰이다. friction, mass·inertia, terrain, reset, action, solver, contact/rest offset, reward, checkpoint, curriculum은 바꾸지 않았다.
- 각 raw run은 Isaac Sim 창을 띄우지 않는 `headless=true`, `render=false`, `fast_shutdown=false`로 실행했다. `Isaac-G009-Recover-Flat-Go2-R0-v0`, seed `42`, 8 environment, 150 physics step, `physics_dt=0.005s`, control period `0.02s`다. source는 실제 `contact_forces.data.force_matrix_w [8,19,1,3]`, `torch.float32`이고 CPU 또는 `cuda:0` 요청 device를 그대로 유지했다.
- adapter는 filter 축을 합산한 world-frame XYZ `[8,19,3]`, XYZ norm인 magnitude `[8,19]`, `magnitude > 1e-6N`인 bool mask `[8,19]`를 만들었다. 150개 step 모두 source shape·dtype·device·stride·storage pointer·version·exact-value hash가 전후 동일했고, 세 output은 source 및 서로의 storage를 alias하지 않았다. normalization, clipping, saturation, zero-fill, dtype cast, device transfer·fallback은 사용하지 않았다. step `1/50/100/150` snapshot도 독립 Torch oracle과 일치했다.

| slot | raw report SHA-256 | execution ID | max magnitude | magnitude integral | zero source vectors |
| --- | --- | --- | ---: | ---: | ---: |
| `cpu.rep1` | `1f01963e09574ec1388669dac75ed44cddd787f18f8dc7d806c72b11951d3660` | `d539a1102a524e0a9ab196668edfb260` | `1386.23046875N` | `112.00597896575924N·s` | `18,471` |
| `cpu.rep2` | `6ccd4ce4cdf681524505c5122a2a0282097afd31f23129e0a30101e27f632844` | `810a50e8837c4ca498f33a73ba592fdc` | `1386.23046875N` | `112.00597896575924N·s` | `18,471` |
| `cuda:0.rep1` | `049487df5fdc6a2c3310fcf65eb1ef15761a1a06c1160e52af619d96293c2446` | `0801f4b50c804d92b3079838f6ba23cc` | `1385.014404296875N` | `119.45139554977413N·s` | `18,494` |
| `cuda:0.rep2` | `273d2242a6bbc55bd169507b35b4bb7db15caf96d082edab98006f8e2b63a7ae` | `136c70463699438d8fd5d3fb24841dae` | `1385.014404296875N` | `119.45139554977413N·s` | `18,494` |

- `repeatable=true`는 CPU 내부 두 반복과 `cuda:0` 내부 두 반복을 각각 판정한 값이며 CPU↔GPU 동등성 gate가 아니다. 장치 간 참고 차이는 max magnitude `1.216064453125N`(`0.0877%`), magnitude integral `7.445416584014893N·s`(`6.6473%`), zero source vector `23`(`0.1245%`)이다. 현재 계약은 이 차이를 PASS/FAIL로 사용하지 않는다.
- [CPU preflight](reports/runs/g009_r0_rev23_matrix_observation_adapter_cpu_preflight_2x_s42.json)의 SHA-256은 `f8e1d203a5f2fce679a7ad7b9bf7e244b68bade845431f1582fc34fa443f29a7`, outcome은 `gpu_stage_authorized`다. CPU 두 실행은 exact field와 허용오차 대상 numeric field에서 반복 가능했다.
- [최종 CPU/GPU 2×2 synthesis](reports/runs/g009_r0_rev23_matrix_observation_adapter_synthesis_2x2_s42.json)의 SHA-256은 `a28120dc697625452beed9f9ad160f4acc94da0178ca598816215796caf7ef25`, input count는 `4`, CPU와 `cuda:0`의 `repeatable=true`다. outcome은 `read_only_matrix_observation_adapter_runtime_2x2_validated`, next step은 `preregister_and_run_gpu_throughput_ladder_before_matrix_gate01`이다.
- 첫 lifecycle 점검에서는 기본 fast shutdown 뒤 canonical report 발행 경로에 도달하지 못해 정식 반복에 포함하지 않았다. 정식 네 실행은 `fast_shutdown=false`로 `SimulationApp.close()` 이후 canonical 발행을 수행했다. Windows 종료 과정에서 Isaac Sim access-violation 경고가 출력됐지만 프로세스 exit code는 `0`이었다. exit code만으로 PASS하지 않고 canonical JSON 존재, strict validator PASS, 150개 sample 완전성, source binding·SHA 일치, 동일 디렉터리 temporary file `0`을 함께 확인했다. 이 검증은 네 canonical artifact의 무결성을 확인한 것이며 종료 경고가 모든 환경에서 무해하다는 일반화가 아니다.
- 이번 source는 world-frame filtered **normal contact-force vector**다. total contact force, tangential friction force, friction 효과의 직접 관측 또는 physics ground-truth authority를 주장하지 않는다. policy observation 연결, reward 계산, PPO·optimizer update는 모두 `0`이고 Gate01과 qualification은 `forbidden/not_run`이다. 따라서 보행, 앞뒤 이동, 좌·우회전, 경사 주행, 전복 자가복구 또는 강화학습 성공 증거가 아니다.
- 번호 `14.01` CPU preflight와 `14.02` CPU/GPU final은 canonical JSON을 움직이는 그래프로 재구성한 telemetry animation이다. simulation camera footage나 새로운 로봇 운동 영상이 아니다. 공개 GIF·PNG는 각각 `1280×720`, GIF 8 frame·`5.6s`다. `14.01` GIF/PNG SHA-256은 `ef172f1f66b526aea9bf32f4f862baa902b6d29d715f6b28f54a3eaaee1836ef` / `7ff7bdda8eec382300472d9dce05688d45bbedfeb20d4bbcacc78f02fb4326e8`, `14.02`는 `c208accf605529bb4e59ca6568d31f8badb598f104892630a779e79809ab5a62` / `4cd42aa7f2a7019780a8c3aff9df7f9dc832506938f768cded41e7e5a0e87959`다. H.264 MP4는 `%USERPROFILE%\IsaacLab\logs\visual_evidence\g009\R0\diagnostic\g009_5_r0_diag_rev23_14_01_cpu_matrix_adapter_telemetry_s42.mp4`와 `g009_5_r0_diag_rev23_14_02_final_matrix_adapter_telemetry_s42.mp4`에만 두며 SHA-256은 `42fe4696ffd7fc4d719a84b2c72a94de8321ef6d270295e2979b7aaae652beb6` / `7482a069275516e34fb82fa85904637f068ab45903ec8e22480d31a76a7f07db`다. 모든 프레임에 `TELEMETRY ANIMATION / NOT CAMERA FOOTAGE / DIAGNOSTIC ONLY / NO PPO / NOT QUALIFIED`를 표시한다.
- rev23 종료 당시에는 correctness timing과 분리된 GPU env ladder `[8,32,128,256,512,1024]`를 계획했지만 실행하지 않았다. 바로 아래 rev24에서 기존 1024-env 실행 증거를 반영해 `1024 → 2048 env` 두 rung으로 범위를 줄였다. throughput 결과가 matrix Gate01을 자동 승인하지 않는 경계는 유지한다.

#### rev24 E017 GPU throughput 실행 직전 체크포인트

- 구현 커밋은 `134b6ee31c5ef220048eaaf632f7872a86122258`이다. active `configs/g009_r0.json`을 `g009_r0_recover_rev24`로 올리고 solver `8/0`, max depenetration `1.0m/s`, action scale `0.70`, EMA alpha `0.2`, PPO initial noise `0.5`를 고정했다. rev15 position `16` 계약은 `configs/g009_r0_rev15.json`의 historical snapshot으로 분리했다.
- throughput smoke는 Isaac Lab 공식 `scripts/benchmarks/benchmark_rsl_rl.py`를 repository wrapper로 호출한다. wrapper는 Isaac Lab commit `90b79bb2d44feb8d833f260f2bf37da3487180ba`, 공식 파일 SHA-256 `2d5a88b9c07bfb38852082a0b9bf00f4213043b16ce0294776646ab06d351c82`, tracked source clean 상태를 시작 전에 확인한다.
- 사전등록은 seed `42`, headless scratch, `5 iterations`, `24 steps/env`, PPO epoch `5`, mini-batch `4`다. `1024 env`는 `122,880 transitions`, `2048 env`는 `245,760 transitions`이며 각 rung의 optimizer mini-batch update는 `100회`다. 1024 report가 모든 gate를 통과한 뒤에만 2048을 실행한다.
- report는 clean repository HEAD, 필수 source 12개와 bundle SHA, 정확히 5개의 양의 유한 `steps/s`, checkpoint·TensorBoard, 단일 visible GPU, VRAM `≤90%`, utilization·temperature·power와 baseline recovery를 요구한다. `numeric_invalid`가 있으면 maximum `0`이어야 하며 metric 부재는 unavailable로 기록한다.
- source 검증은 `test_g009_benchmark_bootstrap.py`, `test_g009_r0_rev24_gpu_throughput.py`, `test_g009_recover_contracts.py`, `test_g009_training_qualification.py`를 묶어 `46 passed`로 마쳤다. Isaac 번들 구성은 `test_g009_recover_config.py`의 `7 passed`다. PowerShell safety gate와 G005/G006 queue regression, canonical contract sync, `py_compile`, PowerShell parser, `git diff --check`, 공식 benchmark source binding도 PASS했다. 독립 검토에서 HIGH/MEDIUM blocker는 없었다.
- 이번 체크포인트에서는 실제 `1024/2048 env` 프로세스를 시작하지 않았다. 새 PPO checkpoint, runtime report, MP4, GIF, PNG가 없고 Garden에도 발행하지 않는다. 다음 세션은 원격 `main`과 일치하는 clean worktree에서 1024 rung부터 재개한다.
- exact 실행 명령, 1024 partial report의 staging, 2048 승인 조건, 번호 `15.01/15.02` 미디어 계획은 `docs/G009_REV24_GPU_THROUGHPUT_CHECKPOINT.md`에 기록했다. throughput PASS는 matrix Gate01 또는 policy qualification PASS가 아니다.

#### rev24 E017 이후 미디어 프레임 계약 보정

- rev23 번호 `14.01/14.02` GIF는 `8 frames / 5.6s`, 약 `1.43fps`라서 각 화면이 `700ms`씩 멈춰 보였다. 로컬 MP4를 30fps로 인코딩해도 원본 key frame이 8장뿐이면 GIF 움직임은 부드러워지지 않는다.
- 다음 번호 `15.01/15.02`부터 원본 MP4는 30fps로 보존하고 GIF는 15fps를 목표로 만든다. hard floor는 12fps다. 카메라는 원본 프레임을 샘플링하고 텔레메트리는 구간별 중간 프레임을 실제로 렌더링한다.
- 10 MiB를 넘으면 길이, 해상도, 팔레트 순서로 조정한다. 용량을 맞추려고 12fps 아래로 낮추거나 같은 frame을 반복해 FPS 숫자만 올리지 않는다. sidecar에 실제 frame count, duration, 최대 frame duration과 temporal strategy를 기록하고, 고정 우선순위 `compression_policy_order`와 실제 적용 이력 `compression_steps_applied`를 분리한다.

#### rev24 E017 1024 첫 실행 기각과 ordinal source-bundle 수정

- 2026-09-04 원격 `main`과 일치하는 clean commit `1c0b35c2f13b77620d57ad62175ddb87f68bf828`에서 `Isaac-G009-Recover-Flat-Go2-R0-v0`, seed `42`, headless scratch, `1024 env × 24 steps × 5 iterations`를 실행했다. Isaac Lab은 `90b79bb2d44feb8d833f260f2bf37da3487180ba`, 공식 benchmark SHA-256은 `2d5a88b9c07bfb38852082a0b9bf00f4213043b16ce0294776646ab06d351c82`였다.
- wrapper는 exit code `0`, `steps/s=[7355,11977,11829,12220,11614]`, mean `10999`, median `11829`, peak VRAM `4397/12288MiB`, peak/mean GPU utilization `64%/9.63%`, peak temperature `55°C`, peak power `61.8W`, numeric invalid maximum `0`, baseline GPU memory recovery를 기록했다. checkpoint `model_4.pt` SHA-256은 `0d745dca05a97dd1849b584a0dae100642990afaf07432f416910507c41b67be`다.
- strict synthesis는 `source_bundle_matches_commit=false`로 fail-closed했다. `run_training.ps1`이 `Sort-Object` 문화권 순서로 source path를 배열한 반면 Python verifier는 ordinal 순서를 사용해, 같은 파일과 개별 SHA라도 bundle payload 순서가 달라졌다. raw report `fd798eb3afd6e79e123f1d85971b6a9b24599f4a4724aa94ca6c06cbfe57c828`와 synthesis `6782044cbd46dadc4f2becba4eb5d5efb27ebcbc182447630980d75f0dbef596`는 `_rejected_ordinal_sort.json` 이름으로 보존했다.
- 이 실행은 rev24 1024 PASS가 아니며 stable maximum, Matrix Gate01, policy qualification, recovery success를 승인하지 않는다. 계약에 따라 2048은 시작하지 않았다. 기각 단계의 `15.01` 미디어도 만들지 않는다.
- 수정은 raw 입력을 repo-relative path로 정규화하고 Git tracked path의 실제 casing으로 canonicalize한 뒤 `OrdinalIgnoreCase`로 alias를 중복 제거하고 `Array.Sort(..., StringComparer.Ordinal)`로 배열하도록 제한했다. 교차언어 순서·absolute/relative·case alias 회귀를 추가했고 rev24/benchmark/contracts/qualification pytest `47 passed`, training safety PowerShell test PASS, PowerShell parser PASS를 확인했다. 수정이 clean commit에 반영된 뒤 1024부터 새 canonical 실행으로 재개한다.

#### rev24 E017 canonical 1024/2048 재실행 PASS

- clean source commit `0437e3766e6ff50a6b05a788a6cc7872ee582b89`에서 Isaac Lab `90b79bb2d44feb8d833f260f2bf37da3487180ba`, 공식 benchmark SHA-256 `2d5a88b9c07bfb38852082a0b9bf00f4213043b16ce0294776646ab06d351c82`를 다시 고정했다.
- seed `42`, headless scratch, `24 steps/env`, `5 iterations`, PPO epoch `5`, mini-batch `4` 계약으로 fresh 1024와 2048 rung을 순차 실행했다. 두 실행의 source bundle은 `5b244f87797754926c30dbc64a30fad9d7220859f160cfef2571c60cc25aaec0`으로 일치했고 canonical synthesis가 PASS해 이번 ladder의 stable maximum을 `2048 env`로 확정했다.
- 1024: run `g009_r0_rev24_throughput_1024_retry01_s42`, steps/s `8062/12549/12941/13128/12670`, 평균 `11870`, peak VRAM `4397MiB`, peak utilization `55%`, peak temperature `54°C`, peak power `63.74W`, numeric-invalid maximum `0`. report SHA-256 `5f39c701bbfd889c2a44f470abcf1d4e4632398e34fd9d9e70406cc7da51fb50`, checkpoint SHA-256 `0d745dca05a97dd1849b584a0dae100642990afaf07432f416910507c41b67be`다.
- 2048: run `g009_r0_rev24_throughput_2048_retry01_s42`, steps/s `14874/22947/23020/23168/22576`, 평균 `21317`, peak VRAM `4859MiB`, peak utilization `67%`, peak temperature `55°C`, peak power `68.54W`, numeric-invalid maximum `0`. report SHA-256 `27da732b114dc2c6432926814777d4335e5df4309acaf760af45288abb1ca8e9`, checkpoint SHA-256 `fce57b96b0c3b0ff50e85cae273b8bc11d91c6f63a6417b69f67e91609a40e41`다.
- final synthesis SHA-256은 `15281e134159974525fc53050e186e8e16d79108f9a068fe717d8ea26b805358`다. wrapper run-health PASS와 canonical decision PASS를 분리해 확인했다. rev24는 처리량 smoke이므로 hard-joint-limit maximum `0.25/0.2083333`을 허용했으며 복구 성공·policy qualification은 주장하지 않는다.
- `15.01/15.02` throughput telemetry MP4/GIF/PNG/sidecar는 아직 생성하지 않았다. 따라서 Garden 공개 글도 보류한다.

#### rev25 E018 Matrix Gate01 사전등록과 실제 PPO 연결 준비

- 검증된 terrain-pair `force_matrix_w [N,19,1,3]`의 raw authority는 filter 축을 합친 world-frame `[N,19,3]`으로 유지한다. policy projection만 inverse base quaternion으로 base frame에 회전하고 nominal body weight `15.019kg × 9.81m/s²`로 나눈 뒤 elementwise `tanh`를 적용해 bounded `57D`로 만든다. actor 입력은 `83+57=140D`, critic은 uncorrupted actor prefix `140D` 뒤에 privileged `24D`를 붙인 `164D`다.
- 기존 solver position/velocity `8/0`, max depenetration `1.0m/s`, action scale `0.70`, EMA `0.2`, PPO noise `0.5`와 단일 GroundPlane effective friction 계약을 유지한다. `1024 env × 24 steps × 1 iteration`, epoch `5`, mini-batch `4`, 총 optimizer mini-batch update `20`이며 numeric-invalid와 hard-joint-limit maximum 모두 정확히 `0`이어야 PASS한다.
- ordered 19 body name과 order hash를 실행 전에 exact-match하고, policy/critic shape, source 불변성, finite·nonzero·variance·bound, live solver/action readback, checkpoint first-layer, Adam step `20`과 actor matrix columns `83:140`의 nonzero optimizer moment를 fail-closed로 검사한다. `.item()` 기반 GPU 동기화와 raw authority adapter finite 검사는 이 1-iteration Gate 전용이다. 장기 학습에는 `collect_gate_telemetry=false`의 static shape/dtype 검사와 projection만 사용하고 수치 finite 안전은 production `numeric_invalid` termination이 담당한다.
- 이 단계의 유일한 변화는 whole-body terrain contact matrix의 actor/critic-prefix 연결이다. privileged suffix, reward, termination, physics와 reset은 baseline 그대로다. 한 번의 Gate01 PASS도 복구 학습 완료나 recovery success는 뜻하지 않는다.

#### rev25 E018 Matrix Gate01 실행 계보와 retry02 PASS

- 첫 실행 `g009_r0_rev25_matrix_gate01_s42`는 clean commit `818791557dd2696ae68114a792de41e805a887b8`에서 시작했지만 bootstrap이 `AppLauncher`보다 먼저 `matrix_gate01 → isaaclab.managers → isaacsim.core`를 import해 `ModuleNotFoundError`로 종료됐다. exit code `1`, wall time `3.249s`이며 rollout, PPO update, TensorBoard와 checkpoint는 생성되지 않았다. [기각 report](reports/runs/g009_r0_rev25_matrix_gate01_training_s42_rejected_pre_app_import.json) SHA-256은 `7b2e3b4b563b189cbaf27c1ff1e97d71854b3aad09765460712869b2e691cbed`다.
- pre-App import를 제거한 commit `b0f3fd7097ebec36997cb6c527e76a7455f2b24a`의 retry01은 wrapper exit code `0`, `1 iteration`, `5,054 steps/s`, numeric-invalid·hard-joint-limit maximum `0`, checkpoint까지 만들었다. 그러나 공식 benchmark가 `SimulationApp.close()`를 마친 뒤 bootstrap이 runtime telemetry를 읽으려 해서 Gate 전용 matrix telemetry가 남지 않았다. 따라서 wrapper report의 `passed=true`를 E018 PASS로 사용하지 않고 [missing-telemetry 기각 report](reports/runs/g009_r0_rev25_matrix_gate01_training_retry01_s42_rejected_missing_telemetry.json)로 보존했다. SHA-256은 `449209f243c03629cfd4fe39bd03bb58d9b3c4b70d30f7d06019e93880e47a14`다.
- lifecycle 수정 commit `086fa82106ffed5de86ff9e9de3e24d44dc3e593`은 공식 benchmark main을 실행한 뒤 `SimulationApp.close()` 전에 callback으로 telemetry를 기록하고, 그 다음 simulator를 닫는다. retry02 `g009_r0_rev25_matrix_gate01_retry02_s42`는 이 clean commit과 15개 source 파일 bundle SHA-256 `2003b07981a7a2fdadb72a7752a374bb0deee75dbee2c1dba04578d56982f66a`에 결합했다. source path manifest SHA-256은 `530f2371633a699e56201d7e4d02061913606bb0139094d1901f402ecd562200`, 27개 pass-gate key manifest SHA-256은 `d1769b57db7df4909f50d0d853a3451096ab0af484c36b1e04b022f73c8056bd`다.
- retry02는 seed `42`, headless scratch, `1024 env × 24 steps × 1 iteration`, PPO epoch `5`, mini-batch `4`, optimizer mini-batch update `20회`를 완료했다. wrapper exit code `0`, wall time `31.154s`, `4,525 steps/s`, final mean reward `-0.48`, final mean episode length `12.92`다. [training report](reports/runs/g009_r0_rev25_matrix_gate01_training_retry02_s42.json) SHA-256은 `690d1640e0fe2645fc8e7e1a10459184abaf01296427d40198ffaf653d3ccf96`다.
- runtime telemetry는 56회 호출에서 source/output finite, source 불변, positive magnitude `168,456`, nonzero output `505,368`, 최대 magnitude `1,966.054931640625N`, projection 범위 `[-0.9213587045669556, 1.0]`, 최대 variance `0.007228894159197807`을 기록했다. source/output은 `1024×19×1×3 → 1024×57`, `torch.float32`, `cuda:0`이고 ordered 19-body hash는 `2df7038571d25fe75680727bb2c9c9e87567d8946aad77d414a41a1b61d24436`이다. live readback은 1,024 articulation 모두 solver `8/0`, 19,456 body 모두 max depenetration `1.0m/s`, action scale `0.70`, EMA `0.2`다. telemetry SHA-256은 `8d62ac962126edd2b681fc49834e10e2b862f0a94ba41ebe59b8ae2898faa195`다.
- checkpoint `model_0.pt` SHA-256은 `8096d5ecc9dd54d4c48f76bc7e51f5aee073b6eac428bcedb51347561ed2ef9e`다. actor first layer는 `[512,140]`, critic first layer는 `[512,164]`이고 Adam step `20`에서 actor matrix columns `83:140`의 moment가 nonzero, L2 `0.002019342267885804`였다. numeric-invalid와 hard-joint-limit maximum은 모두 `0`이다. [canonical E018 synthesis](reports/runs/g009_r0_rev25_matrix_gate01_retry02_s42.json) SHA-256 `e81a71dcc3fc4b607ab60bae1c4d3dc1683d15616f312e0aedf24438b37da67d`가 사전등록한 `27/27` gate를 PASS했다.
- GPU peak 사용량은 `4,430/12,288MiB`, baseline 대비 delta `2,950MiB`, peak/mean utilization `27%/7.93%`, peak temperature `53°C`, peak power `43.51W`였다. 측정은 완전했고 종료 뒤 baseline `1,480MiB`로 회복했다. 이 값은 1-iteration connectivity/safety smoke의 관측값이며 장기 학습 처리량 상한은 rev24 결과를 사용한다.
- 판정은 `matrix_gate01_passed`다. 정확한 경계는 `policy qualification=not_run`, `recovery success=not_measured`다. 새 MP4·GIF·PNG는 만들지 않았고 Garden·포트폴리오 production 발행도 보류한다. Gate 전용 `.item()`/finite/provenance telemetry는 production에서 `collect_gate_telemetry=false`로 비활성화하며, 다음 단계는 seed `42`의 headless scratch `1024 env × 24 steps × 300 iterations` R0 qualification이다.
