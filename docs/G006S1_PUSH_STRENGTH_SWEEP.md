# G006S1 외란 강도 sweep: 기존 G006 정책을 천장에서 끌어내리기

기준일: 2026-09-27. G006에서 baseline과 push curriculum의 차이를 볼 수 없었던 첫 번째 이유는 baseline 회복률이 이미 `99.5370%`로 천장에 붙어 있었다는 점이다. G006S1은 새 학습 없이 기존 G006 checkpoint 6개만 더 강한 push로 평가해, baseline이 천장에서 내려오는 강도를 찾고 그 강도에서 두 정책을 다시 비교한 실험이다.

## 무엇을 바꿨고 무엇을 고정했나

| 구분 | 내용 |
| --- | --- |
| 바꾼 것 | `evaluation_protocol.push_magnitudes_mps` 하나. G006 `[0.5, 1.0, 1.5]` → `[2.0, 2.5, 3.0] m/s` |
| 고정한 것 | 지형·held-out row/col, 초기 상태 10개, 명령 3개, push 방향 4개, push 시점(step 200), 성공 기준, observation noise·평가 randomization 끔 |
| evaluator | `scripts/evaluate_push_recovery.py`와 `src/isaac_walk_g006/evaluation`을 수정하지 않았다. 실행 전 source bundle SHA-256이 G006 기록 `4277f555…8897`과 같은지 검사하고, 다르면 시뮬레이터를 띄우지 않는다 |
| 정책 | G006 `model_1499.pt` 6개(baseline·push curriculum × seed 42/43/44). 실행 전 checkpoint SHA-256을 [g006_queue_state.json](../reports/runs/g006_queue_state.json)과 대조했다 |
| 학습 | 없음. 평가 전용 |

계약은 결과를 보기 전에 [configs/g006s1_push_strength_sweep.json](../configs/g006s1_push_strength_sweep.json)으로 커밋했다(`a4c6cf2`). 격자 선택 규칙도 여기에 고정했다. baseline seed 42 하나로 `2.0/2.5/3.0 m/s`를 먼저 평가하고, 회복률이 `0.95` 이하로 처음 내려가는 강도를 기준으로 연속한 세 강도를 비교 격자로 쓴다.

## 실행

| 단계 | 대상 | 결과 |
| --- | --- | --- |
| A1 첫 시도 | baseline s42 | 평가 프로세스는 exit 0으로 끝났지만 보고서가 남지 않았다. 래퍼가 `SimulationApp.close()` 뒤에 보고서를 쓰도록 되어 있었고, Isaac Sim 4.5는 close에서 프로세스를 끝낸다. 원본 evaluator와 같은 순서로 고치고 회귀 테스트를 추가했다(`c56cb18`) |
| A1 보정 | baseline s42, 1,080 trials | `2.0 m/s` 70.28%, `2.5` 62.50%, `3.0` 60.00%. 첫 강도에서 이미 0.95 아래라 A2는 실행하지 않았고 규칙에 따라 격자 `[2.0, 2.5, 3.0]`이 선택됐다 |
| B 비교 | 6 checkpoints × 1,080 trials | 6/6 `complete`. boundary 위반·auto-reset 제외·push 전 실패 모두 0 |

한 번의 평가는 1,080 env × 600 control steps이며 wall time은 60.6~63.4초, peak VRAM은 7,522~7,775 MiB였다(RTX 3060 12 GB, 기준 9,830 MiB 이하). 모든 GPU 실행은 `CodexSignals\Use-GpuLease.ps1 -Owner G006S1` 안에서 수행했다. 상세 기록은 [g006s1_run_log.json](../reports/runs/g006s1_run_log.json)에 있다.

A1과 B에서 baseline s42를 같은 격자로 두 번 평가했는데, 1,080 trial의 회복 여부·회복 step·생존 여부가 모두 같았고 torque 지표 차이도 0이었다. 평가가 결정적이므로 seed 42의 중복 실행은 같은 증거를 두 번 센 것이 아니라 재현 확인이다.

마지막 실행(push curriculum s44) 뒤 러너의 GPU 회수 게이트가 FAIL을 냈다. 종료 5초 뒤 VRAM이 1,749 MiB로 시작 전 1,480 MiB보다 269 MiB 많아, 허용치 +256 MiB를 넘었다. 확인한 사실은 다음과 같다: 잔류 Isaac·Python 프로세스 없음, lease는 `completed / exit 0`으로 해제됨, 직후 VRAM 1,764 MiB는 같은 날 평가 전 유휴 상태에서 관찰한 1,701~1,821 MiB 범위 안이다. 그래서 GPU 누수가 아니라 데스크톱 앱의 VRAM 변동으로 판단했다. 이 판단은 추론이며, run log의 `gpu_released=false` 기록은 고치지 않았다.

## 결과

정본: [g006s1_summary.json](../reports/runs/g006s1_summary.json)

### baseline은 2.0 m/s에서 천장을 벗어났다

| push 크기 | baseline (3 seeds) | push curriculum (3 seeds) |
| ---: | ---: | ---: |
| 0.5 m/s (G006) | 1080/1080 (100%) | 1080/1080 (100%) |
| 1.0 m/s (G006) | 1071/1080 (99.17%) | 1079/1080 (99.91%) |
| 1.5 m/s (G006) | 1074/1080 (99.44%) | 1068/1080 (98.89%) |
| **2.0 m/s** | 831/1080 (**76.94%**) | 837/1080 (77.50%) |
| **2.5 m/s** | 634/1080 (**58.70%**) | 511/1080 (47.31%) |
| **3.0 m/s** | 459/1080 (**42.50%**) | 385/1080 (35.65%) |

0.5~1.5 m/s 행은 기존 G006 보고서를 같은 방식으로 강도별로 나눈 값이다. 학습 중 push curriculum의 최대 강도는 1.0 m/s였고, 1.5 m/s까지는 두 정책 모두 99% 근처였다. 2.0 m/s부터 회복률이 크게 떨어졌다. 사전 등록한 천장 이탈 조건(knee 강도에서 baseline 3 seed 합산 ≤ 0.95)은 76.94%로 충족됐다.

### 이 강도에서도 두 정책의 차이는 검출되지 않았다

| 지표 | baseline | push curriculum |
| --- | ---: | ---: |
| 격자 전체 회복률 | 1924/3240 (59.38%) | 1733/3240 (53.49%) |
| Wilson 95% | 57.68~61.06% | 51.77~55.20% |
| paired bootstrap 차이 (curriculum − baseline), 3 seeds | | `-6.00%p`, 95% CI `[-26.33, +15.96]%p` |
| 같은 차이, seed 43/44만 | | `+4.34%p`, 95% CI `[-8.43, +17.36]%p` |

합산 회복률은 baseline이 5.9%p 높지만, seed를 다시 뽑는 paired bootstrap의 95% CI가 0을 포함한다. 사전 등록 규칙에 따라 방향을 주장하지 않는다. 격자 선택에 쓴 seed 42를 빼면 부호가 바뀌는 것도 같은 결론을 가리킨다. 원인은 seed 사이 편차다.

| seed별 회복률 | 2.0 m/s | 2.5 m/s | 3.0 m/s |
| --- | ---: | ---: | ---: |
| baseline s42 | 70.28% | 62.50% | 60.00% |
| baseline s43 | 80.28% | 51.11% | 24.44% |
| baseline s44 | 80.28% | 62.50% | 43.06% |
| push curriculum s42 | 75.28% | 24.44% | 13.33% |
| push curriculum s43 | 86.67% | 66.11% | 51.94% |
| push curriculum s44 | 70.56% | 51.39% | 41.67% |

같은 variant 안에서도 3.0 m/s 회복률이 13.33%에서 51.94%까지 벌어진다. variant 사이의 차이보다 training seed 사이의 차이가 크다. torque L2와 mechanical power proxy는 push curriculum이 각각 `249.58` 대 `243.84`, `57.51` 대 `56.42`로 조금 높았다. G006에서 본 방향과 같지만, 이 실험의 판정 지표는 아니다.

## 해석과 한계

- 확인한 것: G006 평가 grid는 두 정책을 가를 수 없는 강도였다. 같은 checkpoint가 2.0 m/s부터 회복에 실패하기 시작하므로, 이후 비교는 2.0~3.0 m/s 범위에서 해야 의미가 있다.
- 확인하지 못한 것: 이 범위에서도 push curriculum이 낫다거나 못하다는 결론은 나오지 않았다. 3 seed로는 seed 편차가 variant 차이를 덮는다.
- 평가 전용이다. push curriculum은 학습 중 최대 1.0 m/s까지만 경험했으므로, 2.0 m/s 이상은 두 정책 모두에게 학습 분포 밖이다. 이 결과는 "강한 push를 학습하면 나아지는가"에 답하지 않는다.
- push는 시뮬레이션에서 root 속도에 순간적으로 더한 값이다. 실제 로봇에 가한 힘이나 충격량으로 보정한 값이 아니다.
- 외란 후 보행 회복이며, 넘어진 자세에서 일어서는 실험이 아니다.

## 다음 단계

1. 재학습 비교는 이 결과를 근거로 설계한다. 평가 grid는 `2.0/2.5/3.0 m/s`로 고정하고, seed는 최소 5개로 늘린다. 이번 seed 편차(같은 variant 안에서 3.0 m/s 기준 최대 38.6%p)를 보면 3개로는 부족하다.
2. push curriculum의 학습 강도 상한을 1.0 m/s에서 평가 범위까지 올린 variant를 추가할지 검토한다. 이 경우 바뀌는 변수가 "curriculum 유무"에서 "curriculum 강도"로 달라지므로, 별도 계약으로 등록한다.
3. seed별 실패가 어느 명령·방향·지형 row에 몰리는지 cell 단위로 먼저 확인하면, 재학습 없이도 seed 편차의 원인 후보를 좁힐 수 있다. 아래 절에서 수행했다.

## cell 분석 (2026-09-27, 추가 시뮬레이션 없음)

정본: [g006s1_cell_analysis.json](../reports/runs/g006s1_cell_analysis.json), 생성 스크립트 [analyze_g006s1_cells.py](../scripts/analyze_g006s1_cells.py). phase B 보고서 6개의 trial을 다시 나눈 기술 통계이며 다중 비교 보정은 하지 않았다.

| 관찰 | 근거 |
| --- | --- |
| 지형 난도는 회복률을 가르지 않는다 | checkpoint마다 row 1/4/8 회복률 차이가 최대 7.2%p다. 예: push curriculum s43 `0.68 / 0.72 / 0.65` |
| 뒤로 미는 push가 공통 약점이다 | 3 seed 합산 backward 회복률: baseline `28.9% / 4.1% / 0.0%`, push curriculum `49.6% / 1.5% / 0.4%` (2.0 / 2.5 / 3.0 m/s). 2.5 m/s부터는 두 정책 모두 바닥이라 이 cell은 variant를 가르지 못한다 |
| push curriculum이 앞선 cell은 backward 2.0 m/s뿐이다 | 같은 강도에서 forward·left·right는 baseline이 높다(forward `95.2%` 대 `86.7%`) |
| seed마다 명령×방향 비대칭이 다르다 | baseline s42는 회전 명령에서 왼쪽 push `9%`, 오른쪽 push `98%`. push curriculum s42는 횡이동 명령에서 오른쪽 push `2%`. push curriculum s44는 횡이동 명령에서 왼쪽 push `9%` |
| 실패 대부분은 넘어짐이다 | 서 있지만 기준을 못 맞춘 trial은 checkpoint당 0~72개이고, 나머지 실패는 horizon 전 넘어짐이다 |

해석: seed 편차는 지형 적응 차이가 아니라, seed마다 다른 좌우·전후 비대칭에서 온다는 가설과 맞는다. 이 가설은 아직 검증하지 않았다. 확인하려면 명령 없이 제자리에서 같은 push를 준 대칭 평가나, 정책의 좌우 대칭성 지표가 필요하다. 재학습 비교(G006S2)는 사전 등록한 격자를 유지하고, 방향별 회복률을 secondary로 함께 보고한다.

## 재현

```powershell
cd "$HOME\worktrees\isaac-walk-rl-g009-r0"
python -m pytest -q tests/test_g006s1_push_strength.py
# GPU 실행은 러너가 CodexSignals lease로 감싼다.
.\scripts\run_g006s1_push_strength.ps1 -Phase A1 -Grid 2.0,2.5,3.0
python scripts\g006s1_push_strength.py select
.\scripts\run_g006s1_push_strength.ps1 -Phase B -Grid 2.0,2.5,3.0
python scripts\g006s1_push_strength.py summarize --grid 2.0 2.5 3.0
```

근거 파일: [계약](../configs/g006s1_push_strength_sweep.json), [격자 manifest](../configs/g006s1_grid_m2p00_m2p50_m3p00.json), [평가 래퍼](../scripts/evaluate_g006s1_push_strength.py), [집계](../scripts/g006s1_push_strength.py), [러너](../scripts/run_g006s1_push_strength.ps1), [테스트](../tests/test_g006s1_push_strength.py), [보고서](../reports/runs/) `g006s1_A1_*`, `g006s1_B_*`.
