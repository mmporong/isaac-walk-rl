# G006 rough·DR·외란 회복 결과

## 결론

동일한 official rough terrain curriculum·domain randomization 조건에서 학습 중 `events.push_robot`만 추가한 `push_curriculum`은 baseline 대비 pooled 회복률이 `+0.0617%p` 높았다. 그러나 10,000회 paired hierarchical bootstrap의 95% 신뢰구간이 `-0.7716%p ~ +0.9568%p`로 0을 포함하므로 통계적으로 유의한 개선을 주장하지 않는다.

두 variant 모두 270회 guardrail trial에서 생존률 `100%`를 기록했다. push curriculum은 추적 오차 proxy를 낮췄지만 torque·mechanical power proxy를 높였다. 따라서 G006의 판정은 “작은 기술적 회복률 증가와 추적 개선, 에너지 proxy 비용이 관측됐으나 우월성은 입증되지 않음”이다.

## 고정 실험 계약

| 항목 | 값 |
| --- | --- |
| 환경 | Isaac Sim 4.5 / Isaac Lab 2.1.1 / RSL-RL 2.3.3 / Windows native |
| 학습 budget | variant별 4096 env × 1500 iterations × seeds 42/43/44 |
| 비교 variant | `baseline`, `push_curriculum` |
| normalized config diff | `events.push_robot`만 허용 |
| push 평가 | seed별 1080 trials, 108 cells × 10 trials |
| guardrail 평가 | seed별 90 trials, 9 cells × 10 trials |
| bootstrap | seed `20260824`, 10,000 draws, 고정 108 strata |

## 통합 결과

| 지표 | baseline | push curriculum | 차이 |
| --- | ---: | ---: | ---: |
| push 회복률 | 3225/3240 (`99.5370%`) | 3227/3240 (`99.5988%`) | `+0.0617%p` |
| 회복률 Wilson 95% CI | `99.2375% ~ 99.7192%` | `99.3147% ~ 99.7654%` | 기술 비교 |
| push horizon 생존률 | 3235/3240 (`99.8457%`) | 3231/3240 (`99.7222%`) | `-0.1235%p` |
| guardrail 생존률 | 270/270 (`100%`) | 270/270 (`100%`) | `0%p` |
| paired bootstrap 회복률 차이 | — | estimate `+0.0619%p` | 95% CI `-0.7716%p ~ +0.9568%p` |

모든 6480개 push trial에서 pre-push failure와 protocol boundary violation은 0건이었고, auto-reset 제외도 0건이었다.

## seed별 회복률

| variant | seed 42 | seed 43 | seed 44 |
| --- | ---: | ---: | ---: |
| baseline | 1069/1080 (`98.9815%`) | 1079/1080 (`99.9074%`) | 1077/1080 (`99.7222%`) |
| push curriculum | 1080/1080 (`100%`) | 1078/1080 (`99.8148%`) | 1069/1080 (`98.9815%`) |

seed별 방향이 일관되지 않으므로 pooled 차이만으로 개선을 일반화하지 않는다.

## 추적·에너지 proxy

| 지표 | baseline | push curriculum | 상대 변화 |
| --- | ---: | ---: | ---: |
| `tracking_error_sq_mean` | 0.029994 | 0.027256 | `-9.1290%` |
| `yaw_error_sq_mean` | 0.014111 | 0.012737 | `-9.7386%` |
| `torque_l2_mean` | 200.289065 | 209.184042 | `+4.4411%` |
| `absolute_mechanical_power_mean` | 35.583149 | 36.868460 | `+3.6121%` |
| `action_rate_l2_mean` | 1.036716 | 1.027271 | `-0.9110%` |

mechanical power는 시뮬레이션 proxy이며 전기 에너지 소비량이 아니다. 이 비교는 동일 rough·공통 official DR 조건에서 push curriculum의 추가 효과만 다루며, flat→rough 또는 DR 단독 인과효과를 주장하지 않는다.

## 런타임 복구 기록

seed 44 push-curriculum 학습은 exit `0`, `1499/1500`, fatal pattern 0건, 최종 checkpoint SHA-256 `cc799d6c1972ccd417e1631916337aaec2b97b62282f5f3f511f0e5396e31440`으로 완료됐다. 학습 도중 새 Codex 앱 GPU context가 생겨 초기 GPU baseline `1235 MiB` 대비 종료 후 `1531 MiB`가 남았고, 기존 총량 기반 `+128 MiB` 회수 게이트만 false-negative가 났다.

`scripts/revalidate_training_gpu_recovery.ps1`은 원본 실패 보고서 SHA-256 `90c497a9d5c95b4a278d42921ea126494eeedaad674849c88171895ce21d8bd9`, checkpoint·raw log 무결성, 해당 training process 0개, fatal pattern 0건을 확인하고 GPU 사용량 5회가 모두 `1519 MiB`로 안정적임을 attestation에 보존했다. queue는 유효한 failed-state 학습 보고서만 재사용해 평가부터 재개했으며, 변조된 complete job은 기존처럼 전체 재실행한다.

## 재현 증거

- strict summary: [`../reports/runs/g006_summary.json`](../reports/runs/g006_summary.json)
- queue state와 job별 해시: [`../reports/runs/g006_queue_state.json`](../reports/runs/g006_queue_state.json)
- 실험 manifest: [`../configs/g006_rough_push.json`](../configs/g006_rough_push.json)
- 최종 seed 44 학습 보고서: [`../reports/runs/g006_production_push_curriculum_e4096_i1500_s44.json`](../reports/runs/g006_production_push_curriculum_e4096_i1500_s44.json)
- 최종 seed 44 push 보고서: [`../reports/runs/g006_production_push_curriculum_e4096_i1500_s44_push.json`](../reports/runs/g006_production_push_curriculum_e4096_i1500_s44_push.json)
- 최종 seed 44 guardrail 보고서: [`../reports/runs/g006_production_push_curriculum_e4096_i1500_s44_guardrail.json`](../reports/runs/g006_production_push_curriculum_e4096_i1500_s44_guardrail.json)

queue state는 6/6 jobs `complete`, top-level failures 0건이며 strict summary SHA-256은 `09e08d496e428418e9c36f294ef6cf4efddadd84ea901685a584dff4d126347c`다.
