# G008 불규칙 도로·공간 마찰 강화학습

- 실행일: 2026-08-26
- 태스크: `Isaac-G008-Velocity-IrregularRoad-Go2-S1-v0`
- 로봇: Isaac Lab 내장 Unitree Go2
- 강화학습: RSL-RL 2.3.3 PPO
- 최종 상태: 환경·학습·평가·시각 증거 생성 완료, 전용 300-iteration checkpoint는 기각

## 2026-08-27 후속 결과

아래에서 계획했던 도로 형상 전용 G0를 실제로 구현하고 다시 학습했다. 높이 형상은 그대로 두고 바닥 전체를 static/dynamic `0.8/0.6`으로 고정했다. 기존 friction S1 정책은 terrain seed 세 개 중 두 개, 전체 12개 방향 조건 중 11개를 통과했다. seed `20260828` 우회전에서 8환경 중 1개가 넘어져 G0 승인 조건은 충족하지 못했다.

G0에서 `128환경 × 300 iterations`, `921,600 transitions`를 추가 학습했지만 최선 checkpoint도 세 지형 모두 우회전에 실패했다. 순수 회전에서 기존 `feet_air_time` 보상이 꺼지는 문제를 확인해 yaw 명령에서도 이 항을 활성화한 T1을 같은 예산으로 학습했으나, 최선 checkpoint의 우회전 yaw RMSE가 세 지형 모두 `0.25rad/s`를 넘었다.

| 후보 | 통과 terrain seed | 방향 PASS | 낙상 | 판정 |
| --- | ---: | ---: | ---: | --- |
| 기존 friction S1을 G0에서 재생 | `2/3` | `11/12` | `1` | 유지 |
| G0 추가 PPO 최선 후보 | `0/3` | `9/12` | `0` | 기각 |
| 회전 air-time T1 최선 후보 | `0/3` | `9/12` | `0` | 기각 |

F1 저마찰 두 구간 단계는 열지 않았다. 보상 수식, 두 번의 실제 학습량, 세 지형 평가, 영상과 다음 고도화 순서는 [G008 보상함수와 불규칙 도로 curriculum](G008_REWARD_AND_ROAD_CURRICULUM.md)에 정리했다. 기계 판독 결과는 `reports/runs/g008_road_curriculum_summary_s20260826.json`을 기준으로 본다.

## 먼저 결론

이번 단계는 움직임을 스크립트로 재생한 것이 아니다. 기존 friction S1 PPO checkpoint에서 시작해 새 불규칙 도로 태스크로 300 iterations를 실제 추가 학습했다. 학습 중에는 64개 환경에서 iteration마다 환경당 24 step을 모았고, 총 `460,800` transitions로 PPO update를 수행했다.

다만 새 checkpoint가 더 좋아지지는 않았다. 기존 friction S1 정책은 불규칙 도로의 전진·후진·좌회전을 통과하고 우회전의 순간 roll만 기준을 조금 넘었다. 추가 학습한 최종 checkpoint는 좌·우 회전에서 5회 넘어졌다. 따라서 새 모델을 억지로 채택하지 않고, 기존 friction S1 checkpoint를 이 단계의 선택 정책으로 유지했다.

![기존 friction S1과 불규칙 도로 PPO 300회 후 비교](media/g008/g008_irregular_road_baseline_vs_trained.gif)

![전진·후진·좌회전·우회전 대표 화면](media/g008/g008_irregular_road_baseline_vs_trained_contact_sheet.png)

영상은 한 환경의 동작을 보여주는 자료다. 통과 여부는 32환경 정량 평가 JSON으로 결정했다.

## 사용자가 요청한 조건을 어떻게 옮겼는가

이전의 규칙적인 고·저마찰 띠를 그대로 늘리지 않았다. 56m × 56m 평면 전체에 상관된 2차원 난수를 만들고, 이를 네 마찰 구간으로 나눴다. 따라서 패턴은 x축이나 y축으로 일정 주기마다 반복되지 않는다. 한 몸체 폭 안에서도 작은 패치가 섞이도록 거친 성분과 미세 성분을 함께 사용했다.

높이는 도로 중앙부의 완만한 crown, 긴 파장 굴곡, 골재처럼 부드럽게 이어지는 요철, 얕은 함몰을 더해서 만들었다. 무작위 계단이나 암벽이 아니라, 포장 상태가 고르지 않은 도로에 가까운 크기로 제한했다.

| 항목 | 실제 생성값 |
| --- | ---: |
| 전체 범위 | x/y 각각 `-28~28m` |
| cell 크기 | `0.25m` |
| cell 수 | `224 × 224 = 50,176` |
| 삼각형 수 | `100,352` |
| static/dynamic 마찰 | `0.25/0.15`, `0.40/0.28`, `0.60/0.45`, `0.80/0.60` |
| 각 마찰 구간 면적 | `12,544 cells`, 정확히 `25%` |
| 높이 최솟값/최댓값 | `-0.03853m / 0.04249m` |
| 전체 높이차 | `0.08103m` |
| 인접 cell 최대 높이차 | `0.01121m` |
| 국부 경사 평균/최대 | `0.8758° / 2.6989°` |
| road crown 크기 | `0.015m` |
| 긴 파장 굴곡 크기 | `0.030m` |
| 표면 요철 크기 | `0.012m` |
| 함몰 깊이 설정 | `0.025m` |

같은 seed에서는 같은 도로가 만들어진다. 이번 학습과 평가는 모두 terrain seed `20260826`을 썼다. 재현 가능한 단일 지형에서 먼저 실패 원인을 확인하려는 선택이며, 여러 도로 seed에 일반화됐다는 뜻은 아니다.

## 네 발이 같은 마찰일 때와 모두 다를 때

네 발의 월드 좌표를 매 step의 마찰 grid에 다시 대응시켜, 발마다 어느 구간을 밟고 있는지 기록했다. 기존 friction S1 정책의 32환경 평가에서는 네 발이 같은 마찰에 놓인 frame과 네 발이 모두 다른 마찰에 놓인 frame이 모두 나왔다.

| 방향 | 네 발이 모두 같은 frame | 네 발이 모두 다른 frame | 동시에 관측된 최대 구간 수 | material 전환 수 |
| --- | ---: | ---: | ---: | ---: |
| 전진 | `30.75%` | `1.58%` | `4` | `367` |
| 후진 | `23.75%` | `0.56%` | `4` | `217` |
| 좌회전 | `20.53%` | `0%` | `3` | `57` |
| 우회전 | `10.31%` | `11.17%` | `4` | `83` |

좌회전 궤적에서는 네 구간 동시 접촉이 나오지 않았지만 1·2·3개 구간 조합은 모두 나왔다. 전체 평가로 보면 “네 발이 같을 수도 있고 모두 다를 수도 있다”는 조건이 실제 runtime 궤적에 포함됐다. 네 마찰 구간의 발 위치 표본 수도 모든 방향에서 0보다 컸고, 지형 밖으로 나간 표본은 없었다.

이 비율은 강제로 짜 넣은 순서가 아니다. 비주기 공간 field 위에서 정책이 이동한 결과다. 앞으로 여러 terrain seed를 평가할 때도 같은 마찰 frame만 우연히 몰리지 않는지 별도 분포 gate로 검사한다.

## 물리적으로 무엇이 달라지는가

### 마찰

발 material은 static/dynamic `1.0/1.0`, combine mode는 `multiply`로 고정했다. 따라서 발과 바닥의 유효 마찰계수는 바닥의 네 값이 된다. 접선방향 접촉력이 마찰원뿔 안에 있을 때는 미끄러지지 않고, 필요한 접선력이 한계를 넘으면 slip이 생긴다.

\[
\lVert F_t \rVert \leq \mu F_n
\]

같은 순간 네 발의 `μ`가 다르면 각 발이 만들 수 있는 접선력 한계도 달라진다. 전후 추력뿐 아니라 좌·우 회전의 yaw moment가 비대칭이 된다.

\[
\tau_z = \sum_i (r_{i,x}F_{i,y} - r_{i,y}F_{i,x})
\]

이번 evaluator는 접촉 force가 `1N`을 넘은 발을 접촉 중으로 보고, 그 발의 평면 slip speed를 따로 기록했다. 기존 정책의 접촉 표본은 방향별 `9,657~14,283`개였고 평균 slip speed는 전진 `0.1440m/s`, 후진 `0.3164m/s`, 좌회전 `0.0853m/s`, 우회전 `0.0953m/s`였다. 다만 방향과 gait가 서로 다르므로 이 네 수치 차이를 마찰 하나의 인과효과로 해석하지 않는다. 동일 형상·동일 명령에서 마찰 field만 nominal로 바꾼 대조군이 있어야 마찰의 성능 영향을 분리할 수 있다.

### 높낮이와 요철

경사진 triangle 위에서는 접촉 법선이 월드 z축과 일치하지 않는다. 같은 지면 반력이라도 수평·수직 성분이 달라지고, 서로 다른 높이에 놓인 네 발은 body roll과 pitch moment를 만든다. 이번 평가에서 네 발 위치의 높이차 최대값은 방향에 따라 약 `0.0098~0.0170m`, 실제 밟은 국부 경사 최대값은 `1.64~2.13°`였다.

정책 observation에는 187개 height scan이 들어간다. 따라서 높이 형상은 발을 딛기 전에 관측할 수 있다. 반면 마찰계수 자체는 observation에 직접 넣지 않았다. 정책은 미끄럼 이후의 base 속도, 각속도, 자세, 관절 상태와 직전 action을 통해서만 마찰의 결과를 간접적으로 느낀다. 현재 구조가 높낮이에는 예측적이지만 마찰에는 반응적이라는 뜻이다.

## Isaac Sim에서 접촉면을 구성한 방식

처음에는 하나의 비평면 triangle mesh에 face별 physics material을 붙였다. 이 구성은 현재 호스트의 Isaac Sim 4.5/PhysX에서 warm start 중 native 종료를 일으켰다. 같은 형상을 단일 재질로 쓰면 동작했고, 재질별 mesh를 분리하면 접촉도 정상적으로 기록됐다.

최종 구현은 다음처럼 나눴다.

1. 마찰 구간별 collision mesh 네 개를 만든다.
2. 각 collision mesh에는 physics material 하나만 연결한다.
3. 네 mesh의 face 집합은 서로 겹치지 않고, 합치면 전체 `100,352`개 triangle이 된다.
4. 높이 scanner용으로 같은 형상의 비충돌 mesh 하나를 따로 만든다.
5. 기존 ground collider는 생성하지 않는다.

Isaac Lab 2.1.1 RayCaster가 static mesh 하나를 대상으로 동작하기 때문에 scan mesh를 합쳐 둔 것이다. 이 mesh에는 Collision API가 없어서 접촉을 이중으로 계산하지 않는다. runtime readback에서 기본 ground collider 부재, scan mesh의 비충돌 상태, 네 collision mesh의 material binding, triangle 총수를 모두 확인했다.

## 학습 구성

### headless가 뜻하는 것

학습은 `--headless`로 실행했다. 화면 창과 실시간 카메라를 끈 것이지 물리를 생략한 것은 아니다. PhysX 접촉, height scan, 정책 추론, reward 계산, rollout 수집과 PPO update는 모두 GPU에서 실행됐다. 영상은 학습과 별도로 카메라를 켠 off-screen headless 재생에서 만들었다.

### PPO와 네트워크

Go2 rough runner의 기존 PPO 설정을 유지했다.

| 항목 | 값 |
| --- | ---: |
| observation | `235`차원 |
| action | `12`개 관절 위치 목표 |
| actor/critic | 각각 `512 → 256 → 128`, ELU |
| 물리 timestep | `0.005s`, 200Hz |
| action decimation | `4` |
| 정책 주기 | `0.02s`, 50Hz |
| rollout | 환경당 iteration마다 `24 steps` |
| PPO epochs | iteration마다 `5` |
| mini-batches | epoch마다 `4` |
| clip parameter | `0.2` |
| discount `γ` | `0.99` |
| GAE `λ` | `0.95` |
| entropy coefficient | `0.01` |
| desired KL | `0.01` |
| learning-rate schedule | adaptive |

정책 입력은 base 선속도 3, base 각속도 3, projected gravity 3, 명령 3, 관절 위치 12, 관절 속도 12, 직전 action 12, height scan 187개다. 좌회전과 우회전을 별도 함수로 하드코딩한 것이 아니라, 같은 PPO actor가 `[v_x, v_y, ω_z]` 명령을 보고 12관절 action을 낸다.

### 실제 추가 학습량

기존 friction S1 `model_2097.pt`를 warm-start checkpoint로 불러왔다. 이 모델은 이전 단계에서 발바닥 마찰을 환경별로 randomize해 학습한 정책이다. 이번에는 로봇 발 material을 중립값으로 놓고, 공간에 고정된 불규칙 도로의 재질과 높이를 직접 밟게 했다.

| 항목 | 측정값 |
| --- | ---: |
| environments | `64` |
| 추가 iterations | `300` |
| rollout batch/iteration | `64 × 24 = 1,536 samples` |
| mini-batch 크기 | `1,536 / 4 = 384 samples` |
| optimizer mini-batch updates/iteration | `5 × 4 = 20` |
| 전체 optimizer mini-batch updates | `6,000` |
| 추가 transitions | `460,800` |
| seed | `20260826` |
| wall time | `440.274s`, 약 7분 20초 |
| 평균 처리량 | `1,132.32 steps/s` |
| peak VRAM | `5,058MiB` |
| final mean reward | `35.84` |
| final mean episode length | `984` steps |
| 마지막 learning rate | 약 `1.0e-5` |

학습 프로세스는 exit code 0, 요청 iteration 도달, TensorBoard·checkpoint 생성, fatal log 부재, GPU 회수를 모두 통과했다. “학습이 실행됐는가”는 PASS지만 “정책이 좋아졌는가”는 별도의 고정 평가에서 FAIL이었다.

## 고정 평가

평가는 32환경을 네 방향에 8개씩 배정했다. 각 환경은 500 control steps, 즉 10초 동안 움직였고 처음 50 steps는 transient로 제외했다. 다음 조건을 모두 만족해야 방향 하나가 통과한다.

- survival rate `1.0`
- linear tracking RMSE `≤0.25m/s`
- yaw tracking RMSE `≤0.25rad/s`
- 최대 절대 roll `≤0.35rad`
- 최대 절대 pitch `≤0.35rad`
- 명령과 평균 속도의 부호 일치
- 모든 발 위치가 생성한 field 안에 있음

### 기존 friction S1 정책

| 방향 | linear RMSE | yaw RMSE | max roll | max pitch | 낙상 | 판정 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 전진 | `0.0383m/s` | `0.0490rad/s` | `0.0354rad` | `0.0443rad` | `0` | PASS |
| 후진 | `0.0590m/s` | `0.0415rad/s` | `0.0660rad` | `0.0473rad` | `0` | PASS |
| 좌회전 | `0.0576m/s` | `0.1188rad/s` | `0.0938rad` | `0.1423rad` | `0` | PASS |
| 우회전 | `0.0457m/s` | `0.1438rad/s` | `0.3739rad` | `0.2298rad` | `0` | FAIL |

우회전은 추종 오차와 생존율은 통과했지만 순간 roll이 기준보다 `0.0239rad` 컸다. 따라서 “불규칙 도로에서 네 방향을 모두 안정적으로 걷는다”고 말할 수 없다. 대신 낙상 없이 3/4 방향 gate를 통과했다고 기록한다.

### 불규칙 도로에서 300 iterations 추가 학습한 정책

| 방향 | linear RMSE | yaw RMSE | max roll | max pitch | 낙상 | 판정 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 전진 | `0.0496m/s` | `0.0615rad/s` | `0.0655rad` | `0.1362rad` | `0` | PASS |
| 후진 | `0.1288m/s` | `0.0368rad/s` | `0.1048rad` | `0.0880rad` | `0` | PASS |
| 좌회전 | `0.1276m/s` | `0.3972rad/s` | `0.9730rad` | `0.4437rad` | `2` | FAIL |
| 우회전 | `0.1410m/s` | `0.3740rad/s` | `0.9979rad` | `0.3793rad` | `3` | FAIL |

평균 reward와 episode length만 보면 학습은 정상처럼 보이지만, 고정 회전 명령에서는 좌우 모두 퇴화했다. 평균 학습 지표가 최악 방향 성능을 보장하지 않는 사례다.

## checkpoint를 고른 규칙

중간 checkpoint `model_2100`, `2150`, `2200`, `2250`, `2300`, `2350`을 먼저 16환경·300 steps로 screening했다. `model_2100`만 네 방향을 모두 통과했지만, 더 엄격한 32환경·500-step 평가에서는 좌회전 pitch `0.3949rad`로 실패했다.

최종 선택은 다음 순서로 정했다.

1. 통과한 방향 수가 많은 정책
2. 낙상이 적은 정책
3. 각 gate 값을 허용 한계로 나눈 값 중 최댓값이 작은 정책

| 후보 | 전 방향 중 PASS | 낙상 | 최악 normalized gate ratio | 선택 |
| --- | ---: | ---: | ---: | --- |
| 기존 friction S1 | `3/4` | `0` | `1.0683` | 유지 |
| 추가 학습 직후 `model_2097` | `2/4` | `0` | `1.2075` | 기각 |
| 중간 `model_2100` | `3/4` | `0` | `1.1283` | 기각 |
| 최종 `model_2396` | `2/4` | `5` | `2.8512` | 기각 |

`model_2100`의 짧은 screening PASS는 최종 판정에 사용하지 않았다. 전체 horizon에서 기준을 다시 통과하지 못했기 때문이다.

## 왜 추가 학습 뒤 회전이 나빠졌는가

현재 증거로 원인 하나를 확정할 수는 없다. 다만 다음 가설은 검증할 가치가 있다.

- 64환경·단일 terrain seed의 `460,800` transitions는 이전 friction S1의 `1,024환경 × 300 iterations`보다 표본이 훨씬 작다.
- 학습 command 분포의 평균 reward는 고정 좌·우 회전의 최악값을 직접 최적화하지 않는다.
- 높이와 네 단계 마찰을 한 번에 넣어, 새 접촉 조건에 적응하는 동안 기존 회전 gait를 잃었을 수 있다.
- 마찰은 명시적 observation이 아니어서 slip이 일어난 뒤에만 간접적으로 반응한다.
- 하나의 terrain seed에 오래 맞추면 다른 시작 자세와 궤적에서 성능이 불안정할 수 있다.

이 항목은 가능한 설명이지 입증된 결론이 아니다. 다음 실험에서는 geometry와 friction 난이도를 분리하고, 학습 중간에 고정 방향 평가를 넣어 어느 시점부터 회전 성능이 떨어지는지 확인한다.

## 논문 조사에서 반영한 원칙

| 논문 | 핵심 근거 | 이번 단계에 반영한 부분 | 아직 하지 않은 부분 |
| --- | --- | --- | --- |
| [Rudin et al., Learning to Walk in Minutes](https://arxiv.org/abs/2109.11978) | 대규모 병렬 RL과 지형 curriculum | 병렬 PPO 하네스, 쉬운 단계부터 넓히는 후속 계획 | 논문의 전체 reward·curriculum 재현 |
| [Lee et al., Learning Quadrupedal Locomotion over Challenging Terrain](https://arxiv.org/abs/2010.11251) | 어려운 지형에서 상태 추정과 강건한 보행 정책의 중요성 | height scan과 proprioception을 함께 유지 | 논문과 같은 teacher/student 구성 |
| [Miki et al., Learning robust perceptive locomotion in the wild](https://arxiv.org/abs/2201.08117) | 외부 지형 관측이 틀릴 때 proprioception과 결합하는 설계 | geometry는 사전 관측하고 접촉 결과도 평가 | perception reliability 모듈과 실외 실기체 검증 |
| [Hwangbo et al., Learning agile and dynamic motor skills for legged robots](https://arxiv.org/abs/1901.08652) | actuator·동역학 모델과 randomization이 실제 전이에 미치는 영향 | 물성 축을 분리해 측정하는 원칙 | actuator network와 실기체 전이 |

현재 결과는 위 논문의 원리를 참고한 Isaac Lab 실험이지, 어느 논문도 그대로 재현한 것이 아니다. 실물 Go2 결과가 없으므로 sim-to-real 완료로 부르지 않는다.

## 다음 단계: 난이도를 다시 나눠서 진행한다

아래 A 단계는 2026-08-27에 실행됐고 3개 terrain seed gate를 통과하지 못했다. B의 F1은 보류 상태다. 현재 실행 순서는 보상 항의 방향별 분해와 air-time threshold 단일축 검증이며, 최신 계획은 `G008_REWARD_AND_ROAD_CURRICULUM.md`에 있다.

이번에는 높이와 네 마찰 구간을 한 번에 넣었고, 전용 학습이 회전 gait를 잃었다. 다음 실행은 한 번에 한 축만 바꾼다.

### A. 도로 형상만 학습

1. 현재 crown·굴곡·요철·함몰 형상을 유지하되 바닥 전체를 `0.8/0.6`으로 고정한다.
2. 전진·후진·좌회전·우회전이 모두 통과하는지 확인한다.
3. terrain seed `20260826`, `20260827`, `20260828`을 같은 checkpoint로 평가한다.
4. 세 seed가 통과한 뒤에만 마찰 변화를 연다.

형상 stage에서는 마찰을 바꾸지 않는다. 이렇게 해야 자세 실패가 높낮이 때문인지 저마찰 때문인지 구분할 수 있다.

### B. 공간 마찰 curriculum

1. F1: `0.60/0.45`와 `0.80/0.60` 두 구간만 사용한다.
2. F2: `0.40/0.28`을 추가해 세 구간으로 넓힌다.
3. F3: 마지막에 `0.25/0.15`를 추가한다.

각 stage는 네 방향 full gate, 낙상 0, field coverage, 네 발 material 다양성, nominal guardrail을 모두 통과해야 다음 단계로 간다. 낮은 마찰에서 한 방향이 실패하면 그 stage를 다시 학습하고 더 낮은 값으로 내려가지 않는다.

### C. 방향 성능을 학습 중에 지킨다

- exact 전진·후진·좌회전·우회전 표본 수를 batch 안에서 확인한다.
- 50 iterations마다 32환경·500-step 방향 평가를 실행한다.
- 평균 reward가 아니라 네 방향 중 최저 성능을 checkpoint 선택 기준에 넣는다.
- 좌·우 회전의 yaw RMSE와 roll/pitch가 연속 두 평가에서 나빠지면 early stop한다.

### D. 마찰 적응이 필요한지 확인한다

height scan만으로는 마찰을 미리 알 수 없다. feed-forward MLP, frame-stacked MLP, GRU 기반 짧은 이력 encoder를 같은 transition budget으로 비교한다. 이력 모델이 여러 terrain seed와 held-out 마찰에서 일관되게 이길 때만 adaptation 모듈을 채택한다.

### E. 링크 질량은 별도 파트로 유지한다

다리 링크 질량은 마찰·높이와 동시에 학습하지 않는다. 기존 leg-mass S1은 우회전 nominal gate를 잃었으므로, hip·thigh·calf·foot 한 그룹씩 `0.95~1.05` 범위에서 다시 시작한다. 그룹별 단일축이 모두 통과한 뒤에만 여러 링크 동시 변화와 불규칙 도로를 결합한다. 상세 한계 시험은 [G008 주기 마찰·링크 질량 한계](G008_PERIODIC_FRICTION_AND_LINK_MASS_LIMITS.md)에 있다.

## 재현 명령

### smoke와 실제 학습

```powershell
cd "$HOME\isaac-walk-rl"

# 등록·접촉·PPO 경로 smoke
.\scripts\run_g008_stage.ps1 `
  -Part irregular_road -Stage 1 -NumEnvs 16 -MaxIterations 1 `
  -Seed 20260826 -RunName g008_irregular_road_s1_smoke_e16_i1_s20260826

# 기존 friction S1 checkpoint에서 300 iterations 추가 학습
.\scripts\run_g008_stage.ps1 `
  -Part irregular_road -Stage 1 -NumEnvs 64 -MaxIterations 300 `
  -Seed 20260826 `
  -ResumeRun 2026-08-26_11-37-54_g008_friction_s1_finetune_command_s42_e1024_i300 `
  -ResumeCheckpoint model_2097.pt `
  -RunName g008_irregular_road_s1_finetune_friction_s1_e64_i300_s20260826
```

### 32환경·네 방향 평가

```powershell
cd "$HOME\isaac-walk-rl"

& "$HOME\IsaacLab\_isaac_sim\python.bat" .\scripts\evaluate_g008_irregular_road.py `
  --checkpoint "$HOME\IsaacLab\logs\rsl_rl\unitree_go2_rough\2026-08-26_11-37-54_g008_friction_s1_finetune_command_s42_e1024_i300\model_2097.pt" `
  --policy-id friction_s1 --num-envs 32 --horizon-steps 500 --warmup-steps 50 `
  --eval-seed 20260826 --terrain-seed 20260826 `
  --output .\reports\runs\g008_irregular_road_baseline_friction_s1_e32_h500_s20260826.json `
  --headless --device cuda:0
```

### 영상과 공개 파생물

```powershell
cd "$HOME\isaac-walk-rl"

& "$HOME\IsaacLab\_isaac_sim\python.bat" .\scripts\record_g008_irregular_road.py `
  --profile irregular_road_baseline_friction_s1 `
  --checkpoint "$HOME\IsaacLab\logs\rsl_rl\unitree_go2_rough\2026-08-26_11-37-54_g008_friction_s1_finetune_command_s42_e1024_i300\model_2097.pt" `
  --output-dir "$HOME\IsaacLab\logs\visual_evidence\g008" `
  --report .\reports\runs\g008_irregular_road_baseline_capture.json `
  --seed 20260826 --headless --device cuda:0

py .\scripts\build_g008_irregular_road_media.py `
  --capture-reports `
    .\reports\runs\g008_irregular_road_baseline_capture.json `
    .\reports\runs\g008_irregular_road_trained_capture.json `
  --local-composite "$HOME\IsaacLab\logs\visual_evidence\g008\g008_irregular_road_baseline_vs_trained_s20260826.mp4" `
  --public-gif .\docs\media\g008\g008_irregular_road_baseline_vs_trained.gif `
  --public-contact-sheet .\docs\media\g008\g008_irregular_road_baseline_vs_trained_contact_sheet.png `
  --output-report .\reports\runs\g008_irregular_road_visual_evidence.json
```

## 증거 파일

- 최종 선택 summary: `reports/runs/g008_irregular_road_summary_s20260826.json`
- 기존 정책 full 평가: `reports/runs/g008_irregular_road_baseline_friction_s1_e32_h500_s20260826.json`
- 최종 PPO checkpoint full 평가: `reports/runs/g008_irregular_road_trained_s1_e32_h500_s20260826.json`
- 실제 300-iteration 학습 보고서: `reports/runs/g008_irregular_road_s1_finetune_friction_s1_e64_i300_s20260826.json`
- 시각 증거 메타데이터: `reports/runs/g008_irregular_road_visual_evidence.json`
- 환경 생성 코드: `src/isaac_walk_g008/irregular_road.py`
- 태스크 설정: `src/isaac_walk_g008/env_cfg.py`
- 정량 evaluator: `scripts/evaluate_g008_irregular_road.py`

원본 MP4는 Git에 넣지 않았다. 경로와 SHA-256은 다음 시각 증거 문서와 JSON에 연결돼 있다.
