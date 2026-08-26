# Isaac Walk RL sim-to-real 포트폴리오 확장안

- 기준일: 2026-08-26
- 상태: G006 이후의 추천 실험
- 현재 주장 범위: `sim-to-real 완료`가 아니라 `sim-to-real readiness`

## 현재 증거

G006은 4,096 environments × 1,500 iterations × 3 seeds로 baseline과 push curriculum을 비교했습니다. 전체 884,736,000 transitions와 6,480개 push trial을 실행했고, pooled 회복률 차이의 95% 신뢰구간이 0을 포함해 push curriculum의 우월성을 주장하지 않았습니다.

G008은 이 실험 하네스를 방향 명령과 물성 단일축 검증으로 확장했습니다.

- G006 warm-start 뒤 `1,024 env × 300 iterations × seed 42`로 학습한 방향 명령 정책은 평면 전진·후진·좌회전·우회전 gate를 모두 통과했습니다. rough terrain에서는 좌·우 회전만 통과했고 전진·후진은 자세 gate를 넘었습니다.
- friction S1은 randomized·nominal 평면 네 방향 gate를 통과했습니다. rough 학습의 terrain level mean이 약 3.45에서 2.27로 내려가 S2 확대는 보류했습니다.
- leg-mass S1은 학습은 끝났지만 randomized·nominal 평면에서 우회전 yaw gate를 잃었습니다. nominal guardrail도 실패해 S2를 중단했습니다.

따라서 `dynamics randomization 단일축 gate`는 구현·측정 단계로 올릴 수 있습니다. 다만 seed 42의 좁은 S1 학습 범위와 nominal 평면 결과이므로, 미관측 동역학에 강하다는 주장은 아직 사용할 수 없습니다.

이 프로젝트의 강점은 높은 reward 하나가 아닙니다. 같은 학습 예산, seed, 평가 조건을 고정하고 아이디어가 효과가 없을 때도 기각할 수 있는 실험 하네스입니다.

## 다음 단계

### 1. 단일축 gate를 held-out dynamics로 확장

G008 S1의 마찰·다리 링크 질량 단일축 평가를 seed 42/43/44와 rough guardrail로 반복합니다. S1이 nominal과 randomized 조건을 모두 통과한 축만 S2로 넓힙니다. 그다음 학습 범위의 경계와 범위 밖을 별도 held-out으로 떼어 평가합니다. 이후 중심 위치, 모터 강도, 관절 지연, 센서 노이즈, 토크 제한을 한 축씩 추가하며 회복률뿐 아니라 추적 오차, torque, mechanical power proxy, fall time을 함께 봅니다.

### 2. cross-simulator

Isaac Sim 정책을 MuJoCo에서 평가해 PhysX 특성에 과적합됐는지 확인합니다. 이 결과는 `sim-to-sim`으로만 표시합니다. 관절·contact·actuator 모델이 다른 상태에서 수치가 떨어져도 실물 전이 실패라고 부르지 않습니다.

### 3. 실물 접근 이후

Go2의 관절 명령과 응답을 기록해 지연, 모터 강도, 감쇠, 마찰 범위를 수정합니다. zero-shot 배포와 소량 실물 데이터로 보정한 결과를 분리합니다. 실물 결과가 생기기 전에는 `sim-to-real 완료`라는 표현을 사용하지 않습니다.

## 논문별 적용 추천

| 논문 | Isaac Walk에 적용할 내용 | 추천 |
| --- | --- | --- |
| [Domain Randomization](https://arxiv.org/abs/1703.06907) | 원 논문의 시각 randomization은 현재 상태 기반 관측에 직접 맞지 않습니다. 향후 카메라 관측을 넣을 때만 조명·재질·시점을 검토합니다. | 현재는 보류 |
| [Dynamics Randomization](https://arxiv.org/abs/1710.06537) | G008에서 마찰과 다리 링크 질량 S1을 분리 적용했습니다. 다중 seed·rough guardrail을 통과한 축만 S2로 넓히고 관절 지연·모터 강도를 후속 단일축으로 추가합니다. | S1 적용 완료, 확장 추천 |
| [Closing the Sim-to-Real Loop](https://arxiv.org/abs/1810.05687) | 소수 실물 rollout의 행동 차이로 randomization 분포를 갱신합니다. 실물 Go2가 없을 때는 구현 완료를 주장하지 않습니다. | 하드웨어 접근 뒤 추천 |
| [ACT](https://arxiv.org/abs/2304.13705) | 실물 시연 기반 정밀 조작용 방법이라 현재 PPO 보행 기준선에 직접 적용할 이유가 없습니다. | 적용 비추천 |
| [Diffusion Policy](https://arxiv.org/abs/2303.04137) | 다봉 조작 궤적 생성의 장점은 현재의 고주기 12관절 보행 정책보다 추론 비용과 비교 부담이 큽니다. | 현재 적용 비추천 |
| [RMA](https://arxiv.org/abs/2107.04034) | privileged dynamics로 base policy를 학습하고 관측·행동 이력으로 extrinsics를 추정하는 adaptation module을 둡니다. current MLP, frame stack, GRU와 같은 예산으로 비교합니다. | 가장 강하게 추천 |
| [See like a Robot](https://arxiv.org/abs/2607.11498) | VLA의 카메라·로봇 프레임 불일치를 다루므로 현재 상태 기반 보행에는 맞지 않습니다. 향후 시각 지형 인식이 생길 때 검토합니다. | 현재는 보류 |

## Transformer의 위치

RMA식 adaptation module의 목적은 이력에서 숨은 동역학을 추정하는 것입니다. 그 인코더가 꼭 Transformer일 필요는 없습니다.

비교 순서는 current MLP, frame-stacked MLP, GRU, 작은 causal Transformer입니다. 모델 크기, transitions, seed, 지형, 외란 조건을 맞추고 회복률·추적 오차·torque·power proxy·추론 지연·VRAM을 비교합니다. Transformer가 GRU를 이기지 못하면 채택하지 않습니다.

## 포트폴리오 산출물

- 첫 화면: G008 네 방향 명령 GIF와 G006의 `유의한 개선 아님` 판정
- 실험 표: G006의 seed·transitions·push trials·신뢰구간과 G008 S1의 통과·중단 gate
- 시스템 그림: manifest → queue → training → evaluator → durable summary
- 다음 실험: G008 S1 다중 seed·rough guardrail → cross-simulator → 실물 동정
- 한계: 실물 Go2 부재, 상태 기반 관측, RBQ 자산 라이선스 blocker

포트폴리오 제목은 `Transformer 보행`보다 `반복 실험과 단일축 물성 gate로 검증하는 사족보행 RL`이 현재 증거에 맞습니다.
