# G007 RBQ 외부 자산 호환성 사전조사

## 판정

G007은 `external_custom_compatibility_spike`로 분류한다. Rainbow Robotics RBQ v1.20.0의 URDF·패키지 매니페스트·STL 6개를 포함한 8개 blob의 공개 위치와 Git 객체는 고정했지만, 자산 blob에 적용되는 라이선스 범위와 로컬 변환 허용 여부를 확인하지 못했다. 따라서 현재 상태는 `license_scope_unresolved` blocker이며 자산 다운로드, 파생물 생성, 변환기 실행, 시뮬레이터 smoke를 진행하지 않는다.

이 blocker는 프로젝트 브리프가 허용한 G007 완료 경로다. G006 학습·평가나 전체 프로젝트를 중단시키는 blocker는 아니다.

## 근거와 추론

### 확인한 사실

- RBQ release tag `v1.20.0`은 annotated tag object `741ce5733dcd7c0babec663bb7e1afbc02a776ca`에서 source commit `68bc33b77719d357b4323fb88549efd905caf721`을 가리킨다.
- source commit에서 `rbq_sdk/ros2/src/rbq_description/` 아래 URDF 1개, `package.xml` 1개, STL 6개 등 정확히 8개 blob의 경로·크기·Git blob SHA-1을 고정했다.
- GitHub repository API의 `license` 값은 `null`이다. 이는 GitHub가 저장소 전체에 적용되는 라이선스를 감지하지 못했다는 뜻일 뿐, 무허가 또는 이용 금지를 증명하지 않는다.
- `rbq_description/package.xml`은 `Apache-2.0`을 선언한다. 이는 package manifest의 선언이며 URDF·STL blob에 적용되는 범위는 현재 근거만으로 확정할 수 없다.
- 공식 Isaac Lab 공개 소스의 v2.1.1, v2.3.2, 조사 시점 `main` commit에서 대상 경로·심볼 일치 항목이 없었다.
- 검증기는 자산 byte가 없는 상태에서 topology 수를 `null`로 유지하고 변환기·smoke를 실행하지 않은 채 차단 보고서를 만든다.

### 이 근거로 내린 추론

- 공개 저장소에서 파일을 볼 수 있다는 사실만으로 로컬 처리, 파생물 생성 또는 재배포 권한을 확대 해석할 수 없다.
- 공식 Isaac Lab 세 기준점에 대상 구현이 없으므로 “상위 버전의 공식 구현을 2.1.1로 이식한다”는 기존 설명은 사실이 아니다.
- 현재 할 수 있는 안전한 작업은 고정 source와 blocker를 재현하는 것까지다. 자산 호환성이나 topology는 아직 판정하지 않는다.

## 고정한 출처

- [RainbowRobotics/RBQ 저장소 API](https://api.github.com/repos/RainbowRobotics/RBQ)
- [v1.20.0 tag object API](https://api.github.com/repos/RainbowRobotics/RBQ/git/tags/741ce5733dcd7c0babec663bb7e1afbc02a776ca)
- [source commit API](https://api.github.com/repos/RainbowRobotics/RBQ/git/commits/68bc33b77719d357b4323fb88549efd905caf721)
- [source commit tree API](https://api.github.com/repos/RainbowRobotics/RBQ/git/trees/68bc33b77719d357b4323fb88549efd905caf721?recursive=1)
- [고정 commit의 RBQ URDF 원문](https://raw.githubusercontent.com/RainbowRobotics/RBQ/68bc33b77719d357b4323fb88549efd905caf721/rbq_sdk/ros2/src/rbq_description/urdf/rbq.urdf)
- [고정 commit의 package.xml 원문](https://raw.githubusercontent.com/RainbowRobotics/RBQ/68bc33b77719d357b4323fb88549efd905caf721/rbq_sdk/ros2/src/rbq_description/package.xml)
- [Isaac Lab v2.1.1 고정 소스](https://github.com/isaac-sim/IsaacLab/tree/90b79bb2d44feb8d833f260f2bf37da3487180ba)
- [Isaac Lab v2.3.2 고정 소스](https://github.com/isaac-sim/IsaacLab/tree/37ddf626871758333d6ed89cf64ad702aef127d0)
- [Isaac Lab 조사 시점 main 고정 소스](https://github.com/isaac-sim/IsaacLab/tree/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8)

## 8개 blob inventory

| 역할 | source commit 기준 경로 | 크기(byte) | Git blob SHA-1 |
| --- | --- | ---: | --- |
| URDF | `rbq_sdk/ros2/src/rbq_description/urdf/rbq.urdf` | 19,249 | `a598350ba21dc521db7bb16cba199ef35507477e` |
| package manifest | `rbq_sdk/ros2/src/rbq_description/package.xml` | 909 | `c631a432aa1e0083e14b60417bf5b6453552338e` |
| mesh | `rbq_sdk/ros2/src/rbq_description/meshes/stl/calf.STL` | 71,284 | `253f2ee9e4bb67485223faa1951ad89f4442c183` |
| mesh | `rbq_sdk/ros2/src/rbq_description/meshes/stl/hip2.stl` | 866,584 | `0078f9689f7fc31389b64a97b4b01e55a41b6b18` |
| mesh | `rbq_sdk/ros2/src/rbq_description/meshes/stl/hip3.stl` | 818,784 | `8bc31dd3a7aed9926108477a03f200c52012339c` |
| mesh | `rbq_sdk/ros2/src/rbq_description/meshes/stl/mid-360.stl` | 4,193,734 | `ae42618a7b09f274f651156a203bfcc203abc354` |
| mesh | `rbq_sdk/ros2/src/rbq_description/meshes/stl/thigh.stl` | 25,784 | `dbfa93ebe31302bb8fa7b383d4dda1b38c132fb6` |
| mesh | `rbq_sdk/ros2/src/rbq_description/meshes/stl/trunk.stl` | 504,284 | `ddc44a3c5f7f41f42a63eb1a6981dfd1d58cd339` |

기계 판독 가능한 원본은 [`configs/g007_rbq_asset_manifest.json`](../configs/g007_rbq_asset_manifest.json)에 보존한다. 위 링크와 빈 검색 결과는 고정 commit에 대한 증거이며, 미래 release의 상태를 뜻하지 않는다.

## 재현 명령

PowerShell 7.x에서 실행한다.

```powershell
cd "$HOME\isaac-walk-rl"
python .\scripts\validate_rbq_assets.py --manifest .\configs\g007_rbq_asset_manifest.json --expect-blocked --report .\reports\g007_rbq_compatibility_spike.json
$LASTEXITCODE
```

예상 결과는 blocker `RBQ-ASSET-LICENSE-001`과 `license_scope_unresolved`, 종료 코드는 `0`이다. 준비 완료를 요구하면 의도대로 실패한다.

```powershell
cd "$HOME\isaac-walk-rl"
python .\scripts\validate_rbq_assets.py --manifest .\configs\g007_rbq_asset_manifest.json --require-ready --report .\reports\g007_rbq_compatibility_spike.json
$LASTEXITCODE
```

예상 종료 코드는 `3`이다. 종료 코드 `3`은 검증기 오류가 아니라 자산 게이트가 아직 열리지 않았다는 계약이다.

## 검증 결과와 해시

2026-08-24에 위 두 경로를 다시 실행해 각각 exit `0`과 `3`을 확인했고, `python -m pytest tests/test_g007_rbq_gate.py -q`는 `46 passed`였다. 코드 검토 판정은 `APPROVE`였다.

| 대상 | SHA-256 | 의미 |
| --- | --- | --- |
| `reports/g007_rbq_compatibility_spike.json` 파일 | `8cace17b61c944c1395bd42bff81c0cdbd8c39e8b041b0b2039f382983d8927d` | 재현 보고서 byte 해시 |
| manifest canonical JSON | `93ec6cfa7f06d7f2c8b43ac5f057aa2e5b09767a11c515ef333b1dcac799edbf` | 보고서가 기록한 매니페스트 의미 해시 |
| `configs/g007_rbq_asset_manifest.json` 파일 | `994d82f203c30db2f63747f6e894c47898ab8c4e908a42f8ddca2c448b47cc8b` | 매니페스트 파일 byte 해시 |
| `scripts/validate_rbq_assets.py` 파일 | `28040254c014e6de99ab99dac578eee9a0ad55e94353cb6fad5d14fe75bfc36b` | 보고서가 기록한 검증기 해시 |

파일 공백이 바뀌면 매니페스트 파일 해시는 달라질 수 있지만 canonical JSON 해시는 의미가 같으면 유지된다.

## blocker 해제에 필요한 증거

Rainbow Robotics가 발행했거나 승인한 다음 자료가 필요하다.

- 고정한 8개 자산 blob에 적용되는 라이선스 또는 명시적 권한
- 로컬 다운로드·처리와 URDF/STL 변환 허용 범위
- 변환된 USD 등 파생물의 생성·보관 허용 범위
- 원본 및 파생물의 Git 저장·재배포 허용 여부와 필요한 고지·귀속 조건

권한이 로컬 처리만 허용한다면 자산과 파생물은 저장소 밖에 두고 해시·검증 결과만 기록해야 한다. 문서 또는 서면 권한의 범위가 위 항목을 명확히 하지 않으면 게이트를 유지한다.

## 허가 후 별도 수행 순서

1. source commit `68bc33b77719d357b4323fb88549efd905caf721`에서 8개 파일을 저장소 밖으로 가져온다.
2. 각 파일의 크기와 Git blob SHA-1을 manifest와 대조하고, 원시 byte SHA-256을 별도로 기록한다.
3. URDF의 asset URI를 해석한 뒤 link·joint·mesh topology와 actuator/contact 요구사항을 검증한다.
4. Isaac Lab 2.1.1 경계에서 사용할 변환기와 설정 매핑을 별도 구현·검증한다.
5. 고정 base로 load·joint·contact 최소 smoke를 수행한 뒤에만 보행 태스크 이식을 검토한다.

이 순서는 아직 실행하지 않았다. 현재 topology, 변환 성공 여부, Isaac Sim 4.5 호환성, 보행 가능성에 대한 증거는 없다.
