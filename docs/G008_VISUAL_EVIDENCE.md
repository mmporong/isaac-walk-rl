# G008 전진·후진·좌우 회전 시각 증거

## 공개 범위

Git에는 GIF와 접촉시트만 넣는다. 원본 MP4는 로컬 Isaac Lab 로그 아래에 보관하고, 이 문서에는 경로와 SHA-256만 기록한다. 공개 파생물은 동작을 빠르게 확인하는 자료이며 64환경 고정 평가 JSON을 대신하지 않는다.

![전진·후진·좌회전·우회전 GIF](media/g008/g008_direction_commands.gif)

![네 방향 접촉시트](media/g008/g008_direction_contact_sheet.png)

접촉시트는 왼쪽 위부터 전진, 후진, 좌회전, 우회전 순서다. 원본 영상의 3.0초, 7.5초, 12.0초, 16.5초 프레임을 중앙 확대해 배치했다.

## 어떤 정책을 촬영했는가

촬영에 사용한 정책은 G006 baseline `model_1499.pt`에서 command distribution으로 300 iterations를 이어 학습한 G008 `model_1798.pt`다.

- checkpoint: `%USERPROFILE%\IsaacLab\logs\rsl_rl\unitree_go2_rough\2026-08-26_11-11-12_g008_command_finetune_g006_s42_e1024_i300\model_1798.pt`
- checkpoint SHA-256: `53cc09043088bcd53618d2ae1f90c7f2e91d01eab7090cc63922486942b2ed47`
- task: `Isaac-G008-Velocity-Rough-Go2-CommandSuite-v0`
- seed: `42`
- control step: `0.02 s`, 50 Hz
- 촬영 환경: nominal friction `0.8/0.6`, base·leg mass randomization 비활성화

영상은 다음 순서로 900 control step을 실행한다.

| 구간 | steps | 시간 | 명령 `[v_x, v_y, ω_z]` |
| --- | ---: | ---: | --- |
| 정지 | 50 | `1.0 s` | `[0, 0, 0]` |
| 전진 | 175 | `3.5 s` | `[0.6, 0, 0]` |
| 정지 | 50 | `1.0 s` | `[0, 0, 0]` |
| 후진 | 175 | `3.5 s` | `[-0.4, 0, 0]` |
| 정지 | 50 | `1.0 s` | `[0, 0, 0]` |
| 좌회전 | 175 | `3.5 s` | `[0, 0, 0.5]` |
| 정지 | 50 | `1.0 s` | `[0, 0, 0]` |
| 우회전 | 175 | `3.5 s` | `[0, 0, -0.5]` |

## 원본과 파생물 무결성

원본은 H.264, 1280×720, 50 fps, 899 frames, 17.98초다. 시뮬레이션 step은 900회였지만 recorder가 저장한 프레임은 899개다. 이 차이를 숨기지 않고 ffprobe 결과로 남긴다.

| 파일 | 공개 여부 | 크기 | SHA-256 |
| --- | --- | ---: | --- |
| `%USERPROFILE%\IsaacLab\logs\visual_evidence\g008\g008_directions_s42.mp4` | 로컬 전용 | `1,822,108 bytes` | `c388648da898b48a6a00d2415c5a9d7e2342b605c8d07fd559b7400525a716ec` |
| `docs/media/g008/g008_direction_commands.gif` | Git 공개 | `6,648,559 bytes` | `91d213654bf2912b8f04a3142391675d7708d6748b3b3ca30508133f1bf06810` |
| `docs/media/g008/g008_direction_contact_sheet.png` | Git 공개 | `604,557 bytes` | `d80e5bb8bc77a2dd9cfa60b04f290c7ccd3e5d09a056361313e7e6a0172899f7` |

GIF와 접촉시트는 FFmpeg `8.1-full_build-www.gyan.dev`로 만들었다. GIF는 중앙 720×540 영역을 잘라 420×315, 6 fps, 96색으로 줄였다. 원본 MP4는 변환 뒤에도 수정하지 않았다.

## 정량 결과와의 관계

같은 checkpoint의 평면 고정 평가에서는 방향당 16개, 총 64개 환경이 모두 생존했고 네 방향 gate를 통과했다. rough terrain에서는 좌우 회전은 통과했지만 전진과 후진의 순간 자세가 `0.35 rad` 기준을 넘었다. 따라서 이 영상은 네 command가 실제 policy action으로 이어지는 모습을 보여주지만, rough terrain 강건성이나 실물 Go2 성능을 입증하지 않는다.

기계적 수치는 `reports/runs/g008_directional_qualification_finetune_g006_s42_plane.json`과 `reports/runs/g008_directional_qualification_finetune_g006_s42_rough.json`, 파일 메타데이터는 `reports/runs/g008_direction_visual_evidence.json`을 기준으로 한다.
