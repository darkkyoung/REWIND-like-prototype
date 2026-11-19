# Train300/400 사전 진단

기존 T4 결과 comparison.csv의 Global / Pelvis-relative / Pelvis Error(raw)와 Mean Absolute Bone Distortion(%p)을 인용한다. 프레임 범위와 연결 순위는 같은 결과의 frame_comparison.csv 및 bone_length_by_edge.csv에서 이번에 다시 집계했다.

## opening130

기존 지표 순서대로 7.908, 7.665, 3.422, 22.160.
프레임별 전체 길이 비율 범위 89.28–93.88%.

| 연결 | 평균 길이 비율 (%) | 프레임 최소–최대 (%) |
|---|---:|---:|
| lowerarm_l → hand_l | 46.84 | 43.38–50.66 |
| calf_l → foot_l | 138.56 | 133.64–142.34 |
| upperarm_l → lowerarm_l | 62.81 | 60.23–65.26 |
| lowerarm_r → hand_r | 68.07 | 41.21–85.93 |
| upperarm_r → lowerarm_r | 68.39 | 61.37–73.28 |
| calf_r → foot_r | 130.94 | 128.63–132.49 |
| pelvis → thigh_l | 127.49 | 123.77–133.30 |
| spine_02 → spine_03 | 126.25 | 122.92–131.13 |
| pelvis → thigh_r | 77.20 | 67.45–90.90 |
| foot_l → ball_l | 122.73 | 115.93–128.79 |
| pelvis → spine_01 | 80.20 | 78.51–81.50 |
| clavicle_r → upperarm_r | 81.19 | 63.66–93.45 |
| foot_r → ball_r | 117.45 | 115.10–118.88 |
| thigh_l → calf_l | 85.08 | 82.75–87.00 |
| neck_01 → head | 85.22 | 81.01–91.03 |
| clavicle_l → upperarm_l | 86.92 | 76.87–96.04 |
| thigh_r → calf_r | 87.02 | 86.12–88.46 |
| spine_01 → spine_02 | 89.85 | 87.68–92.46 |
| spine_03 → clavicle_l | 91.21 | 84.16–97.51 |
| spine_03 → neck_01 | 91.61 | 90.53–92.91 |
| spine_03 → clavicle_r | 100.46 | 94.91–107.68 |

## gunplay47

기존 지표 순서대로 17.995, 17.126, 4.253, 24.122.
프레임별 전체 길이 비율 범위 91.87–101.46%.

| 연결 | 평균 길이 비율 (%) | 프레임 최소–최대 (%) |
|---|---:|---:|
| lowerarm_r → hand_r | 35.91 | 32.19–40.11 |
| pelvis → thigh_l | 154.97 | 139.70–179.70 |
| calf_l → foot_l | 149.20 | 145.08–152.37 |
| lowerarm_l → hand_l | 56.34 | 14.21–98.83 |
| clavicle_l → upperarm_l | 135.97 | 81.75–205.62 |
| pelvis → thigh_r | 134.60 | 123.11–143.23 |
| spine_02 → spine_03 | 132.73 | 125.82–136.93 |
| upperarm_r → lowerarm_r | 73.39 | 69.26–77.24 |
| thigh_l → calf_l | 74.36 | 72.04–76.21 |
| calf_r → foot_r | 124.34 | 121.31–127.45 |
| pelvis → spine_01 | 83.00 | 81.19–85.46 |
| clavicle_r → upperarm_r | 83.85 | 79.19–89.92 |
| upperarm_l → lowerarm_l | 86.95 | 63.79–107.31 |
| foot_r → ball_r | 109.53 | 104.58–116.74 |
| spine_03 → clavicle_l | 91.25 | 79.07–109.35 |
| neck_01 → head | 92.29 | 88.39–95.74 |
| spine_01 → spine_02 | 92.67 | 87.15–97.23 |
| spine_03 → clavicle_r | 93.00 | 81.95–99.15 |
| thigh_r → calf_r | 106.55 | 103.41–108.72 |
| spine_03 → neck_01 | 93.65 | 90.93–96.94 |
| foot_l → ball_l | 102.99 | 97.76–106.55 |

관절별 위치 오차는 공개 CSV만으로 복원할 수 없다. Work 실행에서 저장 예측 NPZ로 관절별 오차와 edge 변동을 계산하고, checkpoint에 양안 x 좌표 +32px를 적용해 정상 입력과 동일 noise의 출력 차이를 측정한다. 이 차이는 해당 변형에 대한 민감도이며 원인 판단에는 추가 근거가 필요하다.
