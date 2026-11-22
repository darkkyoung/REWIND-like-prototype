# 사전 판정 · RR teacher pilot

GPU 결과 확인 전 고정한다. 1차 대조는 동일 24 updates의 Improved 대 Control이다. 각 validation clip에서 RR JPE가 Control보다 1% 이상 낮고, Body global·pelvis·wrist·Hand global·Whole62 MPJPE가 각각 Control의 102% 이하여야 한다. 두 clip 각각 충족해야 전체 PASS다. 지표 누락은 INCOMPLETE다. 1%/2%는 탐색적 공학 기준이다. Original 대비 변화도 별도 공개한다. Hand local·FDE·Motion Ratio·Bone·Spread·프레임별 손실은 관찰 지표다. 단일 clip 또는 평균으로 실패 조건을 숨기지 않는다.

학습은 selection.json의 train 여섯 clip × 50 frames, FP32 batch 1, seed 20261122, Adam LR 1e-5, StepLR(step=1,gamma=1). epoch별 여섯 clip 순열을 독립 seed로 고정한다. λ는 첫 네 calibration draw에서 기존 목적 gradient norm / 최종 x0 RR gradient norm 비율의 중앙값 × 0.1로 고정한다. 모델 선택·sweep 없음. GPU preflight가 성공하면 N=24, 실패하면 양쪽 N=0으로 중단한다. GPU 종류 T4 또는 A100. 원래 크기의 forward/backward/Adam step과 peak reserved VRAM을 검사하고 총 용량의 95%를 초과하면 중단한다. 예산은 두 군 각 24회, 버리는 preflight 1회와 calibration 4회, 평가 3×2 clips다. CPU 축소 모델은 개발 검사용이다.
