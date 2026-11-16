# M2 Dataset Scaling 100

고정된 Train18과 Standing_Up50 + Dodging32를 합친 Train100을 동일한 seed43 초기 가중치부터 각각 400 updates 학습한다. 기존 Position Loss만 사용한다. 모델 A를 B의 초기값으로 사용하지 않는다.

- `prepare_data.py`: 원본 ZIP/pose SHA, 좌표 변환식 일치, 1024 이미지 크기, Body22, causal window와 누출을 검사한다. 원본 보정 없음.
- `run_experiment.py`: 실제 T4/FP32 전용. 각 군 100/200/300/400 저장, 두 군 종료 후 400만 DDIM10/seed10043 평가.
- `test_scaling.py`: 실제 입력과 boundary/gap/mask/미래 누출, 좌표 불일치 거부, 초기 상태와 실제 gradient 검증.
- `config.json`: 결과를 보기 전에 고정한 설정과 8개 항목의 판정 규칙.

```bash
M2_SCALING_INPUT=/path/input python -m unittest experiments.final_m2.dataset_scaling_100.test_scaling -v
python experiments/final_m2/dataset_scaling_100/run_experiment.py --input-dir /path/input --output /path/new_results
```

현재 커밋은 입력·코드 검증 준비본이다. GPU 실행 완료 전 성능은 **NOT VERIFIED**이며, 실제 결과는 후속 보고서로 기록한다. 기존 main/원본 저장소/과거 실험을 변경하지 않는다.

자료는 GT Stereo2D / stored device_pts3d / raw 단위 / zero camera의 진단 계약이다. Opening130은 Historical Holdout이다. 신규82는 cache manifest의 train이며, 기존18은 기존 명시 Train18을 그대로 유지했다. 원본 mocap 독립성과 물리 단위·FPS는 미검증이다. 프레임 증가에는 동작·인물 구성 변화도 포함되며, 고정400 updates에서 프레임당 평균 노출 횟수가 줄어든다.
