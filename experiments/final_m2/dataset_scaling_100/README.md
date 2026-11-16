# M2 Dataset Scaling 100

실제 Tesla T4에서 18/100프레임 모델을 동일 seed43 초기값부터 각각 400 updates 학습·평가했다. **실행 PASS / 개선 PARTIAL / 전체 T4 원본 회수 PASS**.

Opening130에서 Joint Error는 17.54%, 변위 오차는 28.52% 감소했고 Pelvis Error는 36.51% 증가했다. 8개 사전 기준 중 5개를 통과했다. 원본 ZIP에서 예측 8개와 checkpoint 8개를 회수해 검증했다. [결과 보고서](M2_DATASET_SCALING_100.md)

- `prepare_data.py`: 원본 SHA·좌표식·Body22·mask·causal boundary·누출 검사. 원본 보정 없음.
- `run_experiment.py`: T4/FP32, 두 군 각각400 updates, 기존 Position Loss,100/200/300/400 저장, 최종400만 DDIM10/seed10043 평가.
- `test_scaling.py`: 실제 입력9개 검사와 실제 parameter gradient 확인.
- `recover_full_archive.py`: 기존 실행의 원본 ZIP 길이·SHA·CRC 검증 및 해제.
- `validate_results.py`: 회수한 예측·checkpoint·trace를 검증. 기준 Control400 NPZ의 기록된 SHA를 확인하고, 별도 원본이 있으면 배열도 직접 대조한다.
- `plot_results.py`: 실제 Opening 배열의 동일축 global/root-relative/GIF와 실제 관측 loss.
- `config.json`: 결과 확인 전에 고정한 설정·판정 규칙.
- `selected_frames.csv`: 재사용할100개 고유 frame 경로와 SHA.
- `metrics/t4/`: 실제 수치, 두 군의 400-step training CSV, 회수 상태와 검증 기록.
- `REWIND_M2_Scaling100_reproduce.ipynb`: 최초 T4 실행 소스 `6ccf0ab650a225ec0c823b9ff030ce19c5a7dc84`를 사용하는 재현 진입점.

기존 결과 ZIP과 실행 노트북으로 사후 검증한다. 원본 ZIP과 입력 데이터는 개인 전달본에 포함된다.

```bash
python experiments/final_m2/dataset_scaling_100/recover_full_archive.py --archive /path/REWIND_M2_Scaling100_T4_raw.zip --output /path/recovered_results
python experiments/final_m2/dataset_scaling_100/validate_results.py --input /path/recovered_results --output /path/analysis
python experiments/final_m2/dataset_scaling_100/plot_results.py --input /path/recovered_results --notebook /path/REWIND_M2_Scaling100_executed.ipynb --output /path/figures
```

원본 Control400 배열을 직접 비교하려면 검증 명령에 `--previous /path/prior_displacement_results`를 추가한다.

GT Stereo2D / stored device_pts3d / raw 단위 / zero-camera 진단 계약이다. Opening130은 Historical Holdout이다. 원본 mocap 독립성·물리 단위·capture FPS는 미검증이다. 데이터 수와 동작·인물 다양성이 함께 바뀌며 고정400 updates에서 프레임당 노출은 줄어든다. 전체 데이터셋 준비 완료나 통계적 유의성을 주장하지 않는다.

2026-11-16 후속 인계: [독립 재검증 보고서](HANDOFF_REVERIFICATION_20261116.md). 원본 전체 ZIP·checkpoint 8개·예측 8개 및 과거 Control400 원본 배열 직접 비교 PASS. `verify_handoff.py`는 저장 파일만 검사하며 재학습·추론을 하지 않는다.
