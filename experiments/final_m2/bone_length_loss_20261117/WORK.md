# Colab T4 인계

Colab에서 이 디렉터리의 Control800_T4.ipynb를 열고 T4 런타임에서 위에서 아래로 실행한다. 노트북이 고정된 소스 커밋을 checkout하고 Drive를 마운트한다. 기본 경로는 MyDrive/rewind_bone 아래이며 설정 셀에서 경로를 바꿀 수 있다.

| 경로 | 준비할 실물 |
|---|---|
| inputs/ | Scaling300 입력 패키지의 train300.npz, opening130.npz, gunplay47.npz, data_selection_manifest.json |
| parent/ | 기존 결과 ZIP의 train300_400.pt 및 opening130_train300_predictions.npz, gunplay47_train300_predictions.npz |
| poses/ | config.json의 candidate_sequence 전체 경로 아래 json/frame_0.json … frame_56.json |
| candidate/ 및 results/ | 새 출력 경로. 실행 실패 시 기존 산출물을 보존하고 재실행에는 새 경로를 지정한다. |

후보는 Joyful_Jump__16_의 0–56으로 사전 지정했다. 기존 RGB 발표 진단에 쓰인 자료이며 M2 Train300에는 없는 동작이다. 원본 sequence 출처는 과거 사용자 식별에 의존한다(docs/PREDICTION_OVERLAY.md). Work에서 원본 경로와 인벤토리를 대조한다. 동일 motion group, Train/Gunplay 원본 pose SHA 및 Train/Opening/Gunplay의 정확히 같은 입력·GT를 검사한다. 다른 이름으로 재가공된 원본 mocap의 독립성은 미확인이다.

저장소 루트에서 실행하는 명령은 다음과 같다.

```bash
python -m experiments.final_m2.bone_length_loss_20261117.prepare_candidate poses candidate
python -m experiments.final_m2.bone_length_loss_20261117.run_experiment --input-dir inputs --checkpoint-dir parent --unused-motion candidate/joyful57.npz --output results
```

부모 checkpoint SHA는 config.json에, 기존 eval NPZ·저장 예측 SHA는 dataset_scaling_300의 input_checksums.json 및 checksums.json에 고정했다. 정상 조건의 부모 출력이 저장 예측과 최대 좌표차 0.001 raw 이상이면 중단한다. 두 분기 직전 동일한 부모 RNG를 복원한다. Train 8배치의 Loss와 gradient norm을 calibration.json에 기록하고 λ=0.1×median(norm_position/norm_bone)을 한 번 고정한다.

최종 800에서 세 동작 각각 Mean Absolute Bone Distortion이 감소하고 Global/RR/Pelvis/Frame Displacement Error가 각각 Control의 102% 이내여야 한다. Motion, RR Motion, Joint Spread Ratio의 100% 편차는 각각 Control 대비 +2%p 이내여야 한다. 결측 gate는 통과할 수 없다. 평가 동작별 gate와 전체 판정은 evaluation.json에 기록한다.

회수할 결과는 parent_diagnostics.json, calibration.json, experiment_identity.json, 두 checkpoint와 random_trace/training 로그, 예측 NPZ·비교 CSV, evaluation.json, checksums.json이다. 예산은 300 train frames, 8 calibration draws, 군별 400 update로 고정했다. 이 브랜치의 검증 범위는 CPU이며 T4 결과는 NOT VERIFIED다.
