# M2 Bone Length Loss Phase 1

2026-11-19 통제 실험 실행본은 [Work 인계](WORK.md)와 [Colab 노트북](Control800_T4.ipynb)을 따른다. [기존 결과 사전 진단](PARENT_DIAGNOSIS.md)과 CPU 검사 20개를 완료했다. 아래는 이전 Phase 1 기록이다. T4 성능은 **NOT VERIFIED**.

기준은 `cf767233172e2fe7c9514954f177bcfd56c41d17`의 Train300/400이다. 원본 checkpoint 및 기존 M2 코드를 변경하지 않는다. `loss.py`가 x0의 각 Body22 edge 길이와 GT 길이 차이의 제곱을 계산한다. 예측은 기존 `target_scale_raw=100`으로 정규화된 x0이며 GT만 같은 척도로 나눈다. 유효한 두 끝 관절과 time slot에 대해서만 (frame, edge) 관측 평균을 취한다. 관절 좌표를 덮어쓰거나 추론 출력을 보정하지 않는다.

실험은 같은 Train300/400 상태에서 Control(Position)과 Improved(Position+Bone)를 각각 400 update 재개한다. 각 군에서 checkpoint RNG를 다시 복원하고 첫 상태 및 매 update의 anchor, diffusion timestep, noise hash를 비교한다. 가중치는 학습 전 **Train300만** 이용한 8개 결정적 배치에서 Position/Bone 파라미터 gradient norm 비율의 중앙값 × 0.1로 고정해 `calibration.json`에 남긴다. Opening/Gunplay/미사용 동작 값은 이 계산에 접근하지 않는다. 이 8개 배치는 가중치 측정 후 원본 RNG로 다시 복원하므로 군별 학습 draw가 일치한다.

## 입력과 실행

기존 scaling 300 실험의 검증된 `train300.npz`, `opening130.npz`, `gunplay47.npz`, `data_selection_manifest.json` 디렉터리와 원본 `train300_400.pt` 디렉터리가 필요하다. 체크포인트 SHA와 Train300/manifest SHA는 `config.json`에 고정했다. 미사용 동작은 학습 전에 별도로 고른 하나의 연속된 validation 시퀀스 NPZ여야 한다. 같은 stem의 `.selection.json`에 `{"npz_sha256":"...","selected_before_training":true,"sequence_id":"...","selection_reason":"..."}`를 기록한다. 실행 전 Train/Eval 간 frame key 겹침 및 causal window 내용 일치를 검사한다. 원본 mocap 독립성은 이 검사만으로 증명되지 않는다.

T4 Colab에서 저장소를 이 revision으로 checkout하고 입력을 Drive에서 별도 읽기 위치에 준비한다. 실행 결과는 별도 새 출력 디렉터리에 기록한다.

```bash
python -m unittest experiments.final_m2.bone_length_loss_20261117.test_bone -v
python experiments/final_m2/bone_length_loss_20261117/run_experiment.py \
  --input-dir /path/to/scaling300_inputs \
  --checkpoint-dir /path/to/original_checkpoints \
  --unused-motion /path/to/preselected_unused.npz \
  --output /path/to/new_bone_loss_output
```

각 군의 `*_800.pt`, random trace, 400행 training log, `comparison.csv`, edge별 값과 `evaluation.json`을 보존한다. `comparison.csv`의 Bone Distortion을 주목하고 Joint Error, Root-relative Error, Pelvis Error, Motion Ratio, Joint Spread를 함께 비교한다. Opening130과 Gunplay47은 기존 분석에 사용됐으므로 독립 Test로 표시하지 않는다. GPU 실행과 수치 개선은 현재 **NOT VERIFIED**다.

## 연구 근거 및 판정

이전 [Train300 보고서](../dataset_scaling_300/M2_DATASET_SCALING_300.md)는 평균 길이뿐 아니라 특정 edge의 큰 비율 왜곡을 보고했다. 길이 제약은 회전·방향 또는 운동량을 직접 규제하지 않으므로 성능은 평가로 확인해야 한다. 사전 판정은 두 평가 동작과 미사용 동작 모두에서 edge별 절대 길이 비율 편차의 평균이 Control보다 낮아야 하며, Joint Error, Root-relative Error, Pelvis Error가 각 2%를 초과해 악화되지 않고 Motion Ratio와 Joint Spread의 100% 편차도 2%p를 초과해 악화되지 않아야 한다. 미사용 동작의 독립성 확인이 없으면 일반화 결론은 유보한다. 단일 seed 결과는 통계적 유의성 근거가 아니다.

재검토에서 원본 커밋의 checkpoint schema 오류를 수정했다. 현재 준비 범위와 남은 입력 조건은 [REVIEW.md](REVIEW.md)를 따른다. Opening/Gunplay SHA 검증과 미사용 sequence/motion·동일 pose 검사를 실행 전에 적용한다. 사전 판정 결과는 `evaluation.json`의 `quality`에 기록하며 미정의 지표는 통과로 간주하지 않는다.
