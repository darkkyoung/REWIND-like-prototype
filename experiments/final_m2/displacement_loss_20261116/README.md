# M2 Frame Displacement 통제 실험

기존 step300 checkpoint에서 Adam/StepLR/scaler/RNG를 엄격히 복원한 두 분기를 각각 100 update 학습한다. 기존 M2 코어와 resume guard는 수정하지 않는다. 설정은 `config.json`에 문서화한 고정값이다. 임의 튜닝 인자는 제공하지 않는다.

```bash
python -m unittest experiments.final_m2.displacement_loss_20261116.test_displacement -v
python experiments/final_m2/displacement_loss_20261116/run_experiment.py --input-bundle /path/M2_verified_148_input.zip --output /path/new_t4_result
python experiments/final_m2/displacement_loss_20261116/plot_results.py --input /path/new_t4_result --output /path/new_figures
```

실제 Tesla T4만 학습을 허용한다. 출력 폴더가 있으면 중단하며 기존 결과를 덮어쓰지 않는다. 학습은 train18만 사용하고 두 군 종료 후 baseline/control/improved를 동일한 streaming DDIM10/noise10043으로 평가한다. Holdout 중간 평가·best checkpoint 선택은 하지 않는다. 원본 학습 환경과 현재 환경 차이는 experiment identity에 기록하며 원본 학습의 bitwise 연속 실행이라고 주장하지 않는다.

`loss.py`는 raw x0 출력의 연속 프레임 차분 벡터 오차를 4.1444875로 나누고 xyz 제곱합을 유효 joint-pair에 대해 평균한다. 기존 위치 손실에 improved만 0.1배 추가한다. 두 군의 각 update마다 batch/timestep/noise hash를 저장하고 같지 않으면 결과를 거부한다. optimizer state가 불완전하거나 source/schedule/단위 계약이 다르면 학습하지 않는다. NaN/Inf loss/gradient/parameter/Adam 상태를 감지하면 중단한다.

NPZ와 checkpoint, 개인 입력은 Git에 게시하지 않는다. 실제 실행 상태 및 결과는 `M2_DISPLACEMENT_EXPERIMENT.md`에서 별도로 보고한다. 이 준비 코드만으로 성능 개선을 주장하지 않는다.
