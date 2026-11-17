# Dataset Scaling 300

실행 결과와 제한은 [보고서](M2_DATASET_SCALING_300.md), 사전 조건은 [config.json](config.json)을 따른다.

준비: 기존 VERIFIED 패키지의 input과 선택적 Pose 원본을 사용하여 prepare_data.py --old OLD_INPUT --source SOURCE_DIR --output NEW_INPUT을 실행한다. SOURCE_DIR에는 cache_manifest.json, source_records.json, raw_pose가 필요하다. 원본은 비공개 결과 ZIP에만 보존한다.

학습: python experiments/final_m2/dataset_scaling_300/run_experiment.py --input-dir INPUT --baseline-dir BASELINE --output NEW_RESULTS

출력 디렉터리는 새 경로여야 한다. 실제 T4, 입력 SHA, 코어 SHA, 초기 model/optimizer/scheduler 해시가 일치해야 학습한다. 기존 두 모델을 재학습하지 않는다. 사용자 결과 ZIP과 실행 노트북은 최종 보고서에 연결한다.

독립 검산: python experiments/final_m2/dataset_scaling_300/validate_results.py --archive RAW_ZIP --input INPUT --baseline BASELINE --source source_at_execution.zip --output NEW_AUDIT

그림: python experiments/final_m2/dataset_scaling_300/plot_results.py --input NEW_AUDIT/raw --output NEW_FIGURES

Colab의 tests 이름 충돌을 피하기 위해 노트북은 unittest discover로 기존 M2 회귀를 실행한다. 로컬33개·Colab15개 통과 기록과 최초 import 실패 기록을 구분해 보존했다.
