# Phase 1 코드 재검토 — 2026-11-17

검토 대상: `8b9d341b7063071e43c24b867f04689835f08594`. 시작 시 origin의 실험 브랜치와 일치했다. 결론: **원본 커밋은 T4 실행 준비 완료로 볼 수 없다.** 실제 checkpoint schema 오류로 학습 전에 중단되며, 아래 수정 후 CPU 검사는 통과했다. 실제 GPU 실행·실물 checkpoint 재개·성능 개선은 **NOT VERIFIED**다.

## 발견 사항과 수정

| 우선순위 | 발견한 문제 | 수정 |
|---|---|---|
| P1 | runner가 `ck['source']`를 읽지만 Train300을 저장한 `dataset_scaling_100.train`은 `experiment_identity`를 기록한다. 실제 파일은 `KeyError`로 중단된다. | 실제 writer schema로 수정. format/arm/step/samples/config/전체 manifest/FP32 scaler와 핵심 소스 SHA를 검사한다. 실제 저장된 experiment_identity JSON으로 회귀 검사했다. |
| P1 | 미사용 NPZ의 sequence/frame key 중복만 검사하므로 같은 동작의 다른 frame·다른 인물 복제, 다중 sequence 또는 순서가 뒤섞인 배열이 통과할 수 있다. 기존 evaluator는 하나의 연속 sequence를 전제로 한다. Opening/Gunplay NPZ도 SHA를 잠그지 않았다. | 기존 두 NPZ의 기록된 SHA 검증. 모든 NPZ tensor shape/dtype/finite/causal window 검증. 평가 배열은 하나의 연속 sequence만 허용. 미사용 sequence를 selection에 결합하고 기존 motion group·동일 feature/GT anchor를 거부한다. |
| P2 | 빈 optimizer state에 대한 `any(...)`는 false이므로 Adam 재개 검사가 통과한다. 기존 테스트도 빈 Adam을 사용했다. | 모든 파라미터의 step=400, moment shape/finite와 상태 개수를 검사. Adam moment가 있는 합성 fixture, 누락 상태 거부, 3 update 동안 서로 다른 Loss의 난수 일치를 검사한다. |
| P2 | `mean_abs_bone_ratio_deviation_pp`가 결과표에 없고 README의 악화 gate도 실행하지 않았다. | 유효 edge 관측의 절대 비율 편차와 최소 비율을 기록하고 사전 정의된 joint/RR/pelvis 2%, motion/spread 2%p gate를 평가한다. undefined 값은 PASS로 보지 않는다. |
| P2 | 새 실행의 코드/config/input provenance와 프레임 결과 저장이 부족했다. | 학습 전에 experiment_identity 저장, frame_comparison.csv 저장, update 후 parameter/optimizer finite 검사. 추가 진단의 비정의 비율은 JSON에 null로 기록한다. |

## 요청 항목별 판단

1. **계산식·edge·mask:** `L_bone = mean_valid[(||p_i-p_j||₂ - ||g_i-g_j||₂/100)²]`. 여기서 p는 정규화 x0이고 g는 raw GT다. Body22의 기존 21개 edge를 manifest에서 그대로 사용한다. 두 끝 관절의 valid와 해당 time slot이 모두 true인 관측만 평균한다. xyz별 평균이 아니라 길이 스칼라 제곱의 평균이다. 가시성으로 mask를 대체하지 않는다.
2. **scale:** Position Loss는 기존 함수를 그대로 호출하며 x0와 frame head에 기존 scale=100을 적용한다. Bone Loss는 x0만 받는다. 추가 raw 단위 변환·GT 수정·예측 골격 정렬은 없다. raw를 mm라고 주장하지 않는다.
3. **초기 상태:** 각각 같은 모델·Adam·StepLR state를 복원하고 모델 생성 이후 checkpoint RNG를 다시 복원한다. scaler는 기존 FP32의 빈 state만 허용한다. 원본 dict가 update 후에도 변하지 않는 것을 합성 CPU 시험으로 확인했다.
4. **학습 난수:** anchor index, timestep, noise bytes와 window frame/sequence hash를 전체 400 update trace로 비교한다. 두 군은 같은 입력 NPZ를 사용한다. 모델별 optimizer update로 파라미터가 달라져도 stochastic draw는 일치해야 한다. CUDA 실제 일치는 T4 실행에서 확인해야 한다.
5. **가중치:** Train300만 calibrate에 전달한다. 최초 8개 결정적 draw에서 기존 Position Loss와 Bone Loss의 전체 모델 parameter gradient norm 비율 중앙값 ×0.1을 고정한다. optimizer update 없이 계산하며 군별 학습 시작 때 다시 원본 RNG를 복원한다. 8배치 규칙의 대표성이나 0.1의 최적성은 검증되지 않았다. 평가 결과를 보고 재선택하면 이 실험의 사전 고정 조건을 위반한다.
6. **causality:** 재사용한 window_metadata가 gap 이전 history, sequence 경계, padding, 각 history slot과 실제 anchor 내용 일치를 검사한다. 기존 M2의 미래 feature/camera/noise 교란 테스트 및 streaming gap/reset 테스트가 통과했다. Bone 항 자체는 시간 차분이 없는 프레임 내 공간 손실이다. 저장 NPZ의 외부 생성 과정 전체에 미래 정보를 사용하지 않았다는 사실까지 증명하는 것은 아니다.
7. **회귀:** weight=0은 기존 loss tensor 자체를 반환한다. 값뿐 아니라 gradient의 bitwise 동일성도 확인했다. 기존 Position Loss의 mask/수치 검사 및 M2 모델 회귀를 함께 실행했다.
8. **T4 입력:** 아래 입력이 모두 준비되고 사전 고정돼야 한다. 실물 파일이 현재 workspace에 없어 end-to-end T4 준비 검증은 완료하지 못했다.
9. **CPU:** Python 3.12 / torch 2.15.0+cpu, 신규/확장 10개 + 기존 M2 10개 = **20개 통과**, 로그 `cpu_review_tests.log`. CLI `--help`, py_compile 및 diff 검사를 별도 실행했다. 테스트의 Adam step=400 fixture는 실제 400회 학습 결과가 아니다.
10. **성과:** 실제 T4 학습, 800 update 모델 평가, 가중치의 실제 수치 결정은 미실행. 성능 개선 **NOT VERIFIED**.

## T4 실행 입력 점검

- `--input-dir`: 기존 scaling300 입력의 `train300.npz`, `opening130.npz`, `gunplay47.npz`, `data_selection_manifest.json`.
- `--checkpoint-dir`: 수정하지 않은 `train300_400.pt`. SHA256: `8a4a137cb8db7173e557773e13a43385f159fbcea11badba18de7a3631f9e481`.
- Opening SHA256: `5aaf61938292ea6db9259b625ef94b7762dc171bbc12d7a74916dbc55baa208c`.
- Gunplay SHA256: `4c0f3c2ce5064c3bd09a4a517fdfa2a6d2f971704091ed27e592f8231b167a5c`.
- `--unused-motion`: 별도 미사용 validation 동작 NPZ와 같은 stem의 `.selection.json`. 필수 항목은 `npz_sha256`, `selected_before_training: true`, 실제 `sequence_id`, `selection_reason`이다. 이 선언은 과거 사용 여부나 원본 mocap 독립성의 증거가 아니다. 원본 split·motion 출처와 전처리 규격을 확인하고 **학습 전에 selection 파일을 영구 보관**해야 한다. 현재 선택된 실물은 없다.
- 실행 명령은 README와 동일하며 새 출력 경로만 허용한다. calibration/두 군 학습/평가 순이다. Opening/Gunplay는 이미 분석한 validation이고 독립 Test가 아니다.

## 범위와 게시

변경은 이 실험 디렉터리로 한정했다. main·과거 실험 파일·checkpoint는 변경하지 않았다. 현재 요청은 수정 후 커밋을 명시하므로 같은 브랜치에 로컬 커밋한다. AGENTS.md의 원격 게시 범위 규칙을 확인했다. 이전 Phase 1의 Push는 이미 완료됐으며 이번 재검토 수정에 대한 별도 Push 명시는 없어 이번에는 원격 게시하지 않는다.
