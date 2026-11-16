# Dataset Scaling 100 — 인계 후 독립 재검증 (2026-11-16 KST)

## 판정

| 구분 | 판정 | 이번 채팅의 근거 |
| --- | --- | --- |
| 기존 실제 T4 실행 | PASS | 실행 notebook의 완료 wrapper, 두 군 400행 CSV/JSONL, checkpoint·trace·예측 원본 대조 |
| 성능 개선 | PARTIAL | 기존 사전 기준 8개 중 5개 PASS, 3개 FAIL 유지 |
| 전체 T4 결과 회수 | PASS | 지정 원본 ZIP과 8개 checkpoint·8개 NPZ·전체 로그 실제 확보 |
| 이번 독립 파일 검증 | PASS | 아래 재검증 기록 참조 |
| 신규 학습·추론 | 미실행 | optimizer update 0, model inference 0 |

## 인계문 이후 이미 완료된 작업

작업 시작 시 fresh fetch한 사용자 Fork의 실험 브랜치는 `9511979fc0ff340ed4de77cbd738a7ef42484533`이었다.
인계문 기준 `6ccf0ab650a225ec0c823b9ff030ce19c5a7dc84`보다 2개 커밋 앞섰다.

- `572422a6423939eb264b3282af2d0f05a8b37997`: 전체 T4 결과 복구·보고서·그림·로그 게시.
- `9511979fc0ff340ed4de77cbd738a7ef42484533`: 원본 ZIP CRC 항목 수 32개 정정.
- 동료 원본 main fresh fetch: `e94c64d8689adc6186979b56906e1008b53fb76c`.
- 보고된 이전 로컬 `0b75b9a` 객체는 이번 새 clone에서 확인되지 않았다. 손실로 단정하지 않으며 새 SHA로 가장하지 않는다.
- VERIFIED ZIP의 `experiment/` 47개 파일이 위 원격 HEAD의 Git blob과 바이트 단위로 모두 일치했다. 기존 결과를 중복 복원하거나 되돌리지 않았다.
- `config.json`, `prepare_data.py`, `run_experiment.py`, `data_selection_manifest.json`은 실행 당시 소스와 동일하다.

## 이번 채팅에서 새로 검증한 증거

원본 입력 파일은 수정하지 않았다. 기존 지표를 덮어쓰지 않고 새 디렉터리에서 재계산했다.

| 검증 | 결과 |
| --- | --- |
| 제공 PARTIAL ZIP | 21,640,000 bytes, CRC 186개, 내부 SHA 185개 일치 |
| 기존 VERIFIED ZIP | 29,675,025 bytes, CRC 166개, 내부 SHA 165개 일치 |
| 전체 T4 raw ZIP | 12,373,875 bytes, CRC 32개, 내부 SHA 29개 일치 |
| 원본 Pose·선택 manifest | 고유 100개, Brutal18 + Standing50 + Dodging32, Pose SHA 100개 일치 |
| 신규 82개 train 소속 | 보존 cache manifest와 대조 PASS |
| 좌표·입력 | 원본 Pose에서 Body22·Stereo2D·causal window를 다시 구성해 저장 tensor와 완전 일치 |
| 기존 입력 보존 | 과거 148-frame 입력 ZIP의 train.npz·holdout.npz와 현재 train18.npz·opening130.npz 파일 SHA 일치 |
| 확인 가능한 holdout 중복 | sequence/frame 키·Pose SHA·정확한 입력 tensor 중복 0 |
| checkpoint | 양쪽 100/200/300/400 총 8개 SHA·step·Adam·StepLR·finite weights·RNG 확인 |
| 최종 RNG·trace | 최종 RNG 해시, 각 400행 trace 해시, 실제 anchor 빈도와 실행 기록 일치 |
| 저장 예측 | 8개 NPZ, 그룹당 36개 지표 기존 검증기로 재계산 일치 |
| 별도 계산식 | 핵심 10개 수치를 독립 NumPy 구현으로 8개 그룹 재계산, JSON/CSV와 일치 |
| Control400 | 별도 이전 원본 NPZ를 열어 dtype·shape·prediction/GT 바이트 직접 일치 확인 |
| 학습 로그 | 각 400행 CSV/JSONL 완전 대조, notebook의 8개 checkpoint loss 관측점과 일치 |
| 시각화 | PNG 7개 decode, GIF 3개 각각 130프레임·프레임당 100ms, 공통 축 범위 재계산 일치 |

허용오차는 정량 수치 `rtol=1e-12, atol=1e-12`다. 배열 비교는 정확한 원소 또는 바이트 일치다.
Global/RR skeleton 그림을 직접 확인했으며, 모든 GIF 프레임은 decode했다. 공통 축 및 1:1:1 scale은 저장된 축 수치 재계산과 plotting code로 확인했다. 모든 GIF 프레임의 육안 검토를 의미하지 않는다.

원본 raw ZIP SHA-256:
`4ee135ed64654aa563c975fdca748eba102615ccda633b050166b9f93793c024`

검증에 사용한 이전 VERIFIED ZIP SHA-256:
`ea238b9cfc0dd54ae546672dd48d99e99fb15e356a3adab2d459ed2b083a9413`

PARTIAL ZIP SHA-256:
`c04092dc73eb6b5affb3bb1fd73fd5a7c5c8b1bf0d7a09c1370685f9dbc0ed0f`

위 VERIFIED 해시는 이번 추가 보고서를 넣기 전 입력 ZIP의 해시다. 최종 전달 ZIP은 추가 파일 때문에 다른 해시를 가진다.

[독립 재검증 JSON](metrics/t4/handoff_verification.json) · [기존 패키지·원격 대조 JSON](metrics/t4/handoff_package_comparison.json) · [검증 스크립트](verify_handoff.py)

## Notebook의 실패와 완료를 구분

실행 notebook의 0-based cell 2에는 `tests.test_m2`, `tests.test_final_metrics` import 실패 후 AssertionError가 남아 있다.
해당 오류는 학습 프로세스 생성 전 preflight 단계다. 다음 cell 3은 unittest discover 방식으로 회귀 검사를 통과한 뒤 두 군의 실제 학습·평가를 완료했고 `RETURN_CODE 0`을 기록했다.
오류 기록을 삭제하거나 notebook 전체가 무오류였다고 주장하지 않는다. 보존된 최종 checkpoint와 로그가 완료 근거다.

T4 당시 환경은 Python 3.13.16 / PyTorch 2.14.1+cu130 / CUDA 13.0 / Tesla T4다.
이번 검증 환경은 Windows CPU / Python 3.12.14 / PyTorch 2.15.0+cpu / NumPy 2.3.5다.
CPU에서 저장 상태와 배열만 읽었으며 원래 GPU 실행을 재현하거나 resume 학습한 것이 아니다.
이전 unit test·resume 검사 로그는 과거 증거로 보존하고 이번에 새로 실행한 검사와 구분한다.

## Opening130 실측 비교

오차는 annotation raw 단위이며 mm로 해석하지 않는다. Body22 / DDIM10 / noise seed10043 / GT Stereo2D / zero-camera 조건이다.

| 지표 | Train18/400 | Train100/400 |
| --- | ---: | ---: |
| Global Joint Error | 11.2621 | 9.2867 |
| Root-relative Error | 10.1958 | 8.2862 |
| Frame Displacement Error | 0.4741 | 0.3389 |
| Motion Ratio | 58.43% | 73.08% |
| Root-relative Motion | 35.64% | 54.08% |
| Bone Length Ratio | 98.08% | 88.78% |
| Joint Spread Ratio | 96.92% | 92.79% |
| Pelvis Error | 5.3995 | 7.3709 |
| 이동 방향 각도 오차 | 34.89° | 27.57° |
| Bone distortion | 13.5460 pp | 19.0772 pp |

Global/RR/FDE는 각각 17.54%/18.73%/28.52% 감소했다. 129개 인접 구간 중 FDE가 121개 개선, 8개 악화됐다.
RR 평균 자세 편향은 9.2315→7.3279 raw, RR 시간변화 RMS 비율은 44.82→60.40%다.
평균 자세 쏠림과 움직임 과소 추정이 완화됐지만 해결되지는 않았다.

Pelvis Error는 36.51% 증가했고 bone distortion은 5.5312 pp 증가했다. 뼈 길이와 관절 퍼짐도 GT에서 더 멀어졌다.
따라서 Train100의 전면 우월성 또는 개선 완료를 주장하지 않는다.

## 회수 및 남은 한계

기존 인계문의 미회수 주요 파일은 모두 확보됐다: 양쪽 400 checkpoint, Train18 100/200/300 checkpoint, 양쪽 전체 CSV/JSONL, random trace·identity 및 Brutal18 예측 2개.
원본 checksum 목록 기준 누락은 0이다. 실제 checkpoint·prediction·Pose는 개인 전달 ZIP에만 보존한다.

여전히 NOT VERIFIED: 원본 mocap 수준의 독립성, 독립 미사용 동작 일반화, 통계적 유의성, 물리 단위, capture FPS.
선택 RGB의 원본 바이트와 전체 raw cache, Opening 원본 Pose JSON은 이번에 다시 검증하지 않았다. 확보한 과거 Opening 입력 NPZ의 동일성은 검증했다.

## 재검증 실행

새 디렉터리만 출력 대상으로 사용한다. 학습 notebook 실행은 필요 없다.

```bash
python -X utf8 experiments/final_m2/dataset_scaling_100/recover_full_archive.py --archive PACKAGE/t4_original/REWIND_M2_Scaling100_T4_raw.zip --output NEW_RAW
python -X utf8 experiments/final_m2/dataset_scaling_100/verify_handoff.py --package PACKAGE --raw NEW_RAW --prior PACKAGE/prior_reference --source PACKAGE/source_at_execution.zip --output NEW_AUDIT
```

`PACKAGE`는 최종 전달 ZIP을 푼 경로다. `prior_reference/`에는 이번 직접 비교에 사용한 과거 Control400 NPZ와 과거 입력 ZIP의 내용이 있다.
의존성은 NumPy, Pillow, matplotlib, PyTorch CPU다. 검증기는 학습 함수나 모델 추론을 호출하지 않는다.

## 300프레임 확대 판단

탐색적 후속 비교의 근거는 확보했지만, 현재 결과만으로 무조건 확대하거나 Train100을 최종 모델로 채택하지 않는다.
독립 미사용 동작 평가와 Pelvis·형태 악화 제한을 사전에 고정한 별도 실험으로 검토한다.
11월 20일 개발 마감과 11월 23일 발표에는 현재 18/100 비교와 부작용을 함께 제시할 수 있다.
이번 작업에서는 300프레임 학습·모델·Loss 변경·main merge를 실행하지 않았다.
