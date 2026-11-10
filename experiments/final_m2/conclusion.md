# REWIND M2 실제 데이터 DDIM 5/10 비교 완료 보고서

## A. GitHub 및 복구

- 원본 시작 main: `5b390a8459a674b378ada4305a647e6fd3e53178` (마무리 조회에서도 동일).
- 기존 M2 실험: `74851d8c05933c135bf784c15f29dbf684987687`.
- 사용자 Fork 작업 브랜치: `experiment/final-m2-202611`.
- 실제 T4 실행 revision: `093e480af527094aeffa449d129220fa77fa7c82`. 최종 결과 기록 commit은 다운로드 패키지 `publication.json`을 참조한다.
- 이전 로컬 `728449c`/delivery ZIP은 복구하지 못했다. 모델·DDIM·streaming 코어는 재구현하지 않았으며 과거 코드와 체크포인트를 재사용했다. 이번에 복구한 비교 runner와 지표 테스트는 Git 기록에 남긴다.
- 동료 원본·main·기존 실험 브랜치 수정, PR·merge·force push 없음. 대용량 원본/개인 입력/notebook 내 base64 입력은 Git에 올리지 않는다.

## B. 실제 데이터

보존된 두 ZIP의 Pose SHA를 대조하고 원래 검증·추출 경로에서 148개 JSON을 검증했다. Drive 추가 다운로드 없음. `input_validation.json`과 `result_validation.json`이 검증 근거다.

| 용도 | 시퀀스 | 프레임 | 해석 |
|---|---|---:|---|
| Train-fit | ArchVisInterior / Day / Adanna / 025 / Brutal_Assassination | 0–17, 18개 | 과거 학습 데이터 진단 |
| Historical holdout | ArchVisInterior / Day / Adanna / 017 / Opening | 0–129, 130개 | 과거 holdout, 공식 독립 test 아님 |

입력은 RGB가 아닌 **GT stereo 2D keypoints**다. Raw 87 joints 중 기존 manifest의 Body22 순서를 그대로 사용했다. 2D /1024, zero-camera, target raw `device_pts3d` /100 및 출력 ×100을 유지했다. 최신 62관절/mm 변환은 사용하지 않았다. 평가 배열의 valid mask는 전부 유효하며 time/history padding은 기존 causal sampler 규칙을 따른다. 원본 motion 독립성, 실제 거리 단위와 촬영 FPS는 NOT VERIFIED다. raw를 mm/cm로 표기하지 않는다.

## C. 모델·통제 조건

- Checkpoint SHA-256: `6acd1fd96a2dbfe1e681710a6286c3d9d0177ce69402a0b488f8fcce7a4a5c9d`.
- Step300, Body22, window8, hidden64, heads4, temporal2, graph1, diffusion training1000.
- Strict state dict, 구조 buffer, alpha schedule, 기존 코어4파일 hash, manifest 일치 검사 통과.
- 과거 학습 seed43. 기존 평가 방식의 noise seed는 **10043 (=43+10000)** 이며 두 arm에서 sequence/frame별 동일 noise를 사용한다. 학습 seed와 평가 noise seed를 구분한다.
- 변경 변수는 DDIM sampling5 또는10. 재학습/optimizer update **0회**.
- 결과 runtime: Tesla T4, Python3.13.16, PyTorch2.14.1+cu130, CUDA13.0, FP32, batch1, CPU thread1.

## D. 실제 실험 결과

| Split / DDIM | Joint Error (raw) | Displacement Error (raw) | Motion Ratio | Mean ms | P95 ms |
|---|---:|---:|---:|---:|---:|
| train_5 | 7.512979 | 1.826566 | 32.8321% | 31.3065 | 38.0697 |
| train_10 | 7.513105 | 1.825593 | 32.8385% | 59.2983 | 66.6679 |
| holdout_5 | 12.228817 | 0.505108 | 40.5589% | 30.6249 | 38.8896 |
| holdout_10 | 12.232006 | 0.503500 | 40.3147% | 64.5317 | 84.5967 |


10-step Opening의 기존 12.2320013 raw와 이번 12.2320055 raw 차이는 0.00000423 raw다. 최대 예측 좌표 차이는 0.00005341 raw이며 사전 허용치(JPE0.0001, 최대좌표0.001 raw) 안으로 **Baseline 재현 PASS**다. 기존 약40.3% motion ratio도 재현했다.

Opening에서 5-step의 평균 지연은 10-step보다 52.54% 낮았다. 위치 오차 차이는 0.003188 raw(약0.026%)로 작고 10-step은 이동 벡터 오차가 조금 낮다. 여러 seed·여러 독립 motion 검증이 아니므로 정확도 우열/통계적 유의성은 주장하지 않는다. 두 설정 모두 움직임 크기가 정답의 약40%에 그쳐 sampling step 증가가 움직임 축소 문제를 해결했다는 근거는 없다.

지연 측정은 split/arm마다 warm-up8프레임, 전체 sequence3회 반복, arm 순서 교차, 전후 CUDA synchronize, perf_counter_ns다. Opening arm별390표본, train arm별54표본. P95는 개별 프레임 표본의 linear percentile이다. 반복 예측은 bitwise 일치 검사를 통과했다. CPU keyed noise·H2D·history packing·DDIM·raw scaling·sync를 포함하고 파일 I/O·RGB frontend·D2H·지표 계산은 제외한다. RGB 전체 pipeline FPS나 kernel 단독 시간으로 해석하지 않는다. 3회 반복의 프레임들은 통계적으로 독립적인 motion 표본이 아니다.

CPU는 `metrics/cpu/`, GPU는 `metrics/t4/`로 분리했다. T4 전체 평가44.54초, PyTorch peak allocated VRAM34,880,512bytes이며 GPU 총 메모리 사용량과 구분한다.

## E. 그래프와 실패 사례

- `figures/joint_error.png`: 위치 오차.
- `figures/motion_fidelity.png`: 이동 벡터 오차·움직임 비율.
- `figures/latency.png`: 평균·P95 지연.
- `figures/holdout_motion_comparison.png`, `train_motion_comparison.png`: GT/5/10 동일 좌표·축 범위, 별도 정렬 없음.
- `figures/holdout_displacement.png`, `train_displacement.png`: 프레임별 이동량.
- 개인 다운로드 패키지 `t4/figures/*_motion.gif`: 실제 출력 애니메이션, 표시10FPS이며 촬영 FPS가 아님.
- `metrics/t4/large_movement_frames.csv`: GT 이동량 상위10% 프레임의 위치오차와 예측 이동량. 이 선택은 사후 실패 분석이며 학습/모델선택에 사용하지 않았다. Opening frame45가 대표적인 큰 이동 프레임이다.

시각화에는 raw device 좌표를 그대로 사용했으므로 화면 위 방향을 실제 중력 방향으로 해석하지 않는다. GT·예측의 축이 동일하다는 비교 조건을 우선했다. 움직임 축소와 자세 불일치가 남은 실패다.

## F. 완료 판정

| 항목 | 판정 | 근거 |
|---|---|---|
| 148 Pose 검증·checkpoint strict load | PASS | 입력 및 결과 검증 JSON |
| 기존10-step Baseline 재현 | PASS | CPU 및 T4 실측, 사전 허용치 충족 |
| T4 DDIM5/10 실제 비교 | PASS | 예측 NPZ, evaluation, latency CSV, 실행 notebook |
| 저장된 출력으로 지표 재계산 | PASS | 네 split/arm의 지표·지연 일치 |
| 동일 초기 noise·반복 결정성 | PASS | noise SHA 및 runner bitwise 검사 |
| causal 미래 차단·경계/gap·mask·checkpoint 회귀 | PASS | 기존 M2 테스트10개 로그 |
| 지표의 방향·mask·gap·P95·정지 | PASS | 지표 테스트2개 로그 |
| 그래프 및 실제 motion 시각화 | PASS | 저장된 예측만 사용, 주요 PNG 시각 검수 |
| 새 재실행 notebook | PARTIAL | Python 셀 문법·경로·hash 정합 검사, 새 notebook 전체 재실행은 안 함 |
| 독립 test 일반화·예측2D 입력 성능 | NOT VERIFIED | GT2D historical holdout 진단 범위 |
| Reliability routing 통합 효과 | NOT VERIFIED | 동료 점수 미확보·이번 필수 범위 밖 |

잘못된 테스트 파일명으로 0개를 발견한 최초 조회는 성공으로 세지 않았다. 실제 `test_final_metrics.py`의2개 통과를 기록한다. 이전 보고서의14개 테스트를 이번 복구본 테스트 개수로 주장하지 않는다.

## G. 연동 및 발표까지 남은 최소 작업

동료로부터 sequence_id/frame_id, joint_names(기존22개 순서), coordinate_system=`stored_device_pts3d`, input_source, split, model_run_id, frame-level error_probability[0,1], threshold를 받는다. **값이 클수록 오류 가능성이 높음**으로 명시한다. 동료 점수가 confidence면 의미와 보정을 합의한 뒤 연결하며 임의로 확률이라고 간주하지 않는다. GT3D와 오류 정답 label은 평가에만 사용하고 routing 입력에 넣지 않는다. 현재 inference 함수에는 GT3D 인자가 없다.

- 11월13일 Baseline 재현 및 실제 평가, 11월15일 DDIM 비교 목표의 필수 실험은 이번 결과로 충족했다.
- 11월15일 전: 동료 점수의 프레임 key/관절/좌표 계약 확인. 새로운 학습보다 발표용 결과표와 한계 설명을 먼저 확정한다.
- 11월20일까지: 필요한 경우에만 하나의 통합 검증을 수행하고 결과를 동결한다. 지금 작은 step 차이만으로 routing 이득을 보장할 수 없다.
- 11월23일 발표: 기존 결과 재현 → step수에 따른 정확도/지연 trade-off → GT대비 움직임 축소 → 독립 데이터·예측2D 검증의 필요성 순서로 설명한다.

연결이 끊겨도 재추론은 필요 없다. 실행된 원본 notebook, T4 결과 ZIP과 checksum, 입력 bundle, 재실행 notebook을 개인 다운로드 패키지에 함께 보존한다. 원본 Drive를 수정하거나 업로드 완료를 기다릴 필요가 없다. Git용 CSV 사본은 LF 줄바꿈으로 정리했으며 원본 결과 ZIP과 수치 내용은 그대로 보존했다. 패키지 코드는 PyTorch가 설치된 기존 가상환경에서 import 및 입력 준비 검사를 통과했다. 기본 Python에는 PyTorch가 없어 첫 import 검사가 실패했으나 추론 코드를 변경하지 않고 검증 환경을 바로잡았다.
