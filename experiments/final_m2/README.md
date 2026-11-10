# M2 실제148 Pose DDIM5/10 — 2026-11-10

기존 300-update 가중치와 M2 코어를 재사용한 실제 데이터 추론 비교다. 새 학습0회. GT stereo2D / raw device_pts3d / zero-camera / Body22 / window8 조건이다. Train18은 train-fit, Opening130은 historical holdout이며 공식 독립 test가 아니다.

## 복구·검증 근거

Fresh remote: upstream main `5b390a8459a674b378ada4305a647e6fd3e53178`, fork main `4571f30f6bac17d21e775a3a16488c98d44d117d`, 기존 실험 `74851d8c05933c135bf784c15f29dbf684987687`. 이전 로컬 `728449c`와 delivery ZIP/bundle은 작업 공간 정리 후 복구하지 못했고 원격에도 없다. 이전 보고서와 이 대화의 지표 정의를 복구했다. 기존 모델·DDIM·streaming 구현은 재개발하지 않았다. 원본 main과 과거 실험의 해당4개 core 파일은 byte 동일하다.

보존된 원본 ZIP 두 개를 재사용했다. Drive 원본 추가 다운로드0. `results_dec550973d7941039e16da612a49f51c.zip` SHA `f4c9c2d9ed06a76385ec949b94c447a8f0751602ce82a36bb0f2fa5611debb1f`, 300 ZIP SHA `217fe755c4cd347d4e2ce11045f479fde494d1474ce87c3848019277e21b5a3f`.

기존 baseline `5e52aa3`의 별도 worktree에서 기존 `tools.pose_reuse_300.verify_bundle` 및 `prepare_reuse`를 그대로 실행했다. 추출기 코드 hash 검사 우회 없이 ZIP SHA/CRC/파일 SHA/추출 manifest를 검사하고 동일 M2 manifest를 재생성했다. 두 ZIP의 Pose148개 SHA 일치, 원래 target/mask/frame 배열 일치. checkpoint SHA는 `6acd1fd96a2dbfe1e681710a6286c3d9d0177ce69402a0b488f8fcce7a4a5c9d`, step300. strict state dict·buffer·alpha schedule·core source hash 모두 검사한다.

실제 JSON의 body mapping은87 raw→22. 기존2D /1024 및 3D 출력×100을 보존한다. RGB 이미지는 이번 추론에 사용하지 않는다. 최신62관절/mm/camera 계약을 섞지 않는다. 기존 train_ready=false는 변경하지 않았다.

## 실행

검증한 입력 준비본 `M2_verified_148_input.zip`은 개인 산출물로 별도 제공하며 Git에는 올리지 않는다. 파일 전체 SHA와 내부 파일 SHA를 검사한 후만 읽는다. 모델·원본 Pose JSON을 Git에 중복 추가하지 않는다.

```bash
python tools/final_m2_real.py --input-bundle /path/to/M2_verified_148_input.zip --output outputs/new_run_unique --device cpu
python tools/final_m2_real.py --input-bundle /path/to/M2_verified_148_input.zip --output outputs/new_gpu_run_unique --device cuda
```

출력 경로가 이미 있으면 중단한다. CUDA 미탑재 시 자동 CPU 대체 없음. 입력 ZIP8MiB/해제16MiB,148프레임 고정, 추론300초(프레임 사이 검사), 자동 재시도/학습 없음. 실패 보고서와 기존 결과를 보존한다.

## 통제·지표

checkpoint/data/mask/seed10043/FP32/B1/window8 고정. 동일 sequence/frame keyed Gaussian noise를 두 arm에 사용. 각 split/arm warm-up8, 측정3회, arm 순서 교차, CUDA synchronize 전후 및 perf_counter_ns. 중복 측정 예측은 bitwise 일치해야 한다. 시간은 keyed noise 생성·H2D·history packing·DDIM·raw scale·sync 포함, 파일 I/O·RGB frontend·D2H·GT 지표 제외. 전체 RGB pipeline 시간이나 모델 kernel 단독 시간이 아니다. P95는 개별 프레임 표본의 linear percentile이다.

Joint error는 유효 관절 L2 micro 평균. Displacement error는 동일 sequence·연속 frame·양 끝 유효 관절에서 예측/GT 이동 **벡터** 차이의 L2 평균. Motion Ratio는 동일 유효쌍의 평균 예측 이동 크기 / 평균 GT 이동 크기. 물리단위·FPS 미검증이므로 raw/frame displacement로 표시한다. GT 움직임0이면 ratio=null.

CPU 결과는 `metrics/cpu/`에 저장했다. 실측이지만 T4 latency와 혼합하지 않는다. CPU 원래 batch8 evaluate의 Opening12.232002 raw, ratio40.31476%, 저장T4 최대좌표차0.0000534 raw. 현재 B1 DDIM10 재현도 사전 기준(JPE 차0.0001/raw좌표차0.001) 안이다. 기준은 GPU 비교 전 고정했다. 원래 환경과 다른 PyTorch/CPU의 작은 수치차를 기록한다.

## 미완료 범위

독립 평가 sequence, 물리단위/FPS, 예측2D 입력, Reliability routing은 별도다. 점수 연동 계약: sequence_id/frame_id/joint_names/coordinate_system/input_source/split/model_run_id + frame-level error_probability[0,1](클수록 오류) + 사전 고정 threshold. 추론 함수는 GT3D/오류 정답 라벨을 입력받지 않는다. Threshold/GRU 구현 및 효과를 주장하지 않는다.
