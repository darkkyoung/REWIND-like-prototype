# Work 인계 · Tesla T4

Work_T4.ipynb를 Tesla T4에서 실행한다. 첫 셀은 고정된 코드 커밋을 체크아웃한다. 학교 계정 darkk0729@ajou.ac.kr로 Drive를 연결하고 두 입력 경로를 지정한다. 기존에 마운트한 ZIP이나 복구한 디렉터리를 우선한다. 공식 로더가 원본 영수증, 모델 의존성, 캐시 boundary chain 및 선택 샤드를 검증한다. 재시도는 새 임시 작업 폴더를 사용한다.

|입력|Drive ID|검증|
|---|---|---|
|v2 A ZIP 또는 기존 portable bundle 디렉터리|1WiVnkAr6Q9i7eiGLzTDL1qarxCULSRXo|55,068,837,571 bytes. Drive SHA 4591393d3aceb4bb4af9b387c4fda038bdcdd5fd5d613fdcc95f7cb50cdc736a|
|RootAnchored 결과 ZIP 또는 회수한 results/anchored.pt|1TgEUnqwBVjx4G4jY2Lwubpk5MCcW3iQb|ZIP SHA 51704fc55a7447ec3da17a36b5b94f7d584306fb1c9131b14fbfdabf4712b3a3. 멤버 SHA는 pins.json|
|원본 감사 ZIP|1T_DfQuwbwMC3gXJjduiBSmbdKfvecHN5|48249c395ca40384da6ec547fd7dad5aaea98ff00d2dab8843bc97418b020fae|
|Work 감사 보고|165QCZdF2N_w5yPVpM3EfOWQom1oUKZF8|동료 A source 및 이전 strict load 기록|
|A source/metadata|1d3sh9rkRAeXKW7Sk0T25ahIlUSiymfC-|0500a4f71b32147735c32fbc40eefcfdba2a267bd07f7c52e4ffbde7367f2e4c|
|Index metadata|1f4Qf23SrKQexpi6W3n_TXmiViY3atdSy|88b4e69fe13e59127da25cc7d407646fac7dd5d53496ed0d34e26997b387f74c|
|Gunplay frame 0 pose JSON|1FkXyHX1rqVvZ_uGfI9jN1i57L4QzMKUa|87ddf4fe2aa62abd7f998c1c9c6b2b5c0603bc5ba01e2548f38da27be23741ec|

기존 ZIP을 제자리에서 연다. 전체를 다운로드하거나 전체 캐시를 풀 필요는 없다. 디렉터리 방식이면 원래 상대 경로의 BUNDLE_INDEX.json, tools, source, 공식 catalog, 1,063개 boundary JSON, 147개 feature group marker, 최종 모델 복구 ZIP·진단·영수증, 공식 comparison ZIP·영수증, **샤드 4, 5, 185, 186, 782, 783**이 필요하다. 단독 best .pt 세 개만으로 공식 baseline 재현을 인증할 수 없다. source/metadata ZIP과 index metadata ZIP만으로는 여섯 샤드의 RGB가 완비되지 않는다. 데이터와 모델에는 읽기만 수행한다.

### 실행 순서

Python 3.11+, PyTorch, NumPy, Pillow를 사용한다. 로컬 소스와 패키지 소스의 모듈 이름이 겹치므로 실행 디렉터리는 두 저장소의 외부로 둔다. 아래는 노트북과 같은 진입점이다.

```bash
python "$REPO/experiments/v2a_dualcamera/prepare.py" --bundle "$A_BUNDLE" --anchored "$ROOT_INPUT" --dest "$JOB/runtime"
cd "$JOB"
export PYTHONPATH="$JOB/runtime/source:$JOB/runtime/tools:$REPO"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m unittest -v experiments.v2a_dualcamera.test_adapter
python -m experiments.v2a_dualcamera.audit_split
python -m experiments.v2a_dualcamera.run --bundle "$A_BUNDLE" --anchored "$JOB/runtime/anchored.pt" --work "$JOB/restore" --output "$JOB/result"
```

두 경로는 FP32, DDIM10, seed 71에 Body +100, Hand +200을 쓰고 프레임·mask를 공유한다. 고정된 세 클립 모두에서 같은 환경의 원형 A와 정확히 일치하고 배포된 reference와 공식 허용오차(2D 0.005 px, 3D 0.5 mm)를 만족해야 후보 평가가 시작된다. 실패한 경우 parity.json 및 콘솔 trace를 보존하고 중단한다. GT는 공식 reference 확인과 metric 계산에 사용한다.

results.json에는 클립별 두 정책 지표, camera/feature/noise SHA, 동기화한 Body+Hand 처리 시간과 GPU peak allocated bytes가 있다. 공유 특징 생성 시간에는 캐시 읽기 및 전처리가 포함된다. 단일 실행의 프로세스 내 peak와 클립 벽시계이며 스트리밍 FPS 측정은 별도다. frames.csv의 양수 delta는 후보 악화, 음수는 개선이다. 클립별 NPZ에 양쪽 Body22, local/global Hand40, Whole62, GT 및 validity를 저장한다. 노트북 마지막 셀은 결과 ZIP을 학교 계정 My Drive에 새 파일로 저장하고 SHA를 다시 읽는다. 모델·원본 RGB는 결과 ZIP에 포함하지 않는다.

### 직접 교체가 실패할 때의 별도 실험안

공통 v2 A Body e003와 first_midpoint에서 Control과 Improved를 각각 새 Adam으로 시작한다. Control은 원래 position 및 frame head 목적, Improved는 최종 x0에 RR와 pelvis 항을 더한다. Train 전용 4 draw로 계수를 사전 고정하고 동일 6×50 train, LR 1e-5, 24 update, timestep/noise/order를 사용한다. GPU preflight 및 별도 승인 후 실행할 계획이다.
