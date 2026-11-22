# Work 인계 · 같은 T4에서의 짝비교

Work_Paired_T4.ipynb는 코드 SHA를 고정한다. Tesla T4에서 학교 계정 darkk0729@ajou.ac.kr로 인증한 뒤 이전 Work의 복구된 A portable bundle 경로와 후보 파일 경로를 지정한다. 이전 실행에서 MyDrive 최상위 기본 경로가 실패했으므로 실제 파일 또는 바로가기 경로를 입력한다. 코드가 사용자 입력을 검증한 뒤 읽기 전용으로 사용한다.

|입력|Drive 파일 ID|
|---|---|
|동료 A 55 GB portable bundle|1WiVnkAr6Q9i7eiGLzTDL1qarxCULSRXo|
|후보 결과 ZIP (results/anchored.pt)|1TgEUnqwBVjx4G4jY2Lwubpk5MCcW3iQb|
|이전 실패 증거 ZIP|1HV6hkNBHoE66FQNflj7lP05GLe7QVjOQ|
|A source/metadata|1d3sh9rkRAeXKW7Sk0T25ahIlUSiymfC-|
|Index metadata|1f4Qf23SrKQexpi6W3n_TXmiViY3atdSy|

A 원본 ZIP SHA는 4591393d3aceb4bb4af9b387c4fda038bdcdd5fd5d613fdcc95f7cb50cdc736a, 후보 ZIP은 51704fc55a7447ec3da17a36b5b94f7d584306fb1c9131b14fbfdabf4712b3a3이다. 회수한 디렉터리도 공식 로더의 영수증 검증을 통과해야 한다. 여섯 RGB/pose/cache 샤드(4, 5, 185, 186, 782, 783), source/revision/boundary/feature marker, 공식 reference 및 모델 복구 ZIP이 필요하다. 세부 경로는 WORK.md에 있다.

패키지 준비 후 작업 디렉터리에서 실행한다. 기존 strict 명령은 이전과 같은 배포 기준을 적용한다. exploratory는 한 번의 실행에서 strict 판정도 기록하므로 직전에 strict를 중복 실행할 필요가 없다.

```bash
export PYTHONPATH="$JOB/runtime/source:$JOB/runtime/tools:$REPO"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$JOB"
python -m experiments.v2a_dualcamera.run --mode exploratory --bundle "$A_BUNDLE" --anchored "$JOB/runtime/anchored.pt" --work "$JOB/paired-restore" --output "$JOB/paired-result"
```

배포 재현만 엄격히 검사하는 실행에는 --mode strict와 각각 새로운 work/output 디렉터리를 지정한다. 이전 T4와 같은 차이가 관찰되면 strict는 예외로 종료한다. ZIP에 logs, verdict, 생성된 strict_parity.json과 실패 trace를 보관한다. exploratory가 완료되면 같은 ZIP에 exploratory_comparison.json, results.json, frames.csv, 세 NPZ가 들어간다. 마지막 노트북 셀에서 학교 계정을 재확인하고 새 ZIP으로 업로드한 뒤 Drive에서 재다운로드하여 SHA를 검증한다.
