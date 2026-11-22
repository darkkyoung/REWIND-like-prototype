# GPU 실행 인계

[Work_GPU.ipynb](Work_GPU.ipynb)를 T4 또는 A100 Colab에서 위에서부터 실행한다. 실행 revision은 06bb1de0826fab98739f16cb390331b9878f944a다. 학교 계정 darkk0729@ajou.ac.kr로 원본 읽기와 My Drive 쓰기가 가능해야 한다. 노트북은 인증한 Drive 사용자를 검사하고 Drive 쓰기 직전에 다시 확인한다. 기존 모델 폴더를 수정하지 않는다.

1. 모델 ZIP 전체 SHA를 비교한 후 package/source의 원형 strict loader를 사용한다. Frontend e009(19363625849d7fc72232cf3af7e438e0c3c594eb52f17c7c15d5a44296153f38), Body e010(94a072551fd3da40621da5a589fbc3d6f919d3e7b52d29ea1c7f6099ef681305), Hand e009(d6c610ead412cc0495d4adf99ed3872e8c434ebfabecb2e9f551f5c1eb766f8f)의 byte SHA·source/metadata binding 재확인과 실제 가중치 CPU parity를 수행한다. [모델](https://drive.google.com/file/d/1eXSpkVdoLFybz6QBvX-uyrOFfZURfxdt/view)
2. 공식 감사 ZIP과 selection으로 400개 anchor의 실제 좌·우 RGB 및 pose 1,200파일을 내려받고 SHA를 확인한다. 기존 T4에서 사용한 학교 계정 폴더 ID로 파일을 찾는다. 누락·중복 이름·SHA 불일치·split 변경은 중단한다. 모델·원본·새 checkpoint 두 개·보관 ZIP을 위한 로컬 디스크와 학교 Drive 공간이 필요하다. [공식 감사 ZIP](https://drive.google.com/file/d/1bvmfWhFVMGOaFQDidxp_0KQ12seBfFEs/view) · [RGB](https://drive.google.com/drive/folders/1GdhUa-s7Ov_fFGGOSrBona6X5rQtiQvb) · [pose](https://drive.google.com/drive/folders/1G8X753DlLHBI1vBYLYifZcUXPKuKqWkL)
3. 최종 셀이 preflight·calibration·24+24 학습·평가를 수행한다. OOM이면 중단하고 같은 크기와 FP32 설정으로 A100에서 새 run을 시작한다. 실패 후 임의로 길이나 정밀도를 줄이지 않는다. 학습·평가 단계가 실패해도 해당 시점까지의 로그·산출물로 새 결과 ZIP을 생성해 학교 계정 My Drive 루트에 보관한다. 업로드 크기와 SHA256을 대조하고 출력 링크를 회수한다.
4. preflight.json(메모리·gradient와 λ), branch_states.json, control/improved.json(부모·입력·학습 trace·SHA), 새 Body checkpoint 두 개, frames.csv, evaluation.json, checksums.json, pilot.log를 확인한다. 추론 adapter는 원본 strict loader가 만든 동일 V1Body에 trainer.restore로 해당 arm의 SHA·binding을 확인 후 Body state만 strict-load한다. 학습 중간 자동 재개 기능은 제공하지 않는다. 저장된 경계에서 optimizer/scheduler를 포함해 복구하는 함수는 CPU에서 다음 update 일치를 확인했다.

원본을 이미 로컬에 검증해 둔 경우 추출된 package/source에서 같은 runner를 직접 실행할 수 있다.

~~~bash
export PYTHONPATH="$PWD:$FORK_ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m experiments.v12_rr.run --package .. --audit "$AUDIT_ZIP" --rgb "$RGB_ROOT" --pose "$POSE_ROOT" --out new_rr_run
~~~
