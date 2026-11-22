# Work 인계 · v1.2 원형 확인 후 Body/Hand 통제 비교

[실행 노트북](Work_T4.ipynb)은 코드 SHA 447385016e7973ace9871d53781c6f9153f83a57을 사용한다. 새 Colab 런타임에서 위에서 아래로 실행한다. 사용자 포크의 experiments만 추가 경로로 읽고 v1.2 모듈과 strict loader는 ZIP의 source에서 import한다.

|필수 입력|식별과 사용 시점|
|---|---|
|v1.2 공유 ZIP|127,886,815 B. 노트북에서 지정 ID로 읽고 Drive provider SHA256과 로컬 전체 바이트·CRC를 검증한다. package source revision과 outer SHA는 결과 source.json에 기록한다. [모델 ZIP](https://drive.google.com/file/d/1eXSpkVdoLFybz6QBvX-uyrOFfZURfxdt/view)|
|원본 내부 모델|F9 19363625849d7fc72232cf3af7e438e0c3c594eb52f17c7c15d5a44296153f38, B10 94a072551fd3da40621da5a589fbc3d6f919d3e7b52d29ea1c7f6099ef681305, H9 d6c610ead412cc0495d4adf99ed3872e8c434ebfabecb2e9f551f5c1eb766f8f. 원형 load_models가 archive manifest, 원본 receipt, dependency·spec·코드 및 strict state dict를 확인한다.|
|공유 M2 ZIP|42,297,177 B. 이전 Control800의 기록 SHA a70d9efbce4294323e2eca735155a20c12068a9d1e4215ae3dc12f1f862a2185. 통합 실행에 이 모델 ZIP은 필요하지 않다. [M2 ZIP](https://drive.google.com/file/d/1-DVyi4uK6j9r5D6URgy5YCdhUKR7PyoT/view) · [공유 폴더](https://drive.google.com/drive/folders/1Fng6gzWSK9jfFQvDtLFF_vnU-jltgPpV)|
|AUDIT_ZIP|동료 completed_003/reports.zip 10,114,342 B, SHA 30c46b8d3259d9a48eae7f4b94465a5563e3ad6bf4fb74b6b88ef546da8e4c2f. [동료 보존 기록](https://github.com/BattleMarine/REWIND-like-prototype/blob/12ba5e4b2f9f56c4cc1164430674fd969ac094a6/docs/V12_AUDIT_RESUME.md)|
|RGB_ROOT · POSE_ROOT|위 ZIP의 verified/manifest.json에서 선택된 각 파일의 상대 path가 시작되는 추출 루트. 좌·우 native 1024² RGB와 실제 world→camera 필드가 있는 pose JSON. 각 파일 크기/SHA 및 split을 검사한다. 두 공유 모델 ZIP에는 이 원본들이 없다. 접근 가능한 세 데이터 경로는 미확인이다.|

1. 기본 RUN_T4=False로 모델 다운로드·검증, CPU 테스트, 실제 모델 CPU 2프레임 smoke와 원형 parity를 수행한다. 출력에서 각 SHA, 네 tensor shape와 baseline_max_abs_mm=0을 확인한다.
2. 데이터가 준비되면 새 실행에서 AUDIT_ZIP, RGB_ROOT, POSE_ROOT를 마운트 경로로 지정하고 RUN_T4=True로 실행한다. train 자료가 없지만 지정 validation 원본이 있으면 BASELINE_ONLY=True로 실행한다. 불일치한 SHA 또는 다른 clip으로 대체하지 않는다.
3. FP32 DDIM10, train 3×50프레임 calibration과 validation 1×50프레임 평가다. 원본 v1.2 추론과 동일 입력으로 baseline을 별도 대조한 뒤, 보정은 출발 Body를 공유한다. B/C/D는 refined upper 및 wrist, 보조 세 조건은 원래 upper 및 wrist를 쓴다. seed 171/271의 텐서 SHA를 기록한다.
4. 회수 위치는 MyDrive/rewind_v12_m2/실행시각 아래다. CPU와 T4 결과 ZIP은 분리하고 복사 후 SHA를 대조한다. 평가 ZIP 내부에 metrics.csv, evaluation.json, source_frames.json, calibration.json(보정 시), predictions.npz, whole62.gif, SHA256.json과 로그가 있다. 실패 시 부분 출력과 로그를 보관하고 실패 원인을 먼저 확인한다. GIF는 장치축의 고정 투영이며 재생시간은 측정 FPS와 무관하다.

로컬에서 동일하게 호출할 때 아래 명령은 압축 해제한 package/source에서 시작한다. PYTHONPATH는 source가 사용자 포크보다 앞에 있어야 한다.

```bash
export PYTHONPATH="$PWD:$FORK_ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m unittest -v experiments.v12_m2.test_integration
python -m experiments.v12_m2.preflight ..
python -m experiments.v12_m2.evaluate --package .. --audit "$AUDIT_ZIP" --rgb "$RGB_ROOT" --pose "$POSE_ROOT" --out new_results --device cuda
```

연구 상태는 통합 API의 CPU 합성 검증까지다. 실제 원본 가중치 parity, 네 정책 정확도, 추가 지연은 Work 실행 결과로 판정한다. Archived M2 15조건 및 Taunt16은 기존 평가 기록이고 새 whole-body 결과를 대체하지 않는다.
