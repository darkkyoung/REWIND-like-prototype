# 배포 재현과 동결 Body 교체 실험 · 2026-11-22

**기존 Strict Release는 FAIL, 이전 T4의 공식 A ↔ Adapter Baseline은 150/150프레임 정확 일치, 후보 성능은 NOT VERIFIED다.**

|공식 Validation 0–49|배포 대비 2D 최대 절대차(px)|배포 대비 3D 최대 절대차(mm)|사전 기준 결과|
|---|---:|---:|---|
|Standing|0.099709|0.831787|2D·3D 초과|
|Gunplay|0.132862|0.808971|2D·3D 초과|
|Reloading|0.057693|0.340469|2D 초과|

### 기존 중단 원인

실패 ZIP의 SHA256은 f849916e5b9c052fa9cf60451d4471183aafe6709e4848c320cbf4e05891dd52이다. audit_failure.py로 ZIP 내부 영수증 23개 항목과 선택 입력 SHA 1,239개를 패키지 인덱스에 대조했다. pins.json의 내용도 같았다. 실행 코드는 cbf98aa93970b55518bd7afc6a999e3d1c37f609였다. 원래 strict 실행은 Standing에서 중단했다. 이후 Work가 같은 검사식을 세 클립에 적용한 reference_only 진단에서 표의 나머지 값과 loader 예외를 기록했다. prior_t4.json에 두 실행의 해석에 필요한 원본 수치와 binding 차이를 남겼다.

공식 비교 함수가 GT2D·GT3D, mask, joint 이름, 시간 배열의 dtype과 값을 확인했다. 기준 클립·모델 SHA·seed 71 및 Body/Hand offset 100/200을 확인했다. 동일 T4에서는 정규화 Frontend xy, Hand local, Whole62가 원소별로 일치했다. Whole62의 앞 22개와 뒤 40개가 각각 Body와 global Hand이므로 이 두 출력도 정확히 같다. A100 기록과 비교하면 이미 Frontend xy에서 차이가 발생한다. 최종 3D 차이에는 Frontend 차이의 전파와 Teacher 자체 수치 차이가 함께 들어갈 수 있다.

배포 환경은 NVIDIA A100-SXM4-40GB, 기존 Work는 Tesla T4다. 두 기록의 PyTorch는 2.14.1+cu130, CUDA runtime은 13.0이다. Work는 Python 3.13.16, NVIDIA driver 580.82.07을 기록했다. 동일 SHA와 평가 배열은 다른 클립·모델을 사용했을 가능성을 낮춘다. 과거 A100 실행에서의 실제 cuDNN 알고리즘, TF32, cuBLAS workspace, 이미지 전처리 직후 텐서 및 전체 층별 출력은 확보되지 않았다. GPU 기종별 단독 효과는 미확정이다. 새 실행은 runtime 플래그, 좌·우 xy, Body22·global Hand40의 배포 대비 최대 차이를 남긴다. 원인을 더 좁히려면 동일 환경 설정과 입력으로 A100을 재실행해 전처리 영상과 Frontend feature를 단계별로 대조할 수 있다.

별도 loader 실패의 유일한 binding 차이는 /weight의 list와 tuple이었다. 값은 순서대로 0.08310449710117396, 0.03768277692597375이다. 양쪽의 두 유한한 수만 순서 있는 float tuple로 변환한다. 나머지 binding, step 24, anchored.pt 전체 SHA, Body 104 state 항목의 키·shape·dtype·유한성 및 strict=True 검사를 유지한다. 현재 CPU 세션에서는 동결 소스의 전체 크기 V1Body fixture를 실제 파일로 직렬화·strict load하고 104개 state와 파일 SHA를 다시 대조했다. 새 코드로 실제 학습된 후보를 strict load하고 forward하는 검사는 Work에 남아 있다.

### 실행 모드

|조건|strict (CLI 기본값)|exploratory (명시적으로 선택)|
|---|---|---|
|고정 입력·모델·clip·noise·카메라 계약 오류, 같은 T4 exact 불일치|중단|중단|
|배포 기준만 실패|FAIL로 중단|strict FAIL을 기록하고 후보 로드 단계로 진행|
|후보 SHA·binding·strict load 오류|중단|중단|
|모든 필수 검사 충족|기존 비교 수행|동일 T4의 탐색적 짝비교 수행|

배포 기준은 공식 함수 그대로 2D 최대차 ≤0.005 px, 3D 최대차 ≤0.5 mm다. 세 클립의 parity 수집 후 strict_parity.json에 기록한다. 모델·입력 계약이 그 전에 실패하면 verdict.json에 중단 단계와 예외가 기록된다. exploratory_comparison.json은 전체 평가 및 체크포인트 readback 성공 후 생성한다. verdict.json의 completed는 실행 완료를 뜻한다. historical_release는 이전 FAIL을 보존한다.

후보는 frozen A Frontend e003·Hand e002와 기존 v1.2 RootAnchored24의 조합이다. Body는 각각 first_midpoint와 absolute 입력에서 device mm를 반환한다. Hand는 두 경로 모두 first_midpoint를 받고 선택된 Body upper로 조건화되며 local finger를 해당 손목에 한 번 부착한다. 평가 GT는 공식 reference 확인 및 metric 계산에서만 사용한다. 결과 JSON에 Body global/RR/pelvis/wrist, Hand local/global, Whole62, Bone, FDE, Motion Ratio, 동기화된 클립 지연·peak memory를, CSV에 프레임별 Body 악화/개선 delta를 남긴다.

세 동작 모두 같은 인물이며 Standing은 반복 관찰했고 Gunplay와 Reloading은 A 학습에 같은 동작명이 있다. 후보는 v1.2의 전체 품질 사전 기준에서 FAIL이었다. A e003과 후보 학습 당시 e009 Frontend 사이의 특징 분포 이동 가능성도 있다. 해석 범위는 이 세 클립에서의 동결 Body 교체 효과다.
