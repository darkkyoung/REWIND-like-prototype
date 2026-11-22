# Work 실행 인계

새 T4 Colab에서 [Work_GPU.ipynb](Work_GPU.ipynb)를 학교 계정 darkk0729@ajou.ac.kr로 실행한다. 실제 학습 코드 pin은 ed1c1e1ebda25d7f1e5cb495c5e34f06829ac2d2다. 이전 T4와 재현성을 비교하기 위해 같은 torch/CUDA 환경을 우선 사용한다. 코드가 실제 GPU 종류·torch 버전을 기록한다. 다른 환경에서도 A/B 재현 기준은 같다.

|입력|확인 계약|
|---|---|
|[모델 ZIP](https://drive.google.com/file/d/1eXSpkVdoLFybz6QBvX-uyrOFfZURfxdt/view)|fbf1adda7fb63007d799545278e513be7f0a32ae09a9453bad70f675312274a0. package/source에서 e009/e010/e009 원형 strict loader 실행. Body 94a072551fd3da40621da5a589fbc3d6f919d3e7b52d29ea1c7f6099ef681305.|
|[실제 RR24 ZIP](https://drive.google.com/file/d/16zTnwKQ6Ybo6KSCrotuCdd2BpUPFNxu2/view)|c1985fe1624489a58df446c77fa42a469a14a3eb92bf3c19b9d12cb67a848eed. 이전 Control 243f7d7b1649b0f6c6b37c90458aaf0448a9ab1b3d650917bac74504470a77bc, RR 76f63306780af8f9ab44b11d46619aa2dc0484fe0cd2d88df87e4bf20c7cefa3. 바이트 검증 후 state/trace 대조에만 사용.|
|[공식 audit](https://drive.google.com/file/d/1bvmfWhFVMGOaFQDidxp_0KQ12seBfFEs/view)|30c46b8d3259d9a48eae7f4b94465a5563e3ad6bf4fb74b6b88ef546da8e4c2f. manifest SHA·plan·split·선택된 1,500파일의 SHA를 확인한다.|
|[RGB](https://drive.google.com/drive/folders/1GdhUa-s7Ov_fFGGOSrBona6X5rQtiQvb) · [pose](https://drive.google.com/drive/folders/1G8X753DlLHBI1vBYLYifZcUXPKuKqWkL)|Train 여섯 clip, 기존·신규 validation 각각 두 clip. 모두 0–49, 총 500 frames. read-only.|

1. 첫 셀은 계정 및 GPU 확인 후 모델을 내려받고 원형 strict-load synthetic CPU preflight와 신규·기존 CPU tests를 수행한다.
2. 둘째 셀은 audit와 과거 ZIP을 검증하고 실제 고정 파일을 임시 디스크에 준비한다. 모델·과거 ZIP·원본·네 checkpoint 및 결과 ZIP의 업로드·readback 복사 공간을 확보한다.
3. 마지막 셀은 원본 Body에서 실제 gradient 진단, 메모리 preflight, 96 updates, A/B 과거 state 비교, 평가를 수행한다. 변경한 폭·길이·정밀도·계수로 자동 재시도하지 않는다. 조기 실패 시 failure.txt·pilot.log·이미 생성된 결과로 별도 ZIP을 보관한다. 인증/네트워크 자체 실패 시 로컬 evidence 폴더를 확인한다.
4. 결과는 학교 계정 My Drive 루트의 새 ZIP으로 업로드한 뒤 전체 파일을 다시 내려받아 SHA256을 대조한다. 출력 URL과 SHA를 회수한다. gradient.json, branch_states.json, historical_parity.json, 각 arm의 checkpoint·로그, evaluation.json, metrics.csv, factorial.csv, frames.csv, clip별 poses/GT NPZ와 첫 프레임 공통축 PNG를 확인한다. latency는 frontend를 제외한 Body+Hand sampling 구간이다.

사전 기준은 [CRITERIA.md](CRITERIA.md)에 고정했다. A/B 불일치는 historical_parity.json의 입력 trace 차이와 tensor별 최대 차이를 먼저 확인하고 이전 device·torch 및 packaged source를 대조한다. 실제 결과가 회수되기 전에는 발표용 향상 수치가 없다.
