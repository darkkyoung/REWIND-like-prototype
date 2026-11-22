# v1.2 Body teacher의 pelvis-relative x0 학습 pilot

**코드·CPU 검증 완료, GPU 학습 효과 NOT VERIFIED.** 기존 통합의 Standing 0–49 결과 ZIP(학교 Drive, SHA256 cda02d747c222cdc5484a29d7a11b2c721ee8b5c1d4c51fe8f58efd435a4ee06)을 직접 대조했다. 원형 Body global 243.8736 mm, RR 253.3411 mm에서 기존 Phase 2 후처리는 각각 244.1069 mm, 242.8057 mm였다. 여기서는 동결 Hand에 이르는 원형 sampler를 유지하고 Body 학습 목적만 바꾼다. [기존 실제 T4 결과](https://drive.google.com/file/d/1pKxjgh9Uy6MYGOCdq9kCGBTiTHmeMyTj/view)

## 학습 목적과 분모

motion_losses의 기존 최종 position MSE와 중간 frame-head MSE를 그대로 호출한다. 추가 항은 최종 x0 관절의 예측 상대 벡터와 GT 상대 벡터(mm/100) 차이에 대해 xyz 제곱합을 계산한 후 유효한 비골반 joint-frame 쌍 수로 나눈다. 골반은 Body22 index 0이다. 두 joint valid와 time_valid의 교집합을 사용한다. 무효 GT의 NaN은 제외하고 유효 GT·예측의 비유한 값은 오류다. 유효 RR 쌍이 없으면 미분 가능한 0을 반환한다. 기존 position 목적은 모든 GT가 무효인 배치를 원래대로 거부한다. 프레임끼리 차분하지 않으며 연속 full clip마다 새 history로 시작한다. gap은 오류다.

기본 loss는 기존 목적 + λ × RR. λ=0은 원래 loss tensor를 그대로 반환한다. 중간 head의 기존 gradient는 유지되며 추가 항의 직접 대상은 최종 head뿐이다. Train 전용 네 draw에서 전체 학습 가능 Body 파라미터의 gradient L2 norm 비 중앙값 × 0.1을 고정한다. 비율과 λ는 실제 GPU 실행 때 기록한다. 0.1은 탐색적 설계 선택이다. [구현](trainer.py) · [원본 학습 경로](https://github.com/BattleMarine/REWIND-like-prototype/blob/12ba5e4b2f9f56c4cc1164430674fd969ac094a6/rewind_v1/training.py)

## 데이터 및 통제

공식 reports.zip SHA256 30c46b8d3259d9a48eae7f4b94465a5563e3ad6bf4fb74b6b88ef546da8e4c2f 및 원형 full_contract의 manifest/coverage seal을 검증했다. Train 20,888개에서 완전한 clip 중 서로 다른 인물·동작을 순서대로 골라 6명·6시퀀스·6동작 × 50 = 300프레임을 확정했다. 기존 calibration의 Joyful 한 clip과 겹친다. Standing 및 Petting Animal 각 0–49가 공식 validation이다. Petting Animal은 이번 RR 실험에 추가한 motion이며 과거 모든 연구에서 미사용인지 확인되지 않았다. 평가 두 동작은 같은 인물이고 학습 인물과도 겹친다. manifest SHA 기준 RGB-left·right·pose의 train–validation 정확한 파일 중복은 각각 0건. retarget/반전, 근사 pose, 원본 mocap 독립성은 미검증이다. 선택 원본 1,200파일 합은 940,817,816 B. CPU 감사에서는 manifest를 확인했고 원본 RGB 전체 바이트 검증은 Work 실행 시 수행한다. [고정 범위](selection.json) · [검산](input_audit.json)

동일 e010에서 Adam/StepLR을 새로 만든다. FP32, hidden512, batch 1×50, LR 1e−5, 4회 순회하여 각 24 updates다. 매 순회 clip 순서, update별 DDPM timestep 및 Gaussian noise는 명시적 CPU Generator로 재생성한다. 초기 model/optimizer/scheduler/CPU·CUDA RNG 및 update별 feature·camera·target·mask·noise trace를 비교한다. 원래 resume binding은 strict loader가 검증한다. 새 형식에는 arm·원본 SHA·코드 SHA·selection·λ·step·로그·Body·optimizer·scheduler를 저장한다. restore는 파일 SHA·형식·binding 확인 후 strict load한다. 다음 draw는 step에서 결정된다. 동결 Frontend e009/Hand e009의 state hash를 양끝에서 대조한다. 기존 Hand가 새 Body에 맞춰 다시 학습되지 않으므로 local/global 손 오차를 나눠 본다.

A/B/C는 동일 predicted Stereo2D·absolute camera·frame 및 원형 seed 71+100/200으로 DDIM10 추론한다. GT 없이 sample_features를 호출한 뒤 지표를 계산한다. Body global/RR/pelvis/wrist/FDE/이동량·골격·퍼짐, Hand local/global, Whole62, 프레임별 RR·global 변화량, GPU 메모리와 frontend 제외 Body+Hand sampling 시간을 산출한다. 사전 판정은 두 clip 각각 Improved 대 Control에서 RR ≥1% 개선과 다섯 오차 악화 ≤2%를 동시에 요구한다. [판정·예산](CRITERIA.md) · [GPU runner](run.py)

## CPU 실측과 한계

신규 검사 9개, 통합 회귀 9개가 통과했다. 식과 분모, 이동 불변성, root/time/NaN mask, all-invalid RR, λ=0 두 head의 값·gradient bitwise 일치, 실제 V1Body gradient를 통한 분기, 동일 draw, checkpoint SHA·optimizer 복구 후 다음 update 일치, gap과 validation 거부, prefix 및 frozen 모델 검사를 포함한다. 추가로 원래 **15,010,822개 파라미터의 hidden512 V1Body**에서 임의 초기 가중치·합성 50프레임으로 CPU forward/backward/Adam 1회를 실행해 유한 loss와 파라미터 변경을 확인했다. 이때의 0.1은 smoke용이고 실험 λ가 아니다. [18개 검사](cpu.log) · [원래 크기 CPU smoke](full_smoke.log)

이 컴퓨터의 Torch는 CUDA를 사용할 수 없다. 이번 실행에서 배포 checkpoint를 다시 로드하거나 실제 RGB로 학습하지 않았다. 모델 ZIP SHA fbf1adda7fb63007d799545278e513be7f0a32ae09a9453bad70f675312274a0은 기존 T4 실행의 source.json에서도 확인했다. 새 GPU 실행에서 ZIP 전체 SHA, packaged source 및 F9/B10/H9 binding을 다시 검증한다. T4 16 GB 추가 학습 가능성은 미확인이다. A100 40 GB에서 원본 전체 학습이 이루어진 이력은 새 pilot의 성공을 보장하지 않는다.
