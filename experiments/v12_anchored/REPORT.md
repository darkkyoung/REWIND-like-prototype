# Root-Anchored Relative Loss · 2×2 통제 실험

**CPU 구현·회귀 검증 완료. 새 e010 gradient 진단과 네 군 GPU 성능은 NOT VERIFIED다.** 이전 실제 RR24 결과의 경량 JSON을 확인했다. Control24 대비 Standing은 RR −2.27%, pelvis +6.23%, Petting Animal은 RR +3.78%, pelvis +12.08%였다. 이 관찰만으로 gradient 충돌을 원인으로 확정할 수 없다. [이전 GPU 요약](https://drive.google.com/file/d/1twUWt9G4g6xg28Nm1FExHlQORdXU6Q4M/view) · [이전 네 draw 근거](previous.json)

## 목적과 원인 진단

골반을 뺀 상대 자세 목적은 모든 관절에 같은 평행이동을 가해도 같다. 절대 골반을 개선할 직접 제약이 없지만 상대 벡터를 통해 골반과 공유 파라미터에 gradient는 전달된다. 이번 실험에서는 기존 position과 중간 frame loss, 기존 최종 x0 RR 함수를 재사용한다. 새로운 항은 최종 x0 골반 좌표에서 GT mm/100을 뺀 잔차의 xyz 제곱합을 유효 골반 프레임 수로 나눈다. time·joint valid 둘 다 요구하고 유효값이 비유한 경우 오류, 유효 골반이 없을 때 미분 가능한 0이다. 이 항은 기존 위치 목적의 골반 가중치를 높인다.

원래 e010의 네 train-only draw에서 분석한 뒤 동일한 e010으로 매번 초기화한다. 기존 모델 결과나 평가 GT를 coefficient 계산에 넣지 않는다. P, R, H의 norm과 cosine은 gradient.json에, 네 군의 결합 norm은 draw별 combined_norm에 기록한다. 실제 e010에서 어느 쌍의 cosine이 음수인지는 Work 실행 후 확인할 사항이다. 음수여도 그 한 배치의 국소 gradient 관계이며 optimizer 누적 효과나 validation 인과 효과와 다르다. [진단 구현](trainer.py) · [사전 기준](CRITERIA.md)

## 재현과 신규 평가

Train은 이전과 동일한 300프레임이다. 기존 두 평가 외에 공식 teacher validation plan에서 Military_Signaling__6_(Manuel), Standing_2H_Magic_Attack_01(Maya), 각각 0–49를 고정했다. 두 사람은 선택한 여섯 train 인물 및 기존 평가 인물 Adanna와 다르다. 선택 train과 알려진 motion 이름 및 좌/우 RGB·pose 파일 SHA 중복을 거부한다. 원본 motion 계보·retarget 중복은 미확인이다. selection.json SHA256은 d5d2b3f6f90130c1b3d176b09426c4a6769dfed98f6fbda97b0f5cad348ad399이다. [선택과 파일 identity](selection.json)

A·B는 이전 update 함수를 그대로 실행한다. 네 군의 초기 Body SHA·optimizer·scheduler·CPU/CUDA RNG와 24개 실제 feature/camera/target/mask/timestep/noise trace를 비교한다. 과거 ZIP은 별도로 읽고 원본 파일 SHA 검증 후 재생성한 A/B의 Body 전체 state와 trace를 대조한다. 불일치 시 변경 tensor의 최대 차이, 첫 trace 차이 및 환경 로그를 보존하고 성능 해석 전에 중단한다. C/D도 e010에서 독립 분기한다. Frontend와 Hand state는 양끝에서 검증하며 원래 예측 upper 및 wrist 경로를 사용한다.

각 군별 Body/Hand/Whole, per-frame global/RR/pelvis와 Control 대비 signed delta, CI 없는 기술통계를 출력한다. factorial.csv에는 B−A, C−A, D−B 및 (D−B)−(C−A)를 남긴다. pelvis를 고정해도 다른 관절 및 손에 손실이 전파되는지를 local/global 손 오차로 분리한다. 실행은 GT 없이 sampling한 후 평가한다. [평가 코드](evaluation.py)

## 실제 검사 범위

신규 CPU 4개와 기존 RR 9개·통합 9개, 합계 22개 통과. 네 군 각각 축소 폭의 실제 V1Body로 24회 학습해 동일한 draw와 서로 다른 최종 state를 확인했다. A/B를 원래 trainer로 각각 다시 실행했을 때 state SHA가 같았다. Pelvis 수치·mask·gradient, Gram 결합 norm과 직접 backward의 일치, 기존 λ=0·gap·split·strict 복구·동결 모델 검사를 포함한다. 추가로 hidden512, 50프레임의 원래 크기 V1Body에서 세 gradient를 합성 입력으로 네 번 측정했다. 이 합성 수치는 실제 teacher의 계수나 충돌 증거로 사용하지 않는다. [검사 로그](cpu.log) · [합성 진단](synthetic_gradients.json)

기존 GPU ZIP(333,225,573 B)의 Drive 메타데이터와 경량 JSON을 확인했고 공식 audit ZIP의 고정 해시 및 plan을 로컬에서 검증했다. 대형 모델·과거 학습 ZIP 전체 바이트와 checkpoint를 이번 CPU 실행에서 다시 로드하지 않았다. 런타임은 CPU 전용이다. 실제 훈련에 앞서 Colab이 해당 SHA·원형 strict binding을 검사한다. [Work 인계](WORK.md)
