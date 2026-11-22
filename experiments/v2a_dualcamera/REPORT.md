# v2 A × RootAnchored24 · 이중 카메라 추론 어댑터

**카메라 원점 검산 PASS. 실제 동결 A의 전체 cascade parity 및 후보 정확도는 NOT VERIFIED.** 실제 Gunplay pose의 SHA를 검증한 뒤 원본 좌우 카메라 필드만 읽어 재계산했다. 두 2×9 float32 입력이 이전 T4 감사 텐서와 원소별로 같았다. 첫 world anchor는 [48896.465, −257638.75, 2217.94] mm, 최대 input translation 차이는 218774.90625 mm였다. 이 값으로는 Body 오차를 추정할 수 없다.

|경로|RGB 특징|Body 카메라|Hand 카메라|Hand upper·손목|
|---|---|---|---|---|
|Baseline|A Frontend e003의 joint-major 62×14|A e003, first_midpoint|A e002, first_midpoint|A Body의 device-mm 예측|
|Candidate|동일 특징 텐서|RootAnchored24, absolute|A e002, first_midpoint|RootAnchored24의 device-mm 예측|

동료 ZIP 학습 revision은 b7a2d00f3bc755808a451ed4ead873e6513d094b, 동결 소스 SHA는 9a68d8e45bcf3abf6fab3004740faa6805d4ce3fda5118c3a7102011937849d8이다. 동료 main 12ba5e4b2f9f56c4cc1164430674fd969ac094a6, 사용자 RootAnchored 기준 fb4ec39220100f3ac5501ea57defd43fcd4dddb7을 확인했다. pins.json에 원래 세 가중치 SHA, 후보 SHA·학습 binding, 실행 도구 SHA, 고정 평가 키가 있다.

### 좌표·시간 계약

adapter.DualCamera는 CPU float64 world→camera R,t를 받는다. 패키지 CameraOrigin.apply가 첫 유효 양안 카메라 중심의 평균 −mean(Rᵀt)을 world anchor로 정하고 t+R·anchor를 계산한다. absolute는 원래 t를 유지한다. 같은 R의 첫 두 열과 t/100을 좌우 순서로 인코딩한다. 호출은 실제 프레임의 연속 구간 단위다. padding은 제외한다. sequence 변경과 호출 사이 gap은 자동 reset, 연속 번호의 새 공식 클립은 new_clip=True로 reset한다. 청크 안에 gap이 있으면 분할을 요구한다. 이 continuation은 카메라 상태에 해당하며 Teacher 평가는 공식 50프레임 클립 단위다.

predict는 GT 필드가 없는 Observations를 TeacherRun.predict에 전달한다. 학습 좌표 정의에 따라 두 Body 모두 device mm로 반환하므로 Body 출력에 camera anchor를 재가산할 단계가 없다. 패키지의 body_joint_names와 양손 finger 순서를 검사한다. pelvis는 0이다. 선택 Body의 실제 SHA와 spec upper 순서를 UpperCondition에 전달한다. 혼합 경로의 Hand는 별도 추론 인스턴스에 A Hand state를 strict load하고 실제 후보의 upper를 받는다. attach_predicted_hands가 같은 프레임의 예측 좌우 손목을 해당 손 local finger에 한 번 더한다. 원본 A 영수증은 공식 로더가 검증한다.

### CPU 근거 및 해석 범위

- 새 검사 11개, 패키지 원본 camera training 회귀 검사 14개 통과. cpu.log와 regression.log에 개별 검사명이 있다.
- 고정 패키지의 PaperHeatmapFrontend, hidden512 V1Body/V1Hand, DDIM10, 합성 RGB 4프레임과 초기화 가중치로 공식 predict_clip을 실제 호출했다. 동일 환경에서 정규화 xy와 Whole62의 최대 차이는 0이었다. 미래 특징·카메라 교란 시 앞 프레임, 선택 upper, 손목 부착, Body의 104 state 항목, 모델·noise 불변을 검사했다.
- 실제 Gunplay 검산은 witness_replay.json에 기록했다. 원본 pose·학습 데이터·checkpoint는 Git에서 제외했다. 이전 Work 감사 ZIP의 SHA를 직접 확인했다. 거기에 기록된 Tesla T4 strict load 범위는 A Frontend/Body/Hand와 RootAnchored Body 전체다. 이번 CPU 세션에서 실제 가중치 파일을 다시 로드한 결과는 없다. 후보 ZIP 668,248,529 bytes가 커넥터 단일 다운로드 한도를 넘었고 전체 A ZIP은 55 GB다.
- 패키지에 포함된 실제 manifest에서 세 공식 Validation 각 0–49를 점검했다. 150프레임의 좌·우 이미지 및 pose SHA가 A 학습 또는 후보 Train300과 정확히 겹친 건수는 0이다. 세 클립 모두 adanna이며 Gunplay와 Reloading은 A train에도 동작명이 있다. Standing은 기존 후보 연구에서도 관찰했다. 원본 mocap 계보의 독립성과 다른 인물 일반화는 미확인이다.
- 후보 Body와 v2 Hand가 함께 학습된 적은 없다. 스케일·원점·shape의 일치는 손 또는 전신 성능의 이득을 보장하지 않는다. 실제 모델 parity, GPU 성능 및 latency 판정은 Work 실행 결과가 필요하다.
