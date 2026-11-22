# v1.2 Whole62에 M2 보정 연결 | 2026-11-22

**CPU에서 관절·좌표·조건 전달과 원형 추론의 합성 입력 동등성을 확인했다. 원본 학습 가중치의 로컬 strict load, 실제 RGB 정확도 및 T4 지연시간은 미검증이다.** 기준 모델은 동료의 Frontend e009, Body e010, Hand e009이고, M2에서는 세 후처리 함수만 재사용한다. [v1.2 공유 계약](https://github.com/BattleMarine/REWIND-like-prototype/blob/12ba5e4b2f9f56c4cc1164430674fd969ac094a6/docs/V12_MODEL_SHARING.md) · [이전 M2 결과](../final_m2/final_audit/REPORT.md)

## 계약 감사

|항목|v1.2 → 통합 경로|M2에서 달라지는 부분|
|---|---|---|
|모델|hidden 512, graph 3, temporal 3, 과거 attention 20, 연속 1–50프레임, DDIM10 FP32|기존 hidden 64, graph 1, temporal 2, Window8, DDIM10 Body 가중치는 호출하지 않는다.|
|배열 순서|같은 이름·순서의 Body22, pelvis 0, 방향과 순서까지 같은 21개 edge. 양쪽 manifest를 실제 비교한다. Whole은 spec 이름으로 조립한다.|remapping 불필요. spec SHA256 32f1cfda25bc2b05745d2b534f0fd21eec0defbd379cd1d865c152e52c74b187 이외 계약은 중단.|
|3D|양쪽 원본 필드는 device_pts3d. v1.2 converter는 spec의 adopted metric anchor 10 mm/raw를 적용한다. 축 교환·회전 없이 동일한 장치 원점에서 보정한다.|M2의 기존 raw 결과 및 길이 prior를 가져오지 않는다. 기존 Taunt16 JSON 및 NPZ에서 16×22 관절과 mask를 대조했다. v1.2 변환 좌표 /10과 M2 raw의 최대 차이는 0.00001526 raw다. 이 검사는 실제 신체 길이의 계측 검증과 구분한다.|
|2D|좌→우 RGB 1024², align_corners bilinear 128², float32 [0,1]. 모델 feature의 xy는 [−1,1]. refiner에만 (xy+1)×1023/2048을 전달한다.|기존 GT 픽셀/1024와 같은 규약. 양안 7채널 중 index 2는 좌표 제공 여부(항상 1), 3은 heatmap 확률 최댓값이다. 작은 confidence를 누락이나 실제 visibility로 바꾸지 않는다. 누락 경로는 현 frontend에서 발동하지 않는다.|
|카메라|각 뷰 world→camera R의 첫 두 열과 t_mm/100, [T,2,9]. reflected-world convention은 동료 변환기가 담당한다. 원본 target_scale_mm 100은 teacher 정규화 계수.|M2의 zero-camera 입력을 전달하지 않는다. 보정 입력은 device-mm 자세와 현재까지의 predicted 2D다.|
|Hand|upper는 Body index 1–13. 왼손목 index 9, 오른손목 13. Hand40은 left 20 → right 20. refined upper/100을 BoundHand에 넣고 local 결과(mm)에 대응 손목을 딱 한 번 더한다.|보조 조건에서는 원래 upper와 원래 wrist를 모두 사용해 hand 전체 경로를 고정한다. 해당 Whole에서 몸과 손목의 불연속은 분리 실험의 성질이다.|
|시간·noise|원본 schedule, seed 71 + 100(Body) / +200(Hand). sequence/frame별 keyed noise. 한 연속 clip마다 별도 호출, gap은 새 clip으로 분리한다.|같은 Body 출력을 공유한다. 보정기는 매 sequence/gap 초기화, GT·오염 라벨 입력 없음. 조건마다 동일한 Hand noise로 다시 샘플링한다.|

전체 62개 이름·방향은 고정 spec에서 확인할 수 있다. 좌표 검사는 Taunt16을 calibration 및 성능 평가에 사용하지 않았다. [spec](../../configs/real_archvis62_v1.json) · [실제 좌표 대조 로그](coordinates.log) · [추론 코드](pipeline.py)

## 경로와 사전 비교 설계

- baseline은 Body를 그대로 전달한다. 실데이터 실행 때 원본 공유 API로 따로 추론해 Body, hand local/global, whole의 원소별 동등성을 먼저 검증한다.
- Phase 2는 연결 방향의 제한된 길이 보정, Phase 3는 구조·관측 게이트와 과거 길이 추정, Phase 4는 지속 편향과 순간 innovation 경로를 사용한다. 세 구현의 임계값·gain은 기존 코드 그대로다. 통계가 없으면 보정 요청은 오류로 종료되고 baseline만 실행할 수 있다.
- 기존 full training의 고정 audit ZIP에서 manifest·coverage·training plan을 재생성하고 checkpoint run의 manifest/plan SHA를 대조한다. Train에서만 sequence 사전순 앞의 서로 다른 3개 동작별 최초 완전 50프레임을 선택한다. 150프레임의 유효 GT mm 길이 중앙값과 동결 frontend의 양안 시간 잔차 P95를 산출한다. 이는 v1.2 train의 고정 부분집합이다. 원본 관절 prior와 innovation scale의 이식 성능은 실험 질문이다.
- 평가 대상은 같은 공식 plan의 첫 teacher validation clip 50프레임이다. 실제 sequence/frame, 원본 SHA, calibration 통계 및 SHA를 실행 결과에 기록한다. 평가 GT는 지표 계산 시 사용한다. 평가 결과로 선택을 바꾸지 않는다.
- 지표는 Body global/RR/pelvis/wrist, 길이 왜곡, 중심으로부터 평균 관절 반경, 변위 벡터 FDE, 이동량 비율, Hand local/global, Whole62 global, 수정량 및 악화 프레임 비율이다. CPU↔GPU 복사 및 동기화를 포함한 보정 hook 시간만 별도 기록한다. body와 hand의 개선·악화를 각각 판단한다. 이전 M2 raw 수치 및 반복 관찰한 Validation과 절대값을 비교하지 않는다. [평가 코드](evaluate.py)

## 실측 검증 및 남은 차단 요인

CPU 48개 통과. 신규 9개는 축소 폭의 원본 V1Body/Hand 클래스와 동결 무작위 가중치, 원형 frontend를 사용했다. 원본 cross_inference와 통합 baseline의 네 출력은 원소별 동일했다. 조건 전달 검사에서 실제 Hand local 출력 변화, 원래 조건 ablation의 동일성, 손목 1회 덧셈을 확인했다. 모든 보정 정책의 prefix/future 불변·gap/sequence reset·pelvis 보존, strict state dict 합성 검사, 잘못된 카메라/shape/NaN/mask 거부, 부분 mask 지표를 검사했다. 동료 cross-scenario 8개, 기존 M2 보정·선택 21개, M2 core 10개도 통과했다. 테스트명은 기대 동작을, 각 로그의 ok는 실제 결과를 나타낸다. [통합 9개](cpu.log) · [동료 8개](peer_regression.log) · [M2 31개](m2_regression.log)

Drive에서 공유 ZIP 두 개의 ID·파일명·크기(v1.2 127,886,815 B, M2 42,297,177 B)를 확인했다. 실행기 파일 전송 한도는 파일당 32 MiB이고 별도 바이트 다운로드도 성공하지 못했다. 따라서 이번 CPU 결과로 원본 checkpoint strict load 또는 실제 모델 baseline 재현을 인증할 수 없다. 현재 로컬 환경은 CPU 전용이다. 실제 v1.2 train/validation RGB·pose 및 full audit ZIP의 대응 경로도 확인되지 않았다. 고정 selection과 검증 절차를 구현했으나 실데이터 prior·성능·속도는 NOT VERIFIED다. 기존 동료 GPU 결과와 기존 M2 판정은 각 원래 실험에만 적용한다.

Work에서 원본 패키지만으로 실제 가중치 CPU smoke가 가능하다. 실데이터 정책 비교에는 아래 인계 문서의 audit와 원본 RGB·pose가 추가로 필요하다. [Work 인계](WORK.md)
