# 평가 전 고정 — 2026-11-21

주 정책은 신뢰 가능한 과거 예측으로 업데이트하는 sequence prior와 연결별 gate다. 모든 임계값은 세 validation에 적용하기 전에 고정한다. Train300의 anchor GT에서 연결별 중앙 길이, 동일 sequence 연속 3개 anchor의 양안 2D 상대벡터에서 연결별 등속 예측 잔차 P95(하한 1px)를 구한다. 인접한 연결들의 길이 비율과 불일치하고 기준 길이 편차가 5%를 넘으며 관측 잔차가 P95를 넘거나 양 끝 관측이 누락될 때 보정한다. 몸 전체의 일관된 이동은 상대벡터에서 상쇄된다. 분모 바닥값, gain 및 clip은 validation 확인 후 바꾸지 않는다.

prior ablation은 고정 Train 중앙값, 직전 예측 8개 중앙값, 신뢰한 과거 예측의 누적 EMA(0.1)로 한다. 후자 둘은 Train 기준 ±20% 범위로 제한한다. 주 정책에서 최근 두 관측으로 등속 잔차를 실제로 계산할 수 있고 양안 모두 정상이며 P95 이내일 때만 해당 연결 prior를 현재 출력 계산 후 갱신한다. gate on/off는 같은 sequence prior와 수정량 상한을 쓴다. Phase 2는 원본 그대로 독립 비교한다.

판정 단위는 motion × condition의 동일가중 평균. 성공하려면 (1) 세 clean 동작 평균 fast motion ratio의 100% 절대편차가 Phase 2보다 작고 (2) 비오염 수정률과 비오염 JPE 악화율이 모두 낮고 (3) 평균 Global JPE가 Phase 2보다 낮고 (4) Control 대비 Phase 2의 평균 RR/FDE/Bone 개선 폭을 각각 25% 이상 보존해야 한다. 네 항목 모두 필수다. 체크포인트는 동결된 저장 예측으로만 사용한다.
