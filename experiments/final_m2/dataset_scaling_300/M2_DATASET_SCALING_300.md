# M2 Dataset Scaling 300 — 실행 전 고정 기록

2026-11-17. 입력 검증 PASS, CPU 회귀 33개 PASS. 실제 T4 학습·품질은 아직 NOT VERIFIED.

기존100(Brutal18, Standing50, Dodging32)을 그대로 보존했다. Boxing55 0–49, Lifting3 50–99, HipHop24 0–49, PullingRope9 50–99를 추가한다. Gunplay9 0–46은 캐시 Validation 근거가 있는 추가 평가 동작이다. 원본 mocap 독립성·FPS·물리 단위는 NOT VERIFIED다.

Train300만 fresh seed43에서 Position Loss로 400 updates 학습한다. 모델·최적화·추론은 기존 함수를 호출한다. Train18/100은 검증된400 checkpoint를 재사용하며 Opening 예측은 원본을 그대로 사용한다. 새 평가 동작에서는 세 모델을 동일 조건으로 추론한다. 최종 checkpoint400과 여덟 판정 기준은 config.json에 사전 고정했다.

400×8=3200 anchor draws이며 평균 기대 노출은10.67회/프레임이다. Train100의32회와 다르므로 동일 update 예산 비교로 해석한다.

현재 결과 수치·Train300 그림·GIF는 생성하지 않았다. 실제 학습 후 갱신한다.
