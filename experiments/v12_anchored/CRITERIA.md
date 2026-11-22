# 사전 고정 · Root-Anchored24

GPU 결과 확인 전 고정한다. 구간은 selection.json의 연속 0–49번, 집계 단위는 동작이다. 기존 Standing 및 Petting Animal은 진단용이다. 추가 두 동작만 전체 판정에 사용한다.

|대조군|RootAnchored24가 각 신규 클립에서 동시에 충족할 조건|
|---|---|
|RR24|pelvis error ≤ 0.95 × RR24, RR JPE ≤ 1.01 × RR24|
|Control24|RR JPE ≤ 0.99 × Control24, pelvis·body global·wrist·hand global·whole62 각각 ≤ 1.02 × Control24|

두 신규 clip 모두 통과해야 전체 PASS. 지표 누락·비유한·기준 분모 0이면 INCOMPLETE. 5%/1%/2%는 탐색적 공학 기준이다. 신규 motion 선정으로 원본 mocap 독립성이나 통계적 유의성이 확보되지는 않는다.

Train300/6명/6동작은 이전 selection 그대로다. 전 군 hidden512, FP32, 1×50, Adam 1e-5, StepLR(1,1), 24 updates, seed 20261122, 원형 DDIM10과 고정 카메라 입력이다. max 4 gradient draws + 1 폐기 preflight update + 4×24 updates + 4×4 평가 clip. 메모리 예약량 95% 초과 또는 OOM 시 중단한다.

E010 최초 상태에서 이전 draw 0–3(Singing k167, Crouch To Stand k64, Joyful Jump k762, Pulling A Rope k286)에 대해 기존 목적 P, RR R, pelvis H의 전체 Body gradient를 산출한다. 파라미터별 내적 합으로 Gram 행렬 G를 구성하고 미사용 파라미터의 미분은 0으로 둔다. 세 L2 norm, 세 cosine, 각 군 계수 v에 대한 sqrt(vᵀGv)를 기록한다. 음의 cosine은 해당 입력에서의 gradient 반대 방향 성분이다. λRR=0.08310449710117396 고정. λH=0.1×median(||∇P||/||∇H||) 한 번. norm이 0이거나 비유한 경우, coefficient가 비양수/비유한 경우, 결합 norm이 비유한 경우 중단한다. 평가 결과로 재설정하지 않는다.
