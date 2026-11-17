# 상태 — 2026-11-17

- 기준 원격 브랜치와 커밋 `cf767233172e2fe7c9514954f177bcfd56c41d17` 확인.
- Bone Length Loss, 2군 checkpoint 분기, Train-only gradient 가중치 고정, paired RNG trace, T4 실행 스크립트 작성.
- CPU 검증 결과: Python 3.12 / torch 2.15.0+cpu에서 신규 6개 + 기존 `tests.test_m2` 10개, 총 16개 통과. `py_compile` 및 `git diff --check` 통과.
- 실제 Train300 checkpoint와 NPZ는 Git에 없으므로 이 환경에서 실물 재개 및 T4 학습 미실행.
- 미사용 평가 동작 NPZ는 실행 전에 고정·검증 필요.
- 성능 개선: **NOT VERIFIED**.

## 독립 재검토 후 수정

- 원본 `8b9d341`의 checkpoint schema 오류와 입력/상태/지표 검증 누락을 수정했다. 상세 근거는 [REVIEW.md](REVIEW.md).
- 수정본 CPU 20개 검사 통과(신규/확장 10 + 기존 M2 10). 로그는 `cpu_review_tests.log`.
- T4 준비 상태: 실행 코드 수정 완료, 실물 checkpoint/NPZ 및 미사용 동작 사전 고정·검증은 미완료.
- 실제 T4 실행과 성능 개선: **NOT VERIFIED**. 이번 수정은 로컬 커밋만 수행한다.
