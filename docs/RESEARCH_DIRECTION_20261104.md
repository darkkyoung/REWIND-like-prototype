# REWIND / VRION — 연구 방향 결정 기록 (2026-11-04)

> **상태: 사용자 확정 연구 방향 / 구현·성능 개선은 아직 검증 전**
>
> 이 문서는 2026-11-04 중간발표 준비 과정에서 사용자가 확정한 향후 연구 방향의 기준 기록이다. 코드가 구현되거나 성능 향상이 입증되었다는 선언이 아니다. 후속 작업은 이 방향을 우선 참고하되, 변경 시 날짜·이유·실험 근거를 명시한다.

## 1. 한 문장 연구 목표

**기존 REWIND-like의 Causal Body/Hand Motion Diffusion 구조를 활용하되, 관측 가시성과 상태 예측의 불일치로 신뢰도를 판단하고, 정상적인 빠른 동작은 보존하면서 불확실한 프레임에만 필요한 Diffusion 계산을 배정한다.**

연구명(가칭): **Reliability-Aware Adaptive Motion Refinement for Egocentric 3D Tracking**.

연구 질문: 실제 빠른 움직임과 Tracking Error를 구분해 선택적으로 복원할 수 있는가? 또한 어려운 프레임에 추가 계산을 배정하면 정확도·Motion Fidelity·평균/P95 지연 시간의 균형을 개선할 수 있는가?

## 2. 중간발표에서 합의한 세 가지 개선 축

| REWIND의 기존 방식 | 추가 개선하고자 하는 부분 | 참고 연구 | 우리 연구에서 맡을 역할 |
|---|---|---|---|
| 2D Keypoint + Confidence/Uncertainty | **가려진 관절과 실제로 관측 가능한 관절의 가시성 구분 강화** | **EvaPose (CVPR 2026)** | 관절별 visibility/observation reliability 신호 설계 |
| Causal Temporal Motion Diffusion | **과거 상태 예측과 현재 관측이 불일치할 때 어느 쪽을 신뢰하고 어떻게 복원할지 판단** | **DnD Filter (IROS 2025)** | causal prior + observation을 활용한 discrepancy 분석·선택적 보정 |
| One-step Distillation | **정확도 손실을 줄이면서 프레임 난도에 따라 추가 복원 계산량을 동적으로 배정** | **DSA (CVPR Findings 2026)** | confidence-guided adaptive computation 개념을 3D Motion 경로에 적용할지 검증 |

**해석상 주의**

- REWIND는 이미 uncertainty, causal attention, Body–Hand Cascade, one-step distillation을 사용한다. 이를 우리가 새로 발명하거나 처음 도입하는 기술로 서술하지 않는다.
- REWIND의 논문 수치와 자체 REWIND-like 실험 결과를 혼동하지 않는다.
- EvaPose는 관절 가시성에 관한 논문이며, REWIND에서 visibility 정보를 사용하지 않았다는 절대적 주장을 하지 않는다. 기존 uncertainty와 명시적 visibility는 구분한다.
- DnD Filter의 본래 연구는 동적 시스템의 Diffusion 상태 추정이다. 3D Motion의 Reliability Gate 자체가 해당 논문에 구현되어 있다고 주장하지 않는다.
- DSA는 **자기회귀 영상 생성**에서 confidence head로 **프레임별 diffusion steps**를 조절한다. REWIND의 3D Motion 모델에 적용한 사례나 Google 제품에 채택됐다는 뜻은 아니다. 우리 모델에 맞춘 routing·sampler는 별도의 신규 설계와 검증이 필요하다.
- **JEPA, World Model/World Engine** 등의 언급을 프로젝트 필수 개선 지시로 오인하지 않는다. **REPA**는 별도 검토 후보이지 이번 세 축의 필수 구현이 아니다.

### 참고 논문 (원문)

1. **REWIND** — Lee et al., *Real-Time Egocentric Whole-Body Motion Diffusion with Exemplar-Based Identity Conditioning*, CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Lee_REWIND_Real-Time_Egocentric_Whole-Body_Motion_Diffusion_with_Exemplar-Based_Identity_Conditioning_CVPR_2025_paper.html
2. **EvaPose** — Dai et al., *Egocentric Visibility-Aware Human Pose Estimation*, CVPR 2026. https://openaccess.thecvf.com/content/CVPR2026/html/Dai_Egocentric_Visibility-Aware_Human_Pose_Estimation_CVPR_2026_paper.html
3. **DnD Filter** — Wan & Zhao, *Differentiable State Estimation for Dynamic Systems using Diffusion Models*, IROS 2025. https://arxiv.org/abs/2503.01274
4. **DSA** — Le et al., *Dynamic Step Allocation for Fast Autoregressive Video Generation*, CVPR Findings 2026. https://arxiv.org/abs/2606.04432

## 3. 적용할 구체적 설계 (현재는 계획)

### A. 관측 신뢰도 / Visibility-aware Reliability

현재/과거만 사용하여 관절별 신뢰도 `r(j,t)` 또는 프레임 난도 `d(t)`를 산출할 수 있는지 검증한다.

- **관측 가시성**: 예측 2D confidence와 별도 visibility 예측/라벨이 실제로 존재할 때만 사용한다. `valid_mask`는 수치 유효성이지 실제 가시성/가림 정답이 아니다.
- **Stereo consistency**: 검증된 좌우 카메라 calibration, 시차·재투영 등 기하적 일관성(어안 모델 포함)이 있는 경우 활용한다.
- **Temporal innovation**: 직전 상태에서 **현재를 예측한 prior**와 실제 현재 관측의 차이를 평가한다. 실제 빠른 이동량 자체를 곧바로 오류로 판정하지 않는다.
- **Skeleton consistency**: 연결 관절·상대 뼈 길이의 급격한 모순을 보조 단서로 사용한다.
- **Camera motion**: HMD 이동에 의한 좌표 변화와 실제 신체 운동을 혼동하지 않도록 좌표계를 통일하고 변환을 먼저 검증한다.

초기에는 사용 가능한 단서만으로 규칙/가벼운 estimator를 만들고, 예측 2D·실제 confidence/visibility 확보 후 확장한다. Reliability 임계값은 train 내부 개발 구간에서 선택하며 독립 test에 튜닝하지 않는다.

### B. State-aware Selective Refinement

- 직전 모션으로부터 현재 3D 상태의 **causal prior**를 만들고 현재 2D 관측과 비교한다.
- 관측과 prior가 서로 충돌할 때, reliability와 skeleton/temporal evidence를 바탕으로 보정할 관절·프레임을 선택한다.
- 정상적인 빠른 동작에 기계적인 smoothing을 일괄 적용하지 않는다. 필요 시 clean/fast-motion 유지 경로와 오류 복원 경로를 비교한다.
- 초기 검증은 **기존 예측을 그대로 사용 vs 항상 보정 vs 선택적 보정**의 Ablation으로 분리한다. 관절별 선택 결합만으로 계산량 감소가 생겼다고 주장하지 않는다.

### C. Confidence-guided Adaptive Computation (DSA 원리 참고)

- 진짜 추론 시간 절감을 위해서는 **고비용 Diffusion 호출 전에** 저비용 경로와 추가 refinement 경로를 선택해야 한다.
- 가능 후보: 고정 M1 Regression 또는 별도 학습한 One-step Student를 Fast Path로 사용하고, 고비용 M2 Teacher 또는 올바른 sampler를 Refinement Path로 사용한다.
- 자체 M2의 현재 실험은 **10-step DDIM Teacher**다. GT에 노이즈를 추가한 single-x0 reconstruction은 One-step Distilled Student가 아니다.
- One-step 전용 student를 임의로 반복 호출하는 것을 유효한 multi-step sampler라고 간주하지 않는다. 서로 다른 step/solver 지원은 실제 학습 계약과 품질·시간 실측으로 검증한다.
- 실제 배치/시퀀스 스케줄링, GPU workload·sync overhead까지 측정해 gate 비용이 이익을 상쇄하지 않는지 확인한다.

## 4. 평가 가설 및 기준선

| 연구 질문 | 비교 대상 | 핵심 지표 |
|---|---|---|
| RQ1. 오류 탐지 | 큰 이동량 단독 임계값 vs 다중 단서 reliability | corruption 탐지 AUROC/PR, 정상 고속 동작 false-correction rate |
| RQ2. 모션 보존 | 무보정 vs 일괄 smoothing/refinement vs 신뢰도 기반 선택 | 관절 위치 L2/MPJPE(단위 검증 조건), displacement error, 동작 크기 비율, clean/fast-motion 오차 |
| RQ3. 계산량 배분 | 항상 저비용 vs 항상 고비용 vs adaptive routing | 같은 GPU·batch1·precision의 mean/P95 ms/frame, 품질–지연 Pareto |

- 정상 Clean / Jitter / Spike / 단·장기 Missing / Occlusion·Out-of-view / Fast Real Motion을 구분한다. 인위적 관측 오염은 실세계 2D Front-end 오류 검증과 별도로 표시한다.
- Raw 단위와 mm 단위를 혼동하지 않는다. FPS·timestamp 미확인 상태에서는 초당 velocity/acceleration을 주장하지 않고 frame displacement로 보고한다.
- 학습·Validation·최종 Test는 가능한 한 **원본 Motion/Sequence 단위**로 독립 분할한다. 같은 모션 재렌더링 중복을 점검한다. Test는 임계값·모델 선택에 사용하지 않는다.
- 실제 측정 시 3D 모델 단독과 RGB front-end 포함 end-to-end latency를 분리하고 warm-up·GPU 동기화·평균 및 P95를 명시한다.
- 개선 성공 여부는 Clean 성능 손상과 정상 빠른 동작 오판율을 포함해 판단한다. 결과가 좋지 않아도 실패 사례와 가정을 보존한다.

## 5. 현재 진행 사실과 향후 연구의 경계 (2026-11-04 기준)

- 동료 원본: https://github.com/BattleMarine/REWIND-like-prototype (확인 main `40fda70bfbc2101591350d0f59d11136ba0e6995`). M0~M3의 연구용 구현·합성 검증, 실제 Stereo RGB 102쌍/대응 Pose JSON 102개 추출·검사 기록이 있다. **102개 모두 train이며 독립 validation/정식 training release/실제 RGB→전신 3D 최종 성능은 미검증**이다.
- 사용자 Fork: https://github.com/darkkyoung/REWIND-like-prototype (문서 작성 기준 main `4571f30f6bac17d21e775a3a16488c98d44d117d`).
- 별도 M2 실험: `experiment/m2-pose-300`의 `74851d8c05933c135bf784c15f29dbf684987687` 및 `experiments/m2_pose_300/README.md`. 실제 UnrealEgo2 GT2D/GT3D 148프레임 중 train18/holdout130, **300 optimizer updates**, 22관절, raw 좌표, zero-camera, 10-step DDIM Teacher.
- 자체 M2 Holdout DDIM 위치 오차 약 **12.23 raw** vs 평균 자세 baseline **17.27 raw**. GT 대비 프레임별 평균 관절 이동량 **40.3%** 관측. 이는 **소규모 자체 모델에서의 진단**이지 REWIND 논문에 증명된 일반적 결함이 아니다.
- **아직 미구현/미검증**: 실제 신뢰도 추정기, visibility branch, DnD-inspired state update, Adaptive Routing, DSA 방식 step 배분, One-step Student, 본 연구의 accuracy–latency 개선 효과.

## 6. 실행 순서 및 팀 협업

1. **중간발표:** 이 문서의 3축을 개선 방향으로 설명하고, 기존 REWIND-like 구현/데이터 확보와 실제 M2 진단 결과를 별도로 제시한다. 새 방법의 성능 향상은 주장하지 않는다.
2. **공통 데이터·입력 정합:** 동료의 RGB/2D/pose/skeleton·좌표계·validation 준비 수준을 확인하고 기존 22관절 raw 진단과의 호환 차이를 문서화한다.
3. **실패 분석:** 기존 checkpoint 및 추가 독립 시퀀스에서 clean/fast/corruption 유형별 motion fidelity를 측정한다.
4. **RQ1/RQ2 우선:** 저비용 reliability baseline과 관측 선택/복원 정책을 구현·Ablation 한다. GT와 미래 프레임은 추론 입력으로 누출하지 않는다.
5. **RQ3 후속:** 유효한 경량·고비용 추론 경로가 둘 다 준비된 후에만 adaptive compute/latency 개선을 실측한다. 시연과 최종보고에는 성공·미검증·실패를 구분한다.

**역할 분담 제안:** 동료분은 M0~M3/실제 영상 입력·학습 파이프라인 중심, 사용자님은 논문 적용 설계·Reliability 및 선택 정책·평가 계획과 결과 분석 중심. 이는 실행 계획이며 팀의 별도 합의·기술 준비도에 따라 조정한다.

## 7. 변경 관리

- **2026-11-04:** 사용자 중간발표 슬라이드 3에서 세 개선 축과 참고 연구를 최종 선택: **EvaPose(2026) + DnD Filter(2025) + DSA(2026)**. SADA 대신 DSA를 채택.
- 프로젝트의 기본 REWIND-like 구현·데이터 준비·과거 체크포인트는 수정하거나 덮어쓰지 않는다.
- 후속 구현 변경은 별도 브랜치에서 설계/실험/측정값과 함께 기록한다. 연구 방향이 바뀌면 본 문서에 새로운 결정 날짜와 반대 근거를 추가하고 과거 결정을 삭제하지 않는다.

