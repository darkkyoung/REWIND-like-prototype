# M2 Body Diffusion — UnrealEgo2 Pose 300-update 실험

**분리 실험 브랜치:** `experiment/m2-pose-300`  
**기준 원본 커밋:** [`5e52aa3`](https://github.com/BattleMarine/REWIND-like-prototype/commit/5e52aa3b0b1e27e7d0cb2697629e8acfc63d38c3)  
**원격 main에서 2026-10-19 관측한 커밋:** `4571f30` (실험 기준 아님)

이 디렉터리는 **실제 UnrealEgo2 Pose로 300번의 optimizer update를 수행한 M2 Diffusion 진단 실험**을 기록합니다. 학습 데이터나 공식 REWIND 재현 모델이 아니라, **소규모 GT 2D 조건의 연구용 중간 산출물**입니다. 공식 benchmark 성능/실시간성은 입증되지 않았습니다.

## 체크포인트

- [`checkpoints/last.pt`](checkpoints/last.pt) — 학습 완료 **300/300 update**, **1,725,981 bytes**, SHA-256 `6acd1fd96a2dbfe1e681710a6286c3d9d0177ce69402a0b488f8fcce7a4a5c9d`
- 포맷: `m2_raw_pose_diagnostic_v1`. `model`, Adam optimizer, scheduler, scaler, RNG, 설정, 데이터 manifest, 원본 코드 hash 포함. PyTorch 2.11.0+cu130 / Python 3.13.15 / Tesla T4 환경에서 생성.
- 모델 구성: body 22관절, hidden 64, graph 1층, temporal 2층, 1,000 diffusion steps, 10-step DDIM, GT 2D, **camera features zero**. 이 모델은 증류된 one-step student가 아닙니다.

### 로드 / 무결성 검사

프로젝트 root에서 Python과 PyTorch를 준비한 다음:

```bash
python tools/inspect_pose_300.py --checkpoint experiments/m2_pose_300/checkpoints/last.pt
```

체크포인트 직접 사용 예시:

```python
import torch
from rewind_m2.models import BodyDenoiser
path = 'experiments/m2_pose_300/checkpoints/last.pt'
ck = torch.load(path, map_location='cpu', weights_only=True)
names = ck['manifest']['joint_names']
model = BodyDenoiser(ck['config'], len(names), ck['manifest']['edges'])
model.load_state_dict(ck['model'], strict=True)
model.eval()
# 실제 추론에는 별도 관측 feature tensor와 camera/time/noise 입력이 필요합니다.
```

`torch.load(weights_only=True)`를 사용하되 체크포인트는 신뢰할 수 있는 출처에서만 불러오십시오.

## 학습 입력과 비교 가능 범위

- Train: 18프레임 (`Brutal_Assassination`, ArchVisInterior / rp_adanna)
- Holdout: 130프레임 (`Opening`, 같은 환경·캐릭터)
- 기존 5-step 결과 ZIP에서 검증된 148개 pose JSON을 **새 원본 다운로드 없이 재사용**하여 같은 설정으로 처음부터 300 updates 학습.
- 실제 학습 자료에는 공개 데이터셋에서 가져온 annotation의 원본 motion ID 독립성이 검증되지 않았습니다.
- `distance_unit=raw_annotation_unit_unverified`, `input_source=gt_2d`, `camera_policy=zero_features_ablation_not_extrinsics`, 개발용 body 22관절. **mm로 단정하지 않습니다.**
- 본 브랜치에는 원본 Pose/RGB JSON, 추출 ZIP, 프레임별 GT, 개인 Drive 식별 정보를 올리지 않았습니다.

## 평가 (raw 좌표, 낮을수록 좋음)

| 조건 | 5회 학습 | 300회 학습 |
|---|---:|---:|
| Train single x0 | 61.52 | **6.64** |
| Train DDIM-10 | 58.75 | **7.51** |
| Holdout single x0 | 60.25 | **12.27** |
| Holdout DDIM-10 | 56.73 | **12.23** |
| Holdout 평균 자세 baseline | 17.27 | 17.27 |

- 훈련 loss: 0.978046 → 0.003025 (`loss_history.json`).
- GPU 훈련 시간: 약 10.00초(평가, 데이터 준비 제외).
- DDIM: **독립 Gaussian** 시작. Single x0: **GT 3D에 인공 noise를 넣어 예측**한 진단이며 one-step distillation이 아닙니다.
- 작은 홀드아웃에서 평균 자세보다 오류가 낮아졌지만, 빠른 실제 동작의 크기가 작아지는 현상 등 Motion Fidelity, 일반화, RGB end-to-end 및 실시간성은 미확인입니다.
- 원본 별도 시각화와 GIF는 대화에 업로드된 300회 결과 ZIP에 포함되며, 이 브랜치에는 학습 결과의 핵심 지표와 코드만 기록합니다.

## 코드와 실행 재현

- `rewind_m2/pose_pilot.py`: 실제 Pose JSON → M2 Tensor, Diffusion 학습/평가.
- `tools/pose_reuse_300.py`: 검증된 5-step bundle 재사용·300 update·체크포인트 재개, 엄격한 원본 ZIP SHA 검사.
- `tools/pose_reuse_analysis.py`: 5 vs 300 정확도, 평균 자세 baseline, 모션 변화량을 평가.
- `configs/m2_pose_pilot.json` / `configs/m2_pose_selection.json`: 원본 실행 설정.
- `experiments/m2_pose_300/metrics.json`, `comparison.md`, `loss_history.json`, `validation.json`: 실측 자료.

`tools/pose_reuse_300.py`를 재실행하려면 원본 5-step ZIP(`results_dec550973d7941039e16da612a49f51c.zip`, SHA-256 `f4c9c2d9ed06a76385ec949b94c447a8f0751602ce82a36bb0f2fa5611debb1f`)을 Colab에 별도로 공급해야 하며, 코드의 `verify_bundle()`로 검사합니다. **해당 데이터 ZIP은 GitHub에 포함하지 않습니다.** 이전 학습의 GPU/PyTorch/Python 환경과 코드 식별자가 다르면 엄격 재현 검사가 차단될 수 있습니다.

## 다음 단계

여러 독립 시퀀스의 분할·카메라 및 단위 검증 → 실제 예측 2D+Confidence 연결 → 3D Motion 정확도 및 시간적 fidelity 비교 → One-step student / Adaptive Refinement 순서가 필요합니다. 최신 `main`의 62관절·좌표 파이프라인은 **별도 호환성 검증 없이는 이 체크포인트와 섞지 않습니다.**
