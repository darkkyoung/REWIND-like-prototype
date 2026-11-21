# 관측·상태 불일치 기반 선택적 보정 — Phase 1

기준 소스 ac61e4d, Train300 cf76723, 연구 방향 859b3c7. 사용자 제공 기존 결과에서 Bone Loss는 세 동작의 길이 왜곡을 줄였지만 RR JPE가 모두 증가했다. 아래 기준은 신규 게이트 평가 전에 고정했다.

**추론**. 같은 FP32, zero-camera, Window8, DDIM10, sequence/frame noise seed 10043으로 동결 모델 둘을 실행한다. Control의 복원 상태와 현재 GT Stereo2D 입력(관절당 양안 7채널)을 사용한다. 양안별 xy 관측이 연속 세 프레임에 존재할 때만 등속 외삽 잔차를 계산한다. 나머지 신호는 Control pose 등속 외삽 잔차, 직전 대비 평균 연결 길이 변화, 관측 누락 비율이다. sequence 변경이나 gap에서 history를 비운다. valid는 좌표 존재 여부다. FPS와 물리 단위는 미확인이다.

**고정한 규칙**. Train300 clean 신호 P90으로 정규화하고 누락 비율 또는 관측 잔차와 구조 잔차가 동시에 큰 경우에 높은 점수를 준다. 구체식은 max(missing, min(obs, max(innovation, bone))). Train clean 점수 P95를 엄격히 초과하면 해당 프레임 전체를 Improved로 선택한다. magnitude baseline은 양안 2D 이동량에 같은 정규화·분위수 규칙을 쓴다. 빠른 정상 동작 보존 여부는 실측 대상이다. 두 teacher와 route를 모두 계산하며, latency는 GPU 동기화와 CPU 결과 복사를 포함한 arm별 프레임 평균 및 CPU gate/selection 평균이다. selective 시간은 이들의 합이다.

**오염**. seed 2111, 주기 20 native frames, 6–13번 팔 관절의 양안에 적용한다. jitter는 offset 4–8에서 표준편차 8px Gaussian, spike는 offset 9에서 64px Gaussian이다. missing은 offset 4부터 각각 2·8프레임 좌표와 존재 표시를 지운다. GT3D는 그대로 유지한다. 이 실험은 합성 오염을 측정한다. 실제 RGB 오류·visibility confidence·카메라 재투영은 측정하지 않는다.

**판정**. clean fast는 해당 validation에서 GT3D 유효 관절 평균 이동량 상위 사분위다. 이 라벨은 평가에만 쓴다. RQ1 통과는 3 motion × 4 corruption에서 모두 multi AUROC 및 average precision(계단형 AUPRC)이 magnitude보다 높고, 각 clean fast false correction rate가 5% 이하여야 한다. clean 단독 AUROC/AP는 정의되지 않아 빈 값이다. FCR은 오염되지 않은 프레임 중 선택 비율이다. RQ2는 세 motion × 다섯 조건 각각에서 selective의 Global/RR JPE 및 FDE가 양쪽 단일 경로보다 낮고, Motion/RR Motion Ratio가 모두 100%에 더 가까워야 통과한다. 동률은 실패다. clean fast 표본 수와 선택 오류, switch FDE는 trace.csv에서 확인한다. 검증 동작 재사용·단일 seed로 원본 mocap 독립성과 일반화는 미확인이다.

**Work 실행**. Colab에서 이 폴더의 T4.ipynb를 연다. Drive 읽기 인증 및 출력용 Drive mount 후 셀 순서대로 실행한다. 노트북은 Drive 파일 ID와 provider SHA256을 확인하고 원본 ZIP과 연결된 checkpoint·저장 예측의 member SHA를 검사한다. 다운로드에 쓰는 계정에는 두 파일에 대한 읽기 권한이 필요하다.

- 입력 ZIP ID `1V11xsktR_TNybTYXDD6RDF45ZgjiVnqZ`, 2,867,728 bytes, 로컬 실측 SHA256 `d786d36034f7c430d7eb7d3aa6bbee1a26b08308f7f75162623bceb67cec8082`. inputs/train300.npz, opening130.npz, gunplay47.npz는 scaling300 고정 SHA와 대조한다. poses/에서 이전 고정 converter로 joyful57.npz를 생성한다.
- 결과 ZIP ID `1MBVYoI0L77N4fc0ZQfrLKiFfKFNyz5AC`, Drive 메타데이터 42,297,177 bytes. 로컬 전송 한도 32MiB로 실제 ZIP SHA·800 모델·내부 배치는 이번 CPU 환경에서 미검증이다. 노트북은 ZIP 및 중첩 ZIP에서 control_800.pt, improved_800.pt, opening130/gunplay47/unused_motion_preselected 각각의 control/improved predictions.npz와 일치하는 checksums.json을 찾는다. SHA나 clean 재현이 다르면 중단한다.
- 실행 명령은 저장소 루트에서 아래와 같다. 결과 경로는 새 디렉터리다.

```bash
python -m experiments.final_m2.selective_refinement.runner --inputs inputs --candidate candidate/joyful57.npz --results frozen --out results
```

calibration.json은 Train 규칙, trace.csv는 입력 특징·오염·fast·선택·경로별 JPE 및 switch/FDE, metrics.csv와 edges.csv는 pose/동작/골격·latency 지표, detection.csv는 두 detector 비교, spike_scores.png는 gate 시각화, summary.json은 판정·SHA·환경을 담는다. Drive ZIP SHA는 별도 archive_identity.json에 보존한다. 실제 T4 및 RQ1/RQ2 판정은 NOT VERIFIED다.
