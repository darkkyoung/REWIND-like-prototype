# 발표 수치 검증 기록

- [x] 감사 브랜치를 fresh fetch해 기준 SHA 29aa6c5cdf2379cff55f28fcdf7b7493a64a9259를 확인했다. Taunt 실행 코드는 62b411eca4f100d056362e3fec73e5cfc819227c, 후보 고정은 7949e2d다.
- [x] Drive ZIP 546,226바이트의 SHA256은 810b70d455096a9c15a422df00c6e24deaf74b148fbc498bd4a5344dc98bc526이다. 내부 manifest의 32개 파일과 실제 목록·바이트를 대조했다. [원본 ZIP](https://drive.google.com/file/d/1hWxIrPHkVmpSB32PPEI06R-Plrxv-trk/view) · [결과 폴더](https://drive.google.com/drive/folders/1gVAk9s3HjJ5mAyjLN01tNaFZJD5zSSvi)
- [x] 실행 receipt 및 결과 보고서의 동결 Control800 SHA는 a70d9efbce4294323e2eca735155a20c12068a9d1e4215ae3dc12f1f862a2185다. 입력 taunt16.npz는 3dce98899fcb0eaaa281d3089031ee66621c0245767bef7c2ed374c83677abd7, poses.npz는 3e33c6074ad6a1b062082a58cb744955a1e79c86663e7e8d0e227c4d755c1cf4다. checkpoint 바이트는 Work에서 검증했다.
- [x] 릴리스 CPU 감사에서 poses.npz의 frame 0–15, 16×22 유효 mask, pelvis 불변, 네 정책의 다섯 지표와 사전 판정을 다시 계산했다. 원본 evaluation.json과 최대 차이는 8.77×10⁻⁶이다. 그림은 zip의 analysis/harm.png·help.png를 그대로 사용했다. [감사 로그](audit.log)
- [x] 기존 CPU 회귀는 26개, Taunt Work의 보완 회귀 로그는 31/31이다. 이번 릴리스는 저장 결과의 재계산이다. T4를 다시 실행하지 않았다.
- [ ] 상위 mocap 및 변형 중복 여부, 실제 RGB와 카메라, 독립 동작 일반화, 지속 실시간 처리는 미검증이다.

재계산 명령은 저장소 루트에서 실행한다.

```bash
python experiments/final_m2/final_audit/release/audit.py REWIND_M2_Taunt16_T4_VERIFIED_20261122.zip
```

- [x] 최종 PPTX 10장을 LibreOffice로 PDF 렌더링하고 전체 페이지의 제목, 표, 축, 본문, 하단 번호를 시각 검토했다. 9장의 두 줄에 13번과 15번의 다섯 자세가 각각 표시된다. [발표용 PDF](REWIND_M2_Final.pdf)
