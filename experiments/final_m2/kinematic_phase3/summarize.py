"""Generate the report from the fixed CPU replay tables."""
from pathlib import Path
import json
import pandas as pd

here=Path(__file__).parent
results=here/'results'
p=pd.read_csv(results/'policies.csv')
means=p.groupby('policy').mean(numeric_only=True)
p2,p3,control=(means.loc[k] for k in ('phase2','phase3','control'))
retained=(control-p3)/(control-p2)
clean=p[p.condition=='clean'].copy()
fast=(clean.fast_motion_pct-100).abs().groupby(clean.policy).mean()
gates={'fast':bool(fast.phase3<fast.phase2),'protection':bool(p3.uncorrupted_active<p2.uncorrupted_active and p3.uncorrupted_harmful<p2.uncorrupted_harmful),
       'global':bool(p3.jpe<p2.jpe),'retention':bool((retained[['rr','fde','bone_pp']]>=.25).all())}
verdict='PASS' if all(gates.values()) else 'FAIL'
(results/'verdict.json').write_text(json.dumps(dict(gates=gates,verdict=verdict,retained=retained[['rr','fde','bone_pp']].to_dict()),indent=2))
f=pd.read_csv(results/'frames.csv')
e=pd.read_csv(results/'edges.csv').groupby('above_threshold')[['n','on_child_benefits','on_child_harms','corrected']].sum()
lines=['# Phase 3 — 저장된 Control800 예측의 연결별 보정','',
       f'사전 전체 판정 **{verdict}**. 15개 조건을 동일 가중했다. 같은 동작 프레임을 오염 조건별로 반복 사용했다.',
       '',means.round(4).to_markdown(),'',
       f'clean fast 이동량의 100% 절대편차(세 동작 평균)는 {fast.phase2:.2f} → {fast.phase3:.2f}%p. Control 대비 Phase 2 개선 폭 중 유지한 비율은 RR {retained.rr:.1%}, FDE {retained.fde:.1%}, 골격 왜곡 {retained.bone_pp:.1%}다. 세 지표 각각 25% 이상이어야 하는 조건과 clean fast 조건을 충족하지 못했다.','',
       '## 동작별 비교','',p[p.policy.isin(['control','phase2','phase3'])].groupby(['split','policy'])[['jpe','rr','fde','bone_pp','uncorrupted_active','uncorrupted_harmful']].mean().round(4).to_markdown(),'',
       '아래 표는 clean 조건의 빠른 프레임 이동량 비율(%). 100에 가까울수록 GT 크기에 가깝다.','',clean[clean.policy.isin(['control','phase2','phase3'])].pivot(index='policy',columns='split',values='fast_motion_pct').round(3).to_markdown(),'',
       '## 필요성 점수와 유익한 보정','',
       '적용되지 않은 연결도 평가하려고 동일한 sequence prior에서 gate를 항상 켠 정책을 반사실 비교로 사용했다. 각 연결의 child JPE가 Control보다 낮은 경우를 이득으로 센다. 부모에서 받은 이동도 포함하므로 해당 연결만 수정했을 때의 인과효과로 볼 수 없다. 이전 offset이 해제되는 동안 점수 1 이하에서도 일부 수정량이 남는다.','',
       e.assign(benefit_rate=e.on_child_benefits/e.n,harm_rate=e.on_child_harms/e.n).round(4).to_markdown(),'',
       'Phase 2 오류 조건 비교. 비율은 프레임 관측 단위이며 정책 표의 조건 동일가중 평균과 구분한다.','',
       f.assign(p2_harm=f.phase2_jpe_delta>0,p3_harm=f.phase3_jpe_delta>0).groupby(['dirty','fast'])[['p2_harm','p3_harm','local_edges']].mean().round(4).to_markdown(),'',
       '대표 사례는 Phase 3가 Phase 2보다 Global JPE를 가장 많이 개선·악화시킨 프레임을 사후 선택했다.','']
f['p3_minus_p2']=f.phase3_jpe_delta-f.phase2_jpe_delta
lines += [f.loc[[f.p3_minus_p2.idxmin(),f.p3_minus_p2.idxmax()]].round(3).to_markdown(index=False),'',
          '## 신체 비율 진단','',
          '오프라인 GT와 Train prior의 연결별 절대 편차 평균은 Opening/Gunplay/Joyful 모두 약 6.91%다. 이 세 표본에서는 Joyful만의 RR 악화를 개인 비율 차이로 설명할 근거가 없다. 시퀀스 prior는 신뢰한 시점의 예측 길이를 이용해 완만하게 업데이트하므로 모델 고유의 왜곡도 흡수할 수 있다. GT2D와 zero-camera에서 절대 신체 길이 식별은 미검증이다. 평가 GT는 진단에만 썼다.','',
          '## 재현 및 범위','',
          '실행 규칙은 DESIGN.md와 결과 전 커밋 0dc392e에 고정했다. 데이터·모델 코어는 Phase 2와 동일하다. 15개 evidence NPZ의 shape·GT·mask·ID·edge와 원래 hard-switch 선택, corruption label을 검증했다. 결과 ZIP 전체 SHA 및 내부 FILE_SHA256, 기존 T4 summary와 4개 입력 NPZ SHA를 대조했다. 두 checkpoint SHA는 evidence/frozen_checksums 및 summary의 과거 검증 기록과 대조했다. 이번 CPU 분석에서 checkpoint 바이트를 다시 로드하거나 직접 해싱하지 않았다.','',
          '```bash','python -m experiments.final_m2.kinematic_phase3.evaluate --archive REWIND_M2_Selective_T4_20261121_VERIFIED.zip --inputs inputs --candidate candidate/joyful57.npz --out new_results','python -m unittest experiments.final_m2.kinematic_phase3.test_gate experiments.final_m2.kinematic_phase2.test_phase2 -v','```','',
          '측정 시간은 CPU 후처리 단일 실행으로 특징 추출·연결별 선택·자손 이동을 포함하며 teacher 추론을 제외한다. CPU 검증은 Phase 3 5개, Phase 2 6개, 기존 M2 10개다. 이미 관찰한 세 validation을 재사용했고 신규 motion test는 확보하지 않았다. 실제 RGB frontend와 카메라 변화에 대한 일반화는 미검증이다. 이번 저장 예측 비교에 추가 Work T4 실행은 필요하지 않다.','',
          '![15개 조건의 Phase 3 - Phase 2 변화](results/comparison.png)']
(here/'REPORT.md').write_text('\n'.join(lines)+'\n')
