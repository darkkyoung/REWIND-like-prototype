"""Summarize the frozen decision rule from the replay CSV files."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

root=Path(__file__).parent
out=root/'results'
p=pd.read_csv(out/'policies.csv')
f=pd.read_csv(out/'frames.csv')
e=pd.read_csv(out/'edges.csv')
m=p.groupby('policy').mean(numeric_only=True)
c,k,g,d=[m.loc[key] for key in ('control','phase2','phase3','combined')]
clean=p[p.condition=='clean'].copy()
fast=(clean.fast_motion_pct-100).abs().groupby(clean.policy).mean()
retention=((c-d)/(c-k))[['rr','fde','bone_pp']]
protection=((k-d)/(k-g))[['uncorrupted_active','uncorrupted_harmful']]
gates=dict(retention=bool((retention>=.25).all()),protection=bool((protection>=.5).all()),
           global_jpe=bool(d.jpe<=c.jpe),clean_fast=bool(fast.combined<=fast.control))
by=p.groupby(['split','policy']).mean(numeric_only=True)
verdict=dict(status='PASS' if all(gates.values()) else 'FAIL',gates=gates,
             retention=retention.to_dict(),protection=protection.to_dict(),fast_error_pp=fast.to_dict(),
             by_motion=json.loads(by.reset_index().to_json(orient='records')))
(out/'verdict.json').write_text(json.dumps(verdict,indent=2))
fig,axes=plt.subplots(1,3,figsize=(12,4))
for ax,metric in zip(axes,['rr','fde','bone_pp']):
    table=by[metric].unstack()
    (table[['A','B','combined']].sub(table.control,axis=0)).plot.bar(ax=ax,rot=20)
    ax.axhline(0,color='gray');ax.set_title(metric+' minus Control')
fig.tight_layout();fig.savefig(out/'motion.png');plt.close(fig)
cols=['jpe','rr','fde','bone_pp','uncorrupted_active','uncorrupted_harmful','cpu_ms']
def table(data):return data.round(4).fillna('').to_markdown()
text=['# Phase 4 · 지속 구조 오류와 순간 추적 오류','',
'**사전 전체 판정 FAIL.** Global JPE와 clean fast 기준은 충족했다. FDE는 Control보다 악화됐고 비오염 프레임 보호 효과는 기준에 미달했다. 모든 평균은 세 기존 validation × 다섯 조건 동일 가중이다.','',table(m[cols]),'',
'## 경로별 역할','',
'A만으로 평균 RR JPE와 골격 왜곡이 줄었다. 길이 편향은 관측 누락과 독립적으로 지속될 수 있다. B는 누락 또는 양안 연결벡터의 인과적 잔차와 3D 혁신이 같이 나타날 때만 발동했다. B만으로는 수정 범위가 작고 FDE 개선을 얻지 못했다. 결합은 A보다 Global/RR JPE를 소폭 더 낮췄지만 FDE와 골격 왜곡은 더 컸다. A와 B가 각각 도움이 되는 상황이 있다는 것과 결합이 모든 목적에 이득이라는 것은 구분해야 한다.','',
f'Combined가 보존한 Phase 2 개선량은 RR {retention.rr:.1%}, FDE {retention.fde:.1%}, Bone {retention.bone_pp:.1%}였다. 음수는 Control보다 악화된 경우다. Phase 3의 비오염 수정률·악화율 감소 중 보존한 비율은 각각 {protection.uncorrupted_active:.1%}, {protection.uncorrupted_harmful:.1%}였다. clean fast 이동량의 100% 절대편차는 Control {fast.control:.2f}, Phase 2 {fast.phase2:.2f}, Phase 3 {fast.phase3:.2f}, Combined {fast.combined:.2f}%p였다.','',
'## 원인과 전파','',
f'전체 {len(f)} frame-condition, {len(f)*21} edge-frame에서 A 후보 {f.persistent.sum()}, B 후보 {f.transient.sum()}, 중복 {f.both.sum()}. 중복 중 {f.conflict.sum()}회는 반대 방향으로 충돌해 목표 보정을 0으로, 나머지는 B 우선으로 정했다. 변화량 상한 때문에 이전 offset은 즉시 사라지지 않을 수 있다.','',
f'Phase 3 점수 ≤1인데 실제 이전 offset의 절댓값이 1e-7 raw보다 큰 edge-frame은 {f.p3_off_residual.sum()}개였다. A 기준의 지속 편향 중 Phase 3 점수가 꺼진 경우는 {f.p3_missed_persistent.sum()}개다. Opening clean은 A 후보 1,275개 전부가 여기에 해당한다. 이 수는 prior와의 지속 불일치이지 GT 기반 오류 정답이 아니다.','',
f'국소 offset이 적용된 edge-frame {e.active.sum()}개 가운데 해당 이동만 자손에서 제거해 대조했을 때 자손 평균 JPE가 줄어든 경우 {e.benefit.sum()}, 악화된 경우 {e.harm.sum()}였다(1e-6 raw 이내 동률 제외). 가장 긴 연속 적용은 {e.max_consecutive.max()}프레임. 다른 연결의 이동은 고정한 사후 진단이다. 근위 연결 제한으로도 비오염 프레임 수정률이 높은 문제는 남았다.','',
'## 동작 및 사례','',table(by[cols[:-1]]),'',
f.loc[[f.jpe_delta.idxmin(),f.jpe_delta.idxmax()],['split','condition','frame','local_edges','moved_joints','jpe_delta']].to_markdown(index=False),'',
'사례는 Combined−Control Global JPE 최소·최대 프레임으로 사후 선택했다. 각 행 안에서 세 스켈레톤은 동일 축 범위·시점을 사용한다.','',
'![동일 축 사례](results/skeletons.png)','![동작별 오차 변화](results/motion.png)','',
'## 범위 및 재현','',
'원본 T4 ZIP SHA256 6d8e29d88a4e6b1735244b5f302ecf26159841b87d3c7c7a39871312a32d82ee 및 내부 파일 전체, 입력 NPZ 4개 SHA, GT·mask·frame ID·edge와 저장된 선택 출력을 대조했다. checkpoint SHA는 기존 T4 summary와 frozen_checksums 양쪽 기록으로 검증했다. 이번 측정은 저장 예측 CPU 후처리이며 checkpoint 재로딩·GPU 추론은 수행하지 않았다. cpu_ms는 Python/NumPy 단일 실행의 frame 평균이다.','',
'입력 출처는 Phase 2 REPORT와 같다. 로컬 첨부 목록 및 기존 Drive 실험 폴더에서 확인한 데이터는 기존 INPUTS와 결과 패키지다. 새 출처·동작 중복·전처리·동결 모델 출력을 모두 검증할 수 있는 새 sequence는 확보하지 못했다. 기존 3개 동작에서 반복 관찰한 결과이며 새 인물·실제 RGB·실제 카메라 일반화는 미검증이다. 현 오프라인 동작 비교를 위한 추가 Work T4 실행은 필요하지 않다.','',
'```bash','python -m experiments.final_m2.dual_phase4.evaluate --archive REWIND_M2_Selective_T4_20261121_VERIFIED.zip --inputs inputs --candidate candidate/joyful57.npz --out new_results','python -m unittest experiments.final_m2.dual_phase4.test_dual -v','```','',
'결과물은 policies.csv(정확도·골격·이동 비율·CPU 시간), frames.csv(경로 중복·충돌 및 잔여 상태), edges.csv(연결별 자손 전파·이득·손해), verdict.json(사전 판정·동작별 평균)에 보존했다.']
(root/'REPORT.md').write_text('\n'.join(text)+'\n')
