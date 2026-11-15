"""회수한 실제 결과의 checksum·상태·지표를 재검사. 추론/학습 없음."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.final_m2.displacement_loss_20261116.run_experiment import state_hash,success_checks
from experiments.final_m2.diagnosis_20261111.diagnose import calculate
from tools.final_m2_real import sha,write


def validate(src,baseline_input,prior_t4,output):
    src=Path(src);e=json.loads((src/'evaluation.json').read_text())
    for name,h in json.loads((src/'checksums.json').read_text()).items():
        assert Path(name).name==name and sha(src/name)==h,name
    ck=torch.load(baseline_input,map_location='cpu',weights_only=True)
    assert sha(baseline_input)==e['identity']['parent_checkpoint_sha256']
    rows=[];bones=[];frames=[]
    for arm,record in zip(('control','improved'),e['arms']):
        p=src/f'{arm}_400.pt';assert sha(p)==record['checkpoint_sha256']
        c=torch.load(p,map_location='cpu',weights_only=True)
        assert c['step']==400 and c['samples_seen']==3200 and c['scheduler']['last_epoch']==400
        assert all(float(s['step'])==400 for s in c['optimizer']['state'].values())
        assert c['lambda']==(0 if arm=='control' else .1) and c['s_delta_raw']==4.1444875
        for k in ('model','optimizer','scheduler','scaler','rng'):
            assert record['initial_state_sha256'][k]==state_hash(ck[k]),k
        for p,t in c['model'].items():assert torch.isfinite(t).all(),p
        logs=[json.loads(s) for s in (src/f'{arm}_training.jsonl').read_text().splitlines()]
        assert [r['step'] for r in logs]==list(range(301,401)) and len(logs)==100
        assert all(r['lr_used']==.0005 and r['valid_joint_pairs']>0 for r in logs)
        trace=json.loads((src/f'{arm}_random_trace.json').read_text())
        assert len(trace)==100 and state_hash(trace)==record['random_trace_sha256']
        if arm=='control':reference_trace=trace
        else:assert trace==reference_trace
    previous_equal={}
    for split in ('train','holdout'):
        reference=None
        for arm in ('baseline','control','improved'):
            with np.load(src/f'{split}_{arm}_predictions.npz',allow_pickle=False) as z:d=dict(z)
            for k in ('gt','valid','frame_ids','sequence_ids'):
                if reference is not None:np.testing.assert_array_equal(reference[k],d[k])
            reference=d
            summary,b,f=calculate(d['prediction'],d['gt'],d['valid'],d['frame_ids'],ck['manifest']['edges'],0)
            expected=next(r for r in e['results'] if r['split']==split and r['model']==arm)
            for k,v in summary.items():
                if v is not None:np.testing.assert_allclose(v,expected[k],atol=1e-12,rtol=0)
            rows.append({'split':split,'model':arm,**summary});bones.extend({'split':split,'model':arm,**r} for r in b)
            frames.extend({'split':split,'model':arm,**r} for r in f)
            if arm=='baseline':
                with np.load(Path(prior_t4)/f'{split}_10_predictions.npz',allow_pickle=False) as old:
                    previous_equal[split]=np.array_equal(old['prediction'],d['prediction'])
                    assert previous_equal[split]
    assert success_checks(rows,frames,bones)==e['quality']
    assert e['execution_status']=='PASS' and e['baseline_unchanged']
    result={'status':'PASS','checksum_files':len(json.loads((src/'checksums.json').read_text())),
            'actual_updates':{'control':100,'improved':100},'baseline_matches_previous_T4_bitwise':previous_equal,
            'recalculated_metric_rows':len(rows),'metric_absolute_tolerance':1e-12,'pilot_status':e['quality']['pilot_status'],
            'checks':['SHA and CRC checked on acquisition','optimizer step400 and scheduler epoch400','all model parameters finite',
                      'original full initial state hashes equal','all100 batch/timestep/noise records equal',
                      'six NPZ share GT, masks, frame and sequence identities within each split','all summary metrics recalculated'],
            'new_inference_or_training_updates':0}
    write(output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--baseline-checkpoint',required=True)
    p.add_argument('--prior-t4',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(validate(a.input,a.baseline_checkpoint,a.prior_t4,a.output),indent=2))
