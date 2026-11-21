"""Run from the repository root with python -m experiments.final_m2.kinematic_phase3.evaluate."""
import argparse
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
from time import perf_counter_ns
from zipfile import ZipFile

import numpy as np
import matplotlib.pyplot as plt
from experiments.final_m2.selective_refinement.policy import corrupt
from experiments.final_m2.diagnosis_20261111.diagnose import calculate,write_csv
from ..kinematic_phase2.refine import Refiner,train_lengths
from ..kinematic_phase2.oracle import errors
from .gate import Gated,calibrate

RESULT_SHA='6d8e29d88a4e6b1735244b5f302ecf26159841b87d3c7c7a39871312a32d82ee'
SPLITS={'opening130':130,'gunplay47':47,'unused_motion_preselected':57}
CONDITIONS=('clean','jitter','spike','short_missing','long_missing')


def run(archive,inputs,candidate,out):
    out.mkdir(exist_ok=False,parents=True)
    def check(ok, key):
        if not ok: raise ValueError(key)
    check(sha256(archive.read_bytes()).hexdigest()==RESULT_SHA,'result archive')
    rows,frames,edge_rows,identity_rows=[],[],[],[]
    with ZipFile(archive) as z:
        checks=json.loads(z.read('FILE_SHA256.json'))
        for name,h in checks.items():
            check(sha256(z.read(name)).hexdigest()==h,name)
        identity=json.loads(z.read('results/summary.json'))
        expected={'control':'a70d9efbce4294323e2eca735155a20c12068a9d1e4215ae3dc12f1f862a2185','improved':'fa2e875cd0757011ed62755b5adca0185579ea9945e1c714ad8d4497902507ca'}
        frozen=json.loads(z.read('evidence/frozen_checksums.json'))
        check(identity['checkpoint_sha256']==expected and all(frozen[k+'_800.pt']==v for k,v in expected.items()),'frozen provenance')
        trace=list(csv.DictReader(io.StringIO(z.read('results/trace.csv').decode())))
        def read_input(name):
            path=candidate if name=='unused_motion_preselected' else inputs/(name+'.npz')
            check(sha256(path.read_bytes()).hexdigest()==identity['inputs'][name],name)
            with np.load(path,allow_pickle=False) as data:
                return {k:data[k] for k in data}
        train=read_input('train300')
        edges=np.asarray(json.loads((Path(__file__).parents[1]/'dataset_scaling_300/data_selection_manifest.json').read_text())['edges'])
        lengths,scale=calibrate(train,edges)
        (out/'calibration.json').write_text(json.dumps(dict(lengths=lengths.tolist(),scale=scale.tolist(),train_sha=identity['inputs']['train300']),indent=2))
        for split,n in SPLITS.items():
            data=read_input(split)
            seq=data['sequence_ids']; check(len(set(seq))==1,'sequence')
            for cond in CONDITIONS:
                with np.load(io.BytesIO(z.read(f'evidence/{split}_{cond}.npz'))) as evidence:
                    c,i,s,g,v,ids=[evidence[k] for k in ('control','improved','selective','gt','valid','ids')]
                    np.testing.assert_array_equal(edges,evidence['edges'])
                check(all(p.shape==(n,22,3) and np.isfinite(p).all() for p in (c,i,s,g)),'pose')
                check(v.dtype==bool and v.shape==(n,22) and v.all(),'expected full Body22 validity')
                for actual,want in [(ids,data['frame_ids']),(g,data['target_raw'][:,-1]),(v,data['valid'][:,-1])]:
                    np.testing.assert_array_equal(actual,want)
                x,dirty=corrupt(data['features'][:,-1],cond)
                t=[r for r in trace if r['split']==split and r['condition']==cond]
                def col(k):return np.array([int(r[k]) for r in t])
                fast,sel,sw=[col(k).astype(bool) for k in ('fast','selected','switch')]
                np.testing.assert_array_equal(dirty,col('corrupted'))
                np.testing.assert_array_equal(ids,col('frame'))
                np.testing.assert_array_equal(s,np.where(sel[:,None,None],i,c))
                tag=dict(split=split,condition=cond)
                refiner=Refiner(edges,lengths)
                start=perf_counter_ns()
                phase2=np.stack([refiner.push(q,f,p,obs) for q,f,p,obs in zip(seq,ids,c,x)])
                times={'phase2':(perf_counter_ns()-start)/1e6/n}
                policies={'control':c,'improved':i,'switch':s,'phase2':phase2}
                for key,prior,mode in [('phase3','sequence','gated'),('off','sequence','off'),('on','sequence','on'),
                                       ('fixed','fixed','gated'),('temporal','temporal','gated')]:
                    model=Gated(edges,lengths,scale,prior,mode)
                    start=perf_counter_ns()
                    batch=[model.push(q,f,p,obs) for q,f,p,obs in zip(seq,ids,c,x)]
                    times[key]=(perf_counter_ns()-start)/1e6/n
                    pred,score,offset=map(np.stack,zip(*batch))
                    policies[key]=pred
                    if key=='phase3':
                        primary_score,primary_offset=score,offset
                a,b=edges.T
                gt_length=np.linalg.norm(g[:,b]-g[:,a],axis=-1)
                if cond=='clean':
                    for j,(a0,b0) in enumerate(edges):
                        bias=100*(lengths[j]/gt_length[:,j]-1)
                        identity_rows.append(dict(split=split,parent=a0,child=b0,train_length=lengths[j],
                                             gt_median=np.median(gt_length[:,j]),mean_bias_pp=bias.mean(),abs_bias_pp=abs(bias).mean()))
                base=errors(c,g,v)
                for key,p in policies.items():
                    m,_,_=calculate(p,g,v,ids,edges,0)
                    bone=np.abs(np.linalg.norm(p[:,b]-p[:,a],axis=-1)/gt_length-1)
                    speeds=lambda y:np.linalg.norm(np.diff(y,axis=0),axis=-1).mean(1)
                    active=np.any(abs(p-c)>1e-5,axis=(1,2))
                    err=errors(p,g,v)
                    rows.append(dict(**tag,policy=key,jpe=m['global_joint_error_raw'],rr=m['root_relative_error_22_raw'],
                         pelvis=m['pelvis_position_error_raw'],fde=m['frame_displacement_error_raw'],bone_pp=100*bone.mean(),
                         motion=m['motion_magnitude_ratio_pct'],rr_motion=m['root_relative_motion_ratio_pct'],spread=m['joint_spread_ratio_pct'],
                         fast_motion_pct=100*speeds(p)[fast[1:]].mean()/speeds(g)[fast[1:]].mean(),
                         uncorrupted_active=active[~dirty].mean(),uncorrupted_harmful=(err>base)[~dirty].mean(),
                         harmful=(err>base).mean(),cpu_ms=times.get(key)))
                err2=errors(phase2,g,v)-base
                err3=errors(policies['phase3'],g,v)-base
                for j,f in enumerate(ids):
                    frames.append(dict(**tag,frame=f,dirty=int(dirty[j]),fast=int(fast[j]),max_score=primary_score[j].max(),
                                       local_edges=int(np.count_nonzero(primary_offset[j])),phase2_jpe_delta=err2[j],phase3_jpe_delta=err3[j],
                                       selected_children=' '.join(map(str,b[primary_offset[j]!=0]))))
                # Validate the necessity signal against an always-on alternative, including rejected edges.
                for j,(parent,child) in enumerate(edges):
                    on_delta=np.linalg.norm(policies['on'][:,child]-g[:,child],axis=-1)-np.linalg.norm(c[:,child]-g[:,child],axis=-1)
                    for high in (False,True):
                        mask=(primary_score[:,j]>1)==high
                        edge_rows.append(dict(**tag,parent=parent,child=child,above_threshold=high,n=int(mask.sum()),
                            on_child_benefits=int((on_delta[mask]<0).sum()),on_child_harms=int((on_delta[mask]>0).sum()),
                            corrected=int(np.count_nonzero(primary_offset[mask,j])),mean_abs_raw_offset=np.mean(abs(primary_offset[:,j]))))
    for name,table in [('policies',rows),('frames',frames),('edges',edge_rows),('prior_diagnostic',identity_rows)]:
        write_csv(out/(name+'.csv'),table)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,metric in zip(axes,['rr','fde','bone_pp']):
        delta=[a[metric]-b[metric] for a,b in zip(rows[4::9],rows[3::9])]
        ax.bar(range(15),delta,color=['#ad463b' if d>0 else '#357965' for d in delta])
        ax.axhline(0,color='black',lw=.6);ax.set(title=metric,xlabel='condition index (CSV order)',ylabel='Phase 3 minus Phase 2')
    fig.tight_layout();fig.savefig(out/'comparison.png');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('archive','inputs','candidate','out'):parser.add_argument('--'+name,type=Path,required=True)
    run(**vars(parser.parse_args()))
