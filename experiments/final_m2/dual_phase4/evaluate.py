"""Run from the repository root with python -m experiments.final_m2.dual_phase4.evaluate."""
from itertools import groupby
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
from .dual import Dual
from ..kinematic_phase3.gate import Gated,calibrate

RESULT_SHA='6d8e29d88a4e6b1735244b5f302ecf26159841b87d3c7c7a39871312a32d82ee'
SPLITS={'opening130':130,'gunplay47':47,'unused_motion_preselected':57}
CONDITIONS=('clean','jitter','spike','short_missing','long_missing')


def run(archive,inputs,candidate,out):
    out.mkdir(exist_ok=False,parents=True)
    def check(ok, key):
        if not ok: raise ValueError(key)
    check(sha256(archive.read_bytes()).hexdigest()==RESULT_SHA,'result archive')
    rows,frames,edge_rows=[],[],[]
    extremes={}
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
                policies={'control':c,'phase2':phase2}
                p3=Gated(edges,lengths,scale)
                start=perf_counter_ns()
                batch=[p3.push(q,f,p,obs) for q,f,p,obs in zip(seq,ids,c,x)]
                times['phase3']=(perf_counter_ns()-start)/1e6/n
                policies['phase3'],score3,offset3=map(np.stack,zip(*batch))
                for mode in ('A','B','combined'):
                    model=Dual(edges,lengths,scale,mode)
                    start=perf_counter_ns()
                    batch=[model.push(q,f,p,obs) for q,f,p,obs in zip(seq,ids,c,x)]
                    times[mode]=(perf_counter_ns()-start)/1e6/n
                    policies[mode]=np.stack([pair[0] for pair in batch])
                    if mode=='combined':
                        need_a,need_b,conflict,offset,shift=map(np.stack,zip(*(pair[1] for pair in batch)))
                a,b=edges.T
                gt_length=np.linalg.norm(g[:,b]-g[:,a],axis=-1)
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
                combined=policies['combined']
                delta=errors(combined,g,v)-base
                local=np.linalg.norm(offset,axis=-1)>1e-6
                moved=np.linalg.norm(shift,axis=-1)>1e-6
                for t,f in enumerate(ids):
                    frames.append(dict(**tag,frame=f,fast=int(fast[t]),dirty=int(dirty[t]),
                        persistent=int(need_a[t].sum()),transient=int(need_b[t].sum()),both=int((need_a[t]&need_b[t]).sum()),
                        conflict=int(conflict[t].sum()),p3_off_residual=int(((score3[t]<=1)&(abs(offset3[t])>1e-7)).sum()),
                        p3_missed_persistent=int(((score3[t]<=1)&need_a[t]).sum()),local_edges=int(local[t].sum()),
                        moved_joints=int(moved[t].sum()),jpe_delta=delta[t]))
                for j,(parent,child) in enumerate(edges):
                    descendants={child}
                    for u,w in edges:
                        if u in descendants: descendants.add(w)
                    group=sorted(descendants)
                    residual=combined[:,group]-g[:,group]
                    # Counterfactual: remove just this edge's applied translation on its descendants.
                    benefit=(np.linalg.norm(residual-offset[:,j,None,:],axis=-1)-np.linalg.norm(residual,axis=-1)).mean(1)
                    active=local[:,j]
                    edge_rows.append(dict(**tag,parent=parent,child=child,descendants=len(group),active=active.sum(),
                        benefit=((benefit>1e-6)&active).sum(),harm=((benefit< -1e-6)&active).sum(),
                        summed_subtree_jpe_gain=benefit[active].sum(),max_consecutive=max((len(list(run)) for flag,run in groupby(active) if flag),default=0)))
                for label,idx in [('best',np.argmin(delta)),('worst',np.argmax(delta))]:
                    val=float(delta[idx])
                    if label not in extremes or (val-extremes[label][0])*(1 if label=='worst' else -1)>0:
                        extremes[label]=(val,split,cond,int(ids[idx]),c[idx],combined[idx],g[idx])
    for name,table in [('policies',rows),('frames',frames),('edges',edge_rows)]:
        write_csv(out/(name+'.csv'),table)
    fig=plt.figure(figsize=(10,7))
    for row,(label,(err,split,cond,frame,*poses)) in enumerate(sorted(extremes.items())):
        lo=np.min(poses,axis=(0,1));hi=np.max(poses,axis=(0,1));mid=(lo+hi)/2;radius=max(hi-lo)*.55
        for col,(pose,title) in enumerate(zip(poses,['Control','Combined','GT'])):
            ax=fig.add_subplot(2,3,1+row*3+col,projection='3d')
            for a,b in edges:ax.plot(*pose[[a,b]].T,color='#37658a')
            for setter,center in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],mid):setter(center-radius,center+radius)
            ax.set_box_aspect((1,1,1));ax.set_title(f'{title} | {label} {err:+.3f}\n{split} {cond} f{frame}',fontsize=9)
    fig.tight_layout();fig.savefig(out/'skeletons.png');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('archive','inputs','candidate','out'):parser.add_argument('--'+name,type=Path,required=True)
    run(**vars(parser.parse_args()))
