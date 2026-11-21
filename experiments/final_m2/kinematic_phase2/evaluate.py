"""Run from the repository root with python -m experiments.final_m2.kinematic_phase2.evaluate."""
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
from .refine import Refiner,train_lengths
from .oracle import errors,paths

RESULT_SHA='6d8e29d88a4e6b1735244b5f302ecf26159841b87d3c7c7a39871312a32d82ee'
SPLITS={'opening130':130,'gunplay47':47,'unused_motion_preselected':57}
CONDITIONS=('clean','jitter','spike','short_missing','long_missing')


def run(archive,inputs,candidate,out):
    out.mkdir(exist_ok=False,parents=True)
    def check(ok, key):
        if not ok: raise ValueError(key)
    check(sha256(archive.read_bytes()).hexdigest()==RESULT_SHA,'result archive')
    rows,oracles,frames,associations=[],[],[],[]
    with ZipFile(archive) as z:
        checks=json.loads(z.read('FILE_SHA256.json'))
        for name,h in checks.items():
            check(sha256(z.read(name)).hexdigest()==h,name)
        identity=json.loads(z.read('results/summary.json'))
        trace=list(csv.DictReader(io.StringIO(z.read('results/trace.csv').decode())))
        def read_input(name):
            path=candidate if name=='unused_motion_preselected' else inputs/(name+'.npz')
            check(sha256(path.read_bytes()).hexdigest()==identity['inputs'][name],name)
            with np.load(path,allow_pickle=False) as data:
                return {k:data[k] for k in data}
        train=read_input('train300')
        edges=np.asarray(json.loads((Path(__file__).parents[1]/'dataset_scaling_300/data_selection_manifest.json').read_text())['edges'])
        lengths=train_lengths(train['target_raw'][:,-1],train['valid'][:,-1],edges)
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
                good=errors(i,g,v)<errors(c,g,v)
                tag=dict(split=split,condition=cond)
                for group,mask in [('corrupted',dirty),('uncorrupted',~dirty),('clean_fast',fast&~dirty),('gate_on',sel),('gate_off',~sel)]:
                    associations.append(dict(**tag,group=group,n=int(mask.sum()),beneficial=int(good[mask].sum())))
                refiner=Refiner(edges,lengths); start=perf_counter_ns()
                new=np.stack([refiner.push(q,f,p,obs) for q,f,p,obs in zip(seq,ids,c,x)])
                ms=(perf_counter_ns()-start)/1e6/n
                start=perf_counter_ns(); ema=c.copy()
                for j in range(1,n):ema[j]=.75*c[j]+.25*ema[j-1]
                ema_ms=(perf_counter_ns()-start)/1e6/n
                models=np.stack([c,i])
                policies={'control':c,'improved':i,'switch':s,'ema':ema,'kinematic':new}
                for key,path in paths(models,g,v).items():policies[key]=models[path,np.arange(n)]
                for key,p in policies.items():
                    m,_,_=calculate(p,g,v,ids,edges,0)
                    a,b=edges.T
                    bone=np.abs(np.linalg.norm(p[:,b]-p[:,a],axis=-1)/np.linalg.norm(g[:,b]-g[:,a],axis=-1)-1)
                    fde=np.r_[np.nan,np.linalg.norm(np.diff(p-g,axis=0),axis=-1).mean(1)]
                    active=np.any(abs(p-c)>1e-5,axis=(1,2))
                    speeds=lambda y:np.linalg.norm(np.diff(y,axis=0),axis=-1).mean(1)
                    row=dict(**tag,policy=key,jpe=m['global_joint_error_raw'],rr=m['root_relative_error_22_raw'],pelvis=m['pelvis_position_error_raw'],
                             fde=m['frame_displacement_error_raw'],motion=m['motion_magnitude_ratio_pct'],rr_motion=m['root_relative_motion_ratio_pct'],
                             bone_pp=100*bone.mean(),spread=m['joint_spread_ratio_pct'],fast_motion_pct=100*speeds(p)[fast[1:]].mean()/speeds(g)[fast[1:]].mean(),
                             uncorrupted_active=float(active[~dirty].mean()),uncorrupted_harmful=float((errors(p,g,v)>errors(c,g,v))[~dirty].mean()),
                             switch_fde=float(fde[sw].mean()) if sw.any() else None,post_ms={'kinematic':ms,'ema':ema_ms,'control':0,'improved':0}.get(key))
                    (oracles if key.endswith('oracle') else rows).append(row)
                for j in range(n):
                    frames.append(dict(**tag,frame=int(ids[j]),fast=int(fast[j]),dirty=int(dirty[j]),switch=int(sw[j]),beneficial=int(good[j]),
                        new_minus_switch_rr=float((np.linalg.norm((new[j]-new[j,:1])-(g[j]-g[j,:1]),axis=1)-np.linalg.norm((s[j]-s[j,:1])-(g[j]-g[j,:1]),axis=1)).mean())))
    for name,table in [('policies',rows),('oracle',oracles),('benefit',associations),('frames',frames)]:
        write_csv(out/(name+'.csv'),table)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,metric in zip(axes,['rr','fde','bone_pp']):
        delta=[a[metric]-b[metric] for a,b in zip(rows[4::5],rows[2::5])]
        ax.bar(range(15),delta,color=['#ad463b' if d>0 else '#357965' for d in delta])
        ax.axhline(0,color='black',lw=.6);ax.set(title=metric,xlabel='condition index (CSV order)',ylabel='kinematic minus switch')
    fig.tight_layout();fig.savefig(out/'comparison.png');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('archive','inputs','candidate','out'):parser.add_argument('--'+name,type=Path,required=True)
    run(**vars(parser.parse_args()))
