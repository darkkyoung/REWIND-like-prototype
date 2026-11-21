"""Post-hoc Path A attribution. No inference parameters are fitted here."""
import argparse, hashlib, io, json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..dual_phase4.dual import Dual
from ..kinematic_phase2.evaluate import RESULT_SHA, SPLITS, CONDITIONS

HERE=Path(__file__).parent

def audit(archive,out):
    out.mkdir(parents=True,exist_ok=False)
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==RESULT_SHA
    cal=json.loads((HERE.parent/'dual_phase4/results/calibration.json').read_text())
    L=np.array(cal['lengths']); rows=[]; frames=[]; example=None
    reference=pd.read_csv(HERE.parent/'dual_phase4/results/policies.csv').query("policy=='A'").set_index(['split','condition'])
    with ZipFile(archive) as z:
        for name,h in json.loads(z.read('FILE_SHA256.json')).items():
            assert hashlib.sha256(z.read(name)).hexdigest()==h
        for split in SPLITS:
            for cond in CONDITIONS:
                with np.load(io.BytesIO(z.read(f'evidence/{split}_{cond}.npz'))) as d:
                    pred,gt,ids,edges=[d[k] for k in ('control','gt','ids','edges')]
                    assert d['valid'].all()
                model=Dual(edges,L,np.array(cal['scale']),mode='A')
                # A reads only length history. Replay parity with published A is checked below.
                obs=np.zeros((22,2,7));obs[:,:,2]=1
                err=[]
                for t,(frame,p,g) in enumerate(zip(ids,pred,gt)):
                    y,(need,_,_,offset,_)=model.push(split,int(frame),p,obs)
                    base=np.linalg.norm(p-g,axis=1).mean()
                    jpe=np.linalg.norm(y-g,axis=1).mean();err.append(jpe)
                    frames.append((split,cond,frame,jpe-base,int(need.sum())))
                    if cond=='clean' and (example is None or jpe-base>example[0]):
                        example=(jpe-base,split,int(frame),p.copy(),y.copy(),g.copy(),edges.copy())
                    for j,(a,b) in enumerate(edges):
                        if not model.allowed[j] or t<4:continue
                        subtree={b}
                        for u,v in edges:
                            if u in subtree:subtree.add(v)
                        group=sorted(subtree);residual=y[group]-g[group]
                        gain=(np.linalg.norm(residual-offset[j],axis=1)-np.linalg.norm(residual,axis=1)).mean()
                        vec=p[b]-p[a];length=np.linalg.norm(vec)
                        # Same 1%-prior infinitesimal-like finite probe for both proposed and rejected edges.
                        step=.01*L[j]*np.sign(L[j]-length)*vec/max(length,1e-8)
                        pre=p[group]-g[group]
                        probe=(np.linalg.norm(pre,axis=1)-np.linalg.norm(pre+step,axis=1)).mean()
                        rows.append((split,cond,frame,a,b,len(group),bool(need[j]),float(gain),float(probe)))
                assert abs(np.mean(err)-reference.loc[(split,cond),'jpe'])<1e-4
    edges_df=pd.DataFrame(rows,columns='split condition frame parent child descendants persistent actual_gain probe_gain'.split())
    edges_df.to_csv(out/'edge_frame.csv.gz',index=False)
    stats=edges_df.assign(probe_help=edges_df.probe_gain>1e-6,actual_help=edges_df.actual_gain>1e-6)
    stats.groupby(['split','persistent']).agg(n=('frame','size'),probe_help=('probe_help','mean'),actual_help=('actual_help','mean'),gain=('actual_gain','mean')).to_csv(out/'persistence.csv')
    stats[stats.persistent].groupby(['split','parent','child','descendants']).actual_gain.agg(['size','mean','sum']).to_csv(out/'links.csv')
    pd.DataFrame(frames,columns='split condition frame delta_jpe proposals'.split()).to_csv(out/'frames.csv',index=False)
    delta,split,idx,*rest=example
    poses=rest[:3]; edges=rest[-1]
    fig=plt.figure(figsize=(10,3.8));lo=np.min(poses,axis=(0,1));hi=np.max(poses,axis=(0,1));mid=(hi+lo)/2;r=max(hi-lo)*.54
    for k,(pose,title) in enumerate(zip(poses,['Control800','Path A','Ground truth'])):
        ax=fig.add_subplot(1,3,k+1,projection='3d')
        for a,b in edges:ax.plot(*pose[[a,b]].T,color=['#426481','#b15b20','#386b54'][k])
        for setter,m in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),mid):setter(m-r,m+r)
        ax.set_box_aspect((1,1,1));ax.set_title(title)
    fig.suptitle(f'Clean case: {split} / frame {idx} / JPE change {delta:+.3f} raw')
    fig.tight_layout();fig.savefig(out/'clean_failure.png',dpi=150);plt.close(fig)
    m=pd.read_csv(HERE.parent/'dual_phase4/results/policies.csv').groupby('policy').mean(numeric_only=True)
    fig,axs=plt.subplots(1,2,figsize=(8,3.6))
    for ax,x in zip(axs,('rr','bone_pp')):
        for name,row in m.iterrows():
            ax.scatter(row[x],row.fde);ax.annotate(name,(row[x],row.fde),xytext=(2,3),textcoords='offset points',fontsize=8)
        ax.margins(.18);ax.set(xlabel={'rr':'Pelvis-relative JPE (raw)','bone_pp':'Bone distortion (pp)'}[x],ylabel='FDE (raw)')
    fig.tight_layout();fig.savefig(out/'tradeoff.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('out',type=Path);a=p.parse_args();audit(a.archive,a.out)
