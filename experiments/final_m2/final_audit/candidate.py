import argparse
from pathlib import Path
import numpy as np
import torch
from ..dataset_scaling_100.prepare_data import convert, coordinate_check, motion
from ..dataset_scaling_300.prepare_data import read_npz,window_metadata
from ..selective_refinement.runner import frozen,predict
from ..kinematic_phase2.refine import Refiner
from ..kinematic_phase3.gate import calibrate,Gated
from ..dual_phase4.dual import Dual
from ..diagnosis_20261111.diagnose import calculate
from rewind_data.audit import read_json
from tools.final_m2_real import load,write,sha

HERE=Path(__file__).parent
SEQ='CityPark_Showcase/Day/rp_sophia_rigged_003_ue4/SKM_MenReadingGlasses_Shape_01/024/Taunt__2_'
CK='a70d9efbce4294323e2eca735155a20c12068a9d1e4215ae3dc12f1f862a2185'

def run(raw,inputs,joyful,out,checkpoint=None):
    out.mkdir(parents=True,exist_ok=False)
    prior=HERE.parent/'dataset_scaling_300'
    man=load(prior/'data_selection_manifest.json')
    files=load(HERE/'raw_manifest.json');rows=[];coord=[]
    hashes={r['pose_sha256'] for r in man['frames']+man['additional_evaluation']['gunplay47']}
    for f in files:
        path=raw/f['title']
        if sha(path)!=f['sha256'] or f['sha256'] in hashes:raise ValueError('source identity')
        hashes.add(f['sha256']);doc=read_json(path.read_bytes())
        coord.append(coordinate_check(doc,man['joint_names']))
        rows.append((SEQ,len(rows),doc))
    d=convert(rows,man['joint_names']);window_metadata(d)
    for name in ('train300','opening130','gunplay47','joyful57'):
        path=joyful if name=='joyful57' else inputs/(name+'.npz')
        expected='430e0836def41304b57c379dd78f7ac16a16012fc4ff1a38718778f20974d76b' if name=='joyful57' else load(prior/'input_checksums.json')[name+'.npz']
        if sha(path)!=expected:raise ValueError('reference sha')
        ref=read_npz(path)
        if motion(SEQ) in {motion(str(seq)) for seq in ref['sequence_ids']}:raise ValueError('motion overlap')
        for key in ('features','target_raw'):
            seen={v.tobytes() for v in ref[key][:,-1]}
            if any(v.tobytes() in seen for v in d[key][:,-1]):raise ValueError('duplicate anchor')
        if name=='train300':train=ref
    np.savez_compressed(out/'taunt16.npz',**d)
    write(out/'candidate_audit.json',dict(sequence=SEQ,frames=d['frame_ids'].tolist(),input_sha256=sha(out/'taunt16.npz'),
          source_verified=len(files),reference_frames=534,exact_overlap=0,coordinate_max={k:max(r[k] for r in coord) for k in coord[0]},mocap_independence='NOT VERIFIED'))
    if checkpoint is None:return
    if sha(checkpoint)!=CK or not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():raise ValueError('requires frozen Control800 and T4')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    adapter=frozen(checkpoint,'control')
    x=d['features'][:,-1];ids=d['frame_ids'];edges=np.array(adapter[2]);g=d['target_raw'][:,-1];v=d['valid'][:,-1]
    c,_=predict(adapter,x,d['sequence_ids'],ids);length,scale=calibrate(train,edges)
    policies={'control':c}
    for name,model in [('phase2',Refiner(edges,length)),('phase3',Gated(edges,length,scale)),('phase4',Dual(edges,length,scale))]:
        preds=[model.push(SEQ,int(f),p,obs) for f,p,obs in zip(ids,c,x)]
        policies[name]=np.stack(preds if name=='phase2' else [r[0] for r in preds])
    speed=lambda p:np.linalg.norm(np.diff(p,axis=0),axis=-1).mean(1)
    fast=speed(g)>=np.quantile(speed(g),.75);metrics={}
    a,b=edges.T
    for name,p in policies.items():
        m,_,_=calculate(p,g,v,ids,edges,0)
        metrics[name]=dict(jpe=m['global_joint_error_raw'],rr=m['root_relative_error_22_raw'],fde=m['frame_displacement_error_raw'],
            bone=100*float(abs(np.linalg.norm(p[:,a]-p[:,b],axis=-1)/np.linalg.norm(g[:,a]-g[:,b],axis=-1)-1).mean()),
            fast_gap=abs(100*float(speed(p)[fast].mean()/speed(g)[fast].mean())-100))
    base=metrics['control']
    for name in ('phase2','phase3','phase4'):
        m=metrics[name];m['pass']=all(m[k]<base[k] for k in ('jpe','rr','fde','bone')) and m['fast_gap']<=base['fast_gap']
    write(out/'evaluation.json',metrics);np.savez_compressed(out/'poses.npz',**policies,gt=g,valid=v,frame_ids=ids)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for k in ('raw','inputs','joyful','out','checkpoint'):parser.add_argument('--'+k,type=Path,required=k!='checkpoint')
    run(**vars(parser.parse_args()))
