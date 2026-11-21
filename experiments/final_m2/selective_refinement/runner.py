"""Frozen paired Window8 / DDIM10 evaluation, one sequence at a time."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import matplotlib.pyplot as plt
from rewind_m2.models import BodyDenoiser
from rewind_m2.diffusion import DiffusionSchedule
from rewind_m2.streaming import AnchorSampler
from tools.final_m2_real import infer_frame, sha, load, write
from experiments.final_m2.diagnosis_20261111.diagnose import calculate, write_csv
from experiments.final_m2.bone_length_loss_20261117.run_experiment import bone_metrics
from experiments.final_m2.dataset_scaling_300.prepare_data import read_npz, window_metadata
from .policy import Signals, fit, scores, route, corrupt

HERE = Path(__file__).resolve().parent
CONDITIONS = ('clean','jitter','spike','short_missing','long_missing')


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def ranking(labels, values):
    pos = labels.sum()
    neg = len(labels)-pos
    if min(pos,neg)==0:
        return None,None
    tp = fp = auc = ap = 0.
    for level in np.unique(values)[::-1]:
        group = values==level
        hit = labels[group].sum()
        false = group.sum()-hit
        auc += false*(tp+hit/2)/(pos*neg)
        tp += hit
        fp += false
        ap += hit/pos*tp/(tp+fp)
    return float(auc),float(ap)


def frozen(path, arm, device="cuda"):
    ck = torch.load(path,map_location='cpu',weights_only=True)
    cfg,edges = ck['config'],ck['manifest']['edges']
    expected = load(HERE.parent/'dataset_scaling_300/config.json')['model']
    if (ck['arm']!=arm or ck['step']!=800 or cfg!=expected or
        ck['parent_checkpoint_sha256']!=load(HERE.parent/'bone_length_loss_20261117/config.json')['parent_checkpoint_sha256'] or
        edges!=load(HERE.parent/'dataset_scaling_300/data_selection_manifest.json')['edges']):
        raise ValueError('checkpoint contract')
    model = BodyDenoiser(cfg,22,edges)
    model.load_state_dict(ck['model'])
    schedule = DiffusionSchedule(cfg['diffusion_steps'],cfg['beta_start'],cfg['beta_end'])
    if not torch.equal(ck['alpha_bar'],schedule.alpha_bar):
        raise ValueError('schedule mismatch')
    return model.to(device).eval().requires_grad_(False), schedule, edges


@torch.inference_mode()
def predict(adapter, features, sequences, frames):
    model,schedule,_ = adapter
    sampler = AnchorSampler(model,schedule,8,10)
    device = next(model.parameters()).device
    if device.type=='cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    poses = [infer_frame(sampler,x,str(s),int(f),10043,device).cpu().numpy()
             for x,s,f in zip(features,sequences,frames)]
    if device.type=='cuda':
        torch.cuda.synchronize()
    return np.stack(poses), (time.perf_counter()-start)*1000/len(poses)


def signals(edges,x,p,seq,ids):
    extractor = Signals(edges)
    return np.stack([extractor.push(s,f,obs,pose) for obs,pose,s,f in zip(x,p,seq,ids)])


def run(inputs, candidate, results, out):
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise ValueError('Tesla T4 required')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    out.mkdir(parents=True,exist_ok=False)
    hashes = load(results/'checksums.json')
    adapters = []
    for arm in ('control','improved'):
        name = arm+'_800.pt'
        if sha(results/name)!=hashes[name]:
            raise ValueError(name)
        adapters.append(frozen(results/name,arm))
    edges = adapters[0][2]
    paths = {name:inputs/(name+'.npz') for name in ('train300','opening130','gunplay47')}
    paths['unused_motion_preselected'] = candidate
    expected = load(HERE.parent/'dataset_scaling_300/input_checksums.json')
    for name in ('train300','opening130','gunplay47'):
        if sha(paths[name])!=expected[name+'.npz']:
            raise ValueError(name)
    data = {name:read_npz(path) for name,path in paths.items()}
    for d in data.values():
        window_metadata(d)
    if set(data['unused_motion_preselected']['sequence_ids'])!={load(HERE.parent/'bone_length_loss_20261117/config.json')['candidate_sequence']} or not np.array_equal(data['unused_motion_preselected']['frame_ids'],np.arange(57)):
        raise ValueError('Joyful57 identity')
    train = data.pop('train300')
    x,seq,ids = train['features'][:,-1],train['sequence_ids'],train['frame_ids']
    p,_ = predict(adapters[0],x,seq,ids)
    scale,threshold = fit(signals(edges,x,p,seq,ids))
    write(out/'calibration.json',dict(scale=scale.tolist(),threshold=threshold.tolist(),source=sha(paths['train300'])))
    metrics,logs,bones,detectors = [],[],[],[]
    for split,d in data.items():
        seq,ids = d['sequence_ids'],d['frame_ids']
        g,v = d['target_raw'][:,-1],d['valid'][:,-1]
        if len(set(seq))!=1 or not np.all(np.diff(ids)==1):
            raise ValueError('validation sequence')
        speed = np.r_[0,[np.linalg.norm(delta[mask],axis=1).mean() for delta,mask in zip(np.diff(g,axis=0),v[1:]&v[:-1])]]
        fast = speed>=np.quantile(speed[1:],.75)
        for condition in CONDITIONS:
            x,label = corrupt(d['features'][:,-1],condition)
            (control,ms_c),(improved,ms_i) = [predict(a,x,seq,ids) for a in adapters]
            if condition=='clean':
                for arm,pred in [('control',control),('improved',improved)]:
                    name = f'{split}_{arm}_predictions.npz'
                    if sha(results/name)!=hashes[name]:
                        raise ValueError(name)
                    reference = read_npz(results/name)
                    for key,value in [('gt',g),('valid',v),('frame_ids',ids),('sequence_ids',seq)]:
                        np.testing.assert_array_equal(value,reference[key])
                    np.testing.assert_allclose(pred,reference['prediction'],rtol=0,atol=.001)
            start = time.perf_counter()
            feats = signals(edges,x,control,seq,ids)
            mag,multi = scores(feats,scale)
            selected = multi>threshold[1]
            output = route(control,improved,selected)
            gate_ms = 1000*(time.perf_counter()-start)/len(x)
            tag = dict(split=split,condition=condition)
            for mode,score,cut in [('magnitude',mag,threshold[0]),('multi',multi,threshold[1])]:
                auc,ap = ranking(label,score)
                fire = score>cut
                def fraction(mask):
                    return float(fire[mask].mean()) if mask.any() else None
                detectors.append(dict(**tag,mode=mode,auroc=auc,auprc=ap,negative_fcr=fraction(~label),
                                      fast_negative_fcr=fraction(~label & fast),input_sha256=digest(x)))
            for i in range(len(ids)):
                logs.append(dict(**tag,frame=int(ids[i]),corrupted=int(label[i]),fast=int(fast[i]),
                                 magnitude=float(mag[i]),score=float(multi[i]),selected=int(selected[i]),
                                 switch=int(i>0 and selected[i]!=selected[i-1]),
                                 displacement_error_raw=(float(np.linalg.norm((output[i]-output[i-1])-(g[i]-g[i-1]),axis=1)[v[i]&v[i-1]].mean()) if i else None),
                                 features=json.dumps(feats[i].tolist()),control_jpe=float(np.linalg.norm(control[i]-g[i],axis=1)[v[i]].mean()),
                                 improved_jpe=float(np.linalg.norm(improved[i]-g[i],axis=1)[v[i]].mean())))
            for policy,pred,ms in [('none',control,ms_c),('always',improved,ms_i),('selective',output,ms_c+ms_i+gate_ms)]:
                summary,per_edge,_ = calculate(pred,g,v,ids,edges,0)
                summary.update(bone_metrics(dict(prediction=pred,gt=g,valid=v),edges))
                metrics.append(dict(**tag,policy=policy,ms_frame=ms,router_ms=gate_ms if policy=='selective' else 0.,
                                    selected_fraction=float(selected.mean()) if policy=='selective' else float(policy=='always'),**summary))
                bones.extend(dict(**tag,policy=policy,**edge) for edge in per_edge)
    for name,rows in [('metrics',metrics),('trace',logs),('edges',bones),('detection',detectors)]:
        write_csv(out/(name+'.csv'),rows)
    groups = [metrics[i:i+3] for i in range(0,len(metrics),3)]
    def better(group):
        c,a,s = group
        return all(s[k]<min(c[k],a[k]) for k in ('global_joint_error_raw','root_relative_error_22_raw','frame_displacement_error_raw')) and all(abs(s[k]-100)<min(abs(c[k]-100),abs(a[k]-100)) for k in ('motion_magnitude_ratio_pct','root_relative_motion_ratio_pct'))
    for arm in ('control','improved'):
        name = arm+'_800.pt'
        if sha(results/name)!=hashes[name]: raise ValueError('checkpoint changed')
    positive_pairs = [(a,b) for a,b in zip(detectors[::2],detectors[1::2]) if a['auprc'] is not None]
    rq1 = all(b['auprc']>a['auprc'] and b['auroc']>a['auroc'] for a,b in positive_pairs)
    clean_fast = [r for r in detectors if r['condition']=='clean' and r['mode']=='multi']
    rq1 = rq1 and all(r['fast_negative_fcr']<=.05 for r in clean_fast)
    write(out/'summary.json',dict(rq1='PASS' if rq1 else 'FAIL',rq2='PASS' if all(map(better,groups)) else 'FAIL',
         comparisons=[dict(split=g[0]['split'],condition=g[0]['condition'],passed=better(g)) for g in groups],
         revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         checkpoint_sha256={arm:hashes[arm+'_800.pt'] for arm in ('control','improved')},
         inputs={k:sha(v) for k,v in paths.items()},torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),
         corruption_seed=2111,noise_seed=10043))
    fig,ax=plt.subplots(figsize=(10,4))
    for split in data:
        rows=[r for r in logs if r['split']==split and r['condition']=='spike']
        ax.plot([r['score'] for r in rows],label=split)
    ax.axhline(threshold[1],color='black',linestyle='--');ax.set(xlabel='frame',ylabel='gate score')
    ax.legend();fig.tight_layout();fig.savefig(out/'spike_scores.png');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('inputs','candidate','results','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(**vars(parser.parse_args()))
