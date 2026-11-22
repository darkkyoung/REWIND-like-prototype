"""Pinned ArchVis train calibration and fixed validation comparison. No training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scripts.v12_share_smoke import load_models, predict, verify_manifest
from rewind_m2.streaming import keyed_frame_noise
from rewind_v1.full_contract import load_audit, training_plan
from rewind_v1.full_cache import PreparationBudget, _frame
from rewind_v1.data import _hand_indices
from experiments.final_m2.kinematic_phase3.gate import calibrate
from .pipeline import POLICIES, compare, frontend, layout, observations


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def fingerprint(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def metrics(result, baseline, gt, valid, spec):
    if gt.shape!=result['whole'].shape or valid.dtype!=torch.bool or valid.shape!=gt.shape[:2] or not torch.isfinite(gt[valid]).all():
        raise ValueError('invalid evaluation GT/mask')
    idx=[spec['joint_names'].index(n) for n in spec['body_joint_names']]
    fingers,wrists=_hand_indices(spec['joint_names'],spec)
    p,g,v=result['body'],gt[:,idx],valid[:,idx]
    norm=lambda x:x.double().norm(dim=-1)
    def avg(x,mask):
        return x[mask].mean().item() if mask.any() else None
    error=norm(p-g)
    a,b=layout(spec).T
    gl=norm(g[:,a]-g[:,b])
    dp,dg=p.diff(dim=0),g.diff(dim=0)
    pair=v[1:]&v[:-1]
    changed=(p!=baseline).any(-1)
    fast,truth=avg(norm(dp),pair),avg(norm(dg),pair)
    with_gt=v.any(-1)
    worsened=((error-norm(baseline-g)).masked_fill(~v,0).sum(-1)>0)
    center=(p*v[...,None]).sum(1)/v.sum(1,keepdim=True).clamp_min(1)
    return dict(body_mpjpe_mm=avg(error,v),rr_jpe_mm=avg(norm((p-p[:,:1])-(g-g[:,:1])),v&v[:,:1]),
        pelvis_mm=avg(error[:,0],v[:,0]),bone_distortion_pct=avg(100*(norm(p[:,a]-p[:,b])/gl-1).abs(),v[:,a]&v[:,b]&(gl>0)),
        body_spread_mean_radius_mm=avg(norm(p-center[:,None]),v),fde_mm_per_frame=avg(norm(dp-dg),pair),
        motion_ratio_pct=100*fast/truth if truth else None,
        wrist_mm=avg(error[:,[idx.index(wrists[0]),idx.index(wrists[-1])]],valid[:,[wrists[0],wrists[-1]]]),
        hand_local_mpjpe_mm=avg(norm(result['hand_local']-(gt[:,fingers]-gt[:,wrists])),valid[:,fingers]&valid[:,wrists]),
        hand_global_mpjpe_mm=avg(norm(result['hand_global']-gt[:,fingers]),valid[:,fingers]),
        whole_mpjpe_mm=avg(norm(result['whole']-gt),valid),changed_frames=int(changed.any(1).sum()),
        changed_joint_frames=int(changed.sum()),changed_distinct_joints=int(changed.any(0).sum()),
        harmed_frame_fraction=avg(worsened.double(),with_gt))


def animation(path, outputs, spec):
    # Fixed orthographic view for all frames and policies; include depth in the projection.
    projected={name:x['whole'].numpy()@np.array([[.87,.16],[0.,-.95],[.5,-.28]]) for name,x in outputs.items() if name in POLICIES}
    flat=np.concatenate(list(projected.values())).reshape(-1,2)
    center=(flat.max(0)+flat.min(0))/2
    factor=215/max(float(np.ptp(flat,axis=0).max()),1)
    edges=[tuple(spec['joint_names'].index(n) for n in e) for e in spec['edges']]
    frames=[]
    for t in range(len(next(iter(projected.values())))):
        im=Image.new('RGB',(250*len(projected),265),'white');draw=ImageDraw.Draw(im)
        for i,(name,xy) in enumerate(projected.items()):
            pts=(xy[t]-center)*factor+[125+250*i,140]
            draw.text((10+250*i,8),f'{name}  +{t}',fill='black')
            for a,b in edges:
                draw.line([tuple(pts[a]),tuple(pts[b])],fill='#275d70',width=2)
        frames.append(im)
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=120,loop=0)


def run(package,audit,rgb,pose,out,device='cuda',baseline_only=False):
    torch.set_num_threads(1)
    if device=='cuda' and (not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name()):
        raise ValueError('Tesla T4 required')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    models=load_models(package,device)
    spec=models['spec'];edges=layout(spec)
    manifest,coverage,_=load_audit(audit)
    plan=training_plan(manifest,coverage)
    binding=json.loads((package/'metadata/run.json').read_text())['input']
    if (binding['plan_sha256']!=plan['content_sha256'] or binding['manifest_sha256']!=manifest['content_sha256']):
        raise ValueError('training split binding')
    rows={(r['sequence_id'],r['frame_index']):r for r in manifest['records']}
    budget=PreparationBudget()
    receipt=[]
    def read(seq,start,count,split):
        images,metadata,keys=[],[],[]
        for f in range(start,start+count):
            row=rows[seq,f]
            if row['split']!=split:
                raise ValueError('split mismatch')
            raw={role:budget.read_source(pose if role=='pose' else rgb,pin) for role,pin in row['files'].items()}
            img,meta=_frame(raw,f,spec)
            images.append(img);metadata.append(meta);keys.append((seq,f))
            receipt.append(dict(key=[seq,f],split=split,files=row['files']))
        return torch.stack(images),{k:torch.stack([m[k] for m in metadata]) for k in metadata[0]},keys
    out.mkdir(parents=True,exist_ok=False)
    prior=None
    if not baseline_only:
        # First complete training clip for each of first three eligible distinct sequences.
        chosen={}
        for row in sorted(plan['train_clips'],key=lambda r:(r['sequence'],r['start_frame'])):
            if row['real_frames']==50:
                chosen.setdefault(row['sequence'],row['start_frame'])
        xs,ys,vs,ks=[],[],[],[]
        indices=[spec['joint_names'].index(n) for n in spec['body_joint_names']]
        for seq,start in list(chosen.items())[:3]:
            images,meta,keys=read(seq,start,50,'train')
            xs.extend(observations(frontend(models,images),spec))
            ys.extend(meta['target_mm'][:,indices].numpy());vs.extend(meta['target_valid'][:,indices].numpy());ks.extend(keys)
        if len(ks)!=150:
            raise ValueError('incomplete train-only calibration')
        prior=calibrate(dict(target_raw=np.array(ys)[:,None],valid=np.array(vs)[:,None],features=np.array(xs)[:,None],
                             sequence_ids=[k[0] for k in ks],frame_ids=[k[1] for k in ks]),edges)
        if not all(np.isfinite(v).all() and (v>0).all() for v in prior):
            raise ValueError('calibration failed')
        write(out/'calibration.json',dict(keys=ks,edges=edges.tolist(),length_mm=prior[0].tolist(),innovation=prior[1].tolist()))
    seq,start=plan['teacher_validation_clips'][0]
    images,meta,keys=read(seq,start,50,'validation')
    reference=predict(models,images,meta['camera'],keys)
    features=frontend(models,images)
    outputs,times=compare(models,features,meta['camera'],keys,('baseline',) if baseline_only else POLICIES,prior,True)
    if any(not torch.equal(v,outputs['baseline'][k]) for k,v in reference.items()):
        raise ValueError('original baseline parity failed')
    results=[]
    arrays={}
    for name,pred in outputs.items():
        row=metrics(pred,reference['body'],meta['target_mm'],meta['target_valid'],spec)
        row.update(policy=name,body_hook_ms_per_frame=times[name.split('_')[0]])
        results.append(row)
        arrays.update({name+'_'+k:v.numpy() for k,v in pred.items()})
    np.savez_compressed(out/'predictions.npz',**arrays,gt=meta['target_mm'].numpy(),valid=meta['target_valid'].numpy(),frames=[k[1] for k in keys])
    write(out/'source_frames.json',receipt)
    package_manifest=verify_manifest(package)
    noise_sha={role:fingerprint(torch.stack([keyed_frame_noise(s,f,j,seed) for s,f in keys]))
               for role,j,seed in [('body',22,171),('hand',40,271)]}
    write(out/'evaluation.json',dict(source=package_manifest['source_revision'],model_sha={k:v['sha256'] for k,v in package_manifest['files'].items() if k.startswith('models/')},
        noise_sha=noise_sha,plan_sha256=plan['content_sha256'],keys=keys,results=results,device=str(device),torch=str(torch.__version__),
        parity='exact',noise_seeds=[171,271],input_sha={k:fingerprint(v) for k,v in dict(images=images,camera=meta['camera'],features=features,body=reference['body']).items()}))
    with (out/'metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=results[0]);writer.writeheader();writer.writerows(results)
    animation(out/'whole62.gif',outputs,spec)
    write(out/'SHA256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('package','audit','rgb','pose','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--baseline-only',action='store_true')
    run(**vars(parser.parse_args()))
