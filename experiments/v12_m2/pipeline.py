"""Frozen v1.2 cascade with an explicit device-mm body-to-hand boundary."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from rewind_m2.diffusion import DiffusionSchedule
from rewind_m2.streaming import keyed_frame_noise
from rewind_m3.models import BoundHand
from rewind_v1.data import attach_predicted_hands
from rewind_v1.frontend import predicted_features
from scripts.visualize_archvis_v1 import _group_names
from experiments.final_m2.kinematic_phase2.refine import Refiner
from experiments.final_m2.kinematic_phase3.gate import Gated
from experiments.final_m2.dual_phase4.dual import Dual

ROOT = Path(__file__).resolve().parents[2]
POLICIES = ('baseline', 'phase2', 'phase3', 'phase4')


def layout(spec):
    pinned = (ROOT/'configs/real_archvis62_v1.json').read_bytes()
    if hashlib.sha256(pinned).hexdigest() != '32f1cfda25bc2b05745d2b534f0fd21eec0defbd379cd1d865c152e52c74b187' or spec != json.loads(pinned):
        raise ValueError('v1.2 spec identity')
    body = spec['body_joint_names']
    edges = [[body.index(a), body.index(b)] for a,b in spec['edges'] if a in body and b in body]
    old = json.loads((ROOT/'experiments/final_m2/dataset_scaling_300/data_selection_manifest.json').read_text())
    if body != old['joint_names'] or edges != old['edges'] or body[0] != 'pelvis':
        raise ValueError('directed Body22 mapping')
    return np.array(edges)


def observations(features, spec):
    indices = [spec['joint_names'].index(n) for n in spec['body_joint_names']]
    v = features.reshape(len(features),62,2,7)[:,indices].detach().cpu().numpy().copy()
    if (not np.isfinite(v).all() or np.any(abs(v[...,:2])>1.00001)
            or not np.all(v[...,2]==1) or not np.all(v[...,4]==1)
            or not np.all(v[...,5:]==0) or np.any((v[...,3]<0)|(v[...,3]>1))):
        raise ValueError('predicted stereo feature contract')
    v[...,:2] = (v[...,:2]+1)*(1023/(2*1024))
    return v


def corrected(body, obs, keys, policy, edges, prior):
    if policy not in POLICIES:
        raise ValueError('unknown policy')
    if body.shape != (len(keys),22,3) or not torch.isfinite(body).all():
        raise ValueError('body contract')
    if policy == 'baseline':
        return body
    if prior is None:
        raise ValueError('refinement disabled: missing train calibration')
    lengths, scale = map(np.asarray,prior)
    for x in (lengths,scale):
        if x.shape != (21,) or not np.isfinite(x).all() or np.any(x<=0):
            raise ValueError('invalid mm length or innovation scale')
    if obs.shape != (len(keys),22,2,7) or not np.isfinite(obs).all() or not np.isin(obs[...,2],[0,1]).all():
        raise ValueError('refiner observation contract')
    model = Refiner(edges,lengths) if policy=='phase2' else (Gated if policy=='phase3' else Dual)(edges,lengths,scale)
    rows = []
    for (seq,frame),pose,xy in zip(keys,body.detach().cpu().numpy(),obs):
        result = model.push(seq,frame,pose,xy)
        rows.append(result if policy=='phase2' else result[0])
    out = torch.as_tensor(np.stack(rows),device=body.device,dtype=body.dtype)
    if not torch.isfinite(out).all() or not torch.equal(out[:,0],body[:,0]):
        raise ValueError('invalid corrected pelvis/pose')
    return out


@torch.inference_mode()
def frontend(models, images):
    if (images.dtype != torch.float32 or images.shape[1:] != (2,3,128,128)
            or not 1<=len(images)<=50 or not torch.isfinite(images).all() or images.min()<0 or images.max()>1):
        raise ValueError('stereo RGB contract')
    return torch.cat([predicted_features(models['frontend'],x.to(models['device'])).cpu() for x in images.split(2)])


@torch.inference_mode()
def compare(models, features, camera, keys, policies=POLICIES, prior=None, hand_ablation=False):
    spec,device = models['spec'], models['device']
    edges = layout(spec)
    n = len(keys)
    if (not 1<=n<=50 or features.shape!=(n,62*14) or camera.shape!=(n,2,9) or
            camera.dtype!=torch.float32 or features.dtype!=torch.float32 or not torch.isfinite(camera).all()):
        raise ValueError('clip contract')
    u,v = camera[...,:3],camera[...,3:6]
    if any(not torch.allclose(x.norm(dim=-1),torch.ones_like(x[...,0]),atol=1e-5,rtol=0) for x in (u,v)) or (u*v).sum(-1).abs().max()>1e-5:
        raise ValueError('absolute camera rotation')
    if any(k!=(keys[0][0],keys[0][1]+i) for i,k in enumerate(keys)):
        raise ValueError('supply each contiguous sequence separately')
    if any(models[r].training or any(p.requires_grad for p in models[r].parameters()) for r in ('frontend','body','hand')):
        raise ValueError('requires frozen models')
    obs=observations(features,spec)
    cfg=models['config']['teacher']
    schedule=DiffusionSchedule(cfg['diffusion_steps'],cfg['beta_start'],cfg['beta_end'])
    feat=features.reshape(1,n,62,14).to(device)
    cam=camera[None].to(device)
    valid=torch.ones((1,n),dtype=torch.bool,device=device)
    scale=cfg['target_scale_mm']
    def sample(role,model,offset):
        names=_group_names(spec,role)
        indices=[spec['joint_names'].index(name) for name in names]
        noise=torch.stack([keyed_frame_noise(seq,fr,len(names),models['config']['seed']+offset) for seq,fr in keys])[None].to(device)
        return schedule.sample(model,feat[...,indices,:].flatten(-2),cam,valid,noise,cfg['sampling_steps'])*scale
    def sync():
        if torch.device(device).type=='cuda':
            torch.cuda.synchronize(device)
    body=sample('body',models['body'],100)[0]
    upper_indices=[spec['body_joint_names'].index(n) for n in spec['upper_joint_names']]
    names=spec['body_joint_names']+_group_names(spec,'hand')
    order=[names.index(n) for n in spec['joint_names']]
    outputs,timings={},{}
    for policy in policies:
        sync()
        start=time.perf_counter()
        refined=corrected(body,obs,keys,policy,edges,prior)
        sync()
        timings[policy]=(time.perf_counter()-start)*1000/n
        for original in ([False,True] if hand_ablation and policy!='baseline' else [False]):
            anchor=body if original else refined
            upper=anchor[None,:,upper_indices,:]
            bound=BoundHand(models['hand'],upper/scale,valid[...,None].expand(upper.shape[:-1])).eval()
            local=sample('hand',bound,200)[0]
            hand=attach_predicted_hands(anchor,local,spec)
            whole=torch.cat((refined,hand),dim=1)[:,order]
            result=dict(body=refined,hand_local=local,hand_global=hand,whole=whole)
            if any(x.shape!=(n,j,3) or not torch.isfinite(x).all() for x,j in zip(result.values(),(22,40,40,62))):
                raise ValueError('invalid output skeleton')
            outputs[policy+('_original_hand' if original else '')]={k:t.cpu() for k,t in result.items()}
    return outputs,timings
