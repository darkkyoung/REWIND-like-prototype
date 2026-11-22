"""Warm start a fresh Body optimizer. Original v1.2 loader remains untouched."""
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median

import torch
from rewind_v1.teacher import motion_losses
from rewind_m2.diffusion import DiffusionSchedule

SEED=20261122
STEPS=24
FORMAT='v12_rr_warm_start_v1'
PARENT='94a072551fd3da40621da5a589fbc3d6f919d3e7b52d29ea1c7f6099ef681305'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def state_hash(model):
    return hashlib.sha256(json.dumps([(k,tensor_hash(v)) for k,v in model.state_dict().items()]).encode()).hexdigest()

def rr_loss(x,target,valid,time,scale):
    if (x.shape!=target.shape or x.ndim!=4 or x.shape[-2:]!=(22,3) or valid.shape!=x.shape[:-1]
            or time.shape!=x.shape[:2] or valid.dtype!=torch.bool or time.dtype!=torch.bool or not math.isfinite(scale) or scale<=0):
        raise ValueError('RR input contract')
    mask=valid & time[...,None]
    if not torch.isfinite(x).all() or not torch.isfinite(target[mask]).all():
        raise ValueError('nonfinite x0 or valid target')
    pairs=mask[...,1:] & mask[...,:1]
    if not pairs.any():
        return x.reshape(-1)[:0].sum()
    gt=torch.where(mask[...,None],target,0.)/scale
    residual=(x[...,1:,:]-x[...,:1,:])-(gt[...,1:,:]-gt[...,:1,:])
    return residual[pairs].square().sum(-1).mean()

def objective(existing,rr,weight):
    if not math.isfinite(weight) or weight<0:
        raise ValueError('invalid weight')
    return existing if weight==0 else existing+weight*rr

def draw(step,n,calibration=False):
    offset=1000000 if calibration else 0
    order=torch.randperm(n,generator=torch.Generator().manual_seed(SEED+offset+step//n))
    return int(order[step%n]), torch.Generator().manual_seed(SEED+offset+10000+step)

def forward_terms(body,batch,cfg,generator):
    if batch['split']!='train':
        raise ValueError('requires training split')
    keys=batch['keys']
    if not keys or any(k!=(keys[0][0],keys[0][1]+i) for i,k in enumerate(keys)):
        raise ValueError('gap or sequence boundary')
    device=next(body.parameters()).device
    target=batch['target'][:,:22].to(device)[None]
    valid=batch['valid'][:,:22].to(device)[None]
    time=torch.ones(target.shape[:2],dtype=torch.bool,device=device)
    scale=cfg['target_scale_mm']
    clean=torch.where(valid[...,None],target/scale,0.)
    k=torch.randint(1,cfg['diffusion_steps']+1,(1,),generator=generator)
    noise=torch.randn(clean.shape,generator=generator)
    features=batch['features'].reshape(len(keys),62,14)[:,:22].flatten(1)[None].to(device)
    camera=batch['camera'][None].to(device)
    schedule=DiffusionSchedule(cfg['diffusion_steps'],cfg['beta_start'],cfg['beta_end'])
    noisy=schedule.q_sample(clean,noise.to(device),k.to(device))
    x,frame=body(noisy,features,camera,k.to(device),time)
    old,_=motion_losses(x,frame,target,valid,time,scale_mm=scale,frame_weight=cfg['frame_loss_weight'])
    rr=rr_loss(x,target,valid,time,scale)
    return old,rr,dict(key=keys[0],k=int(k.item()),noise=tensor_hash(noise),features=tensor_hash(features),
                       target=tensor_hash(target),mask=tensor_hash(valid),camera=tensor_hash(camera))

def calibrate(body,batches,cfg):
    records=[]
    for step in range(4):
        i,g=draw(step,len(batches),True)
        pos,rr,trace=forward_terms(body,batches[i],cfg,g)
        params=tuple(body.parameters())
        def norm(term,retain):
            grads=torch.autograd.grad(term,params,retain_graph=retain,allow_unused=True)
            return torch.stack([v.detach().float().norm() for v in grads if v is not None]).norm().item()
        p,r=norm(pos,True),norm(rr,False)
        if not all(math.isfinite(x) and x>0 for x in (p,r)):
            raise ValueError('invalid calibration gradient norm')
        records.append(dict(trace=trace,position_grad=p,rr_grad=r))
    return .1*median(row['position_grad']/row['rr_grad'] for row in records),records

def optimizer(body):
    opt=torch.optim.Adam(body.parameters(),lr=1e-5)
    return opt,torch.optim.lr_scheduler.StepLR(opt,1,1.)

def update(body,opt,scheduler,batches,cfg,step,weight):
    index,generator=draw(step,len(batches))
    opt.zero_grad(set_to_none=True)
    pos,rr,trace=forward_terms(body,batches[index],cfg,generator)
    loss=objective(pos,rr,weight)
    if not torch.isfinite(loss):
        raise ValueError('nonfinite objective')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(body.parameters(),float('inf'),error_if_nonfinite=True)
    opt.step();scheduler.step()
    return dict(step=step+1,position=pos.item(),rr=rr.item(),total=loss.item(),trace=trace)

def save(path,body,opt,sched,binding,log):
    path=Path(path);tmp=path.with_suffix('.tmp')
    torch.save(dict(format=FORMAT,binding=binding,body=body.state_dict(),optimizer=opt.state_dict(),
                    scheduler=sched.state_dict(),step=len(log),log=log),tmp)
    os.replace(tmp,path)
    return sha(path)

def restore(path,pin,body,binding,opt=None,sched=None):
    if sha(path)!=pin:
        raise ValueError('checkpoint SHA')
    doc=torch.load(path,map_location='cpu',weights_only=True)
    if doc['format']!=FORMAT or doc['binding']!=binding or doc['step']!=len(doc['log']):
        raise ValueError('warm-start binding')
    body.load_state_dict(doc['body'],strict=True)
    if opt is not None:
        opt.load_state_dict(doc['optimizer']);sched.load_state_dict(doc['scheduler'])
    return doc['log']
