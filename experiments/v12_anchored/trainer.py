"""Measure gradients at one shared x0 and add a root-coordinate term."""
import math
from statistics import median
import torch
from experiments.v12_rr import trainer as rr

RR_WEIGHT=0.08310449710117396
ARMS=('control','rr','pelvis','anchored')

def weights(pelvis):
    if not math.isfinite(pelvis) or pelvis<=0:
        raise ValueError('invalid pelvis calibration')
    return dict(control=(0.,0.),rr=(RR_WEIGHT,0.),pelvis=(0.,pelvis),anchored=(RR_WEIGHT,pelvis))

def pelvis_loss(x,target,valid,time,scale):
    if (x.shape!=target.shape or x.ndim!=4 or x.shape[-2:]!=(22,3) or valid.shape!=x.shape[:-1]
            or time.shape!=x.shape[:2] or valid.dtype!=torch.bool or time.dtype!=torch.bool
            or not math.isfinite(scale) or scale<=0):
        raise ValueError('pelvis input contract')
    keep=valid[...,0]&time
    pred=x[...,0,:];gt=target[...,0,:]
    if not torch.isfinite(x).all() or not torch.isfinite(gt[keep]).all():
        raise ValueError('nonfinite prediction or valid pelvis')
    if not keep.any():
        return pred.reshape(-1)[:0].sum()
    return (pred[keep]-gt[keep]/scale).square().sum(-1).mean()

def terms(body,batch,cfg,generator):
    # Observe the exact forward used by the original RR trainer, including its intermediate head.
    output=[]
    def capture(module,args,pair):
        output.append(pair[0])
    with body.register_forward_hook(capture):
        position,relative,trace=rr.forward_terms(body,batch,cfg,generator)
    x,=output
    target=batch['target'][None,:,:22].to(x.device)
    valid=batch['valid'][None,:,:22].to(x.device)
    time=torch.ones(x.shape[:2],dtype=torch.bool,device=x.device)
    return (position,relative,pelvis_loss(x,target,valid,time,cfg['target_scale_mm'])),trace

def gram_matrix(losses,params):
    grads=[torch.autograd.grad(loss,params,retain_graph=i<len(losses)-1,allow_unused=True)
           for i,loss in enumerate(losses)]
    gram=torch.zeros((3,3),dtype=torch.float64,device=losses[0].device)
    for group in zip(*grads):
        vectors=[None if g is None else g.detach().flatten().double() for g in group]
        for i,u in enumerate(vectors):
            if u is not None:
                for j in range(i+1):
                    if vectors[j] is not None:
                        gram[i,j]+=u.dot(vectors[j])
    gram=gram+gram.tril(-1).T
    if not torch.isfinite(gram).all() or (gram.diag()<=0).any():
        raise ValueError('undefined diagnostic gradient')
    return gram.cpu()

def diagnose(body,batches,cfg):
    draws=[];grams=[]
    for step in range(4):
        i,g=rr.draw(step,len(batches),True)
        losses,trace=terms(body,batches[i],cfg,g)
        gram=gram_matrix(losses,tuple(body.parameters()));grams.append(gram)
        norms=gram.diag().sqrt()
        cosine=gram/(norms[:,None]*norms[None,:])
        draws.append(dict(draw=step,clip_index=i,trace=trace,norms=norms.tolist(),
                          cos_position_rr=cosine[0,1].item(),cos_position_pelvis=cosine[0,2].item(),
                          cos_rr_pelvis=cosine[1,2].item()))
    wp=.1*median(d['norms'][0]/d['norms'][2] for d in draws)
    arms=weights(wp)
    for record,gram in zip(draws,grams):
        record['combined_norm']={}
        for arm,(wr,wa) in arms.items():
            v=gram.new_tensor([1.,wr,wa]);norm=(v@gram@v).sqrt().item()
            if not math.isfinite(norm):
                raise ValueError('invalid combined gradient')
            record['combined_norm'][arm]=norm
    return wp,draws

def update(body,opt,scheduler,batches,cfg,step,coefficients):
    wr,wp=coefficients
    if wp==0:
        return rr.update(body,opt,scheduler,batches,cfg,step,wr)
    i,g=rr.draw(step,len(batches))
    opt.zero_grad(set_to_none=True)
    (old,relative,root),trace=terms(body,batches[i],cfg,g)
    loss=rr.objective(rr.objective(old,relative,wr),root,wp)
    if not torch.isfinite(loss):
        raise ValueError('nonfinite loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(body.parameters(),float('inf'),error_if_nonfinite=True)
    opt.step();scheduler.step()
    return dict(step=step+1,position=old.item(),rr=relative.item(),pelvis=root.item(),total=loss.item(),trace=trace)
