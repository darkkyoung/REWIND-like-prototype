"""FP32 single-device pilot. Fresh warm-start checkpoints live in the run directory."""
import argparse
import gc
import json
from pathlib import Path

import torch
from scripts.v12_share_smoke import load_models,verify_manifest
from . import trainer as tr
from .data import HERE,read_clip,selection

from experiments.v12_rr.run import write
from experiments.v12_rr import trainer as base
from .evaluation import evaluate
import zipfile

PRIOR_SHA='c1985fe1624489a58df446c77fa42a469a14a3eb92bf3c19b9d12cb67a848eed'
PINS={'control':'243f7d7b1649b0f6c6b37c90458aaf0448a9ab1b3d650917bac74504470a77bc',
      'rr':'76f63306780af8f9ab44b11d46619aa2dc0484fe0cd2d88df87e4bf20c7cefa3'}

def historical(archive):
    if base.sha(archive)!=PRIOR_SHA:
        raise ValueError('previous run archive SHA')
    result={}
    with zipfile.ZipFile(archive) as z:
        for arm,name in [('control','control.pt'),('rr','improved.pt')]:
            member,=[p for p in z.namelist() if p.split('/')[-1]==name]
            import io, hashlib
            raw=z.read(member)
            if hashlib.sha256(raw).hexdigest()!=PINS[arm]:
                raise ValueError('previous arm SHA')
            result[arm]=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    return result

def run(package,audit,rgb,pose,out,previous):
    if not torch.cuda.is_available() or not any(s in torch.cuda.get_device_name() for s in ('T4','A100')):
        raise ValueError('requires T4 or A100')
    out.mkdir(parents=True,exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    past=historical(previous)
    models=load_models(package,'cuda')
    cfg=models['config']['teacher']
    if (cfg['hidden'],cfg['dropout'],cfg['target_scale_mm'],cfg['sampling_steps'])!=(512,0.,100.,10):
        raise ValueError('teacher contract')
    spec=models['spec']
    if spec['body_joint_names']!=spec['joint_names'][:22] or spec['body_joint_names'][0]!='pelvis':
        raise ValueError('Body22 layout')
    selected,manifest=selection(audit)
    bound=json.loads((package/'metadata/run.json').read_text())['input']
    if selected!=json.loads((HERE/'selection.json').read_text()) or (bound['manifest_sha256'],bound['plan_sha256'])!=(selected['manifest'],selected['plan']):
        raise ValueError('original training binding')
    lookup={(r['sequence_id'],r['frame_index']):r for r in manifest['records']}
    def batches(split):
        return [read_clip(s,f,split,lookup,spec,rgb,pose,models) for s,f in selected[split]]
    train=batches('train')
    body=models['body']
    start={k:v.cpu().clone() for k,v in body.state_dict().items()}
    frozen={k:base.state_hash(models[k]) for k in ('frontend','hand')}
    def reset():
        body.load_state_dict(start,strict=True)
        body.train().requires_grad_(True)
        body.zero_grad(set_to_none=True)
        torch.manual_seed(base.SEED)
    reset()
    initial=base.state_hash(body)
    props=torch.cuda.get_device_properties(0)
    torch.cuda.reset_peak_memory_stats()
    wp,cal=tr.diagnose(body,train,cfg)
    write(out/'gradient.json',dict(order=['position','rr','pelvis'],lambda_rr=tr.RR_WEIGHT,lambda_pelvis=wp,draws=cal))
    historical_diagnosis=json.loads((HERE/'previous.json').read_text())
    if historical_diagnosis['lambda_rr']!=tr.RR_WEIGHT:
        raise ValueError('RR coefficient identity')
    for record,old_draw in zip(cal,historical_diagnosis['draws'],strict=True):
        for key in ('key','k','noise','target','mask','camera'):
            if json.dumps(record['trace'][key])!=json.dumps(old_draw['trace'][key]):
                raise ValueError('historical calibration draw mismatch: '+key)
    opt,sched=base.optimizer(body)
    tr.update(body,opt,sched,train,cfg,0,(tr.RR_WEIGHT,wp))
    del opt,sched
    reset()
    reserved=torch.cuda.max_memory_reserved()
    write(out/'preflight.json',dict(device=props.name,total_bytes=props.total_memory,peak_reserved=reserved,
                                    lambda_rr=tr.RR_WEIGHT,lambda_pelvis=wp,calibration=cal,torch=str(torch.__version__)))
    if reserved>.95*props.total_memory:
        raise MemoryError('preflight exceeds 95% VRAM')
    common=dict(initial=initial,parent=base.PARENT,spec=bound['spec_sha256'],selection=base.sha(HERE/'selection.json'),
                package='fbf1adda7fb63007d799545278e513be7f0a32ae09a9453bad70f675312274a0',
                frozen=frozen,cfg=cfg,seed=base.SEED,steps=base.STEPS,learning_rate=1e-5,
                implementation={f:base.sha(HERE/f) for f in ('trainer.py','run.py','data.py','evaluation.py','CRITERIA.md')})
    checkpoints={}; traces={}; initial_states={}
    for arm,w in tr.weights(wp).items():
        reset()
        if base.state_hash(body)!=initial:
            raise ValueError('initial state mismatch')
        opt,sched=base.optimizer(body);log=[]
        initial_states[arm]=dict(body=base.state_hash(body),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
                                cpu_rng=base.tensor_hash(torch.get_rng_state()),cuda_rng=base.tensor_hash(torch.cuda.get_rng_state()))
        if arm!='control' and initial_states['control']!=initial_states[arm]:
            raise ValueError('optimizer/scheduler/RNG branch mismatch')
        binding=dict(common,experiment='root_anchored_2x2',arm=arm,weight=w)
        for step in range(base.STEPS):
            log.append(tr.update(body,opt,sched,train,cfg,step,w))
        path=out/(arm+'.pt')
        pin=base.save(path,body,opt,sched,binding,log)
        checkpoints[arm]=(path,pin,binding)
        traces[arm]=[r['trace'] for r in log]
        write(out/(arm+'.json'),dict(sha256=pin,binding=binding,log=log))
        del opt,sched
    write(out/'branch_states.json',initial_states)
    if any(t!=traces['control'] for t in traces.values()):
        raise ValueError('paired training traces differ')
    parity={}
    for arm,old in past.items():
        path,pin,binding=checkpoints[arm]
        base.restore(path,pin,body,binding)
        state=body.state_dict()
        changed={k:float((v.detach().cpu().double()-old['body'][k].double()).abs().max())
                 for k,v in state.items() if not torch.equal(v.detach().cpu(),old['body'][k])}
        current=json.loads(json.dumps(traces[arm]))
        previous_trace=json.loads(json.dumps([r['trace'] for r in old['log']]))
        same_trace=current==previous_trace
        first_diff=next((dict(step=i+1,fields=[k for k in t if t[k]!=v.get(k)])
                         for i,(t,v) in enumerate(zip(current,previous_trace)) if t!=v),None)
        parity[arm]=dict(old_checkpoint=PINS[arm],new_state_hash=base.state_hash(body),
                         changed_tensors=changed,first_trace_mismatch=first_diff,same_trace=same_trace,exact=not changed and same_trace)
    write(out/'historical_parity.json',parity)
    if not all(v['exact'] for v in parity.values()):
        raise ValueError('prior reproduction mismatch; inspect historical_parity and environment before interpreting C/D')
    del past,train
    gc.collect()
    evaluate(models,batches('validation'),selected,checkpoints,out)
    verify_manifest(package)
    if any(base.state_hash(models[k])!=v for k,v in frozen.items()):
        raise ValueError('frozen dependency changed')
    write(out/'checksums.json',{p.name:base.sha(p) for p in out.iterdir()})

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for arg in ('package','audit','rgb','pose','out','previous'):
        parser.add_argument('--'+arg,type=Path,required=True)
    run(**vars(parser.parse_args()))
