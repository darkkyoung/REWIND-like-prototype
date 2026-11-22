"""FP32 single-device pilot. Fresh warm-start checkpoints live in the run directory."""
import argparse
import csv
import gc
import json
import os
import time
from pathlib import Path

import torch
from scripts.v12_share_smoke import load_models,verify_manifest
from experiments.v12_m2.evaluate import metrics
from rewind_v1.cross_inference import sample_features
from . import trainer as tr
from .data import HERE,read_clip,selection

def write(path, value):
    tmp=path.with_suffix('.part')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False))
    os.replace(tmp,path)

def decision(rows):
    for k in ('rr_jpe_mm','body_mpjpe_mm','pelvis_mm','wrist_mm','hand_global_mpjpe_mm','whole_mpjpe_mm'):
        c,i=rows['control'][k],rows['improved'][k]
        if c is None or i is None:
            return 'INCOMPLETE'
    guard=('body_mpjpe_mm','pelvis_mm','wrist_mm','hand_global_mpjpe_mm','whole_mpjpe_mm')
    return 'PASS' if rows['improved']['rr_jpe_mm']<=.99*rows['control']['rr_jpe_mm'] and all(
        rows['improved'][k]<=1.02*rows['control'][k] for k in guard) else 'FAIL'

def run(package,audit,rgb,pose,out):
    if not torch.cuda.is_available() or not any(s in torch.cuda.get_device_name() for s in ('T4','A100')):
        raise ValueError('requires T4 or A100')
    out.mkdir(parents=True,exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
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
    frozen={k:tr.state_hash(models[k]) for k in ('frontend','hand')}
    def reset():
        body.load_state_dict(start,strict=True)
        body.train().requires_grad_(True)
        body.zero_grad(set_to_none=True)
        torch.manual_seed(tr.SEED)
    reset()
    initial=tr.state_hash(body)
    props=torch.cuda.get_device_properties(0)
    torch.cuda.reset_peak_memory_stats()
    opt,sched=tr.optimizer(body)
    # Exercise both loss graphs, gradients and Adam allocations, then discard the step.
    tr.update(body,opt,sched,train,cfg,0,.1)
    del opt,sched
    reset()
    weight,cal=tr.calibrate(body,train,cfg)
    reserved=torch.cuda.max_memory_reserved()
    write(out/'preflight.json',dict(device=props.name,total_bytes=props.total_memory,peak_reserved=reserved,
                                    lambda_rr=weight,calibration=cal,torch=str(torch.__version__)))
    if reserved>.95*props.total_memory:
        raise MemoryError('preflight exceeds 95% VRAM')
    common=dict(initial=initial,parent=tr.PARENT,spec=bound['spec_sha256'],selection=tr.sha(HERE/'selection.json'),
                package='fbf1adda7fb63007d799545278e513be7f0a32ae09a9453bad70f675312274a0',
                frozen=frozen,cfg=cfg,seed=tr.SEED,steps=tr.STEPS,learning_rate=1e-5,
                implementation={f:tr.sha(HERE/f) for f in ('trainer.py','run.py','data.py','CRITERIA.md')})
    checkpoints={}; traces={}; initial_states={}
    for arm,w in [('control',0.),('improved',weight)]:
        reset()
        if tr.state_hash(body)!=initial:
            raise ValueError('initial state mismatch')
        opt,sched=tr.optimizer(body);log=[]
        initial_states[arm]=dict(body=tr.state_hash(body),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
                                cpu_rng=tr.tensor_hash(torch.get_rng_state()),cuda_rng=tr.tensor_hash(torch.cuda.get_rng_state()))
        if arm=='improved' and initial_states['control']!=initial_states[arm]:
            raise ValueError('optimizer/scheduler/RNG branch mismatch')
        binding=dict(common,arm=arm,weight=w)
        for step in range(tr.STEPS):
            log.append(tr.update(body,opt,sched,train,cfg,step,w))
        path=out/(arm+'.pt')
        pin=tr.save(path,body,opt,sched,binding,log)
        checkpoints[arm]=(path,pin,binding)
        traces[arm]=[r['trace'] for r in log]
        write(out/(arm+'.json'),dict(sha256=pin,binding=binding,log=log))
        del opt,sched
    write(out/'branch_states.json',initial_states)
    if traces['control']!=traces['improved']:
        raise ValueError('paired training inputs differ')
    del train
    gc.collect()
    clips=batches('validation')
    summary=[];frames=[]
    for batch in clips:
        results={};preds={}
        for arm in ('original','control','improved'):
            if arm=='original':
                body.load_state_dict(start,strict=True)
            else:
                path,pin,binding=checkpoints[arm]
                tr.restore(path,pin,body,binding)
            body.eval().requires_grad_(False)
            torch.cuda.synchronize();begin=time.perf_counter()
            p=sample_features(models,batch['features'],batch['camera'],batch['keys'],models['config'],'cuda')
            torch.cuda.synchronize()
            elapsed=(time.perf_counter()-begin)*1000/len(batch['keys'])
            preds[arm]=p
            results[arm]=metrics(p,preds['original']['body'],batch['target'],batch['valid'],spec)
            results[arm]['body_hand_ms_frame']=elapsed
        for i,(seq,f) in enumerate(batch['keys']):
            row=dict(sequence=seq,frame=f)
            gt=batch['target'][i,:22];v=batch['valid'][i,:22]
            for arm,p in preds.items():
                err=p['body'][i]-gt
                row[arm+'_global']=float(err[v].norm(dim=-1).mean()) if v.any() else None
                row[arm+'_rr']=float((err-err[:1])[v].norm(dim=-1).mean()) if v[0] else None
            for metric in ('global','rr'):
                c,ierr=row['control_'+metric],row['improved_'+metric]
                row['improved_minus_control_'+metric]=ierr-c if c is not None else None
            frames.append(row)
        summary.append(dict(sequence=batch['keys'][0][0],results=results,decision=decision(results)))
    verify_manifest(package)
    if any(tr.state_hash(models[k])!=v for k,v in frozen.items()):
        raise ValueError('frozen dependency changed')
    with (out/'frames.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=frames[0]);writer.writeheader();writer.writerows(frames)
    states=[r['decision'] for r in summary]
    write(out/'evaluation.json',dict(clips=summary,decision='INCOMPLETE' if 'INCOMPLETE' in states else
          ('PASS' if all(v=='PASS' for v in states) else 'FAIL'),peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    write(out/'checksums.json',{p.name:tr.sha(p) for p in out.iterdir()})

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for arg in ('package','audit','rgb','pose','out'):
        parser.add_argument('--'+arg,type=Path,required=True)
    run(**vars(parser.parse_args()))
