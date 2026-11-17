"""T4 only: immutable Train300/400 -> paired 400-update arms."""
import argparse
import copy
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from rewind_m1.training import rng_state,restore_rng
from rewind_m2.models import BodyDenoiser
from rewind_m2.diffusion import DiffusionSchedule
from rewind_m2.training import position_losses
from experiments.final_m2.dataset_scaling_300.prepare_data import read_npz
from experiments.final_m2.displacement_loss_20261116.loss import window_metadata
from experiments.final_m2.displacement_loss_20261116.run_experiment import state_hash,evaluate,grad_norm,ensure_finite
from experiments.final_m2.diagnosis_20261111.diagnose import write_csv
from tools.final_m2_real import sha,load,write
from experiments.final_m2.bone_length_loss_20261117.loss import bone_length_loss,combined_loss

HERE=Path(__file__).resolve().parent


def restore_arm(ck,device):
    cfg=ck['config'];model=BodyDenoiser(cfg,22,ck['manifest']['edges']);model.load_state_dict(ck['model'],strict=True);model.to(device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate']);opt.load_state_dict(copy.deepcopy(ck['optimizer']))
    scheduler=torch.optim.lr_scheduler.StepLR(opt,cfg['scheduler_step'],cfg['scheduler_gamma']);scheduler.load_state_dict(copy.deepcopy(ck['scheduler']))
    for name,obj in [('model',model.state_dict()),('optimizer',opt.state_dict()),('scheduler',scheduler.state_dict())]:
        if state_hash(obj)!=state_hash(ck[name]):raise ValueError('restore_'+name)
    if scheduler.last_epoch!=400 or any(float(s['step'])!=400 for s in opt.state.values()):raise ValueError('parent_step')
    restore_rng(copy.deepcopy(ck['rng']))
    if state_hash(rng_state())!=state_hash(ck['rng']):raise ValueError('rng_restore')
    return model,opt,scheduler


def draw(model,ck,schedule,data,frames,sequences,device):
    cfg=ck['config'];ids=torch.randint(len(data['frame_ids']),(8,))
    x,y,m,t=[torch.from_numpy(data[k][ids]).to(device) for k in ('features','target_raw','valid','time')]
    k=torch.randint(1,cfg['diffusion_steps']+1,(8,),device=device);noise=torch.randn_like(y)
    clean=torch.where((m&t[...,None])[...,None],y/cfg['target_scale_raw'],0.)
    p,f=model(schedule.q_sample(clean,noise,k),x,torch.zeros(8,8,2,9,device=device),k,t)
    existing,main,aux=position_losses(p,f,y,m,t,cfg['target_scale_raw'],cfg['frame_loss_weight'])
    bone,pairs=bone_length_loss(p,y,m,t,ck['manifest']['edges'],cfg['target_scale_raw'])
    trace={'anchors':ids.tolist(),'timesteps':k.cpu().tolist(),'noise_sha256':state_hash(noise),
           'frames_sha256':state_hash(frames[ids]),'sequences_sha256':state_hash(sequences[ids])}
    return existing,main,aux,bone,pairs,trace


def calibrate(ck,data,schedule,frames,sequences,device,settings):
    model,_,_=restore_arm(ck,device);model.train();params=list(model.parameters());ratios=[];trace=[]
    for _ in range(settings['calibration_batches']):
        existing,_,_,bone,pairs,row=draw(model,ck,schedule,data,frames,sequences,device)
        gp=grad_norm(torch.autograd.grad(existing,params,retain_graph=True,allow_unused=True))
        gb=grad_norm(torch.autograd.grad(bone,params,allow_unused=True))
        if gp<=0 or gb<=0 or pairs<=0:raise ValueError('calibration_gradient')
        ratios.append(gp/gb);trace.append(row)
    weight=float(settings['calibration_fraction']*np.median(ratios))
    if not np.isfinite(weight) or weight<=0:raise ValueError('calibration_weight')
    return weight,{'train_only_gradient_ratios':ratios,'weight':weight,'draw_trace_sha256':state_hash(trace)}


def train_arm(arm,ck,data,schedule,frames,sequences,device,weight,out,settings):
    model,opt,scheduler=restore_arm(ck,device);model.train();initial={k:state_hash(v) for k,v in
        [('model',model.state_dict()),('optimizer',opt.state_dict()),('scheduler',scheduler.state_dict()),('rng',rng_state())]}
    trace=[];rows=[];start=time.monotonic();w=0. if arm=='control' else weight
    with (out/f'{arm}_training.jsonl').open('x') as log:
        for update in range(1,settings['additional_updates']+1):
            existing,main,aux,bone,pairs,row=draw(model,ck,schedule,data,frames,sequences,device)
            trace.append(row);total=combined_loss(existing,bone,w);ensure_finite([total],'loss')
            opt.zero_grad(set_to_none=True);total.backward();ensure_finite([p.grad for p in model.parameters()],'gradient')
            opt.step();scheduler.step()
            item={'step':400+update,'arm':arm,'loss':float(total.detach()),'position_loss':float(existing.detach()),
                  'x0_loss':float(main.detach()),'frame_loss':float(aux.detach()),'bone_loss':float(bone.detach()),
                  'bone_weight':w,'valid_edge_observations':pairs,'elapsed_seconds':time.monotonic()-start}
            rows.append(item);log.write(json.dumps(item,allow_nan=False)+'\n');log.flush()
            if update%100==0:print(json.dumps(item),flush=True)
    if scheduler.last_epoch!=800 or any(float(s['step'])!=800 for s in opt.state.values()):raise ValueError('final_step')
    end={'format_version':'m2_bone_length_20261117_v1','parent_checkpoint_sha256':settings['parent_checkpoint_sha256'],
         'arm':arm,'weight':w,'step':800,'config':ck['config'],'manifest':ck['manifest'],'model':model.state_dict(),
         'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':copy.deepcopy(ck['scaler']),
         'rng':rng_state(),'alpha_bar':schedule.alpha_bar,'initial_state_sha256':initial,'random_trace_sha256':state_hash(trace)}
    torch.save(end,out/f'{arm}_800.pt');write(out/f'{arm}_random_trace.json',trace);write_csv(out/f'{arm}_training.csv',rows)
    return {'arm':arm,'initial':initial,'random_trace_sha256':state_hash(trace),'checkpoint_sha256':sha(out/f'{arm}_800.pt')}


def run(input_dir,checkpoint_dir,unused_path,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);settings=load(HERE/'config.json')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():raise ValueError('requires_actual_T4')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    inp=Path(input_dir);parent=Path(checkpoint_dir)/settings['parent_checkpoint']
    for path,want in [(parent,settings['parent_checkpoint_sha256']),(inp/'train300.npz',settings['train_input_sha256']),
                      (inp/'data_selection_manifest.json',settings['selection_sha256'])]:
        if sha(path)!=want:raise ValueError('input_sha_'+path.name)
    ck=torch.load(parent,map_location='cpu',weights_only=True);manifest=load(inp/'data_selection_manifest.json')
    if ck['step']!=400 or ck['manifest']['edges']!=manifest['edges'] or ck['config']['window_size']!=8:raise ValueError('parent_contract')
    for name in ('models','diffusion','streaming','training'):
        path=f'rewind_m2/{name}.py'
        if sha(ROOT/path)!=ck['source']['code_sha256'][path]:raise ValueError('core_sha_'+name)
    data={name:read_npz(inp/f'{name}.npz') for name in ('train300','opening130','gunplay47')}
    unused=read_npz(Path(unused_path));data['unused_motion_preselected']=unused
    selection=load(Path(unused_path).with_suffix('.selection.json'))
    if selection['npz_sha256']!=sha(unused_path) or selection['selected_before_training'] is not True:raise ValueError('unused_selection_lock')
    train_keys=set(zip(map(str,data['train300']['sequence_ids']),map(int,data['train300']['frame_ids'])))
    other=set()
    for name,d in data.items():
        frames,sequences=window_metadata(d)
        if name=='train300':train_frames,train_sequences=frames,sequences;continue
        keys=set(zip(map(str,d['sequence_ids']),map(int,d['frame_ids'])))
        if train_keys&keys or other&keys:raise ValueError('evaluation_overlap')
        other|=keys
    schedule=DiffusionSchedule(ck['config']['diffusion_steps'],ck['config']['beta_start'],ck['config']['beta_end'])
    if not torch.equal(schedule.alpha_bar,ck['alpha_bar']):raise ValueError('schedule')
    weight,calibration=calibrate(ck,data['train300'],schedule,train_frames,train_sequences,'cuda',settings)
    write(out/'calibration.json',calibration)
    arms=[train_arm(arm,ck,data['train300'],schedule,train_frames,train_sequences,'cuda',weight,out,settings) for arm in ('control','improved')]
    if arms[0]['initial']!=arms[1]['initial'] or arms[0]['random_trace_sha256']!=arms[1]['random_trace_sha256']:
        raise ValueError('paired_randomness')
    rows=[];bones=[];frames=[];noises={}
    for arm in ('control','improved'):
        end=torch.load(out/f'{arm}_800.pt',map_location='cpu',weights_only=True)
        r,b,f,n,_=evaluate(arm,end['model'],ck,{k:v for k,v in data.items() if k!='train300'},schedule,out,'cuda')
        rows+=r;bones+=b;frames+=f;noises[arm]=n
    if noises['control']!=noises['improved']:raise ValueError('evaluation_noise')
    for split in ('opening130','gunplay47','unused_motion_preselected'):
        a=read_npz(out/f'{split}_control_predictions.npz');b=read_npz(out/f'{split}_improved_predictions.npz')
        for key in ('gt','valid','frame_ids','sequence_ids'):
            np.testing.assert_array_equal(a[key],b[key])
    if sha(parent)!=settings['parent_checkpoint_sha256']:raise ValueError('parent_mutated')
    from experiments.final_m2.dataset_scaling_100.run_experiment import extra_metrics
    for row in rows:
        z=read_npz(out/f"{row['split']}_{row['model']}_predictions.npz")
        row.update(extra_metrics(z['prediction'],z['gt'],z['valid'],z['frame_ids']))
    write_csv(out/'comparison.csv',rows);write_csv(out/'bone_length_by_edge.csv',bones)
    write(out/'evaluation.json',{'execution_status':'PASS','performance_improvement':'NOT VERIFIED until T4 results reviewed',
        'arms':arms,'calibration':calibration,'evaluation_noise':noises,'results':rows,'unused_selection':selection,
        'parent_checkpoint_sha256_after':sha(parent),'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name()}})
    write(out/'checksums.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);p.add_argument('--checkpoint-dir',required=True)
    p.add_argument('--unused-motion',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run(a.input_dir,a.checkpoint_dir,a.unused_motion,a.output)
