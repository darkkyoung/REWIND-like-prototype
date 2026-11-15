"""고정 T4 pilot: baseline300 -> control400 / improved400. 기존 resume 우회 없음."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from rewind_m1.training import rng_state, restore_rng
from rewind_m2.models import BodyDenoiser
from rewind_m2.diffusion import DiffusionSchedule
from rewind_m2.training import position_losses
from rewind_m2.streaming import AnchorSampler, keyed_frame_noise
from tools.final_m2_real import prepare, infer_frame, sha, load, write
from experiments.final_m2.displacement_loss_20261116.loss import displacement_loss, window_metadata, LAMBDA, S_DELTA
from experiments.final_m2.diagnosis_20261111.diagnose import calculate, write_csv

HERE = Path(__file__).resolve().parent
FORMAT = 'm2_displacement_pilot_v1'


def state_hash(obj):
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            a=x.detach().cpu().contiguous();h.update(str((str(a.dtype),list(a.shape))).encode());h.update(a.numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str): h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(str(type(x).__name__).encode())
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(obj);return h.hexdigest()


def ensure_finite(values,label):
    if any(v is not None and not torch.isfinite(v).all() for v in values):
        raise ValueError('nonfinite_'+label)


def make_arm(ck,device):
    cfg=ck['config']
    model=BodyDenoiser(cfg,22,ck['manifest']['edges'])
    for name,b in model.named_buffers():
        if not torch.equal(b,ck['model'][name]):raise ValueError('structural_buffer')
    model.load_state_dict(ck['model'],strict=True);model.to(device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    scheduler=torch.optim.lr_scheduler.StepLR(opt,cfg['scheduler_step'],cfg['scheduler_gamma'])
    opt.load_state_dict(copy.deepcopy(ck['optimizer']))
    scheduler.load_state_dict(copy.deepcopy(ck['scheduler']))
    scaler=torch.amp.GradScaler('cuda',enabled=False);scaler.load_state_dict(ck['scaler'])
    for key,value in [('model',model.state_dict()),('optimizer',opt.state_dict()),('scheduler',scheduler.state_dict()),('scaler',scaler.state_dict())]:
        if state_hash(value)!=state_hash(ck[key]):raise ValueError('restore_'+key)
    if len(opt.state)!=len(list(model.parameters())) or any(float(s['step'])!=300 for s in opt.state.values()):
        raise ValueError('incomplete_adam_state')
    return model,opt,scheduler,scaler


def grad_norm(grads):
    ensure_finite(grads,'diagnostic_gradient')
    return float(torch.stack([g.detach().square().sum() for g in grads if g is not None]).sum().sqrt())


def train_arm(arm,ck,data,schedule,out,device,identity):
    model,opt,scheduler,scaler=make_arm(ck,device)
    initial={k:state_hash(v) for k,v in [('model',model.state_dict()),('optimizer',opt.state_dict()),('scheduler',scheduler.state_dict()),('scaler',scaler.state_dict())]}
    # 모델 생성 시 소비된 RNG까지 원본 checkpoint 상태로 되돌린다.
    restore_rng(copy.deepcopy(ck['rng']))
    if state_hash(rng_state())!=state_hash(ck['rng']):raise ValueError('rng_restore')
    initial['rng']=state_hash(rng_state())
    cfg=ck['config'];d={k:torch.from_numpy(data[k]) for k in ('features','target_raw','valid','time')}
    frames,sequences=window_metadata(data); logs=[];trace=[];start=time.monotonic()
    params=list(model.parameters());weight=0. if arm=='control' else LAMBDA
    model.train()
    with (out/f'{arm}_training.jsonl').open('x') as stream:
        for update in range(1,101):
            if time.monotonic()-start>300:raise ValueError('training_budget_300_seconds')
            ids=torch.randint(len(data['frame_ids']),(8,))
            x,y,m,t=[d[k][ids].to(device) for k in ('features','target_raw','valid','time')]
            cam=torch.zeros(8,8,2,9,device=device)
            k=torch.randint(1,cfg['diffusion_steps']+1,(8,),device=device)
            noise=torch.randn_like(y)
            trace_row={'step':300+update,'anchors':ids.tolist(),'timesteps':k.cpu().tolist(),'noise_sha256':state_hash(noise)}
            trace.append(trace_row)
            clean=torch.where((m&t[...,None])[...,None],y/cfg['target_scale_raw'],0.)
            noisy=schedule.q_sample(clean,noise,k)
            opt.zero_grad(set_to_none=True)
            p,f=model(noisy,x,cam,k,t)
            existing,main,aux=position_losses(p,f,y,m,t,cfg['target_scale_raw'],cfg['frame_loss_weight'])
            disp,pairs=displacement_loss(p*cfg['target_scale_raw'],y,m,t,frames[ids].to(device),sequences[ids].to(device))
            total=existing+weight*disp
            ensure_finite([existing,disp,total],'loss')
            # 추가 항의 실제 파라미터 gradient 크기도 기록. optimizer/RNG에는 영향 없음.
            g_existing=grad_norm(torch.autograd.grad(existing,params,retain_graph=True,allow_unused=True))
            g_disp=grad_norm(torch.autograd.grad(disp,params,retain_graph=True,allow_unused=True))
            total.backward();ensure_finite([p.grad for p in params],'gradient')
            lr=float(opt.param_groups[0]['lr']);opt.step();scheduler.step()
            ensure_finite(params,'parameter')
            ensure_finite([v for s in opt.state.values() for v in s.values() if isinstance(v,torch.Tensor)],'optimizer_state')
            row={'arm':arm,'step':300+update,'additional_updates':update,'loss':float(total.detach()),'existing_loss':float(existing.detach()),
                 'x0_loss':float(main.detach()),'aux_position_loss':float(aux.detach()),'displacement_loss':float(disp.detach()),
                 'weighted_disp_loss':weight*float(disp.detach()),'valid_joint_pairs':pairs,'lr_used':lr,'lr_next':float(opt.param_groups[0]['lr']),
                 'existing_gradient_norm':g_existing,'displacement_gradient_norm':g_disp,'weighted_disp_gradient_norm':weight*g_disp,
                 'elapsed_seconds':time.monotonic()-start}
            logs.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if update%10==0:print(json.dumps(row),flush=True)
    if scheduler.last_epoch!=400 or any(float(s['step'])!=400 for s in opt.state.values()):raise ValueError('final_update_count')
    end={'format_version':FORMAT,'experiment_identity':identity,'arm':arm,'lambda':weight,'s_delta_raw':S_DELTA,
         'parent_checkpoint_sha256':identity['parent_checkpoint_sha256'],'config':ck['config'],'manifest':ck['manifest'],
         'source':identity['source'],'environment':identity['environment'],'step':400,'samples_seen':3200,
         'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
         'rng':rng_state(),'alpha_bar':schedule.alpha_bar,'initial_state_sha256':initial,'random_trace_sha256':state_hash(trace)}
    checkpoint=out/f'{arm}_400.pt';torch.save(end,checkpoint)
    write(out/f'{arm}_random_trace.json',trace)
    write_csv(out/f'{arm}_training.csv',logs)
    record={'arm':arm,'checkpoint_sha256':sha(checkpoint),'step':400,'additional_optimizer_updates':100,'initial_state_sha256':initial,
            'random_trace_sha256':state_hash(trace),'end_rng_sha256':state_hash(end['rng']),'seconds':time.monotonic()-start}
    write(out/f'{arm}_identity.json',record)
    return record


def evaluate(arm,state,ck,data,schedule,out,device):
    model=BodyDenoiser(ck['config'],22,ck['manifest']['edges']);model.load_state_dict(state,strict=True)
    model.to(device).eval().requires_grad_(False)
    rows=[];bones=[];frames=[];noise_hashes={};reproduction={}
    with torch.inference_mode():
        for split,d in data.items():
            sampler=AnchorSampler(model,schedule,8,10);pred=[]
            keys=[(str(s),int(f)) for s,f in zip(d['sequence_ids'],d['frame_ids'])]
            noise_hashes[split]=state_hash([keyed_frame_noise(s,f,22,10043) for s,f in keys])
            for i,(seq,frame) in enumerate(keys):
                pred.append(infer_frame(sampler,d['features'][i,-1],seq,frame,10043,device).cpu().numpy())
            p=np.stack(pred);g=d['target_raw'][:,-1];v=d['valid'][:,-1]
            np.savez_compressed(out/f'{split}_{arm}_predictions.npz',prediction=p,gt=g,valid=v,frame_ids=d['frame_ids'],sequence_ids=d['sequence_ids'])
            summary,br,fr=calculate(p,g,v,d['frame_ids'],ck['manifest']['edges'],0)
            rows.append({'split':split,'model':arm,**summary});bones.extend({'split':split,'model':arm,**b} for b in br)
            for i,r in enumerate(fr):
                pair=v[i]&v[i-1] if i else np.zeros(22,dtype=bool)
                dg=g[i].astype(float)-g[i-1] if i else None;dp=p[i].astype(float)-p[i-1] if i else None
                r.update(gt_displacement_raw=float(np.linalg.norm(dg[pair],axis=-1).mean()) if pair.any() else None,
                         pred_displacement_raw=float(np.linalg.norm(dp[pair],axis=-1).mean()) if pair.any() else None,
                         displacement_error_raw=float(np.linalg.norm((dp-dg)[pair],axis=-1).mean()) if pair.any() else None)
                frames.append({'split':split,'model':arm,**r})
            if arm=='baseline':
                old=np.load(out/'private_input'/f'historical_{split}_motion.npz',allow_pickle=False)
                difference=float(np.max(np.abs(p-old['ddim'])))
                olderr=float(np.linalg.norm(old['ddim'].astype(float)-g,axis=-1)[v].mean())
                error_delta=abs(summary['global_joint_error_raw']-olderr)
                reproduction[split]={'max_coordinate_difference_raw':difference,'global_error_difference_raw':error_delta,
                                     'status':'PASS' if difference<.001 and error_delta<.0001 else 'FAIL'}
            print(json.dumps({'evaluation':rows[-1]},allow_nan=False),flush=True)
    return rows,bones,frames,noise_hashes,reproduction


def success_checks(rows,frames,bones):
    hold={r['model']:r for r in rows if r['split']=='holdout'};i=hold['improved'];comparisons={}
    for other in ('baseline','control'):
        c=hold[other];comparisons[other]={
            'displacement_reduction_pct':100*(1-i['frame_displacement_error_raw']/c['frame_displacement_error_raw']),
            'displacement_10pct_pass':i['frame_displacement_error_raw']<=.9*c['frame_displacement_error_raw'],
            'motion_closer_to_100_pass':abs(i['motion_magnitude_ratio_pct']-100)<abs(c['motion_magnitude_ratio_pct']-100),
            'global_within_2pct_pass':i['global_joint_error_raw']<=1.02*c['global_joint_error_raw'],
            'root_relative_within_2pct_pass':i['root_relative_error_22_raw']<=1.02*c['root_relative_error_22_raw']}
    # 사전 고정한 보수적인 collapse screening. 정상 체형을 보증하는 기준 아님.
    selected=[r for r in frames if r['model']=='improved']
    minspread=min(r['spread_ratio_pct'] for r in selected);minbone=min(r['min_frame_ratio_pct'] for r in bones if r['model']=='improved')
    shape=minspread>=50 and minbone>=50
    passed=all(all(v for k,v in c.items() if k.endswith('_pass')) for c in comparisons.values()) and shape
    return {'pilot_status':'PASS' if passed else 'FAIL','comparisons':comparisons,
            'shape_screen_pass':shape,'min_frame_spread_ratio_pct':minspread,'min_frame_edge_ratio_pct':minbone,
            'shape_screen':'All improved frames spread and each bone >=50% GT; inspect plots separately; exploratory not anatomical norm'}


def run(bundle,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    cfg=load(ROOT/'experiments/final_m2/configs/real148.json')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():raise ValueError('requires_actual_T4')
    device='cuda';torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    ck,data,private=prepare(bundle,out,cfg)
    if ck['config']['batch_size']!=8 or ck['config']['window_size']!=8 or ck['config']['dropout']!=0 or ck['samples_seen']!=2400:
        raise ValueError('locked_training_config')
    if ck['scheduler']['last_epoch']!=300 or ck['scaler']!={} or not ck['rng']['torch_cuda']:raise ValueError('incomplete_training_state')
    core={f'rewind_m2/{n}.py':sha(ROOT/f'rewind_m2/{n}.py') for n in ('models','diffusion','streaming','training')}
    if any(ck['source']['code_sha256'][p]!=h for p,h in core.items()):raise ValueError('core_code_mismatch')
    schedule=DiffusionSchedule(1000,ck['config']['beta_start'],ck['config']['beta_end'])
    if not torch.equal(schedule.alpha_bar,ck['alpha_bar']):raise ValueError('alpha_schedule')
    for d in data.values():window_metadata(d)
    train_gt=data['train']['target_raw'][:,-1].astype(float)
    measured=float(np.sqrt(np.mean(np.sum(np.diff(train_gt,axis=0)**2,axis=-1))))
    if abs(measured-S_DELTA)>1e-6:raise ValueError('s_delta_units')
    source={'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'core_code_sha256':core,'experiment_files_sha256':{p.name:sha(p) for p in HERE.glob('*.py')}}
    environment={'python':platform.python_version(),'torch':str(torch.__version__),'cuda':torch.version.cuda,
                 'gpu':torch.cuda.get_device_name(),'gpu_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
                 'precision':'FP32','device':'cuda','threads':1}
    identity={'format':FORMAT,'parent_checkpoint_sha256':cfg['checkpoint_sha256'],'input_bundle_sha256':cfg['input_bundle_sha256'],
              'source':source,'environment':environment,'parent_environment':ck['environment'],
              'additional_updates_per_arm':100,'lambda':LAMBDA,'s_delta_raw':S_DELTA,'measured_train_s_delta_raw':measured,
              'sampling_steps':10,'noise_seed':10043,'training_contract':'GT2D / Body22 / zero-camera / raw / train18 only',
              'branching_policy':'New experiment derived from immutable parent; original resume guard untouched',
              'shape_screen_min_pct':50}
    write(out/'experiment_identity.json',identity);write(out/'data_manifest.json',ck['manifest'])
    records=[]
    for arm in ('control','improved'):records.append(train_arm(arm,ck,data['train'],schedule,out,device,identity))
    a,b=records
    for key in ('initial_state_sha256','random_trace_sha256','end_rng_sha256'):
        if a[key]!=b[key]:raise ValueError('unfair_'+key)
    allrows=[];allbones=[];allframes=[];noises={};reproduction={}
    for arm in ('baseline','control','improved'):
        state=ck['model'] if arm=='baseline' else torch.load(out/f'{arm}_400.pt',map_location='cpu',weights_only=True)['model']
        r,b,f,n,repro=evaluate(arm,state,ck,data,schedule,out,device)
        allrows+=r;allbones+=b;allframes+=f;noises[arm]=n;reproduction.update(repro)
    if any(noises[a]!=noises['baseline'] for a in ('control','improved')):raise ValueError('evaluation_noise_mismatch')
    checks=success_checks(allrows,allframes,allbones)
    report={'execution_status':'PASS','quality':checks,'identity':identity,'arms':records,'results':allrows,
            'baseline_reproduction':reproduction,'evaluation_noise_sha256':noises,
            'baseline_unchanged':sha(private/'historical_last.pt')==cfg['checkpoint_sha256']}
    if not report['baseline_unchanged'] or any(r['status']!='PASS' for r in reproduction.values()):
        report['execution_status']='FAIL'
    write(out/'evaluation.json',report)
    write_csv(out/'comparison.csv',allrows);write_csv(out/'bone_length_by_edge.csv',allbones);write_csv(out/'frame_comparison.csv',allframes)
    write(out/'checksums.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'execution_status':report['execution_status'],'quality':checks},indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input-bundle',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args()
    try:run(a.input_bundle,a.output)
    except Exception as exc:
        out=Path(a.output)
        if out.exists() and not isinstance(exc,FileExistsError):
            write(out/'failure.json',{'status':'FAIL','error_type':type(exc).__name__,'message':str(exc)})
        raise
