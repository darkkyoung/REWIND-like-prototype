"""T4 only: immutable Train300/400 -> paired 400-update arms."""
import argparse
import copy
import json
import os
import platform
import subprocess
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
    if ck.get('scaler') != {}:raise ValueError('fp32_scaler_required')
    if len(opt.state)!=len(list(model.parameters())):raise ValueError('incomplete_adam_state')
    for param,state in opt.state.items():
        if not {'step','exp_avg','exp_avg_sq'} <= state.keys() or state['exp_avg'].shape!=param.shape or state['exp_avg_sq'].shape!=param.shape:
            raise ValueError('incomplete_adam_state')
        ensure_finite([state['exp_avg'],state['exp_avg_sq']], 'adam_state')
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
    trace={'input_sha256':state_hash([x,y,m,t]),'anchors':ids.tolist(),'timesteps':k.cpu().tolist(),'noise_sha256':state_hash(noise),
           'frames_sha256':state_hash(frames[ids]),'sequences_sha256':state_hash(sequences[ids])}
    return existing,main,aux,bone,pairs,trace


def calibrate(ck,data,schedule,frames,sequences,device,settings):
    model,_,_=restore_arm(ck,device);model.train();params=list(model.parameters());ratios=[];trace=[]
    for _ in range(settings['calibration_batches']):
        existing,_,_,bone,pairs,row=draw(model,ck,schedule,data,frames,sequences,device)
        gp=grad_norm(torch.autograd.grad(existing,params,retain_graph=True,allow_unused=True))
        gb=grad_norm(torch.autograd.grad(bone,params,allow_unused=True))
        if gp<=0 or gb<=0 or pairs<=0:raise ValueError('calibration_gradient')
        ratios.append(gp/gb);trace.append({**row,'position_loss':existing.item(),'bone_loss':bone.item(),'position_gradient':gp,'bone_gradient':gb})
    weight=float(settings['calibration_fraction']*np.median(ratios))
    if not np.isfinite(weight) or weight<=0:raise ValueError('calibration_weight')
    return weight,{'train_only_gradient_ratios':ratios,'weight':weight,'batches':trace,'draw_trace_sha256':state_hash(trace)}


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
            ensure_finite(model.parameters(),'parameter')
            ensure_finite([v for state in opt.state.values() for v in state.values() if isinstance(v,torch.Tensor)],'optimizer')
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


def validate_parent(ck,manifest):
    locked=load(HERE.parent/'dataset_scaling_300/config.json')
    if (ck.get('format_version')!='m2_dataset_scaling_v1' or ck.get('arm')!='train300'
        or ck['step']!=400 or ck.get('samples_seen')!=3200 or ck['config']!=locked['model']
        or ck['manifest']!=manifest or ck.get('scaler')!={}):
        raise ValueError('parent_contract')
    # This is the schema written by dataset_scaling_100.train and reused by Train300.
    for path,want in locked['core_code_sha256'].items():
        if ck['experiment_identity']['code_sha256'][path]!=want or sha(ROOT/path)!=want:
            raise ValueError('core_sha_'+path)


def validate_evaluation(data,selection):
    from experiments.final_m2.dataset_scaling_100.prepare_data import motion
    unused=data['unused_motion_preselected']
    for name,d in data.items():
        n=len(d['frame_ids'])
        if d['sequence_ids'].shape!=(n,) or d['frame_ids'].shape!=(n,) or d['frame_ids'].dtype.kind not in 'iu':
            raise ValueError('frame_identity_shape')
        for key,shape,dtype in [('features',(n,8,308),np.float32),('target_raw',(n,8,22,3),np.float32),
                                ('valid',(n,8,22),bool),('time',(n,8),bool)]:
            if d[key].shape!=shape or d[key].dtype!=dtype or not np.isfinite(d[key]).all():raise ValueError('tensor_contract_'+name)
        window_metadata(d)
        if name!='train300':
            if n<2 or len(set(d['sequence_ids']))!=1 or not np.all(np.diff(d['frame_ids'])==1):
                raise ValueError('evaluation_single_contiguous_sequence')
    if selection.get('sequence_id')!=str(unused['sequence_ids'][0]) or not selection.get('selection_reason'):
        raise ValueError('unused_sequence_binding')
    prior=[d for name,d in data.items() if name!='unused_motion_preselected']
    if motion(selection['sequence_id']) in {motion(str(s)) for d in prior for s in d['sequence_ids']}:
        raise ValueError('unused_motion_overlap')
    for key in ('features','target_raw'):
        seen={x.tobytes() for d in prior for x in d[key][:,-1]}
        if any(x.tobytes() in seen for x in unused[key][:,-1]):raise ValueError('unused_exact_pose_overlap')


def bone_metrics(z,edges):
    e=np.asarray(edges);p=z['prediction'].astype(float);g=z['gt'].astype(float);v=z['valid']
    mask=v[:,e[:,0]] & v[:,e[:,1]]
    gl=np.linalg.norm(g[:,e[:,1]]-g[:,e[:,0]],axis=-1)
    pl=np.linalg.norm(p[:,e[:,1]]-p[:,e[:,0]],axis=-1)
    if not mask.any() or np.any(gl[mask]<=1e-12):raise ValueError('undefined_bone_ratio')
    ratio=pl[mask]/gl[mask]
    return {'mean_abs_bone_ratio_deviation_pp':float(np.mean(abs(ratio-1))*100),
            'min_bone_ratio_pct':float(ratio.min()*100)}


def quality_checks(rows):
    result={}
    for split in sorted({r['split'] for r in rows}):
        c=next(r for r in rows if r['split']==split and r['model']=='control')
        i=next(r for r in rows if r['split']==split and r['model']=='improved')
        gates={'bone_distortion_decreases':i['mean_abs_bone_ratio_deviation_pp']<c['mean_abs_bone_ratio_deviation_pp']}
        for key in ('global_joint_error_raw','root_relative_error_22_raw','pelvis_position_error_raw','frame_displacement_error_raw'):
            gates[key]=None if c[key] is None or i[key] is None else i[key]<=1.02*c[key]
        for key in ('motion_magnitude_ratio_pct','joint_spread_ratio_pct','root_relative_motion_ratio_pct'):
            gates[key]=None if c[key] is None or i[key] is None else abs(i[key]-100)<=abs(c[key]-100)+2
        result[split]=gates
    return {'gates':result,'all_pass':all(v is True for gates in result.values() for v in gates.values()),
            'claim':'Exploratory validation; performance improvement requires result review'}


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
    validate_parent(ck,manifest)
    expected=load(HERE.parent/'dataset_scaling_300/input_checksums.json')
    for name in ('opening130.npz','gunplay47.npz'):
        if sha(inp/name)!=expected[name]:raise ValueError('evaluation_input_sha_'+name)
    data={name:read_npz(inp/f'{name}.npz') for name in ('train300','opening130','gunplay47')}
    unused=read_npz(Path(unused_path));data['unused_motion_preselected']=unused
    selection=load(Path(unused_path).with_suffix('.selection.json'))
    if selection['npz_sha256']!=sha(unused_path) or selection['selected_before_training'] is not True:raise ValueError('unused_selection_lock')
    if selection['sequence_id'] != settings['candidate_sequence'] or not np.array_equal(unused['frame_ids'], np.arange(57)):
        raise ValueError('preselected_candidate')
    source_hashes = selection.get('source_pose_sha256', [])
    used_hashes = {row['pose_sha256'] for row in manifest['frames'] + manifest['additional_evaluation']['gunplay47']}
    if len(source_hashes)!=57 or len(set(source_hashes))!=57 or used_hashes.intersection(source_hashes):
        raise ValueError('candidate_source_overlap_or_missing')
    validate_evaluation(data,selection)
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
    identity={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [*HERE.glob('*.py'),ROOT/'experiments/final_m2/displacement_loss_20261116/run_experiment.py']},
              'config_sha256':sha(HERE/'config.json'),'parent_checkpoint_sha256':sha(parent),
              'input_sha256':{name:sha(inp/f'{name}.npz') for name in ('train300','opening130','gunplay47')},
              'unused_selection':selection,'unused_selection_sha256':sha(Path(unused_path).with_suffix('.selection.json'))}
    write(out/'experiment_identity.json',identity)
    from experiments.final_m2.bone_length_loss_20261117.diagnostics import measure
    measure(ck,schedule,data,Path(checkpoint_dir),out,load(HERE.parent/'dataset_scaling_300/checksums.json'))
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
        with np.errstate(divide='ignore',invalid='ignore'):
            extra=extra_metrics(z['prediction'],z['gt'],z['valid'],z['frame_ids'])
        row.update({k:(v if np.isfinite(v) else None) for k,v in extra.items()})
        row.update(bone_metrics(z,manifest['edges']))
    write_csv(out/'frame_comparison.csv',frames)
    write_csv(out/'comparison.csv',rows);write_csv(out/'bone_length_by_edge.csv',bones)
    write(out/'evaluation.json',{'execution_status':'PASS','performance_improvement':'NOT VERIFIED until T4 results reviewed',
        'quality':quality_checks(rows),'identity':identity,'arms':arms,'calibration':calibration,'evaluation_noise':noises,'results':rows,'unused_selection':selection,
        'parent_checkpoint_sha256_after':sha(parent),'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name()}})
    write(out/'checksums.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);p.add_argument('--checkpoint-dir',required=True)
    p.add_argument('--unused-motion',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run(a.input_dir,a.checkpoint_dir,a.unused_motion,a.output)
