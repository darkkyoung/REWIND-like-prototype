"""고정 seed부터 각각 400 update: 데이터만 바꾸는 T4 전용 실험."""
import argparse,copy,hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from rewind_m1.training import seed_run,rng_state,restore_rng
from rewind_m2.models import BodyDenoiser
from rewind_m2.diffusion import DiffusionSchedule
from rewind_m2.training import position_losses
from experiments.final_m2.displacement_loss_20261116.run_experiment import state_hash,ensure_finite,evaluate
from experiments.final_m2.displacement_loss_20261116.loss import window_metadata
from experiments.final_m2.diagnosis_20261111.diagnose import calculate,write_csv
from tools.final_m2_real import sha,load,write
HERE=Path(__file__).resolve().parent

def new_model(cfg,manifest,device):
    seed_run(cfg['seed']);m=BodyDenoiser(cfg,22,manifest['edges']).to(device)
    o=torch.optim.Adam(m.parameters(),lr=cfg['learning_rate']);s=torch.optim.lr_scheduler.StepLR(o,cfg['scheduler_step'],cfg['scheduler_gamma'])
    return m,o,s

def train(arm,cfg,manifest,data,schedule,out,identity,device='cuda',updates=400):
    model,opt,scheduler=new_model(cfg,manifest,device);initial={'model':state_hash(model.state_dict()),'optimizer':state_hash(opt.state_dict()),'scheduler':state_hash(scheduler.state_dict()),'rng':state_hash(rng_state())}
    d={k:torch.from_numpy(data[k]) for k in ('features','target_raw','valid','time')};trace=[];logs=[];start=time.monotonic();counts=np.zeros(len(data['frame_ids']),int);saved=[]
    model.train()
    with (out/f'{arm}_training.jsonl').open('x') as stream:
        for step in range(1,updates+1):
            if time.monotonic()-start>300:raise ValueError('arm_300_seconds_budget')
            ids=torch.randint(len(data['frame_ids']),(8,));np.add.at(counts,ids.numpy(),1)
            x,y,m,t=[d[k][ids].to(device) for k in ('features','target_raw','valid','time')];cam=torch.zeros(8,8,2,9,device=device)
            k=torch.randint(1,cfg['diffusion_steps']+1,(8,),device=device);noise=torch.randn_like(y)
            trace.append({'step':step,'anchors':ids.tolist(),'timesteps':k.cpu().tolist(),'noise_sha256':state_hash(noise)})
            clean=torch.where((m&t[...,None])[...,None],y/100.,0.);noisy=schedule.q_sample(clean,noise,k)
            opt.zero_grad(set_to_none=True);p,f=model(noisy,x,cam,k,t);loss,main,aux=position_losses(p,f,y,m,t,100.,cfg['frame_loss_weight'])
            ensure_finite([loss,main,aux],'loss');loss.backward();ensure_finite([p.grad for p in model.parameters()],'gradient')
            lr=float(opt.param_groups[0]['lr']);opt.step();scheduler.step();ensure_finite(model.parameters(),'parameter')
            ensure_finite([v for state in opt.state.values() for v in state.values() if isinstance(v,torch.Tensor)],'optimizer')
            row={'model':arm,'step':step,'loss':float(loss.detach()),'x0_loss':float(main.detach()),'aux_position_loss':float(aux.detach()),'lr_used':lr,'lr_next':float(opt.param_groups[0]['lr']),'elapsed_seconds':time.monotonic()-start}
            logs.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if step%100==0 or step==updates:
                ck={'format_version':'m2_dataset_scaling_v1','step':step,'samples_seen':step*8,'arm':arm,'config':cfg,'manifest':manifest,'experiment_identity':identity,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':{},'rng':rng_state(),'alpha_bar':schedule.alpha_bar,'initial_state_sha256':initial}
                path=out/f'{arm}_{step}.pt';torch.save(ck,path);saved.append({'step':step,'file':path.name,'sha256':sha(path)});print(json.dumps(row),flush=True)
    if scheduler.last_epoch!=updates or any(float(s['step'])!=updates for s in opt.state.values()):raise ValueError('update_count')
    write_csv(out/f'{arm}_training.csv',logs);write(out/f'{arm}_random_trace.json',trace)
    record={'model':arm,'updates':updates,'samples_seen':int(counts.sum()),'distinct_anchors_seen':int((counts>0).sum()),'anchor_sample_counts':counts.tolist(),'initial_state_sha256':initial,'diffusion_trace_sha256':state_hash([{k:v for k,v in r.items() if k!='anchors'} for r in trace]),'full_trace_sha256':state_hash(trace),'end_rng_sha256':state_hash(rng_state()),'checkpoints':saved,'seconds':time.monotonic()-start}
    write(out/f'{arm}_identity.json',record);return record

def extra_metrics(p,g,v,ids):
    p,g=p.astype(float),g.astype(float);pr=p-p[:,:1];gr=g-g[:,:1];dp=np.diff(p,axis=0);dg=np.diff(g,axis=0);drp=np.diff(pr,axis=0);drg=np.diff(gr,axis=0)
    mask=v[1:]&v[:-1]&(np.diff(ids)==1)[:,None];pm=np.linalg.norm(dp,axis=-1);gm=np.linalg.norm(dg,axis=-1);direction=mask&(pm>1e-6)&(gm>1e-6)
    cos=np.sum(dp*dg,axis=-1)/np.maximum(pm*gm,1e-30);result={'direction_cosine':float(cos[direction].mean()),'direction_angle_deg':float(np.degrees(np.arccos(np.clip(cos[direction],-1,1))).mean()),'direction_pairs':int(direction.sum())}
    rr_mask=mask&v[1:,:1]&v[:-1,:1]
    for label,joints in [('arms',[6,7,8,9,10,11,12,13]),('legs',[14,15,16,17,18,19,20,21]),('nonroot',list(range(1,22)))]:
        a=np.linalg.norm(drp[:,joints]-drg[:,joints],axis=-1);b=rr_mask[:,joints]
        result[label+'_rr_displacement_error_raw']=float(a[b].mean())
        result[label+'_rr_motion_ratio_pct']=100*float(np.linalg.norm(drp[:,joints],axis=-1)[b].mean()/np.linalg.norm(drg[:,joints],axis=-1)[b].mean())
    # Time-varying pose amplitude about each joint's own sequence mean, after pelvis removal.
    pc=pr-pr.mean(0,keepdims=True);gc=gr-gr.mean(0,keepdims=True)
    result['rr_temporal_pose_rms_ratio_pct']=100*float(np.sqrt(np.mean(pc[v]**2))/np.sqrt(np.mean(gc[v]**2)))
    result['rr_temporal_pose_error_raw']=float(np.linalg.norm(pc-gc,axis=-1)[v].mean())
    return result

def quality(rows):
    a=next(r for r in rows if r['model']=='train18' and r['split']=='opening130');b=next(r for r in rows if r['model']=='train100' and r['split']=='opening130')
    gates={k:b[k]<a[k] for k in ('global_joint_error_raw','root_relative_error_22_raw','frame_displacement_error_raw')}
    gates.update({k:abs(b[k]-100)<abs(a[k]-100) for k in ('motion_magnitude_ratio_pct','root_relative_motion_ratio_pct')})
    gates['pelvis_no_more_than_2pct_worse']=b['pelvis_position_error_raw']<=1.02*a['pelvis_position_error_raw']
    gates['bone_distortion_no_more_than_2pp_worse']=b['mean_abs_bone_ratio_deviation_pp']<=a['mean_abs_bone_ratio_deviation_pp']+2
    gates['spread_no_more_than_2pp_farther']=abs(b['joint_spread_ratio_pct']-100)<=abs(a['joint_spread_ratio_pct']-100)+2
    primary=all(gates[k] for k in ('global_joint_error_raw','root_relative_error_22_raw','frame_displacement_error_raw'))
    status='PASS' if all(gates.values()) else 'PARTIAL' if any(gates[k] for k in ('global_joint_error_raw','root_relative_error_22_raw','frame_displacement_error_raw')) else 'FAIL'
    return {'status':status,'all_predeclared_gates_pass':all(gates.values()),'primary_three_errors_improved':primary,'gates':gates,'interpretation':'Exploratory single seed; no statistical significance claim; all metrics reported.'}

def run(input_dir,output):
    inp=Path(input_dir);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():raise ValueError('requires_actual_T4')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    for name,h in load(inp/'input_checksums.json').items():
        if Path(name).name!=name or sha(inp/name)!=h:raise ValueError('input_sha')
    locked=load(HERE/'config.json')
    assert sha(inp/'input_checksums.json')==locked['input_checksums_sha256']
    for name,h in locked['core_code_sha256'].items():assert sha(ROOT/name)==h
    validation=load(inp/'input_validation.json');assert validation['status']=='PASS' and validation['unique_train_frames']==100
    manifest=load(inp/'data_selection_manifest.json');cfg=load(HERE/'config.json')['model'];data={}
    for name in ('train18','train100','standing50','dodging32','opening130'):
        with np.load(inp/f'{name}.npz',allow_pickle=False) as z:data[name]={k:z[k] for k in z.files}
        window_metadata(data[name])
        for k in ('features','target_raw'):
            assert data[name][k].dtype==np.float32 and np.isfinite(data[name][k]).all()
    for k in data['train18']:np.testing.assert_array_equal(data['train18'][k],data['train100'][k][:18])
    assert len(set(zip(data['train100']['sequence_ids'],data['train100']['frame_ids'])))==100
    schedule=DiffusionSchedule(cfg['diffusion_steps'],cfg['beta_start'],cfg['beta_end'])
    identity={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in list(HERE.glob('*.py'))+[ROOT/f'rewind_m2/{x}.py' for x in ('models','diffusion','training','streaming')]},'config_sha256':sha(HERE/'config.json'),'input_checksums_sha256':sha(inp/'input_checksums.json'),'initialization':'seed43 fresh random weights; no parent checkpoint loaded','precision':'FP32','device':torch.cuda.get_device_name(),'python':platform.python_version(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'training_frames':[18,100],'updates_each':400,'evaluation_noise_seed':10043,'evaluation_steps':10}
    write(out/'experiment_identity.json',identity);records=[]
    # No evaluation until both preassigned terminal models have been saved.
    for arm in ('train18','train100'):records.append(train(arm,cfg,manifest,data[arm],schedule,out,identity))
    assert records[0]['initial_state_sha256']==records[1]['initial_state_sha256']
    assert records[0]['diffusion_trace_sha256']==records[1]['diffusion_trace_sha256']
    assert records[1]['distinct_anchors_seen']==100 and records[0]['distinct_anchors_seen']==18
    evaldata={k:data[k] for k in ('train18','standing50','dodging32','opening130')};allrows=[];allbones=[];allframes=[];noises={}
    for arm in ('train18','train100'):
        ck=torch.load(out/f'{arm}_400.pt',map_location='cpu',weights_only=True)
        rows,bones,frames,noise,_=evaluate(arm,ck['model'],{'config':cfg,'manifest':manifest},evaldata,schedule,out,'cuda');noises[arm]=noise
        for row in rows:
            z=np.load(out/f"{row['split']}_{arm}_predictions.npz",allow_pickle=False);row.update(extra_metrics(z['prediction'],z['gt'],z['valid'],z['frame_ids']))
            e=np.array(manifest['edges']);g=z['gt'].astype(float);p=z['prediction'].astype(float);br=np.linalg.norm(p[:,e[:,1]]-p[:,e[:,0]],axis=-1)/np.linalg.norm(g[:,e[:,1]]-g[:,e[:,0]],axis=-1)
            row['mean_abs_bone_ratio_deviation_pp']=float(np.mean(abs(br-1))*100);row['min_bone_ratio_pct']=float(br.min()*100)
            row['evaluation_use']='historical_holdout' if row['split']=='opening130' else 'train_fit' if arm=='train100' or row['split']=='train18' else 'not_trained_by_control_but_training_data_for_experiment'
        allrows+=rows;allbones+=bones;allframes+=frames
    assert noises['train18']==noises['train100']
    write_csv(out/'comparison.csv',allrows);write_csv(out/'bone_length_by_edge.csv',allbones);write_csv(out/'frame_comparison.csv',allframes)
    report={'execution_status':'PASS','quality':quality(allrows),'arms':records,'evaluation_noise_sha256':noises,'results':allrows,'identity':identity}
    write(out/'evaluation.json',report);write(out/'checksums.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()});print(json.dumps({'execution_status':'PASS','quality':report['quality']},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    try:run(a.input_dir,a.output)
    except Exception as e:
        out=Path(a.output)
        if out.exists() and not isinstance(e,FileExistsError):write(out/'failure.json',{'status':'FAIL','error_type':type(e).__name__,'message':str(e)})
        raise
