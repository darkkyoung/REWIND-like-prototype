"""검증된 과거148 Pose 입력으로 고정 M2 추론. 학습 및 Drive 읽기 없음."""
import argparse,csv,hashlib,json,os,platform,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from rewind_m2.models import BodyDenoiser
from rewind_m2.diffusion import DiffusionSchedule
from rewind_m2.streaming import AnchorSampler,keyed_frame_noise
from rewind_m2.final_metrics import evaluate_motion,latency_summary


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def load(path):return json.loads(Path(path).read_text())


def prepare(bundle,output,cfg):
    """고정된 준비본 SHA, 모든 파일 SHA, checkpoint/manifest/tensor 계약 검사."""
    if Path(bundle).stat().st_size>8*1024**2 or sha(bundle)!=cfg['input_bundle_sha256']:raise ValueError('input_bundle_sha')
    dest=output/'private_input';dest.mkdir(exist_ok=False)
    with zipfile.ZipFile(bundle) as z:
        entries=z.infolist()
        if len(entries)>20 or sum(i.file_size for i in entries)>16*1024**2:raise ValueError('input_budget')
        if len({i.filename for i in entries})!=len(entries) or any(Path(i.filename).name!=i.filename or '\\' in i.filename for i in entries):raise ValueError('unsafe_zip')
        if z.testzip() is not None:raise ValueError('input_crc')
        z.extractall(dest)
    for p,h in load(dest/'input_checksums.json').items():
        if Path(p).name!=p or sha(dest/p)!=h:raise ValueError('input_file_sha')
    checkpoint=dest/'historical_last.pt'
    if sha(checkpoint)!=cfg['checkpoint_sha256']:raise ValueError('checkpoint_sha')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    m=load(dest/'data_manifest.json');mc=load(dest/'config.json')
    body={k:v for k,v in m.items() if k!='sha256'}
    if (hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()!=cfg['manifest_sha256'] or ck['manifest']!=m
        or ck['config']!=mc or ck['step']!=300 or ck['format_version']!='m2_raw_pose_diagnostic_v1'):raise ValueError('checkpoint_manifest_config')
    if len(m['joint_names'])!=22 or m['coordinate_frame']!='stored_device_pts3d' or mc['pixel_scale']!=1024 or mc['target_scale_raw']!=100:raise ValueError('raw_contract')
    data={}
    for split,count in [('train',18),('holdout',130)]:
        with np.load(dest/f'{split}.npz',allow_pickle=False) as z:data[split]={k:z[k] for k in z.files}
        d=data[split];old=np.load(dest/f'historical_{split}_motion.npz',allow_pickle=False)
        if d['features'].shape!=(count,8,308) or d['target_raw'].shape!=(count,8,22,3) or d['valid'].dtype!=bool or d['time'].dtype!=bool:raise ValueError('tensor_contract')
        for k in ('features','target_raw'):
            if d[k].dtype!=np.float32 or not np.isfinite(d[k]).all():raise ValueError('dtype_finite')
        if len(set(d['sequence_ids']))!=1 or not np.array_equal(d['frame_ids'],np.arange(count)):raise ValueError('sequence_frames')
        seq=str(d['sequence_ids'][0]);expected=m['sequences'][seq]
        if expected['split']!=split or expected['frames']!=d['frame_ids'].tolist():raise ValueError('split_manifest')
        np.testing.assert_array_equal(d['target_raw'][:,-1],old['gt']);np.testing.assert_array_equal(d['valid'][:,-1],old['valid'])
    return ck,data,dest


def infer_frame(sampler,features,sequence_id,frame_id,seed,device):
    """GT3D/정답 라벨을 받을 인자가 없다. Reliability routing은 별도 미구현."""
    x=torch.as_tensor(features,dtype=torch.float32,device=device)
    cam=torch.zeros(2,9,device=device)
    n=keyed_frame_noise(sequence_id,frame_id,22,seed).to(device)
    return sampler.push(sequence_id,frame_id,x,cam,n)*100.


def run(bundle,output,device):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    cfg=load(ROOT/'experiments/final_m2/configs/real148.json');write(out/'run_config.json',cfg)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('cuda_unavailable')
    ck,data,private=prepare(bundle,out,cfg);mc=ck['config']
    model=BodyDenoiser(mc,22,ck['manifest']['edges'])
    for name,b in model.named_buffers():
        if not torch.equal(b,ck['model'][name]):raise ValueError('structural_buffer')
    model.load_state_dict(ck['model'],strict=True);model.to(device).eval().requires_grad_(False)
    schedule=DiffusionSchedule(mc['diffusion_steps'],mc['beta_start'],mc['beta_end'])
    if not torch.equal(schedule.alpha_bar,ck['alpha_bar']):raise ValueError('alpha_schedule')
    # 원래 소스와 같은지 별도 검사. 신규 모델이 아니다.
    core={f'rewind_m2/{n}.py':sha(ROOT/f'rewind_m2/{n}.py') for n in ['models','diffusion','streaming','training']}
    if any(ck['source']['code_sha256'][n]!=h for n,h in core.items()):raise ValueError('core_code_mismatch')
    def sync():
        if device=='cuda':torch.cuda.synchronize()
    started=time.monotonic();records=[];results={};predictions={};baseline={}
    noises={s:hashlib.sha256(b''.join(keyed_frame_noise(str(seq),int(f),22,cfg['noise_seed']).numpy().tobytes() for seq,f in zip(d['sequence_ids'],d['frame_ids']))).hexdigest() for s,d in data.items()}
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    for split,d in data.items():
        ids=[(str(s),int(f)) for s,f in zip(d['sequence_ids'],d['frame_ids'])]
        samplers={steps:AnchorSampler(model,schedule,8,steps) for steps in cfg['sampling_steps']}
        with torch.inference_mode():
            for sampler in samplers.values():
                for i in range(cfg['warmup_frames']):
                    infer_frame(sampler,d['features'][i,-1],*ids[i],cfg['noise_seed'],device)
                sync();sampler.reset()
            for repeat in range(cfg['repeats']):
                for steps in ([5,10] if repeat%2==0 else [10,5]):
                    sampler=samplers[steps];sampler.reset();ps=[]
                    for i,(seq,frame) in enumerate(ids):
                        if time.monotonic()-started>cfg['max_inference_seconds']:raise ValueError('inference_time_budget')
                        sync();t=time.perf_counter_ns()
                        p=infer_frame(sampler,d['features'][i,-1],seq,frame,cfg['noise_seed'],device)
                        sync();ms=(time.perf_counter_ns()-t)/1e6
                        ps.append(p.cpu().numpy())
                        records.append(dict(split=split,sampling_steps=steps,repeat=repeat,frame_id=frame,latency_ms=ms))
                    ps=np.stack(ps);key=f'{split}_{steps}'
                    if key in predictions and not np.array_equal(ps,predictions[key]):raise ValueError('repeat_not_deterministic')
                    predictions[key]=ps
                    write(out/'progress.json',dict(status='running',split=split,repeat=repeat,sampling_steps=steps))
        old=np.load(private/f'historical_{split}_motion.npz',allow_pickle=False)
        target,valid=d['target_raw'][:,-1],d['valid'][:,-1]
        for steps in [5,10]:
            key=f'{split}_{steps}';v=evaluate_motion(predictions[key],target,valid,ids)
            v.update(latency_summary([r['latency_ms'] for r in records if r['split']==split and r['sampling_steps']==steps]))
            results[key]={'split':split,'sampling_steps':steps,'frames':len(ids),**v}
            np.savez_compressed(out/f'{key}_predictions.npz',prediction=predictions[key],gt=target,valid=valid,frame_ids=d['frame_ids'])
        metric=results[f'{split}_10'];historical=load(private/'historical_metrics.json')['final'][split]['ddim_mpjpe_raw']
        delta=float(np.max(np.abs(predictions[f'{split}_10']-old['ddim'])))
        baseline[split]={'stored_joint_error_raw':historical,'recomputed_joint_error_raw':metric['joint_position_error_raw'],
                         'absolute_joint_error_difference_raw':abs(metric['joint_position_error_raw']-historical),
                         'max_coordinate_difference_raw':delta,'stored_motion_ratio_recomputed':evaluate_motion(old['ddim'],old['gt'],old['valid'],ids)['motion_magnitude_ratio'],
                         'status':'PASS' if abs(metric['joint_position_error_raw']-historical)<cfg['reproduction_joint_error_tolerance_raw'] and delta<cfg['reproduction_max_coordinate_tolerance_raw'] else 'FAIL'}
    try:revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:revision=None
    env={'python':platform.python_version(),'torch':str(torch.__version__),'device':device,'gpu':torch.cuda.get_device_name() if device=='cuda' else None,'cuda':torch.version.cuda,'cpu_threads':1,'precision':'float32','batch_size':1,'cpu':platform.processor()}
    report={'status':'completed','data_kind':'real_unrealego2_pose','new_optimizer_updates':0,'checkpoint_sha256':cfg['checkpoint_sha256'],
            'manifest_sha256':cfg['manifest_sha256'],'input_bundle_sha256':cfg['input_bundle_sha256'],'environment':env,'code_revision':revision,
            'core_code_sha256':core,'runner_sha256':sha(__file__),'initial_noise_sha256':noises,'results':results,'baseline_reproduction':baseline,
            'evaluation_seconds':time.monotonic()-started,'peak_vram_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
            'timing_scope':'B1 streaming emitted frame; CPU keyed noise, H2D, history packing, DDIM, scaling, sync included; disk, RGB frontend, D2H, metrics excluded',
            'measurement':'8 warmups/arm/split,3 repeats,alternating arm order,perf_counter_ns,CUDA synchronize before/after,linear P95 of individual frames',
            'limitations':['GT stereo 2D, no RGB frontend','raw units and capture FPS unverified','zero camera ablation','train-fit18, historical holdout130, not official independent test','no routing','CPU timing is not T4 timing']}
    write(out/'evaluation.json',report);write(out/'data_manifest.json',ck['manifest'])
    for name,rows in [('comparison.csv',list(results.values())),('frame_latency.csv',records)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(out/'progress.json',{'status':'completed'});return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-bundle',required=True);p.add_argument('--output',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    try:
        r=run(a.input_bundle,a.output,a.device);print(json.dumps({'status':r['status'],'environment':r['environment'],'results':r['results']},indent=2))
    except Exception as e:
        dest=Path(a.output)
        if dest.exists() and not isinstance(e,FileExistsError):write(dest/'failure.json',{'status':'failed','error_type':type(e).__name__,'error_code':str(e) if isinstance(e,ValueError) and str(e).replace('_','').isalnum() else 'local_validation_failed'})
        raise
