"""UnrealEgo2 raw-unit Pose 진단. 기존 M2 denoiser/schedule/loss 재사용.

별도 diagnostic 계약: GT2D, zero camera ablation, 개발용 skeleton.
기존 dataset_sample 검증 gate를 수정하거나 verified=true로 위장하지 않는다.
"""
import argparse
import hashlib
import json
import platform
import re
import time
from pathlib import Path

import torch

from rewind_archive.matching import verify_extraction, identify, contiguous_runs
from rewind_data.audit import FrameSample, _validate_joints
from rewind_data.windows import CausalWindowStream
from rewind_m1.training import seed_run, rng_state, restore_rng, save_checkpoint
from rewind_m2.models import BodyDenoiser
from rewind_m2.diffusion import DiffusionSchedule
from rewind_m2.training import position_losses, source_identity
from rewind_m2.streaming import keyed_frame_noise

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'm2_raw_pose_diagnostic_v1'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_data(extraction, selection, config, allow_historical_extractor=False):
    """Hash-verified files only; explicit sequence split; numerical masks; causal gap reset."""
    report = verify_extraction(extraction,allow_code_change=allow_historical_extractor)
    if report['archive_manifest']['kind'] != 'pose' or report['archive_manifest']['source_scope'] != 'user_drive':
        raise ValueError('사용자 Drive에서 추출된 실제 pose만 허용; synthetic fallback 없음')
    if set(selection) != {'train','holdout'} or any(not isinstance(v,list) or not v for v in selection.values()):
        raise ValueError('명시적 train/holdout sequence 목록 필요')
    chosen = [s for values in selection.values() for s in values]
    if len(set(chosen)) != len(chosen):
        raise ValueError('sequence split 중복')
    # Conservative motion-name group only; not a proof of source mocap independence.
    motion = lambda s: re.sub(r'__\d+_$','',s.rsplit('/',1)[-1]).casefold()
    if {motion(s) for s in selection['train']} & {motion(s) for s in selection['holdout']}:
        raise ValueError('motion-name group split 중복')
    skeleton = json.loads((ROOT/'configs/skeleton_unrealego2_dev.json').read_text())
    names = skeleton['groups']['body']
    edges = [[names.index(a),names.index(b)] for a,b in skeleton['edges'] if a in names and b in names]
    records = {s:[] for s in chosen}
    payloads = []
    for entry in report['files']:
        seq,frame,_ = identify(entry['member'],'pose')
        if seq not in records:
            continue
        doc = json.loads((Path(extraction)/entry['relative_path']).read_text())
        if str(doc.get('frame')) != str(frame):
            raise ValueError('raw frame과 파일 번호 불일치')
        records[seq].append((frame,doc))
        payloads.append({'member':entry['member'],'sha256':entry['sha256']})
    packed, inventory = {}, {}
    for split,sequences in selection.items():
        xs,ys,ms,ts,anchors = [],[],[],[],[]
        for seq in sequences:
            rows = sorted(records[seq],key=lambda r:r[0])
            indices = [r[0] for r in rows]
            if len(set(indices)) != len(indices) or not rows:
                raise ValueError('sequence 누락 또는 중복 frame: '+seq)
            runs = contiguous_runs(indices)
            if split=='train' and max(b-a+1 for a,b in runs)<config['window_size']:
                raise ValueError('train에 window_size 이상의 연속 프레임이 필요')
            inventory[seq] = {'split':split,'frames':indices,'runs':runs,'sequence_complete':False,
                              'official_split':'not_revalidated_in_this_run','motion_group':motion(seq)}
            stream = CausalWindowStream(config['window_size'],names)
            for index,doc in rows:
                sample = FrameSample(f'{seq}/json/frame_{index}.json',seq,index,raw_document=doc)
                _validate_joints(sample,names)
                window = stream.push(sample)
                x=torch.zeros(config['window_size'],len(names),14)
                y=torch.zeros(config['window_size'],len(names),3)
                valid=torch.zeros(config['window_size'],len(names),dtype=torch.bool)
                for slot,frame in enumerate(window.frames):
                    if frame is None: continue
                    fields={f.name:f for f in frame.fields}
                    for j in range(len(names)):
                        f=fields['device_pts3d']
                        if f.valid_mask[j]:
                            y[slot,j]=torch.tensor(f.coordinates[j]); valid[slot,j]=True
                        for v,side in enumerate(('left','right')):
                            f2=fields[f'camera_{side}_pts2d']
                            if f2.valid_mask[j]:
                                # Fixed pixel scale; not learned from validation/future data.
                                x[slot,j,v*7:v*7+2]=torch.tensor(f2.coordinates[j])/config['pixel_scale']
                                x[slot,j,v*7+2]=1
                            # confidence/visibility value and known bits stay 0 (unknown).
                if not valid[-1].any() or not torch.isfinite(x).all() or not torch.isfinite(y).all():
                    raise ValueError('유효 GT 누락 또는 float32 비유한 값')
                xs.append(x.flatten(1));ys.append(y);ms.append(valid)
                ts.append(torch.tensor(window.to_dict()['time_valid'],dtype=torch.bool));anchors.append((seq,index))
        packed[split]={'features':torch.stack(xs),'target_raw':torch.stack(ys),'valid':torch.stack(ms),
                       'time':torch.stack(ts),'anchors':anchors}
    manifest = {'schema_version':FORMAT,'data_kind':'unrealego2_archive_pose',
        'source_scope':'user_drive','input_source':'gt_2d','camera_policy':'zero_features_ablation_not_extrinsics',
        'coordinate_frame':'stored_device_pts3d','distance_unit':'raw_annotation_unit_unverified',
        'pixel_scale':config['pixel_scale'],'target_scale_raw':config['target_scale_raw'],
        'skeleton_policy':'existing_development_22_body_joints_not_official_benchmark',
        'joint_names':names,'edges':edges,'selection':selection,'sequences':inventory,
        'payloads':sorted(payloads,key=lambda x:x['member']),
        'extraction_manifest_sha256':sha(Path(extraction)/'extraction_manifest.json'),
        'historical_extractor_accepted':allow_historical_extractor,
        'extractor_code_hashes':report['code_sha256'],
        'original_motion_independence':'unverified','rgb_alignment':'not_evaluated_pose_only',
        'full_m2_dataset_gate_passed':False,'whole_archive_integrity_verified':report['whole_archive_integrity_verified']}
    manifest['sha256']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    return packed,manifest


class PosePilot:
    def __init__(self,config,data,manifest,device='cuda'):
        self.config,self.data,self.manifest=config,data,manifest
        if config['target_scale_raw']<=0 or config['pixel_scale']<=0: raise ValueError('positive scale required')
        seed_run(config['seed'])
        self.device=torch.device(device)
        self.model=BodyDenoiser(config,len(manifest['joint_names']),manifest['edges']).to(self.device)
        self.diffusion=DiffusionSchedule(config['diffusion_steps'],config['beta_start'],config['beta_end'])
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=config['learning_rate'])
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,config['scheduler_step'],config['scheduler_gamma'])
        self.scaler=torch.amp.GradScaler('cuda',enabled=False)
        self.step=0
        self.source=source_identity()
        self.environment={'python':platform.python_version(),'torch':str(torch.__version__),'device':str(self.device),
            'cuda_runtime':torch.version.cuda,'precision':'float32','threads':torch.get_num_threads(),
            'gpu':torch.cuda.get_device_name(self.device) if self.device.type=='cuda' else None,
            'gpu_total_memory_bytes':torch.cuda.get_device_properties(self.device).total_memory if self.device.type=='cuda' else None}

    def inputs(self,split,indices):
        d=self.data[split]
        x,y,m,t=[d[key][indices].to(self.device) for key in ('features','target_raw','valid','time')]
        camera=torch.zeros(len(indices),self.config['window_size'],2,9,device=self.device)
        return x,y,m,t,camera

    def train_step(self):
        cfg=self.config
        ids=torch.randint(len(self.data['train']['anchors']),(cfg['batch_size'],))
        x,y,m,t,camera=self.inputs('train',ids)
        k=torch.randint(1,cfg['diffusion_steps']+1,(len(ids),),device=self.device)
        clean=torch.where((m&t[...,None])[...,None],y/cfg['target_scale_raw'],0.)
        noisy=self.diffusion.q_sample(clean,torch.randn_like(clean),k)
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        p,f=self.model(noisy,x,camera,k,t)
        # Existing M2 loss is unit-agnostic algebra; raw scale is used explicitly here.
        loss,main,aux=position_losses(p,f,y,m,t,cfg['target_scale_raw'],cfg['frame_loss_weight'])
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise ValueError('nonfinite loss/gradient')
        self.optimizer.step();self.scheduler.step();self.step+=1
        return {'step':self.step,'loss':float(loss.detach()),'x0_mse':float(main.detach()),'frame_mse':float(aux.detach())}

    def noise(self,split):
        d=self.data[split];w=self.config['window_size'];j=len(self.manifest['joint_names'])
        result=torch.zeros(len(d['anchors']),w,j,3)
        for row,(seq,index) in enumerate(d['anchors']):
            for slot in range(w):
                if d['time'][row,slot]:
                    result[row,slot]=keyed_frame_noise(seq,index-w+1+slot,j,self.config['seed']+10000)
        return result

    @torch.no_grad()
    def evaluate(self,split):
        cfg=self.config;d=self.data[split];noise=self.noise(split)
        self.model.eval(); arrays={k:[] for k in ('gt','noisy','reconstructed','ddim','valid')}
        if self.device.type=='cuda':torch.cuda.synchronize()
        started=time.perf_counter()
        for start in range(0,len(d['anchors']),cfg['batch_size']):
            ids=torch.arange(start,min(start+cfg['batch_size'],len(d['anchors'])))
            x,y,m,t,cam=self.inputs(split,ids);n=noise[ids].to(self.device)
            k=torch.full((len(ids),),cfg['diagnostic_k'],device=self.device,dtype=torch.long)
            clean=torch.where(m[...,None],y/cfg['target_scale_raw'],0.)
            noisy=self.diffusion.q_sample(clean,n,k)
            reconstructed,_=self.model(noisy,x,cam,k,t)
            ddim=self.diffusion.sample(self.model,x,cam,t,n,cfg['sampling_steps'])
            for key,value in [('gt',y),('noisy',noisy*cfg['target_scale_raw']),
                              ('reconstructed',reconstructed*cfg['target_scale_raw']),('ddim',ddim*cfg['target_scale_raw'])]:
                arrays[key].append(value[:,-1].cpu())
            arrays['valid'].append(m[:,-1].cpu())
        if self.device.type=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        arrays={k:torch.cat(v) for k,v in arrays.items()}
        mask=arrays['valid'];gt=arrays['gt'];metrics={}
        for key in ('noisy','reconstructed','ddim'):
            delta=arrays[key][mask]-gt[mask]
            metrics[key+'_mpjpe_raw']=float(delta.norm(dim=-1).mean())
            metrics[key+'_normalized_mse']=float((delta/cfg['target_scale_raw']).square().mean())
        metrics.update(split=split,frames=len(gt),valid_joints=int(mask.sum()),evaluation_seconds=elapsed,
            evaluation_kind='real_archive_pose_gt2d_raw_unit_diagnostic',diagnostic_k=cfg['diagnostic_k'],
            reconstruction_protocol='single_x0_call_on_q_sample_gt_not_distilled',
            ddim_protocol='existing_ddim_from_independent_gaussian_not_noisy_gt',
            sampling_steps=cfg['sampling_steps'],noise_seed=cfg['seed']+10000,alignment='none')
        return metrics,arrays

    def checkpoint(self):
        return {'format_version':FORMAT,'config':self.config,'manifest':self.manifest,'source':self.source,
                'environment':self.environment,'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),
                'scheduler':self.scheduler.state_dict(),'scaler':self.scaler.state_dict(),'rng':rng_state(),
                'step':self.step,'alpha_bar':self.diffusion.alpha_bar,'samples_seen':self.step*self.config['batch_size']}

    def resume(self,path):
        ck=torch.load(path,map_location='cpu',weights_only=True)
        for key,value in [('format_version',FORMAT),('config',self.config),('manifest',self.manifest),('environment',self.environment)]:
            if ck[key]!=value:raise ValueError('resume mismatch: '+key)
        if ck['source']['code_sha256']!=self.source['code_sha256'] or not torch.equal(ck['alpha_bar'],self.diffusion.alpha_bar):
            raise ValueError('resume code/schedule mismatch')
        self.model.load_state_dict(ck['model']);self.optimizer.load_state_dict(ck['optimizer'])
        self.scheduler.load_state_dict(ck['scheduler']);self.scaler.load_state_dict(ck['scaler'])
        self.step=ck['step'];restore_rng(ck['rng'])


def visualize(output,split,arrays,data,manifest,history):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    out=Path(output)
    np.savez_compressed(out/f'{split}_motion.npz',**{k:v.numpy() for k,v in arrays.items()},
                        frame_indices=np.array([i for _,i in data['anchors']]))
    fig,ax=plt.subplots(figsize=(8,3.5))
    ax.plot([r['step'] for r in history],[r['loss'] for r in history],lw=1,label='x0 + frame MSE')
    ax.set(xlabel='Optimizer update',ylabel='Normalized training loss',title='Real archive Pose diagnostic')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'training_loss.png',dpi=160);plt.close(fig)
    # Animate only the first contiguous run, never bridge frame gaps.
    anchors=data['anchors']; end=1
    while end<len(anchors) and anchors[end][0]==anchors[0][0] and anchors[end][1]==anchors[end-1][1]+1:end+=1
    end=min(end,120)
    keys=['gt','noisy','reconstructed']; titles=['GT','Noisy q(x_t | GT)','Reconstructed x0']
    combined=torch.cat([arrays[k][:end] for k in keys]).numpy()
    lo=combined.min(axis=(0,1));hi=combined.max(axis=(0,1));center=(lo+hi)/2;radius=max(float((hi-lo).max()/2),1.)*1.1
    fig=plt.figure(figsize=(12,4)); axes=[fig.add_subplot(1,3,i+1,projection='3d') for i in range(3)]
    def draw(frame):
        for ax,key,title in zip(axes,keys,titles):
            ax.clear();pose=arrays[key][frame].numpy();valid=arrays['valid'][frame].numpy()
            for a,b in manifest['edges']:
                if valid[a] and valid[b]:ax.plot(*pose[[a,b]].T,lw=2,color='#246cbd')
            ax.scatter(*pose[valid].T,s=12,color='#e48239')
            ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius),title=title,xlabel='raw x',ylabel='raw y',zlabel='raw z')
            ax.set_box_aspect((1,1,1));ax.view_init(elev=-70,azim=-90)
        fig.suptitle(f'{split}: frame {anchors[frame][1]} | raw units; display FPS=10, capture FPS unknown')
        return []
    draw(0);fig.tight_layout();fig.savefig(out/f'{split}_motion.png',dpi=140)
    FuncAnimation(fig,draw,frames=end,interval=100).save(out/f'{split}_motion.gif',writer=PillowWriter(fps=10))
    plt.close(fig)


def run(extraction,selection,config,output,steps=300,device='cuda',allow_historical_extractor=False,resume=None):
    started=time.perf_counter();out=Path(output)
    out.mkdir(parents=True,exist_ok=False)  # Resume also uses a new segment folder; previous logs stay intact.
    data,manifest=prepare_data(extraction,selection,config,allow_historical_extractor)
    write_json(out/'data_manifest.json',manifest);write_json(out/'config.json',config)
    trainer=PosePilot(config,data,manifest,device)
    if resume:trainer.resume(resume)
    history=[];initial={s:trainer.evaluate(s)[0] for s in data}
    if trainer.device.type=='cuda':torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    train_start=time.perf_counter()
    while trainer.step<steps:
        if time.perf_counter()-train_start>config['max_train_seconds']:break
        history.append(trainer.train_step())
        if trainer.step%25==0 or trainer.step==steps:
            print(json.dumps(history[-1]),flush=True)
            save_checkpoint(trainer.checkpoint(),out/'last.pt')
            write_json(out/'loss_history.json',history)
    if trainer.device.type=='cuda':torch.cuda.synchronize()
    train_seconds=time.perf_counter()-train_start
    save_checkpoint(trainer.checkpoint(),out/'last.pt');write_json(out/'loss_history.json',history)
    final={}
    for split in data:
        final[split],arrays=trainer.evaluate(split)
        visualize(out,split,arrays,data[split],manifest,history)
    metrics={'status':'completed' if trainer.step==steps else 'time_budget_stop','data_kind':manifest['data_kind'],
        'scope':FORMAT,'resume_from':str(resume) if resume else None,'step':trainer.step,'requested_steps':steps,'initial':initial,'final':final,
        'training_seconds':train_seconds,'total_run_seconds':time.perf_counter()-started,
        'preparation_seconds_excluded':True,'environment':trainer.environment,'config':config,
        'source':trainer.source,'manifest_sha256':manifest['sha256'],'checkpoint_sha256':sha(out/'last.pt'),
        'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated() if trainer.device.type=='cuda' else None,
        'loss_definition':'mean valid window joint xyz normalized x0 MSE + weight * intermediate frame MSE',
        'limitations':['GT 2D input','camera features zero ablation','raw physical unit unknown','development skeleton',
                      'small named-motion holdout; not official validation/test','capture FPS unknown','not full M2 acceptance']}
    write_json(out/'metrics.json',metrics)
    print(json.dumps(metrics,ensure_ascii=False,indent=2),flush=True)
    return trainer,metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extraction',required=True);p.add_argument('--selection',required=True)
    p.add_argument('--config',default='configs/m2_pose_pilot.json');p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=300);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--allow-historical-extractor',action='store_true');p.add_argument('--resume')
    a=p.parse_args();run(a.extraction,json.loads(Path(a.selection).read_text()),json.loads(Path(a.config).read_text()),
                        a.output,a.steps,a.device,a.allow_historical_extractor,a.resume)

if __name__=='__main__':main()
