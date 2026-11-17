"""기존 trainer/evaluator를 재사용한다. 새 학습은 Train300 한 군뿐이다."""
import argparse,copy,json,os,platform,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from experiments.final_m2.dataset_scaling_100.run_experiment import new_model,train,state_hash,extra_metrics,quality
from experiments.final_m2.displacement_loss_20261116.run_experiment import evaluate
from experiments.final_m2.dataset_scaling_300.prepare_data import read_npz
from experiments.final_m2.displacement_loss_20261116.loss import window_metadata
from experiments.final_m2.diagnosis_20261111.diagnose import calculate,write_csv
from rewind_m1.training import rng_state
from rewind_m2.diffusion import DiffusionSchedule
from rewind_m2.streaming import keyed_frame_noise
from tools.final_m2_real import sha,load,write
HERE=Path(__file__).resolve().parent

def quality300(rows,split='opening130'):
    # 기존 8개 판정식을 그대로 사용하고 비교군 이름만 매핑한다.
    mapped=[]
    for r in rows:
        if r['split']==split and r['model'] in ('train100','train300'):
            mapped.append({**r,'split':'opening130','model':'train18' if r['model']=='train100' else 'train100'})
    return quality(mapped)

def summarize(folder,manifest):
    rows=[];bones=[];frames=[]
    for split in ('opening130','gunplay47'):
        for arm in ('train18','train100','train300'):
            z=read_npz(folder/f'{split}_{arm}_predictions.npz');p,g,v,ids=[z[k] for k in ('prediction','gt','valid','frame_ids')]
            assert len(set(z['sequence_ids']))==1 and np.all(np.diff(ids)==1)
            s,b,f=calculate(p,g,v,ids,manifest['edges'],0);s.update(extra_metrics(p,g,v,ids))
            e=np.array(manifest['edges']);pf,gf=p.astype(float),g.astype(float)
            br=np.linalg.norm(pf[:,e[:,1]]-pf[:,e[:,0]],axis=-1)/np.linalg.norm(gf[:,e[:,1]]-gf[:,e[:,0]],axis=-1)
            s.update(mean_abs_bone_ratio_deviation_pp=float(np.mean(abs(br-1))*100),min_bone_ratio_pct=float(br.min()*100),evaluation_use='historical_holdout' if split=='opening130' else 'additional_validation_motion_not_proven_mocap_independent')
            rows.append({'split':split,'model':arm,**s});bones.extend({'split':split,'model':arm,**r} for r in b)
            for i,r in enumerate(f):
                mask=v[i]&v[i-1] if i else np.zeros(22,bool)
                dg=gf[i]-gf[i-1] if i else None;dp=pf[i]-pf[i-1] if i else None
                r.update(gt_displacement_raw=float(np.linalg.norm(dg[mask],axis=-1).mean()) if mask.any() else None,pred_displacement_raw=float(np.linalg.norm(dp[mask],axis=-1).mean()) if mask.any() else None,displacement_error_raw=float(np.linalg.norm((dp-dg)[mask],axis=-1).mean()) if mask.any() else None)
                frames.append({'split':split,'model':arm,**r})
    return rows,bones,frames

def run(inp,baseline,output):
    inp,baseline,out=map(Path,(inp,baseline,output));out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    lock=load(HERE/'config.json');cfg=lock['model']
    assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(),'requires_actual_T4'
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    assert sha(inp/'input_checksums.json')==lock['input_checksums_sha256']
    for n,h in load(inp/'input_checksums.json').items():assert Path(n).name==n and sha(inp/n)==h
    for n,h in lock['baseline_files'].items():assert sha(baseline/n)==h
    for n,h in lock['core_code_sha256'].items():assert sha(ROOT/n)==h,'core_code_mismatch'
    manifest=load(inp/'data_selection_manifest.json');assert sha(inp/'data_selection_manifest.json')==lock['selection_sha256']
    validation=load(inp/'input_validation.json');assert validation['status']=='PASS' and validation['unique_train_frames']==300
    data={n:read_npz(inp/(n+'.npz')) for n in ('train300','train100','opening130','gunplay47')}
    for d in data.values():window_metadata(d)
    for k in data['train100']:np.testing.assert_array_equal(data['train300'][k][:100],data['train100'][k])
    schedule=DiffusionSchedule(cfg['diffusion_steps'],cfg['beta_start'],cfg['beta_end'])
    model,opt,scheduler=new_model(cfg,manifest,'cuda')
    initial={'model':state_hash(model.state_dict()),'optimizer':state_hash(opt.state_dict()),'scheduler':state_hash(scheduler.state_dict()),'rng':state_hash(rng_state())}
    initial_match={k:initial[k]==v for k,v in lock['expected_initial_state_sha256'].items()}
    environment={'python':platform.python_version(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'precision':'FP32'}
    identity={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [*HERE.glob('*.py'),ROOT/'experiments/final_m2/dataset_scaling_100/run_experiment.py',*[ROOT/n for n in lock['core_code_sha256']]]},'config_sha256':sha(HERE/'config.json'),'input_checksums_sha256':sha(inp/'input_checksums.json'),'environment':environment,'baseline_environment':load(baseline/'experiment_identity.json'),'initial_state_sha256':initial,'initial_state_matches_baseline':initial_match,'initialization':'fresh seed43; no baseline weights loaded for training','training_frames':300,'updates':400,'final_checkpoint_preselected':400,'anchor_draws':3200,'expected_draws_per_frame':3200/300,'evaluation_noise_seed':10043,'ddim_steps':10}
    write(out/'experiment_identity.json',identity)
    assert all(initial_match[k] for k in ('model','optimizer','scheduler')),'initial_state_mismatch_review_required'
    del model,opt,scheduler
    record=train('train300',cfg,manifest,data['train300'],schedule,out,identity)
    assert record['initial_state_sha256']==initial
    assert record['distinct_anchors_seen']==300
    prior=load(baseline/'train100_identity.json')
    fairness={'initial_state':initial_match,'diffusion_trace_matches_train100':record['diffusion_trace_sha256']==prior['diffusion_trace_sha256']}
    noises={}
    for arm in ('train300','train18','train100'):
        assert time.monotonic()-start<lock['max_total_gpu_seconds'],'total_gpu_budget'
        ck=torch.load((out if arm=='train300' else baseline)/f'{arm}_400.pt',map_location='cpu',weights_only=True)
        assert ck['step']==400 and ck['config']==cfg and ck['manifest']['edges']==manifest['edges']
        assert torch.equal(ck['alpha_bar'],schedule.alpha_bar)
        for k in ('model','optimizer','scheduler'):assert ck['initial_state_sha256'][k]==lock['expected_initial_state_sha256'][k]
        evaldata={k:data[k] for k in (('opening130','gunplay47') if arm=='train300' else ('gunplay47',))}
        _,_,_,noise,_=evaluate(arm,ck['model'],ck,evaldata,schedule,out,'cuda');noises[arm]=noise
        if arm!='train300':shutil.copy2(baseline/f'opening130_{arm}_predictions.npz',out/f'opening130_{arm}_predictions.npz')
    assert len({noises[a]['gunplay47'] for a in noises})==1
    for split in ('opening130','gunplay47'):
        reference=read_npz(out/f'{split}_train18_predictions.npz')
        for arm in ('train100','train300'):
            z=read_npz(out/f'{split}_{arm}_predictions.npz')
            for k in ('gt','valid','frame_ids','sequence_ids'):np.testing.assert_array_equal(reference[k],z[k])
    rows,bones,frames=summarize(out,manifest)
    priorrows=load(baseline/'evaluation.json')['results']
    for r in rows:
        if r['split']=='opening130' and r['model']!='train300':
            old=next(a for a in priorrows if a['split']==r['split'] and a['model']==r['model'])
            for k,v in r.items():
                if isinstance(v,(float,int)) and k in old:assert abs(v-old[k])<1e-9,(k,v,old[k])
    write_csv(out/'comparison.csv',rows);write_csv(out/'bone_length_by_edge.csv',bones);write_csv(out/'frame_comparison.csv',frames)
    report={'execution_status':'PASS','quality':quality300(rows),'additional_validation_quality':quality300(rows,'gunplay47'),'generalization_claim':'Additional validation motion only; original mocap independence NOT VERIFIED','arms':[record],'fairness':fairness,'evaluation_noise_sha256':noises,'identity':identity,'results':rows}
    if not all(initial_match.values()) or not fairness['diffusion_trace_matches_train100']:
        report['quality_unqualified']=report['quality'];report['quality']={'status':'NOT VERIFIED','reason':'RNG/trace mismatch; numerical gates retained separately'}
    write(out/'evaluation.json',report);write(out/'checksums.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'execution_status':report['execution_status'],'quality':report['quality'],'additional_validation_quality':report['additional_validation_quality']},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);p.add_argument('--baseline-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    try:run(a.input_dir,a.baseline_dir,a.output)
    except Exception as e:
        out=Path(a.output)
        if out.exists() and not isinstance(e,FileExistsError):write(out/'failure.json',{'status':'NOT VERIFIED','error_type':type(e).__name__,'message':str(e)})
        raise
