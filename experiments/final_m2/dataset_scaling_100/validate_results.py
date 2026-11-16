"""완전히 회수된 실제 파일만 검증. 누락 checkpoint·trace를 재생성하지 않는다."""
import argparse,json,sys,hashlib,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.final_m2.dataset_scaling_100.run_experiment import extra_metrics,quality,state_hash
from experiments.final_m2.diagnosis_20261111.diagnose import calculate,write_csv
from tools.final_m2_real import sha,write,load

def run(src,old,out):
    src,out=map(Path,(src,out));out.mkdir(parents=True,exist_ok=True);report=load(src/'evaluation.json');checks=load(src/'checksums.json');manifest=load(Path(__file__).parent/'data_selection_manifest.json')
    recovered={p.name:sha(p) for p in src.iterdir() if p.name in checks}
    assert all(checks[n]==h for n,h in recovered.items())
    rows=[];frame_rows=[];e=np.array(manifest['edges']);names=manifest['joint_names'];missing=[]
    for r in report['results']:
        pth=src/f"{r['split']}_{r['model']}_predictions.npz"
        if not pth.exists():missing.append(pth.name);continue
        z=dict(np.load(pth,allow_pickle=False));p,g,v,ids=[z[k] for k in ('prediction','gt','valid','frame_ids')];assert len(set(z['sequence_ids']))==1
        s,b,f=calculate(p,g,v,ids,e,0);s.update(extra_metrics(p,g,v,ids));br=np.linalg.norm(p.astype(float)[:,e[:,1]]-p.astype(float)[:,e[:,0]],axis=-1)/np.linalg.norm(g.astype(float)[:,e[:,1]]-g.astype(float)[:,e[:,0]],axis=-1)
        s['mean_abs_bone_ratio_deviation_pp']=float(abs(br-1).mean()*100);s['min_bone_ratio_pct']=float(br.min()*100)
        for k,value in s.items():np.testing.assert_allclose(value,r[k],atol=1e-12,rtol=1e-12)
        rows.append({'model':r['model'],'split':r['split'],'recomputed_metrics':len(s),'status':'PASS'})
        if r['split']=='opening130':
            dg=np.diff(g.astype(float),axis=0);dp=np.diff(p.astype(float),axis=0)
            for i,x in enumerate(f):
                x.update(model=r['model'],frame_displacement_error_raw=float(np.linalg.norm(dp[i-1]-dg[i-1],axis=-1).mean()) if i else None,gt_motion_raw=float(np.linalg.norm(dg[i-1],axis=-1).mean()) if i else None,pred_motion_raw=float(np.linalg.norm(dp[i-1],axis=-1).mean()) if i else None)
                frame_rows.append(x)
    a,b=report['arms'];assert a['updates']==b['updates']==400 and a['samples_seen']==b['samples_seen']==3200
    assert a['initial_state_sha256']==b['initial_state_sha256'];assert a['diffusion_trace_sha256']==b['diffusion_trace_sha256'];assert report['evaluation_noise_sha256']['train18']==report['evaluation_noise_sha256']['train100']
    assert sorted(a['anchor_sample_counts']) and sum(a['anchor_sample_counts'])==sum(b['anchor_sample_counts'])==3200 and min(b['anchor_sample_counts'])>0
    ck_status=[]
    for arm in report['arms']:
        for c in arm['checkpoints']:
            path=src/c['file'];item={'model':arm['model'],**c,'local_byte_verification':'NOT VERIFIED'}
            if path.exists():
                assert sha(path)==c['sha256'];ck=torch.load(path,map_location='cpu',weights_only=True)
                optimizer_state = ck['optimizer']['state']
                assert ck['step']==c['step'] and ck['scheduler']['last_epoch']==c['step']
                assert optimizer_state and all(float(v['step'])==c['step'] for v in optimizer_state.values())
                assert ck['initial_state_sha256']==arm['initial_state_sha256'];item['local_byte_verification']='PASS'
            ck_status.append(item)
    reference_name = 'holdout_control_predictions.npz'
    prior_checks = Path(__file__).resolve().parent.parent / 'displacement_loss_20261116' / 'metrics' / 't4' / 'checksums.json'
    current_path = src / 'opening130_train18_predictions.npz'
    expected_control_sha = load(prior_checks)[reference_name]
    assert sha(current_path) == expected_control_sha, 'Opening Train18 NPZ differs from the archived Control400 SHA-256'
    direct_control_comparison = 'NOT VERIFIED'
    if old is not None:
        reference_path = Path(old) / reference_name
        assert sha(reference_path) == expected_control_sha, 'Unexpected Control400 reference NPZ'
        with np.load(reference_path,allow_pickle=False) as previous, np.load(current_path,allow_pickle=False) as current:
            for key in ('prediction', 'gt'):
                assert previous[key].dtype == current[key].dtype
                assert previous[key].shape == current[key].shape
                assert previous[key].tobytes() == current[key].tobytes()
        direct_control_comparison = 'PASS'
    assert quality(report['results'])==report['quality']
    traces=[]
    for arm in report['arms']:
        path=src/f"{arm['model']}_random_trace.json"
        if path.exists():
            trace=load(path);assert len(trace)==400
            assert [x['step'] for x in trace]==list(range(1,401))
            assert state_hash(trace)==arm['full_trace_sha256']
            assert state_hash([{k:v for k,v in r.items() if k!='anchors'} for r in trace])==arm['diffusion_trace_sha256']
            traces.append(arm['model'])
    absent=sorted(set(checks)-set(recovered))
    validation={'execution_observed':'PASS','opening_array_recomputation':'PASS','prior_control400_opening_prediction_bitwise_identical':True if old is not None else None,'prior_control400_opening_npz_sha256_match':expected_control_sha,'prior_control400_direct_array_comparison':direct_control_comparison,'verified_prediction_groups':rows,'unrecovered_prediction_groups':missing,'checkpoint_verification':ck_status,'training_contract_in_saved_report':'PASS','full_random_traces_local_check':'PASS' if len(traces)==2 else 'NOT VERIFIED; not recovered','quality':report['quality'],'complete_artifact_recovery':'PASS' if not absent and (src/'console.log').exists() else 'PARTIAL','missing_checksummed_files':absent,'verified_file_sha256':recovered}
    write(out/'result_validation.json',validation);write_csv(out/'opening_frame_analysis.csv',frame_rows);write_csv(out/'checkpoint_identifiers.csv',ck_status)
    # Frame and limb evidence from actual Opening arrays.
    ds={arm:dict(np.load(src/f'opening130_{arm}_predictions.npz')) for arm in ('train18','train100')}
    p1,p2,g=[ds['train18']['prediction'].astype(float),ds['train100']['prediction'].astype(float),ds['train18']['gt'].astype(float)]
    de=lambda p:np.linalg.norm(np.diff(p,axis=0)-np.diff(g,axis=0),axis=-1).mean(1)
    dif=de(p2)-de(p1);rr=lambda p:p-p[:,:1]
    bone_rows=[]
    for j,(a,b) in enumerate(e):
        gl=np.linalg.norm(g[:,b]-g[:,a],axis=-1);pl=np.linalg.norm(p2[:,b]-p2[:,a],axis=-1);rat=100*pl/gl
        bone_rows.append({'bone':names[a]+' -> '+names[b],'mean_ratio_pct':float(pl.mean()/gl.mean()*100),'min_ratio_pct':float(rat.min()),'min_frame':int(rat.argmin())})
    detail={'fde_improved_segments':int((dif<0).sum()),'fde_worsened_segments':int((dif>0).sum()),'best_frames':[{'frame':int(i+1),'control_fde':float(de(p1)[i]),'expanded_fde':float(de(p2)[i])} for i in np.argsort(dif)[:5]],'worst_frames':[{'frame':int(i+1),'control_fde':float(de(p1)[i]),'expanded_fde':float(de(p2)[i])} for i in np.argsort(dif)[-5:][::-1]],'root_relative_mean_pose_bias_raw':{a:float(np.linalg.norm(rr(p).mean(0)-rr(g).mean(0),axis=-1).mean()) for a,p in [('train18',p1),('train100',p2)]},'bone_rows':bone_rows}
    write(out/'detailed_analysis.json',detail);print(json.dumps({'validation':validation['opening_array_recomputation'],'quality':report['quality']['status'],'details':detail},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--previous',help='Directory with the original Control400 NPZ, for a direct array comparison');p.add_argument('--output',required=True);a=p.parse_args();run(a.input,a.previous,a.output)
