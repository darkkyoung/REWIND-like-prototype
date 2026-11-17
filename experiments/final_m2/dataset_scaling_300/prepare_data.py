"""Train100 prefix 보존; 캐시로 식별한 Pose 247개만 변환한다."""
import argparse, hashlib, json, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.final_m2.dataset_scaling_100.prepare_data import convert,coordinate_check,motion,MANIFEST_SHA
from experiments.final_m2.displacement_loss_20261116.loss import window_metadata
from experiments.final_m2.diagnosis_20261111.diagnose import write_csv
from rewind_data.audit import read_json
from tools.final_m2_real import sha,load,write
SELECTION=[('Boxing__55_',0,50,'train'),('Lifting__3_',50,50,'train'),('Hip_Hop_Dancing__24_',0,50,'train'),('Pulling_A_Rope__9_',50,50,'train'),('Gunplay__9_',0,47,'validation')]
def read_npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def run(old,source,out):
    old,source,out=map(Path,(old,source,out));out.mkdir(parents=True,exist_ok=False)
    assert sha(source/'cache_manifest.json')==MANIFEST_SHA
    for n,h in load(old/'input_checksums.json').items():assert sha(old/n)==h
    prior=load(old/'data_selection_manifest.json');names=prior['joint_names'];cache=load(source/'cache_manifest.json')
    records=load(source/'source_records.json');bykey={tuple(r['key']):r for p in cache['parts'] for r in p['records']}
    trainrows=[];valrows=[];frames=list(prior['frames']);evalframes=[];audit=[]
    for action,start,count,split in SELECTION:
        chosen=sorted([r for r in records if r['key'][0].endswith('/'+action)],key=lambda r:r['key'][1])
        assert len(chosen)==count and [r['key'][1] for r in chosen]==list(range(start,start+count))
        for r in chosen:
            seq,frame=r['key'];ref=bykey[(seq,frame)]
            assert ref['split']==split==r['split'] and ref['files']['pose']==r['files']['pose']
            f=r['files']['pose'];raw=(source/'raw_pose'/f['path']).read_bytes()
            assert len(raw)==f['size_bytes'] and hashlib.sha256(raw).hexdigest()==f['sha256']
            doc=read_json(raw);audit.append({'sequence_id':seq,'frame_id':frame,**coordinate_check(doc,names)})
            (trainrows if split=='train' else valrows).append((seq,frame,doc))
            rows=frames if split=='train' else evalframes
            rows.append({'index':len(rows),'sequence_id':seq,'motion_group':motion(seq),'frame_id':frame,'split':split,'split_evidence':'cache_manifest_'+split,'source_part':r['source_part'],'pose_path':f['path'],'pose_sha256':f['sha256'],'pose_size_bytes':len(raw)})
    base=read_npz(old/'train100.npz');new=convert(trainrows,names);val=convert(valrows,names)
    train={k:np.concatenate([base[k],new[k]],axis=0) for k in base}
    assert frames[:100]==prior['frames'] and len(frames)==300
    for k in base:
        np.testing.assert_array_equal(train[k][:100],base[k])
        if base[k].dtype.kind not in 'US':assert train[k][:100].tobytes()==base[k].tobytes() and train[k].dtype==base[k].dtype
    assert len({(r['sequence_id'],r['frame_id']) for r in frames})==300
    assert len({r['pose_sha256'] for r in frames})==300
    trainkeys={(r['sequence_id'],r['frame_id']) for r in frames};trainsha={r['pose_sha256'] for r in frames}
    other=[r for p in cache['parts'] for r in p['records'] if r['split']!='train']
    assert not trainkeys&{tuple(r['key']) for r in other}
    assert not trainsha&{r['files']['pose']['sha256'] for r in other}
    assert not {motion(r['sequence_id']) for r in evalframes}&{motion(r['sequence_id']) for r in frames}
    assert motion(prior['holdout_sequence']) not in {motion(r['sequence_id']) for r in frames+evalframes}
    opening=read_npz(old/'opening130.npz')
    for data in (train,val,opening):window_metadata(data)
    for a,b in ((train,val),(train,opening),(val,opening)):
        assert not set(zip(a['sequence_ids'],a['frame_ids']))&set(zip(b['sequence_ids'],b['frame_ids']))
        for k in ('features','target_raw'):
            assert not any(np.array_equal(x,y) for x in a[k][:,-1] for y in b[k][:,-1])
    for data in (train,val):
        for i in range(len(data['frame_ids'])):
            if i==0 or data['sequence_ids'][i]!=data['sequence_ids'][i-1]:
                assert data['time'][i].sum()==1
            else:assert data['frame_ids'][i]==data['frame_ids'][i-1]+1
    for n,d in [('train300',train),('gunplay47',val)]:np.savez_compressed(out/(n+'.npz'),**d)
    for n in ('train100.npz','train18.npz','opening130.npz'):shutil.copy2(old/n,out/n)
    manifest={**prior,'schema':'m2_scaling_selection_v2','base_commit':'d54e3f84acb42e029082bb8058b9c229ade75a83','parent_selection_sha256':sha(old/'data_selection_manifest.json'),'selection_policy':'Prior100 verbatim; Boxing55 0..49, Lifting3 50..99, HipHop24 0..49, PullingRope9 50..99. Gunplay9 validation0..46. Fixed before training or inference.','frames':frames,'additional_evaluation':{'gunplay47':evalframes},'source_binding':cache['binding'],'new_source_parts':sorted({r['source_part'] for r in records})}
    write(out/'data_selection_manifest.json',manifest);write_csv(out/'selected_frames.csv',frames);write_csv(out/'evaluation_frames.csv',evalframes);write_csv(out/'coordinate_audit.csv',audit)
    validation={'status':'PASS','unique_train_frames':300,'new_train_frames':200,'validation_frames':47,'train100_prefix_tensor_bitwise_identical':True,'train100_prefix_manifest_identical':True,'train100_npz_sha256':sha(old/'train100.npz'),'new_pose_sha_and_size_verified':247,'cache_manifest_sha256':sha(source/'cache_manifest.json'),'known_cache_nontrain_key_or_sha_overlap':0,'opening_and_gunplay_key_or_exact_tensor_overlap':0,'body22_finite':True,'sequence_boundary_reset_verified':True,'consecutive_native_frame_ids':True,'future_leakage':'causal inherited converter; regression tests required','coordinate_maxima':{k:max(a[k] for a in audit) for k in audit[0] if k not in ('sequence_id','frame_id')},'pixel_scale':1024,'target_scale_raw':100,'camera':'zero','rgb_downloaded':False,'stereo_image_alignment':'NOT VERIFIED in this run; GT Stereo2D only','original_mocap_independence':'NOT VERIFIED; distinct names, keys, SHA and exact poses checked','prior_train18_split':'historical assignment retained; no new split proof','fps':'NOT VERIFIED','physical_units':'raw; NOT VERIFIED as mm','person_confound':'four added persons; Carla already in Train100, Janna/Joyce/Serena are additional; Gunplay person Adanna is already trained','cache_blockers':cache['blockers']}
    write(out/'input_validation.json',validation);write(out/'input_checksums.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(validation,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.old,a.source,a.output)
