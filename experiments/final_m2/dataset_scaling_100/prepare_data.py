"""선택된 원본 pose만 검증·변환. 기존 Train18 tensor를 그대로 보존한다."""
import argparse,hashlib,json,math,re,shutil,sys,zipfile,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from rewind_data.audit import FrameSample,_validate_joints,read_json
from rewind_data.windows import CausalWindowStream
from rewind_prepare.coordinates import camera_transform,world_to_camera,maximum_error,mv,F,sub
from experiments.final_m2.displacement_loss_20261116.loss import window_metadata
from tools.final_m2_real import sha,write,load
from experiments.final_m2.diagnosis_20261111.diagnose import write_csv
MANIFEST_SHA='caa0956d4432e93fa2f257a7176f568fd50863819b1d2ada93b040ca8e8cee7e'
BUNDLE_SHA='184967a9732702fc0459350974b87e67f6499144dcfdbec48b18cb657805d74d'
SELECTION=(('part-0000.zip','Standing_Up__1_',50),('part-0005.zip','Dodging',32))
def motion(seq):return re.sub(r'__\d+_$','',seq.rsplit('/',1)[-1]).casefold()

def coordinate_check(doc,names):
    # 검사만 수행한다. factor=1: 단위 변환과 좌표 보정 없음.
    result={'world_camera_max_abs_raw':0.,'camera_device_max_abs_raw':0.,'metadata_max_abs_raw':0.};cams={};tol=1e-3
    for side in ('left','right'):cams[side]=camera_transform(doc['camera_'+side],1.,tol)
    l,r=doc['camera_left'],doc['camera_right'];mid=(np.array(l['trans'])+r['trans'])/2
    if not np.allclose(cams['left'][0]['rotation_matrix'],cams['right'][0]['rotation_matrix'],atol=tol,rtol=0):raise ValueError('stereo_rotation')
    result['baseline_raw']=float(np.linalg.norm(np.array(l['trans'])-r['trans']))
    if not 11.99<result['baseline_raw']<12.01:raise ValueError('raw_scale_baseline')
    if maximum_error(doc['device']['open3d_trans'],mv(F,mid))>tol:raise ValueError('device_origin')
    if maximum_error(doc['device']['open3d_rot'],[math.radians(x) for x in l['open3d_rot']])>tol:raise ValueError('device_rotation')
    for side in ('left','right'):
        c,t=cams[side];meta=doc['camera_'+side];offset=mv(c['rotation_matrix'],mv(F,sub(meta['trans'],mid)))
        result['metadata_max_abs_raw']=max(result['metadata_max_abs_raw'],maximum_error(offset,meta['device_pts3d']))
        for name in names:
            j=doc['joints'][name];p=j['camera_'+side+'_pts3d'];d=j['device_pts3d']
            if not np.isfinite(np.array([*j['trans'],*p,*d],float)).all():raise ValueError('nonfinite_coordinate')
            result['world_camera_max_abs_raw']=max(result['world_camera_max_abs_raw'],maximum_error(world_to_camera(j['trans'],c['rotation_matrix'],t),p))
            result['camera_device_max_abs_raw']=max(result['camera_device_max_abs_raw'],maximum_error(np.array(p)+meta['device_pts3d'],d))
    if any(v>tol for k,v in result.items() if k!='baseline_raw'):raise ValueError('coordinate_incompatibility')
    return result

def convert(rows,names,window=8):
    stream=CausalWindowStream(window,names);packed={k:[] for k in ('features','target_raw','valid','time','sequence_ids','frame_ids')}
    for seq,index,doc in rows:
        if str(doc['frame'])!=str(index):raise ValueError('frame_mismatch')
        sample=FrameSample(f'{seq}/json/frame_{index}.json',seq,index,raw_document=doc);_validate_joints(sample,names)
        win=stream.push(sample);x=torch.zeros(window,22,14);y=torch.zeros(window,22,3);m=torch.zeros(window,22,dtype=torch.bool)
        for slot,frame in enumerate(win.frames):
            if frame is None:continue
            fields={f.name:f for f in frame.fields}
            for j in range(22):
                f=fields['device_pts3d']
                if f.valid_mask[j]:y[slot,j]=torch.tensor(f.coordinates[j]);m[slot,j]=True
                for v,side in enumerate(('left','right')):
                    f2=fields[f'camera_{side}_pts2d']
                    if f2.valid_mask[j]:x[slot,j,v*7:v*7+2]=torch.tensor(f2.coordinates[j])/1024.;x[slot,j,v*7+2]=1
        if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not m[-1].all():raise ValueError('finite_full_body22')
        for k,v in [('features',x.flatten(1).numpy()),('target_raw',y.numpy()),('valid',m.numpy()),('time',np.array(win.time_valid)),('sequence_ids',seq),('frame_ids',index)]:packed[k].append(v)
    packed={k:np.array(v) for k,v in packed.items()};window_metadata(packed);return packed

def run(cache,old,raw_old,output):
    cache,old,raw_old,out=map(Path,(cache,old,raw_old,output));out.mkdir(parents=True,exist_ok=False)
    assert sha(cache/'cache_manifest.json')==MANIFEST_SHA
    manifest=load(cache/'cache_manifest.json');original=load(old/'data_manifest.json');names=original['joint_names'];edges=original['edges']
    for name,h in load(old/'input_checksums.json').items():assert sha(old/name)==h
    rows=[];selected=[];audit=[];source_parts=[];rawdir=out/'raw_pose';rawdir.mkdir()
    def add(seq,frame,raw,expected,part,evidence):
        assert hashlib.sha256(raw).hexdigest()==expected
        doc=read_json(raw);assert str(doc['frame'])==str(frame);audit.append({'sequence_id':seq,'frame_id':frame,**coordinate_check(doc,names)})
        dest=rawdir/seq/'json'/f'frame_{frame}.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        selected.append({'index':len(selected),'sequence_id':seq,'motion_group':motion(seq),'frame_id':frame,'split':'train','split_evidence':evidence,'source_part':part,'pose_path':f'{seq}/json/frame_{frame}.json','pose_sha256':expected,'pose_size_bytes':len(raw)})
        rows.append((seq,frame,doc))
    trainseq=original['selection']['train'][0];holdseq=original['selection']['holdout'][0];old_hashes={r['member']:r['sha256'] for r in original['payloads']}
    for i in range(18):
        member=f'{trainseq}/json/frame_{i}.json';add(trainseq,i,(raw_old/member).read_bytes(),old_hashes[member],'original_148','immutable_historical_train18')
    rebuilt=convert(rows,names);base=np.load(old/'train.npz',allow_pickle=False)
    for k in base.files:np.testing.assert_array_equal(rebuilt[k],base[k])
    for partname,action,count in SELECTION:
        part=next(p for p in manifest['parts'] if p['name']==partname);p=cache/partname
        assert p.stat().st_size==part['bytes'] and sha(p)==part['sha256'];source_parts.append({k:part[k] for k in ('name','bytes','sha256')})
        chosen=sorted([r for r in part['records'] if r['key'][0].endswith('/'+action)],key=lambda r:r['key'][1])[:count]
        assert len(chosen)==count and [r['key'][1] for r in chosen]==list(range(count))
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None
            for r in chosen:
                assert r['split']=='train';seq,frame=r['key'];f=r['files']['pose'];assert f['path']==f'{seq}/json/frame_{frame}.json'
                raw=z.read(f['path']);assert len(raw)==f['size_bytes'];add(seq,frame,raw,f['sha256'],partname,'cache_manifest_train')
                for side in ('left','right'):
                    b=z.read(r['files'][side]['path']);assert hashlib.sha256(b).hexdigest()==r['files'][side]['sha256'];assert b[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',b[16:24])==(1024,1024)
    assert len(rows)==100 and len({(s,f) for s,f,d in rows})==100 and len({r['pose_sha256'] for r in selected})==100
    assert motion(holdseq) not in {motion(s) for s,f,d in rows}
    assert not set(r['pose_sha256'] for r in selected)&{h for n,h in old_hashes.items() if n.startswith(holdseq+'/')}
    hold_audit=[]
    for i in range(130):
        member=f'{holdseq}/json/frame_{i}.json';raw=(raw_old/member).read_bytes();assert hashlib.sha256(raw).hexdigest()==old_hashes[member];hold_audit.append(coordinate_check(read_json(raw),names))
    data=convert(rows,names)
    for k in base.files:np.testing.assert_array_equal(data[k][:18],base[k])
    hold=np.load(old/'holdout.npz',allow_pickle=False)
    for key in ('features','target_raw'):
        assert not any(np.array_equal(x,y) for x in data[key][:,-1] for y in hold[key][:,-1])
    for name,d in [('train100',data),('standing50',{k:v[18:68] for k,v in data.items()}),('dodging32',{k:v[68:] for k,v in data.items()})]:np.savez_compressed(out/f'{name}.npz',**d)
    shutil.copy2(old/'train.npz',out/'train18.npz');shutil.copy2(old/'holdout.npz',out/'opening130.npz')
    write_csv(out/'selected_frames.csv',selected);write_csv(out/'coordinate_audit.csv',audit)
    validation={'status':'PASS','train18_exact_tensor_equality':True,'train18_file_sha256':sha(old/'train.npz'),'unique_train_frames':100,'unique_payload_hashes':100,'new_train_frames':82,'holdout_overlap_keys':0,'holdout_overlap_payload_sha':0,'holdout_overlap_exact_pose':0,'motion_name_overlap':False,'body22_complete_finite':True,'stereo_native_resolution':[1024,1024],'coordinate_tolerance_raw':1e-3,'coordinate_maxima':{k:max(a[k] for a in audit+hold_audit) for k in audit[0] if k not in ('sequence_id','frame_id')},'raw_coordinates_modified':False,'train18_original_split_revalidation':'not available in this cache; historical explicit train18 retained','original_mocap_independence':'NOT VERIFIED; names/keys/hash/pose equality checked','physical_unit':'raw only; not mm','capture_fps':'unknown','full_dataset_train_ready':False,'cache_original_blockers':manifest['blockers']}
    write(out/'input_validation.json',validation)
    write(out/'data_selection_manifest.json',{'schema':'m2_scaling_selection_v1','cache_manifest_sha256':MANIFEST_SHA,'original_input_bundle_sha256':BUNDLE_SHA,'base_commit':'635d9424878cae2a031efc14588eb80a45b25391','source_parts':source_parts,'source_binding':manifest['binding'],'selection_policy':'Train18 unchanged; Standing_Up frame0..49; Dodging first contiguous frame0..31. Fixed before model results.','joint_names':names,'edges':edges,'frames':selected,'holdout_sequence':holdseq,'coordinate_frame':'stored_device_pts3d','distance_unit':'raw_annotation_unit_unverified','pixel_scale':1024,'target_scale_raw':100,'camera':'zero','window':8})
    write(out/'input_checksums.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()});print(json.dumps(validation,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--old',required=True);p.add_argument('--raw-old',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.cache,a.old,a.raw_old,a.output)
