"""검증된 5-step 실제 Pose bundle 재사용. 기존 M2 구현은 수정하지 않는다."""
import hashlib
import json
from pathlib import Path
import shutil
import stat
import time
import zipfile

EXPECTED_ZIP_SHA = 'f4c9c2d9ed06a76385ec949b94c447a8f0751602ce82a36bb0f2fa5611debb1f'
BASE_SHA = '5e52aa3b0b1e27e7d0cb2697629e8acfc63d38c3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_bundle(path, destination):
    """Exact approved ZIP, CRC, every member hash, original extractor and payload contracts."""
    path, dest = Path(path), Path(destination)
    if path.stat().st_size > 20*1024**2 or sha(path) != EXPECTED_ZIP_SHA:
        raise ValueError('5-step 결과 ZIP SHA-256 불일치: 지정한 원본 ZIP을 업로드하세요.')
    dest.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(path) as z:
        entries = z.infolist()
        if len(entries)>500 or sum(i.file_size for i in entries)>64*1024**2:
            raise ValueError('ZIP member/해제량 상한')
        seen=set()
        for i in entries:
            target=dest/i.filename
            if (not target.resolve().is_relative_to(dest.resolve()) or '\\' in i.filename
                    or stat.S_ISLNK(i.external_attr>>16) or i.filename.casefold() in seen):
                raise ValueError('ZIP 경로/중복 오류')
            seen.add(i.filename.casefold())
        if z.testzip() is not None: raise ValueError('ZIP CRC 오류')
        z.extractall(dest)
    checks=load(dest/'checksums.json')
    for relative, expected in checks.items():
        p=dest/relative
        if not p.resolve().is_relative_to(dest.resolve()) or not p.is_file() or sha(p)!=expected:
            raise ValueError('bundle file hash 불일치')
    from rewind_archive.matching import verify_extraction
    extraction=verify_extraction(dest/'pilot')  # no historical/code-change bypass
    m=load(dest/'data_manifest.json'); metrics=load(dest/'metrics.json')
    body={k:v for k,v in m.items() if k!='sha256'}
    assert hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()==m['sha256']
    assert metrics['step']==metrics['requested_steps']==5 and metrics['status']=='completed'
    assert metrics['manifest_sha256']==m['sha256'] and metrics['checkpoint_sha256']==sha(dest/'last.pt')
    assert m['extraction_manifest_sha256']==sha(dest/'pilot/extraction_manifest.json')
    assert len(extraction['files'])==148 and len(m['joint_names'])==22
    assert {v['split']:len(v['frames']) for v in m['sequences'].values()}=={'train':18,'holdout':130}
    assert metrics['source']['revision']==BASE_SHA
    from rewind_m2.training import source_identity
    assert source_identity()['code_sha256']==metrics['source']['code_sha256'], 'M2/data code hash mismatch'
    return dest


def prepare_reuse(bundle):
    from rewind_m2.pose_pilot import prepare_data
    cfg=load(bundle/'config.json'); previous=load(bundle/'data_manifest.json')
    data,manifest=prepare_data(bundle/'pilot',previous['selection'],cfg)
    assert manifest==previous, '재생성 M2 입력 manifest 불일치'
    import numpy as np
    for split,d in data.items():
        prior=np.load(bundle/f'{split}_motion.npz',allow_pickle=False)
        np.testing.assert_array_equal(d['target_raw'][:,-1].numpy(),prior['gt'])
        np.testing.assert_array_equal(d['valid'][:,-1].numpy(),prior['valid'])
        np.testing.assert_array_equal([f for _,f in d['anchors']],prior['frame_indices'])
    return cfg,data,manifest


def run300(bundle, output, *, device='cuda', resume=None, steps=300, enforce_baseline_environment=True):
    import torch
    import numpy as np
    from rewind_m2.pose_pilot import PosePilot, write_json
    from rewind_m1.training import save_checkpoint
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter(); cfg,data,manifest=prepare_reuse(bundle)
    trainer=PosePilot(cfg,data,manifest,device)
    prior=load(bundle/'metrics.json')
    env_delta={k:{'previous':v,'current':trainer.environment.get(k)} for k,v in prior['environment'].items() if v!=trainer.environment.get(k)}
    if enforce_baseline_environment:
        assert device=='cuda' and trainer.environment['gpu']=='Tesla T4', 'T4 GPU 런타임 필요'
        assert not env_delta, '기존 환경과 다름: '+json.dumps(env_delta)
    write_json(output/'environment_comparison.json',{'differences':env_delta,'strict':enforce_baseline_environment})
    write_json(output/'config.json',cfg);write_json(output/'data_manifest.json',manifest)
    write_json(output/'input_provenance.json',{'mode':'verified_5step_result_reuse','bundle_sha256':EXPECTED_ZIP_SHA,
        'baseline_revision':BASE_SHA,'fresh_archive_read_this_run':False,'source_files':148,
        'same_input_manifest':True,'same_config_seed_model':True,'train_frames':18,'holdout_frames':130})
    history=[]
    if resume:
        resume=Path(resume); trainer.resume(resume)
        h=load(resume.parent/'loss_history.json');history=[r for r in h if r['step']<=trainer.step]
        assert [r['step'] for r in history]==list(range(1,trainer.step+1)), 'resume loss log missing'
    initial={s:trainer.evaluate(s)[0] for s in data}
    old_history=load(bundle/'loss_history.json')
    old_checkpoint=torch.load(bundle/'last.pt',map_location='cpu',weights_only=True)
    assert old_checkpoint['step']==5 and old_checkpoint['manifest']==manifest and old_checkpoint['config']==cfg
    write_json(output/'metrics.json',{'status':'running','step':trainer.step,'requested_steps':steps})
    if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    t=time.perf_counter(); segment_start=trainer.step
    while trainer.step<steps and time.perf_counter()-t<=cfg['max_train_seconds']:
        history.append(trainer.train_step())
        if trainer.step==5 and segment_start==0:
            equal=history==old_history and all(torch.equal(v.cpu(),old_checkpoint['model'][k]) for k,v in trainer.model.state_dict().items())
            write_json(output/'five_step_replay.json',{'loss_and_model_bitwise_equal':equal,'environment_equal':not env_delta})
            if enforce_baseline_environment and not equal:
                save_checkpoint(trainer.checkpoint(),output/'last.pt');write_json(output/'loss_history.json',history)
                raise ValueError('5-step 재현 불일치: 300 비교 학습을 중단했습니다.')
        if trainer.step%25==0 or trainer.step==steps:
            save_checkpoint(trainer.checkpoint(),output/f'step_{trainer.step:03d}.pt')
            save_checkpoint(trainer.checkpoint(),output/'last.pt')
            write_json(output/'loss_history.json',history)
            print(json.dumps(history[-1]),flush=True)
    if device=='cuda':torch.cuda.synchronize()
    train_seconds=time.perf_counter()-t
    save_checkpoint(trainer.checkpoint(),output/'last.pt');write_json(output/'loss_history.json',history)
    peak_train=torch.cuda.max_memory_allocated() if device=='cuda' else None
    final={}
    for split in data:
        final[split],arrays=trainer.evaluate(split)
        np.savez_compressed(output/f'{split}_motion.npz',**{k:v.numpy() for k,v in arrays.items()},
            frame_indices=np.array([i for _,i in data[split]['anchors']]))
    metrics={'status':'completed' if trainer.step==steps else 'time_budget_stop','step':trainer.step,'requested_steps':steps,
        'data_kind':manifest['data_kind'],'scope':prior['scope'],'initial':initial,'final':final,
        'training_seconds':train_seconds,'training_segment_start':segment_start,'resume_from_sha256':sha(resume) if resume else None,
        'evaluation_seconds':sum(v['evaluation_seconds'] for v in final.values()),
        'total_run_seconds_excluding_visualization':time.perf_counter()-started,
        'environment':trainer.environment,'environment_diff_vs5':env_delta,'source':trainer.source,'config':cfg,
        'manifest_sha256':manifest['sha256'],'checkpoint_sha256':sha(output/'last.pt'),
        'peak_training_allocated_vram_bytes':peak_train,
        'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
        'loss_definition':prior['loss_definition'],'limitations':prior['limitations']}
    write_json(output/'metrics.json',metrics)
    # Resume evidence is isolated from the published step300 weights/history.
    a=PosePilot(cfg,data,manifest,device);a.resume(output/'last.pt');la=a.train_step()
    wa={k:v.detach().cpu().clone() for k,v in a.model.state_dict().items()}
    b=PosePilot(cfg,data,manifest,device);b.resume(output/'last.pt');lb=b.train_step()
    assert la==lb and all(torch.equal(v.cpu(),wa[k]) for k,v in b.model.state_dict().items())
    write_json(output/'validation.json',{'checkpoint_step':trainer.step,'resume_next_update':'bitwise_identical',
        'extra_validation_updates_saved_to_last':False,'actual_data':True,'device':device})
    return trainer,metrics


def package(output,bundle,repo):
    from rewind_m2.pose_pilot import write_json
    output,bundle,repo=map(Path,(output,bundle,repo))
    m=load(output/'metrics.json')
    assert m['step']==m['requested_steps']==300 and m['status']=='completed'
    assert load(output/'validation.json')['checkpoint_step']==300
    shutil.copytree(bundle/'pilot',output/'pilot')
    shutil.copytree(bundle/'metadata',output/'metadata')
    prior=output/'previous_5step';prior.mkdir()
    for n in ('metrics.json','loss_history.json','config.json','data_manifest.json','last.pt','train_motion.npz','holdout_motion.npz','validation.json'):
        shutil.copy2(bundle/n,prior/n)
    for relative in list(m['source']['code_sha256'])+['configs/skeleton_unrealego2_dev.json','tools/pose_reuse_300.py','tools/pose_reuse_analysis.py','tools/pose_drive_diagnostics.py']:
        dest=output/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/relative,dest)
    write_json(output/'checksums.json',{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name!='checksums.json'})
    archive=shutil.make_archive(str(output),'zip',output)
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None
    return archive
