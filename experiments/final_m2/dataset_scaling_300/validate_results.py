"""실제 저장 결과만 독립 검산한다. 모델 추론/학습은 하지 않는다."""
import argparse,csv,hashlib,json,platform,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.final_m2.dataset_scaling_100.verify_handoff import independent_metrics
from experiments.final_m2.dataset_scaling_300.prepare_data import read_npz
from experiments.final_m2.dataset_scaling_300.run_experiment import summarize,quality300,state_hash
from tools.final_m2_real import sha,load,write
def run(archive,inp,baseline,source,out):
 archive,inp,baseline,source,out=map(Path,(archive,inp,baseline,source,out));out.mkdir(parents=True,exist_ok=False)
 raw=out/'raw';raw.mkdir()
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None and all(Path(n).name==n for n in z.namelist());z.extractall(raw);entries=len(z.infolist())
 hashes=load(raw/'checksums.json')
 assert set(hashes)=={p.name for p in raw.iterdir() if p.is_file()}-{'checksums.json'}
 for n,h in hashes.items():assert sha(raw/n)==h,n
 identity=load(raw/'experiment_identity.json');evaluation=load(raw/'evaluation.json');manifest=load(inp/'data_selection_manifest.json')
 with zipfile.ZipFile(source) as z:
  assert z.testzip() is None
  for n,h in identity['code_sha256'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
  assert hashlib.sha256(z.read('experiments/final_m2/dataset_scaling_300/config.json')).hexdigest()==identity['config_sha256']
 assert sha(inp/'input_checksums.json')==identity['input_checksums_sha256']
 for n,h in load(inp/'input_checksums.json').items():assert sha(inp/n)==h
 rows,bones,frames=summarize(raw,manifest);maxdelta=0.;independent=[]
 csvrows=list(csv.DictReader((raw/'comparison.csv').open()))
 for r in rows:
  saved=next(s for s in evaluation['results'] if s['split']==r['split'] and s['model']==r['model'])
  cr=next(s for s in csvrows if s['split']==r['split'] and s['model']==r['model'])
  for k,v in r.items():
   if isinstance(v,(float,int)):
    maxdelta=max(maxdelta,abs(v-saved[k]));assert abs(v-saved[k])<1e-9 and abs(v-float(cr[k]))<1e-9
  z=read_npz(raw/f"{r['split']}_{r['model']}_predictions.npz");d=read_npz(inp/(r['split']+'.npz'))
  assert z['prediction'].shape==(len(d['frame_ids']),22,3) and np.isfinite(z['prediction']).all()
  for k,dk in [('gt','target_raw'),('valid','valid')]:np.testing.assert_array_equal(z[k],d[dk][:,-1])
  for k in ('frame_ids','sequence_ids'):np.testing.assert_array_equal(z[k],d[k])
  im=independent_metrics(z,np.array(manifest['edges']))
  p,g=z['prediction'].astype(float),z['gt'].astype(float);pr=p-p[:,:1];gr=g-g[:,:1]
  im['rr_temporal_pose_rms_ratio_pct']=100*float(np.sqrt(np.mean((pr-pr.mean(0))**2))/np.sqrt(np.mean((gr-gr.mean(0))**2)))
  for k,v in im.items():assert abs(v-r[k])<1e-9
  independent.append({'split':r['split'],'model':r['model'],**im})
 for arm in ('train18','train100'):assert sha(raw/f'opening130_{arm}_predictions.npz')==sha(baseline/f'opening130_{arm}_predictions.npz')
 log=[json.loads(l) for l in (raw/'train300_training.jsonl').read_text().splitlines()]
 csvlog=list(csv.DictReader((raw/'train300_training.csv').open()))
 assert len(log)==len(csvlog)==400 and [r['step'] for r in log]==list(range(1,401))
 for i,(a,b) in enumerate(zip(log,csvlog),1):
  for k,v in a.items():
   if isinstance(v,(int,float)):assert np.isfinite(v) and float(b[k])==v
   else:assert b[k]==v
  assert a['lr_used']==(.001 if i<=200 else .0005)
 trace=load(raw/'train300_random_trace.json');record=load(raw/'train300_identity.json')
 assert len(trace)==400 and [r['step'] for r in trace]==list(range(1,401))
 counts=np.bincount([a for r in trace for a in r['anchors']],minlength=300)
 assert len(counts)==300 and (counts>0).all() and counts.sum()==3200 and counts.tolist()==record['anchor_sample_counts']
 assert state_hash(trace)==record['full_trace_sha256']
 assert state_hash([{k:v for k,v in r.items() if k!='anchors'} for r in trace])==record['diffusion_trace_sha256']
 checkpoints=[]
 for step in (100,200,300,400):
  p=raw/f'train300_{step}.pt';ck=torch.load(p,map_location='cpu',weights_only=True)
  assert ck['step']==step and ck['samples_seen']==step*8 and ck['arm']=='train300'
  assert ck['scheduler']['last_epoch']==step and ck['scaler']=={} and ck['rng']['torch_cuda']
  assert ck['experiment_identity']==identity and ck['initial_state_sha256']==record['initial_state_sha256']
  assert all(float(s['step'])==step for s in ck['optimizer']['state'].values())
  assert all(torch.isfinite(v).all() for v in ck['model'].values())
  assert all(torch.isfinite(v).all() for s in ck['optimizer']['state'].values() for v in s.values() if isinstance(v,torch.Tensor))
  expected=next(c for c in record['checkpoints'] if c['step']==step);assert sha(p)==expected['sha256']
  checkpoints.append({'file':p.name,'sha256':sha(p),'step':step,'optimizer_state_count':len(ck['optimizer']['state']),'status':'PASS'})
 assert state_hash(ck['rng'])==record['end_rng_sha256']
 gates=quality300(rows)
 assert gates==evaluation.get('quality_unqualified',evaluation['quality'])
 result={'status':'PASS','audit_date_kst':'2026-11-17','execution_status':evaluation['execution_status'],'quality':evaluation['quality'],'additional_validation_quality':evaluation['additional_validation_quality'],'archive':{'bytes':archive.stat().st_size,'sha256':sha(archive),'crc_entries':entries,'checksummed_files':len(hashes)},'source_commit':identity['source_commit'],'environment':identity['environment'],'fairness':evaluation['fairness'],'prediction_groups':6,'metric_max_absolute_difference':maxdelta,'independent_metrics':independent,'checkpoint_verification':checkpoints,'training_log_rows':400,'trace_anchor_draws':3200,'distinct_anchors':300,'baseline_opening_npz_byte_identity':True,'new_inference_calls':0,'new_optimizer_updates':0,'missing_files':[]}
 write(out/'result_validation.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('archive','input','baseline','source','output'):p.add_argument('--'+n,required=True)
 a=p.parse_args();run(a.archive,a.input,a.baseline,a.source,a.output)
