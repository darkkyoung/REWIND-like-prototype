"""원본 verifier/prepare_reuse 재사용. 격리된 baseline worktree에서만 동작한다."""
import argparse,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
BASE='5e52aa3b0b1e27e7d0cb2697629e8acfc63d38c3'
OLD='74851d8c05933c135bf784c15f29dbf684987687'
SHA300='217fe755c4cd347d4e2ce11045f479fde494d1474ce87c3848019277e21b5a3f'

def main():
 p=argparse.ArgumentParser();p.add_argument('--legacy-repo',required=True);p.add_argument('--zip5',required=True);p.add_argument('--zip300',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(a.legacy_repo).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==BASE
 zpath=Path(a.zip300)
 assert zpath.stat().st_size<32*1024**2 and hashlib.sha256(zpath.read_bytes()).hexdigest()==SHA300
 with zipfile.ZipFile(zpath) as z:
  assert z.testzip() is None
  for rel in ('rewind_m2/pose_pilot.py','tools/pose_reuse_300.py'):
   content=z.read('source/'+rel)
   assert content==subprocess.check_output(['git','show',f'{OLD}:{rel}'],cwd=root)
   dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():assert dest.read_bytes()==content
   else:dest.write_bytes(content)
  sys.path.insert(0,str(root))
  from tools.pose_reuse_300 import verify_bundle,prepare_reuse
  verified=verify_bundle(a.zip5,out/'original_5step')
  cfg,data,m=prepare_reuse(verified)
  import numpy as np
  for split,d in data.items():
   np.savez_compressed(out/f'{split}.npz',**{k:v.numpy() for k,v in d.items() if k!='anchors'},sequence_ids=np.array([s for s,f in d['anchors']]),frame_ids=np.array([f for s,f in d['anchors']]))
  (out/'config.json').write_text(json.dumps(cfg));(out/'data_manifest.json').write_text(json.dumps(m))
  for n in ('last.pt','metrics.json','train_motion.npz','holdout_motion.npz'):(out/('historical_'+n)).write_bytes(z.read(n))
  for item in m['payloads']:assert hashlib.sha256(z.read('pilot/files/'+item['member'])).hexdigest()==item['sha256']
 (out/'input_checksums.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},indent=2))
 print('PASS: original verifier, extraction, manifest, targets, masks and 148 payload hashes')
 print('새 ZIP을 만들면 컨테이너 SHA가 달라질 수 있으므로 실행용 pinned 입력 ZIP과 혼동하지 마세요.')
if __name__=='__main__':main()
