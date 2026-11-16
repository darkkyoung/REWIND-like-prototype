"""실제 입력 호환성 및 모델 update 계약 검사. 전체 학습은 수행하지 않는다."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from experiments.final_m2.dataset_scaling_100.prepare_data import convert,coordinate_check
from experiments.final_m2.dataset_scaling_100.run_experiment import new_model,train,state_hash,extra_metrics,quality
from experiments.final_m2.displacement_loss_20261116.loss import window_metadata
from rewind_m2.diffusion import DiffusionSchedule
from rewind_m1.training import rng_state
ROOT=Path(__file__).resolve().parents[3]
class ScalingTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import os
  cls.inp=Path(os.environ['M2_SCALING_INPUT']);cls.m=json.loads((cls.inp/'data_selection_manifest.json').read_text());cls.cfg=json.loads((Path(__file__).parent/'config.json').read_text())['model'];cls.names=cls.m['joint_names']
  cls.d={}
  for name in ('train18','train100','opening130'):
   with np.load(cls.inp/f'{name}.npz') as z:cls.d[name]={k:z[k] for k in z.files}
  cls.frames=[]
  for r in cls.m['frames'][:10]:cls.frames.append((r['sequence_id'],r['frame_id'],json.loads((cls.inp/'raw_pose'/r['pose_path']).read_text())))
 def test_exact_prefix_and_cardinality(self):
  for k in self.d['train18']:np.testing.assert_array_equal(self.d['train18'][k],self.d['train100'][k][:18])
  self.assertEqual(len(set(zip(self.d['train100']['sequence_ids'],self.d['train100']['frame_ids']))),100)
 def test_no_holdout_leakage(self):
  self.assertFalse(set(self.d['train100']['sequence_ids'])&set(self.d['opening130']['sequence_ids']))
  self.assertEqual(len(set(r['pose_sha256'] for r in self.m['frames'])),100)
 def test_causal_no_future(self):
  a=convert(self.frames,self.names);rows=copy.deepcopy(self.frames)
  rows[9][2]['joints']['pelvis']['device_pts3d'][0]+=100
  b=convert(rows,self.names)
  for k in ('features','target_raw','valid','time'):np.testing.assert_array_equal(a[k][:9],b[k][:9])
 def test_gap_and_sequence_reset(self):
  rows=[self.frames[0],self.frames[1],self.frames[3]];a=convert(rows,self.names);self.assertEqual(a['time'][2].sum(),1)
  d=self.d['train100'];self.assertEqual(d['time'][18].sum(),1);self.assertEqual(d['time'][68].sum(),1);window_metadata(d)
 def test_coordinate_mismatch_rejected(self):
  d=copy.deepcopy(self.frames[0][2]);coordinate_check(d,self.names);d['joints']['pelvis']['device_pts3d'][0]*=10
  with self.assertRaises(ValueError):coordinate_check(d,self.names)
 def test_missing_2d_mask_and_invalid_3d(self):
  rows=copy.deepcopy(self.frames[:1]);rows[0][2]['joints']['pelvis']['camera_left_pts2d']=None
  a=convert(rows,self.names);np.testing.assert_array_equal(a['features'][0,-1,:3],[0,0,0]);self.assertTrue(a['valid'][0,-1,0])
  rows[0][2]['joints']['pelvis']['device_pts3d']=[float('nan'),0,0]
  with self.assertRaises(ValueError):convert(rows,self.names)
 def test_initial_weights_rng_exact(self):
  a,o,s=new_model(self.cfg,self.m,'cpu');h=state_hash(a.state_dict());r=state_hash(rng_state());b,ob,sb=new_model(self.cfg,self.m,'cpu')
  self.assertEqual(h,state_hash(b.state_dict()));self.assertEqual(r,state_hash(rng_state()));self.assertEqual(state_hash(o.state_dict()),state_hash(ob.state_dict()));self.assertEqual(s.last_epoch,0)
 def test_real_parameter_update_finite(self):
  with tempfile.TemporaryDirectory() as t:
   m,_,_=new_model(self.cfg,self.m,'cpu');initial=state_hash(m.state_dict());schedule=DiffusionSchedule(1000,.0001,.02)
   r=train('cpu_unit',self.cfg,self.m,self.d['train18'],schedule,Path(t),{'unit_test_only':True},device='cpu',updates=2)
   ck=torch.load(Path(t)/'cpu_unit_2.pt',weights_only=True);self.assertNotEqual(initial,state_hash(ck['model']));self.assertEqual(ck['step'],2);self.assertTrue(all(float(v['step'])==2 for v in ck['optimizer']['state'].values()))
 def test_direction_vs_magnitude(self):
  g=self.d['train18']['target_raw'][:,-1];v=self.d['train18']['valid'][:,-1];ids=self.d['train18']['frame_ids'];a=extra_metrics(g,g,v,ids);b=extra_metrics(-g,g,v,ids)
  self.assertAlmostEqual(a['direction_cosine'],1);self.assertAlmostEqual(b['direction_cosine'],-1);self.assertEqual(a['nonroot_rr_displacement_error_raw'],0)
if __name__=='__main__':unittest.main()
