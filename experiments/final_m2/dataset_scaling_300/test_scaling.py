"""실제 Train300 입력으로 prefix/인과성/경계/gradient를 검사한다."""
import copy,json,os,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from experiments.final_m2.dataset_scaling_300.prepare_data import read_npz
from experiments.final_m2.dataset_scaling_100.prepare_data import convert,coordinate_check
from experiments.final_m2.dataset_scaling_100.run_experiment import train,new_model,state_hash
from experiments.final_m2.dataset_scaling_300.run_experiment import quality300
from rewind_m2.diffusion import DiffusionSchedule
class Scaling300Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);cls.inp=Path(os.environ['M2_SCALING300_INPUT']);cls.raw=Path(os.environ['M2_SCALING300_RAW'])
  cls.man=json.loads((cls.inp/'data_selection_manifest.json').read_text());cls.cfg=json.loads((Path(__file__).parent/'config.json').read_text())['model']
  cls.d=read_npz(cls.inp/'train300.npz');cls.base=read_npz(cls.inp/'train100.npz')
  cls.rows=[(r['sequence_id'],r['frame_id'],json.loads((cls.raw/r['pose_path']).read_text())) for r in cls.man['frames'][100:110]]
 def test_prefix_and_unique(self):
  self.assertEqual(len(set(zip(self.d['sequence_ids'],self.d['frame_ids']))),300)
  for k in self.base:np.testing.assert_array_equal(self.base[k],self.d[k][:100])
  for k in ('features','target_raw','valid','time'):self.assertEqual(self.base[k].tobytes(),self.d[k][:100].tobytes())
 def test_new_raw_reproduces_tensor(self):
  d=convert(self.rows,self.man['joint_names'])
  for k in d:np.testing.assert_array_equal(d[k],self.d[k][100:110])
  for s,f,d in self.rows:coordinate_check(d,self.man['joint_names'])
 def test_no_future_and_gap(self):
  a=convert(self.rows,self.man['joint_names']);rows=copy.deepcopy(self.rows);rows[-1][2]['joints']['pelvis']['device_pts3d'][0]+=100
  b=convert(rows,self.man['joint_names'])
  for k in a:np.testing.assert_array_equal(a[k][:-1],b[k][:-1])
  c=convert([self.rows[0],self.rows[2]],self.man['joint_names']);self.assertEqual(c['time'][1].sum(),1)
  for i in (0,18,68,100,150,200,250):self.assertEqual(self.d['time'][i].sum(),1)
 def test_two_cpu_updates_only(self):
  with tempfile.TemporaryDirectory() as tmp:
   m,o,s=new_model(self.cfg,self.man,'cpu');initial=state_hash(m.state_dict())
   result=train('cpu_unit',self.cfg,self.man,self.d,DiffusionSchedule(1000,.0001,.02),Path(tmp),{'unit_test_only':True},device='cpu',updates=2)
   ck=torch.load(Path(tmp)/'cpu_unit_2.pt',weights_only=True)
   self.assertEqual(ck['step'],2);self.assertEqual(result['samples_seen'],16);self.assertNotEqual(initial,state_hash(ck['model']))
   self.assertTrue(all(float(s['step'])==2 for s in ck['optimizer']['state'].values()))
 def test_gate_comparison_is_100_to_300(self):
  a={'split':'opening130','model':'train100','global_joint_error_raw':10.,'root_relative_error_22_raw':10.,'frame_displacement_error_raw':1.,'motion_magnitude_ratio_pct':70.,'root_relative_motion_ratio_pct':50.,'pelvis_position_error_raw':7.,'mean_abs_bone_ratio_deviation_pp':19.,'joint_spread_ratio_pct':93.}
  b={**a,'model':'train300','global_joint_error_raw':9.,'root_relative_error_22_raw':9.,'frame_displacement_error_raw':.9,'motion_magnitude_ratio_pct':80.,'root_relative_motion_ratio_pct':60.}
  self.assertEqual(quality300([a,b])['status'],'PASS')
  b['pelvis_position_error_raw']=8.;self.assertEqual(quality300([a,b])['status'],'PARTIAL')
if __name__=='__main__':unittest.main()
