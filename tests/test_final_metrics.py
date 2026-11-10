import unittest
import numpy as np
from rewind_m2.final_metrics import evaluate_motion,latency_summary

class FinalMetrics(unittest.TestCase):
    def test_direction_mask_gap(self):
        g=np.array([[[0.,0,0]],[[2,0,0]],[[100,0,0]]]);p=-g;m=np.ones((3,1),bool)
        r=evaluate_motion(p,g,m,[('a',0),('a',1),('a',3)])
        self.assertEqual(r['frame_displacement_error_raw'],4)
        self.assertEqual(r['motion_magnitude_ratio'],1)
        self.assertEqual(r['adjacent_valid_joint_pairs'],1)
        m[1]=False;g[1]=np.nan
        self.assertIsNone(evaluate_motion(p,g,m,[('a',0),('a',1),('a',3)])['motion_magnitude_ratio'])
    def test_p95_stationary(self):
        self.assertEqual(latency_summary([1,2,3,4,5])['p95_latency_ms_per_frame'],4.8)
        z=np.zeros((2,1,3))
        self.assertIsNone(evaluate_motion(z,z,np.ones((2,1),bool),[('a',0),('a',1)])['motion_magnitude_ratio'])
