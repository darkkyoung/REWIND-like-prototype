"""Analytic fixtures verify metric invariants; never used as experiment data."""
import unittest
import numpy as np
from diagnose import calculate


class DiagnosisMetricsTests(unittest.TestCase):
    def setUp(self):
        self.g=np.array([[[0.,0,0],[2,0,0],[0,3,0]],[[1.,0,0],[4,0,0],[1,4,0]],[[2.,0,0],[6,0,0],[2,5,0]]])
        self.mask=np.ones((3,3),bool);self.ids=np.arange(3);self.edges=[[0,1],[0,2]]

    def calc(self,p,g=None,mask=None,ids=None):
        return calculate(p,self.g if g is None else g,self.mask if mask is None else mask,self.ids if ids is None else ids,self.edges,0)[0]

    def test_rigid_translation_does_not_change_shape(self):
        s=self.calc(self.g+np.array([3,4,0]))
        for key in ['global_joint_error_raw','pelvis_position_error_raw']:self.assertAlmostEqual(s[key],5)
        self.assertAlmostEqual(s['root_relative_error_22_raw'],0)
        for key in ['bone_length_ratio_pct','joint_spread_ratio_pct','motion_magnitude_ratio_pct']:self.assertAlmostEqual(s[key],100)

    def test_half_scale_halves_length_and_spread(self):
        s=self.calc(self.g*.5)
        for key in ['bone_length_ratio_pct','mean_individual_bone_ratio_pct','joint_spread_ratio_pct','motion_magnitude_ratio_pct']:self.assertAlmostEqual(s[key],50)

    def test_static_pose_is_temporal_not_spatial_collapse(self):
        # Translating rigid object: freezing it preserves shape but destroys motion.
        g=np.stack([self.g[0]+[i,0,0] for i in range(3)])
        s=self.calc(np.repeat(g[:1],3,0),g=g)
        self.assertAlmostEqual(s['bone_length_ratio_pct'],100)
        self.assertAlmostEqual(s['joint_spread_ratio_pct'],100)
        self.assertAlmostEqual(s['motion_magnitude_ratio_pct'],0)
        self.assertAlmostEqual(s['root_relative_error_22_raw'],0)

    def test_invalid_joint_and_frame_gap_excluded(self):
        mask=self.mask.copy();mask[1,2]=False
        p=self.g.copy();p[1,2]=np.nan
        s=self.calc(p,mask=mask,ids=np.array([0,1,3]))
        self.assertAlmostEqual(s['global_joint_error_raw'],0)
        self.assertEqual(s['valid_adjacent_joint_pairs'],2)
        self.assertAlmostEqual(s['bone_length_ratio_pct'],100)


if __name__=='__main__':unittest.main()
