import copy
import json
import unittest

import numpy as np
import torch
from rewind_v1.teacher import V1Body,V1Hand
from rewind_v1.frontend import PaperHeatmapFrontend
from rewind_v1.cross_inference import predict as original
from scripts.visualize_archvis_v1 import _group_names
from .pipeline import ROOT, POLICIES, compare, corrected, frontend, layout, observations
from .evaluate import metrics


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(101)
        cls.spec=json.loads((ROOT/'configs/real_archvis62_v1.json').read_text())
        cls.edges=layout(cls.spec)
        cfg=dict(hidden=16,heads=2,graph_layers=1,temporal_layers=1,attention_past_frames=20,
                 diffusion_steps=1000,beta_start=.0001,beta_end=.02,dropout=0.,target_scale_mm=100.,sampling_steps=10)
        hand=_group_names(cls.spec,'hand')
        hedges=[(hand.index(a),hand.index(b)) for a,b in cls.spec['edges'] if a in hand and b in hand]
        cls.models=dict(spec=cls.spec,config=dict(seed=71,teacher=cfg),device='cpu',
            frontend=PaperHeatmapFrontend(62),body=V1Body(cfg,22,cls.edges.tolist()),
            hand=V1Hand(cfg,40,hedges,len(cls.spec['upper_joint_names'])))
        for key in ('frontend','body','hand'):
            cls.models[key].eval().requires_grad_(False)
        cls.keys=[('A',f) for f in range(7)]
        cls.images=torch.rand(7,2,3,128,128)
        cls.camera=torch.tensor([[[1,0,0,0,1,0,-.6,0,0],[1,0,0,0,1,0,.6,0,0]]],dtype=torch.float32).repeat(7,1,1)
        cls.features=frontend(cls.models,cls.images)
        cls.prior=(np.ones(21)*30,np.ones(21)*.01)
        cls.ref,_=original(cls.models,cls.images,cls.camera,cls.keys,cls.models['config'],'cpu')
        cls.out,_=compare(cls.models,cls.features,cls.camera,cls.keys,prior=cls.prior,hand_ablation=True)

    def test_baseline_exact_all_outputs(self):
        for k,v in self.ref.items():
            self.assertTrue(torch.equal(v,self.out['baseline'][k]),k)

    def test_upper_reaches_hand_and_wrist_added_once(self):
        for name in ('phase2','phase4'):
            r=self.out[name];orig=self.out[name+'_original_hand']
            self.assertFalse(torch.equal(r['hand_local'],orig['hand_local']))
            self.assertTrue(torch.equal(orig['hand_local'],self.ref['hand_local']))
            indices=[9]*20+[13]*20
            self.assertTrue(torch.equal(r['hand_global'],r['body'][:,indices]+r['hand_local']))
            self.assertTrue(torch.equal(r['whole'],torch.cat([r['body'],r['hand_global']],dim=1)))

    def test_mm_identity_and_directed_mapping(self):
        self.assertEqual(self.spec['coordinates']['mm_per_raw_unit'],10)
        self.assertEqual(self.spec['body_joint_names'][0],'pelvis')
        spec=copy.deepcopy(self.spec);spec['edges'][0].reverse()
        with self.assertRaises(ValueError):layout(spec)

    def test_noop_all_policies(self):
        pose=self.ref['body'][:1].repeat(7,1,1)
        a,b=self.edges.T
        prior=((pose[0,b]-pose[0,a]).norm(dim=-1).numpy(),self.prior[1])
        obs=observations(self.features,self.spec)
        for policy in POLICIES:
            self.assertTrue(torch.allclose(corrected(pose,obs,self.keys,policy,self.edges,prior),pose,atol=1e-5))

    def test_future_perturbation_reset_and_pelvis(self):
        p=self.ref['body'];obs=observations(self.features,self.spec)
        for policy in POLICIES:
            def run(x,o,k):
                return corrected(x,o,k,policy,self.edges,self.prior)
            original_pose=run(p,obs,self.keys)
            self.assertTrue(torch.equal(original_pose[:,0],p[:,0]))
            future=p.clone();future[-1]*=4
            other=obs.copy();other[-1,...,:2]=0
            self.assertTrue(torch.equal(original_pose[:-1],run(future,other,self.keys)[:-1]))
            self.assertTrue(torch.equal(original_pose[:-1],run(p[:-1],obs[:-1],self.keys[:-1])))
            for tail in ([('B',0),('B',1)],[('A',10),('A',11)]):
                self.assertTrue(torch.equal(run(p,obs,self.keys[:5]+tail)[5:],run(p[5:],obs[5:],tail)))

    def test_bad_inputs_fail_closed(self):
        for pos,val in [(0,float('nan')),(2,.5),(3,-1.)]:
            f=self.features.clone();f[0,pos]=val
            with self.assertRaises(ValueError):observations(f,self.spec)
        with self.assertRaises(ValueError):compare(self.models,self.features,self.camera*0,self.keys)
        with self.assertRaises(ValueError):compare(self.models,self.features[:-1],self.camera,self.keys)
        with self.assertRaisesRegex(ValueError,'calibration'):compare(self.models,self.features,self.camera,self.keys)
        with self.assertRaises(ValueError):metrics(self.ref,self.ref['body'],self.ref['whole'],torch.ones(7,62),self.spec)

    def test_confidence_is_separate_from_availability(self):
        f=self.features.reshape(7,62,2,7).clone();f[...,3]=0
        obs=observations(f.flatten(1),self.spec)
        self.assertTrue((obs[...,2]==1).all())
        self.assertTrue((obs[...,3]==0).all())
        np.testing.assert_allclose(obs[...,:2],((f[:,:22,:,:2]+1)*1023/2048).numpy())

    def test_strict_synthetic_state_and_frozen_weights(self):
        state={k:v.clone() for k,v in self.models['body'].state_dict().items()}
        clone=copy.deepcopy(self.models['body'])
        clone.load_state_dict(state,strict=True)
        with self.assertRaises(RuntimeError):clone.load_state_dict({},strict=True)
        compare(self.models,self.features,self.camera,self.keys,prior=self.prior)
        self.assertTrue(all(torch.equal(v,self.models['body'].state_dict()[k]) for k,v in state.items()))

    def test_metric_identity_and_partial_masks(self):
        valid=torch.ones(7,62,dtype=torch.bool);valid[1,9]=False
        row=metrics(self.ref,self.ref['body'],self.ref['whole'],valid,self.spec)
        for k in ('whole_mpjpe_mm','fde_mm_per_frame','harmed_frame_fraction','body_mpjpe_mm'):
            self.assertEqual(row[k],0)
        self.assertAlmostEqual(row['motion_ratio_pct'],100)
