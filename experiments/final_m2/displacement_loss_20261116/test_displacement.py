"""합성 fixture는 구현 검사 전용. 성능 결과에는 포함하지 않는다."""
import copy
import unittest
import numpy as np
import torch
from experiments.final_m2.displacement_loss_20261116.loss import displacement_loss, window_metadata, S_DELTA
from experiments.final_m2.displacement_loss_20261116.run_experiment import ensure_finite, state_hash, make_arm
from rewind_m2.models import BodyDenoiser

class DisplacementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(43)
        self.g=torch.randn(2,8,22,3);self.m=torch.ones(2,8,22,dtype=torch.bool)
        self.t=torch.ones(2,8,dtype=torch.bool);self.f=torch.arange(8).repeat(2,1);self.s=torch.zeros(2,8,dtype=torch.long)
        self.cfg=dict(hidden=64,heads=4,graph_layers=1,temporal_layers=2,window_size=8,attention_past_frames=7,
                      dropout=0.,diffusion_steps=1000,learning_rate=.001,scheduler_step=200,scheduler_gamma=.5)
    def loss(self,p,g=None):
        return displacement_loss(p,self.g if g is None else g,self.m,self.t,self.f,self.s)[0]
    def test_identical_displacement_translation_zero(self):
        p=(self.g+8).requires_grad_();self.assertLess(float(self.loss(p).detach()),1e-12)
    def test_xyz_sum_reduction_and_raw_scale(self):
        self.g.zero_();p=torch.arange(8).reshape(1,8,1,1).expand_as(self.g)*S_DELTA
        self.assertAlmostEqual(float(self.loss(p)),3.,places=5)
    def test_static_output_not_motion_match(self):
        self.assertGreater(float(self.loss(torch.zeros_like(self.g))),0)
    def test_mask_padding_gap_sequence(self):
        self.g.zero_();p=self.g.clone();p[:,3]=100
        self.m[:,3]=False
        self.assertEqual(float(self.loss(p)),0)
        self.m[:]=True;self.t[:,3]=False
        self.assertEqual(float(self.loss(p)),0)
        self.t[:]=True;self.f[:,3]=20 # both incident pairs excluded
        self.assertEqual(float(self.loss(p)),0)
        self.f=torch.arange(8).repeat(2,1);self.s[:,3]=1
        self.assertEqual(float(self.loss(p)),0)
    def test_masked_nan_and_empty_pairs_backward(self):
        self.m[:]=False;self.g[:]=float('nan');p=torch.randn_like(self.g,requires_grad=True)
        l=self.loss(p);l.backward();self.assertEqual(float(l.detach()),0);self.assertTrue(torch.isfinite(p.grad).all())
    def test_nonfinite_rejected_before_update(self):
        p=self.g.clone();p[0,0,0,0]=float('inf')
        with self.assertRaisesRegex(ValueError,'nonfinite'):self.loss(p)
        with self.assertRaisesRegex(ValueError,'nonfinite'):ensure_finite([torch.tensor(float('nan'))],'gradient')
    def test_real_model_gradient_and_future_causality(self):
        model=BodyDenoiser(self.cfg,22,[(i,i+1) for i in range(21)]).eval()
        x=torch.randn(2,8,308);cam=torch.zeros(2,8,2,9);noise=torch.randn_like(self.g);k=torch.tensor([80,400])
        p,_=model(noise,x,cam,k,self.t);self.loss(p*100).backward()
        self.assertGreater(float(model.final_head.weight.grad.norm()),0)
        self.assertGreater(float(model.temporal[0].qkv.weight.grad.norm()),0)
        ensure_finite([v.grad for v in model.parameters()],'gradient')
        x2=x.clone();x2[:,4:]+=50;n2=noise.clone();n2[:,4:]-=50
        changed,_=model(n2,x2,cam,k,self.t)
        torch.testing.assert_close(p[:,:4],changed[:,:4],rtol=0,atol=0)
    def test_metadata_checks_actual_slots_and_gaps(self):
        d={'sequence_ids':np.array(['a','a','a','b']), 'frame_ids':np.array([0,1,3,0]),
           'time':np.array([[0,1],[1,1],[0,1],[0,1]],dtype=bool)}
        for key in ('features','target_raw','valid'):
            d[key]=np.zeros((4,2,1),dtype=bool if key=='valid' else np.float32)
            d[key][:,-1,0]=True if key=='valid' else np.arange(1,5)
            d[key][1,0]=d[key][0,-1]
        frames,seq=window_metadata(d);self.assertEqual(frames.tolist(),[[-1,0],[0,1],[-1,3],[-1,0]])
        self.assertNotEqual(int(seq[0,-1]),int(seq[3,-1]))
        bad=copy.deepcopy(d);bad['features'][1,0]=99
        with self.assertRaisesRegex(ValueError,'content'):window_metadata(bad)
        bad=copy.deepcopy(d);bad['time'][2,0]=True
        with self.assertRaisesRegex(ValueError,'gap'):window_metadata(bad)
    def test_checkpoint_optimizer_scheduler_exact_restore(self):
        # Synthetic state at update300 tests branch restore; does not train real data.
        model=BodyDenoiser(self.cfg,22,[(i,i+1) for i in range(21)])
        opt=torch.optim.Adam(model.parameters(),lr=.001);sch=torch.optim.lr_scheduler.StepLR(opt,200,.5)
        for p in model.parameters():p.grad=torch.ones_like(p)
        opt.step();sch.step()
        for s in opt.state.values():s['step'].fill_(300)
        sch.last_epoch=300
        ck={'config':self.cfg,'manifest':{'edges':[(i,i+1) for i in range(21)]},'model':model.state_dict(),
            'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'scaler':{}}
        a=make_arm(ck,'cpu');b=make_arm(ck,'cpu')
        for aa,bb in zip(a,b):self.assertEqual(state_hash(aa.state_dict()),state_hash(bb.state_dict()))
        bad=copy.deepcopy(ck);bad['optimizer']['state'].pop(0)
        with self.assertRaisesRegex(ValueError,'incomplete'):make_arm(bad,'cpu')

if __name__=='__main__':unittest.main()
