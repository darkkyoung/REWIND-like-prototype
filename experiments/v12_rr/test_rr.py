import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from rewind_v1.teacher import V1Body,motion_losses
from . import trainer as tr
from .data import HERE
from .run import decision

class RRTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(921)
        self.x=torch.randn(1,4,22,3,requires_grad=True)
        self.gt=torch.randn_like(self.x)*100
        self.v=torch.ones(1,4,22,dtype=torch.bool)
        self.t=torch.ones(1,4,dtype=torch.bool)
        self.cfg=dict(hidden=16,heads=2,graph_layers=1,temporal_layers=1,attention_past_frames=20,dropout=0.,
                      target_scale_mm=100.,diffusion_steps=1000,beta_start=.0001,beta_end=.02,frame_loss_weight=1.)
        self.body=V1Body(self.cfg,22,[(i,i+1) for i in range(21)])
        self.batch=dict(keys=[('s',i) for i in range(4)],features=torch.randn(4,62*14),camera=torch.randn(4,2,9),
                        target=torch.randn(4,62,3),valid=torch.ones(4,62,dtype=torch.bool),split='train')

    def test_exact_pair_denominator_and_mm(self):
        self.v.zero_();self.v[0,0,:3]=True
        self.x.data.zero_();self.gt.zero_()
        self.gt[0,0,1]=torch.tensor([100.,200.,200.])
        self.assertEqual(tr.rr_loss(self.x,self.gt,self.v,self.t,100).item(),4.5)

    def test_translation_padding_invalid_root_and_nan(self):
        self.v[0,0,0]=False;self.t[0,-1]=False
        a=tr.rr_loss(self.x,self.gt,self.v,self.t,100)
        self.gt[0,0,0]=float('nan');self.gt[0,-1]=float('nan')
        b=tr.rr_loss(self.x+2,self.gt+500,self.v,self.t,100)
        torch.testing.assert_close(a,b)
        self.gt[0,1,1]=float('nan')
        with self.assertRaises(ValueError):tr.rr_loss(self.x,self.gt,self.v,self.t,100)

    def test_no_valid_pair_zero_gradient(self):
        self.v[:,:,0]=False
        z=tr.rr_loss(self.x,self.gt,self.v,self.t,100)
        z.backward()
        self.assertEqual(z.item(),0);self.assertFalse(self.x.grad.any())

    def test_zero_weight_original_value_and_both_head_gradients(self):
        frame=self.x.detach().clone().requires_grad_()
        ref,_=motion_losses(self.x,frame,self.gt,self.v,self.t,scale_mm=100,frame_weight=1.)
        old,_=motion_losses(self.x,frame,self.gt,self.v,self.t,scale_mm=100,frame_weight=1.)
        combined=tr.objective(old,tr.rr_loss(self.x,self.gt,self.v,self.t,100),0.)
        self.assertTrue(torch.equal(ref,combined))
        for a,b in zip(torch.autograd.grad(ref,(self.x,frame)),torch.autograd.grad(combined,(self.x,frame))):
            self.assertTrue(torch.equal(a,b))

    def test_paired_steps_and_actual_body_gradient(self):
        b=copy.deepcopy(self.body)
        self.assertEqual(tr.state_hash(b),tr.state_hash(self.body))
        weight,records=tr.calibrate(b,[self.batch],self.cfg)
        self.assertGreater(weight,0);self.assertEqual(len(records),4)
        refs=[]
        for model,w in [(self.body,0.),(b,weight)]:
            opt,sched=tr.optimizer(model)
            refs.append(tr.update(model,opt,sched,[self.batch],self.cfg,0,w))
        self.assertEqual(refs[0]['trace'],refs[1]['trace'])
        self.assertNotEqual(tr.state_hash(b),tr.state_hash(self.body))

    def test_resume_replays_next_update_and_rejects_wrong_binding(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'body.pt';opt,sch=tr.optimizer(self.body)
            log=[tr.update(self.body,opt,sch,[self.batch],self.cfg,0,.1)]
            pin=tr.save(path,self.body,opt,sch,{'arm':'improved'},log)
            expected=tr.update(self.body,opt,sch,[self.batch],self.cfg,1,.1)
            resumed=copy.deepcopy(self.body);o,s=tr.optimizer(resumed)
            hist=tr.restore(path,pin,resumed,{'arm':'improved'},o,s)
            self.assertEqual(tr.update(resumed,o,s,[self.batch],self.cfg,len(hist),.1),expected)
            self.assertEqual(tr.state_hash(resumed),tr.state_hash(self.body))
            with self.assertRaises(ValueError):tr.restore(path,pin,resumed,{'arm':'control'})
            with self.assertRaises(ValueError):tr.restore(path,'0'*64,resumed,{'arm':'improved'})

    def test_gap_and_eval_leakage_rejected(self):
        for delta in [dict(split='validation'),dict(keys=[('s',0),('s',1),('s',4),('s',5)])]:
            with self.assertRaises(ValueError):tr.forward_terms(self.body,self.batch|delta,self.cfg,tr.draw(0,1)[1])

    def test_causal_teacher_prefix(self):
        x=torch.randn(1,4,22,3);f=torch.randn(1,4,308);c=torch.randn(1,4,2,9);k=torch.tensor([40])
        before=self.body(x,f,c,k,self.t)[0]
        f[:,3]+=99
        torch.testing.assert_close(before[:,:3],self.body(x[:,:3],f[:,:3],c[:,:3],k,self.t[:,:3])[0])

    def test_frozen_selection_and_failure_rule(self):
        doc=json.loads((HERE/'selection.json').read_text())
        self.assertEqual(len({s.split('/')[2] for s,_ in doc['train']}),6)
        self.assertFalse({s for s,_ in doc['train']}&{s for s,_ in doc['validation']})
        m={k:10. for k in ('rr_jpe_mm','body_mpjpe_mm','pelvis_mm','wrist_mm','hand_global_mpjpe_mm','whole_mpjpe_mm')}
        self.assertEqual(decision(dict(control=m,improved=m)), 'FAIL')
        self.assertEqual(decision(dict(control=m,improved=m|{'rr_jpe_mm':9.8})), 'PASS')
