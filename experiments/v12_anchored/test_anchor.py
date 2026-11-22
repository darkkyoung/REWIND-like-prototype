import copy
import json
import math
import unittest

import torch
from rewind_v1.teacher import V1Body
from experiments.v12_rr import trainer as rr
from . import trainer as tr
from .data import HERE
from .evaluation import decide


class AnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(91)
        cls.cfg=dict(hidden=16,heads=2,graph_layers=1,temporal_layers=1,attention_past_frames=20,dropout=0.,
                     target_scale_mm=100.,diffusion_steps=1000,beta_start=.0001,beta_end=.02,frame_loss_weight=1.)
        cls.body=V1Body(cls.cfg,22,[(i,i+1) for i in range(21)])
        cls.batches=[dict(keys=[(str(b),i) for i in range(4)],features=torch.randn(4,62*14),camera=torch.randn(4,2,9),
                     target=torch.randn(4,62,3)*100,valid=torch.ones(4,62,dtype=torch.bool),split='train') for b in range(6)]

    def test_pelvis_units_mask_nan_and_zero(self):
        x=torch.zeros(1,3,22,3,requires_grad=True)
        gt=torch.zeros_like(x);gt[:,0,0]=torch.tensor([100.,200.,200.]);gt[:,1:,0]=float('nan')
        valid=torch.ones(x.shape[:-1],dtype=torch.bool);valid[:,1,0]=False
        time=torch.tensor([[True,True,False]])
        loss=tr.pelvis_loss(x,gt,valid,time,100)
        self.assertEqual(loss.item(),9.)
        loss.backward();self.assertEqual(x.grad[0,0,0].tolist(),[-2,-4,-4])
        self.assertFalse(x.grad[:,:,1:].any())
        time[:]=False;x.grad=None
        tr.pelvis_loss(x,gt,valid,time,100).backward();self.assertFalse(x.grad.any())
        time[:]=True
        with self.assertRaises(ValueError):tr.pelvis_loss(x,gt,valid,time,100)

    def test_cosines_and_joint_norm_match_single_backward(self):
        b=copy.deepcopy(self.body)
        wp,records=tr.diagnose(b,self.batches,self.cfg)
        i,g=rr.draw(0,6,True);terms,_=tr.terms(b,self.batches[i],self.cfg,g)
        for arm,(wr,wa) in tr.weights(wp).items():
            grad=torch.autograd.grad(terms[0]+wr*terms[1]+wa*terms[2],tuple(b.parameters()),retain_graph=True,allow_unused=True)
            direct=torch.cat([v.flatten() for v in grad if v is not None]).double().norm().item()
            self.assertTrue(math.isclose(records[0]['combined_norm'][arm],direct,rel_tol=2e-6))
        self.assertTrue(all(abs(v)<=1.000001 for row in records for k,v in row.items() if k.startswith('cos_')))

    def test_four_arm_24_step_trace_and_original_parity(self):
        saved={}
        for name,coef in tr.weights(.1).items():
            model=copy.deepcopy(self.body);torch.manual_seed(rr.SEED)
            init=rr.state_hash(model);opt,sch=rr.optimizer(model)
            self.assertFalse(opt.state);self.assertEqual(init,rr.state_hash(self.body))
            traces=[]
            for step in range(24):
                traces.append(tr.update(model,opt,sch,self.batches,self.cfg,step,coef)['trace'])
            saved[name]=rr.state_hash(model)
            if name=='control':reference=traces
            else:self.assertEqual(reference,traces)
        self.assertEqual(len(set(saved.values())),4)
        for name,w in [('control',0.),('rr',tr.RR_WEIGHT)]:
            model=copy.deepcopy(self.body);opt,sch=rr.optimizer(model)
            for step in range(24):rr.update(model,opt,sch,self.batches,self.cfg,step,w)
            self.assertEqual(rr.state_hash(model),saved[name])

    def test_selection_and_gate_boundaries(self):
        doc=json.loads((HERE/'selection.json').read_text())
        self.assertTrue(set(x[0] for x in doc['new']).isdisjoint(x[0] for x in doc['diagnostic']+doc['train']))
        self.assertEqual(len(doc['new']),2)
        control=dict.fromkeys(('rr_jpe_mm','pelvis_mm','body_mpjpe_mm','wrist_mm','hand_global_mpjpe_mm','whole_mpjpe_mm'),100.)
        r=dict(control,pelvis_mm=110.,rr_jpe_mm=99.)
        d=dict(control,pelvis_mm=102.,rr_jpe_mm=99.)
        rows=dict(rr=r,control=control,anchored=d)
        self.assertEqual(decide(rows)[0],'PASS')
        d['pelvis_mm']+=.01;self.assertEqual(decide(rows)[0],'FAIL')
        d['pelvis_mm']=None;self.assertEqual(decide(rows)[0],'INCOMPLETE')
