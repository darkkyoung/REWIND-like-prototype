import copy
import unittest
import numpy as np
import torch
from rewind_m2.training import position_losses
from experiments.final_m2.bone_length_loss_20261117.loss import bone_length_loss,combined_loss
from experiments.final_m2.bone_length_loss_20261117.run_experiment import state_hash,restore_arm,draw
from experiments.final_m2.displacement_loss_20261116.loss import window_metadata


class BoneTests(unittest.TestCase):
    def setUp(self):
        self.p=torch.tensor([[[[0.,0,0],[2.,0,0],[9.,0,0]]]],requires_grad=True)
        self.g=torch.tensor([[[[0.,0,0],[4.,0,0],[9.,0,0]]]])
        self.v=torch.ones((1,1,3),dtype=torch.bool);self.t=torch.ones((1,1),dtype=torch.bool)

    def test_formula_scale_and_gradient(self):
        loss,count=bone_length_loss(self.p,self.g,self.v,self.t,[(0,1)],2)
        self.assertEqual(count,1);self.assertEqual(float(loss.detach()),0.)  # prediction length 2 normalized; GT 4/2
        loss,_=bone_length_loss(self.p,self.g,self.v,self.t,[(0,1)],1)
        self.assertEqual(float(loss.detach()),4.);loss.backward();self.assertTrue(torch.isfinite(self.p.grad).all())

    def test_mask_and_zero(self):
        v=self.v.clone();v[...,1]=False
        loss,count=bone_length_loss(self.p,self.g,v,self.t,[(0,1)],1)
        self.assertEqual(count,0);self.assertEqual(float(loss),0.)
        t=self.t.clone();t[:]=False
        self.assertEqual(bone_length_loss(self.p,self.g,self.v,t,[(0,1)],1)[1],0)
        g=self.g.clone();g[...,1,:]=float('nan')
        self.assertEqual(bone_length_loss(self.p,g,v,self.t,[(0,1)],1)[1],0)

    def test_frame_independence_and_partial_edge(self):
        p=self.p.detach().repeat(1,2,1,1).requires_grad_()
        g=self.g.repeat(1,2,1,1);v=self.v.repeat(1,2,1);t=torch.ones((1,2),dtype=torch.bool)
        t[:,1]=False
        base,n=bone_length_loss(p,g,v,t,[(0,1)],1)
        p2=p.detach().clone();p2[:,1]=1000000
        changed,m=bone_length_loss(p2,g,v,t,[(0,1)],1)
        self.assertEqual((n,m),(1,1));self.assertEqual(float(base),float(changed))
        v[:,:,1]=False
        self.assertEqual(bone_length_loss(p,g,v,t,[(0,1)],1)[1],0)

    def test_position_regression_zero_weight(self):
        f=self.p*0.5
        old,_,_=position_losses(self.p,f,self.g,self.v,self.t,10,1)
        bone,_=bone_length_loss(self.p,self.g,self.v,self.t,[(0,1)],10)
        self.assertIs(combined_loss(old,bone,0),old)
        self.assertEqual(float(combined_loss(old,bone,0)),float(old))
        g1=torch.autograd.grad(old,self.p,retain_graph=True)[0]
        g2=torch.autograd.grad(combined_loss(old,bone,0),self.p)[0]
        self.assertTrue(torch.equal(g1,g2))

    def test_window_gap_boundary_future(self):
        d={'sequence_ids':np.array(['a','a','b']), 'frame_ids':np.array([0,2,0]),'time':np.array([[0,1],[0,1],[0,1]],bool)}
        for k,shape in [('features',(3,2,1)),('target_raw',(3,2,1,3)),('valid',(3,2,1))]:
            d[k]=np.zeros(shape,dtype=bool if k=='valid' else np.float32)
        frames,seqs=window_metadata(d)
        self.assertTrue(torch.equal(frames,torch.tensor([[-1,0],[-1,2],[-1,0]])))
        self.assertNotEqual(int(seqs[0,-1]),int(seqs[2,-1]))
        bad=copy.deepcopy(d);bad['time'][1]=[True,True]
        with self.assertRaises(ValueError):window_metadata(bad)
        bad=copy.deepcopy(d);bad['features'][0,0,0]=1
        with self.assertRaises(ValueError):window_metadata(bad)

    def test_draw_randomness_and_restore(self):
        # Minimal synthetic checkpoint exercises paired model/optimizer/scheduler/RNG restoration.
        from rewind_m2.models import BodyDenoiser
        from rewind_m2.diffusion import DiffusionSchedule
        from rewind_m1.training import rng_state
        cfg={'hidden':16,'heads':4,'graph_layers':1,'temporal_layers':1,'window_size':8,'attention_past_frames':7,'dropout':0,
             'diffusion_steps':10,'beta_start':.0001,'beta_end':.02,'learning_rate':.001,'scheduler_step':200,'scheduler_gamma':.5,
             'target_scale_raw':100,'frame_loss_weight':1}
        edges=[(i,i+1) for i in range(21)];torch.manual_seed(123)
        model=BodyDenoiser(cfg,22,edges);opt=torch.optim.Adam(model.parameters(),lr=.001)
        scheduler=torch.optim.lr_scheduler.StepLR(opt,200,.5)
        # Synthetic parent with populated Adam moments; step labels model a 400-update checkpoint.
        sum(p.square().sum() for p in model.parameters()).backward();opt.step()
        for state in opt.state.values():state['step'].fill_(400)
        ck={'config':cfg,'manifest':{'edges':edges},'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),
            'scheduler':copy.deepcopy(scheduler.state_dict()),'rng':copy.deepcopy(rng_state()),'scaler':{}}
        scheduler.last_epoch=400;ck['scheduler']=copy.deepcopy(scheduler.state_dict())
        data={'frame_ids':np.arange(4),'features':np.zeros((4,8,308),np.float32),
              'target_raw':np.zeros((4,8,22,3),np.float32),'valid':np.ones((4,8,22),bool),
              'time':np.ones((4,8),bool)}
        frames=torch.tensor([list(range(i-7,i+1)) for i in range(4)]);sequences=torch.zeros_like(frames)
        parent_hash=state_hash(ck)
        first=[]
        for _ in range(2):
            restored,optimizer,stepper=restore_arm(ck,'cpu')
            initial=[state_hash(restored.state_dict()),state_hash(optimizer.state_dict()),state_hash(stepper.state_dict()),state_hash(rng_state())]
            losses=draw(restored,ck,DiffusionSchedule(10,.0001,.02),data,frames,sequences,'cpu')
            first.append((initial,losses[-1],float(losses[0].detach())))
        self.assertEqual(first[0],first[1])
        bad=copy.deepcopy(ck);bad['optimizer']['state']={}
        with self.assertRaisesRegex(ValueError,'incomplete_adam_state'):restore_arm(bad,'cpu')
        # Multi-update diverging objectives must retain identical stochastic draws.
        traces=[]
        for weight in (0.,.1):
            restored,optimizer,stepper=restore_arm(ck,'cpu');trace=[]
            for _ in range(3):
                existing,_,_,bone,_,row=draw(restored,ck,DiffusionSchedule(10,.0001,.02),data,frames,sequences,'cpu')
                optimizer.zero_grad();combined_loss(existing,bone,weight).backward();optimizer.step();stepper.step();trace.append(row)
            traces.append(trace)
        self.assertEqual(traces[0],traces[1]);self.assertEqual(parent_hash,state_hash(ck))
        from experiments.final_m2.bone_length_loss_20261117.run_experiment import calibrate
        a=calibrate(ck,data,DiffusionSchedule(10,.0001,.02),frames,sequences,'cpu',{'calibration_batches':2,'calibration_fraction':.1})
        b=calibrate(ck,data,DiffusionSchedule(10,.0001,.02),frames,sequences,'cpu',{'calibration_batches':2,'calibration_fraction':.1})
        self.assertEqual(a,b);self.assertEqual(parent_hash,state_hash(ck))
        from rewind_m1.training import restore_rng
        restore_rng(copy.deepcopy(ck['rng']));a=torch.randint(100,(8,));n=torch.randn(8,2,22,3)
        restore_rng(copy.deepcopy(ck['rng']));b=torch.randint(100,(8,));m=torch.randn(8,2,22,3)
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(n,m))

    def test_parent_schema_matches_actual_writer(self):
        from experiments.final_m2.bone_length_loss_20261117.run_experiment import HERE,load,validate_parent
        manifest=load(HERE.parent/'dataset_scaling_300/data_selection_manifest.json')
        ck={'format_version':'m2_dataset_scaling_v1','arm':'train300','step':400,'samples_seen':3200,
            'config':load(HERE.parent/'dataset_scaling_300/config.json')['model'],'manifest':manifest,'scaler':{},
            'experiment_identity':load(HERE.parent/'dataset_scaling_300/experiment_identity.json')}
        validate_parent(ck,manifest)  # No `source` key in real scaling checkpoints.
        bad=copy.deepcopy(ck);bad['experiment_identity']['code_sha256']['rewind_m2/models.py']='bad'
        with self.assertRaises(ValueError):validate_parent(bad,manifest)

    def test_bone_metric_masks_and_gate(self):
        from experiments.final_m2.bone_length_loss_20261117.run_experiment import bone_metrics,quality_checks
        z={'prediction':np.array([[[0.,0,0],[2.,0,0],[20.,0,0]]]),
           'gt':np.array([[[0.,0,0],[4.,0,0],[0.,0,0]]]),'valid':np.array([[True,True,False]])}
        result=bone_metrics(z,[(0,1),(0,2)])
        self.assertEqual(result['mean_abs_bone_ratio_deviation_pp'],50.)
        c={'split':'s','model':'control',**result,'global_joint_error_raw':10.,'root_relative_error_22_raw':10.,
           'pelvis_position_error_raw':10.,'motion_magnitude_ratio_pct':90.,'joint_spread_ratio_pct':110.}
        i={**c,'model':'improved','mean_abs_bone_ratio_deviation_pp':40.}
        self.assertTrue(quality_checks([c,i])['all_pass'])
        i['pelvis_position_error_raw']=10.3
        self.assertFalse(quality_checks([c,i])['all_pass'])

    def test_evaluation_boundary_and_motion_rejection(self):
        from experiments.final_m2.bone_length_loss_20261117.run_experiment import validate_evaluation
        def sample(seq,value):
            d={'sequence_ids':np.array([seq,seq]),'frame_ids':np.array([0,1]),
               'features':np.zeros((2,8,308),np.float32),'target_raw':np.zeros((2,8,22,3),np.float32),
               'valid':np.zeros((2,8,22),bool),'time':np.zeros((2,8),bool)}
            for i in range(2):
                d['time'][i,-i-1:]=True
                for key in ('features','target_raw'):d[key][i,-i-1:]=value
                d['valid'][i,-i-1:]=True
            return d
        data={name:sample(seq,i+1) for i,(name,seq) in enumerate([
            ('train300','train/Walk__1_'),('opening130','val/Open__1_'),
            ('gunplay47','val/Gun__1_'),('unused_motion_preselected','val/Jump__1_')])}
        selection={'sequence_id':'val/Jump__1_','selection_reason':'preselected validation motion'}
        validate_evaluation(data,selection)
        bad=copy.deepcopy(data);bad['unused_motion_preselected']['sequence_ids'][:]='val/Open__1_'
        with self.assertRaisesRegex(ValueError,'unused_motion_overlap'):
            validate_evaluation(bad,{**selection,'sequence_id':'val/Open__1_'})
        bad=copy.deepcopy(data);bad['unused_motion_preselected']['features'][:]=data['train300']['features']
        with self.assertRaisesRegex(ValueError,'unused_exact_pose_overlap'):validate_evaluation(bad,selection)
        bad=copy.deepcopy(data);bad['unused_motion_preselected']['features'][1,-2,0]=999
        with self.assertRaisesRegex(ValueError,'window_frame_content_mismatch'):validate_evaluation(bad,selection)
        bad=copy.deepcopy(data);bad['unused_motion_preselected']['features']=bad['unused_motion_preselected']['features'].astype(np.float64)
        with self.assertRaisesRegex(ValueError,'tensor_contract'):validate_evaluation(bad,selection)

    def test_zero_length_and_nan_mask_backward(self):
        p=torch.zeros_like(self.p,requires_grad=True);g=self.g.clone();g[...,2,:]=float('nan')
        v=self.v.clone();v[...,2]=False
        loss,n=bone_length_loss(p,g,v,self.t,[(0,1),(1,2)],1)
        loss.backward();self.assertEqual(n,1);self.assertTrue(torch.isfinite(p.grad).all())

if __name__=='__main__':unittest.main()
