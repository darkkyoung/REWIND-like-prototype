import json
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import torch

from rewind_v1.camera_training import CameraOrigin
from rewind_v1.data import Clip
from rewind_v1.frontend import PaperHeatmapFrontend, predicted_features
from scripts.v2_gpu_training_pipeline import CommonEvaluation
from v2_gpu128_a_bundle_predict import predict_clip
from .adapter import DualCamera, PINS, predict


def extrinsics(n):
    angles = torch.arange(n*2,dtype=torch.float64).reshape(n,2)*0.15
    r = torch.eye(3,dtype=torch.float64).expand(n,2,3,3).clone()
    r[...,0,0] = r[...,1,1] = angles.cos()
    r[...,1,0] = angles.sin()
    r[...,0,1] = -angles.sin()
    t = torch.arange(n*6,dtype=torch.float64).reshape(n,2,3)*231.1+100035.
    return r,t


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.r,self.t = extrinsics(4)
        self.enc = DualCamera()

    def test_both_official_encoders(self):
        out = self.enc.encode(self.r,self.t,"s",range(4))
        for mode in ("absolute","first_midpoint"):
            expected,_ = CameraOrigin(mode).apply(self.r,self.t,"s",0,list(range(4)))
            self.assertTrue(torch.equal(getattr(out,mode),expected))

    def test_translation_and_rotation_channels(self):
        out = self.enc.encode(self.r,self.t,"s",range(4))
        anchor = -(self.r[0].mT @ self.t[0,...,None]).squeeze(-1).mean(0)
        center = self.t+(self.r@anchor)
        self.assertTrue(torch.equal(out.absolute[...,6:],(self.t/100).float()))
        self.assertTrue(torch.equal(out.first_midpoint[...,6:],(center/100).float()))
        self.assertTrue(torch.equal(out.first_midpoint[...,:6],out.absolute[...,:6]))

    def test_chunk_and_prefix(self):
        a = self.enc.encode(self.r[:2],self.t[:2],"s",[0,1])
        b = self.enc.encode(self.r[2:],self.t[2:],"s",[2,3])
        full = DualCamera().encode(self.r,self.t,"s",range(4))
        for k in ("absolute","first_midpoint"):
            self.assertTrue(torch.equal(torch.cat((getattr(a,k),getattr(b,k))),getattr(full,k)))

    def test_future_perturbation(self):
        before = self.enc.encode(self.r,self.t,"s",range(4))
        self.t[2:] += 766.
        after = DualCamera().encode(self.r,self.t,"s",range(4))
        self.assertTrue(torch.equal(before.first_midpoint[:2],after.first_midpoint[:2]))

    def test_gap_sequence_and_explicit_clip_reset(self):
        for seq,f,new in (("s",12,False),("other",2,False),("s",2,True)):
            e = DualCamera()
            e.encode(self.r[:2],self.t[:2],"s",[0,1])
            actual = e.encode(self.r[2:],self.t[2:],seq,[f,f+1],new_clip=new)
            fresh = DualCamera().encode(self.r[2:],self.t[2:],seq,[f,f+1])
            self.assertTrue(torch.equal(actual.first_midpoint,fresh.first_midpoint))

    def test_reject_bad_camera_transactionally(self):
        self.enc.encode(self.r[:1],self.t[:1],"s",[0])
        for r,t,frames in ((self.r.float(),self.t,range(4)),(self.r,self.t,[2,3,5,6]),
                            (self.r,self.t*float("nan"),range(4)),(-self.r,self.t,range(4))):
            with self.assertRaises(ValueError):
                self.enc.encode(r,t,"s",frames)
            self.assertEqual(self.enc.state["next_frame"],1)


class CascadeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(307)
        torch.set_num_threads(1)
        import rewind_v1
        spec = json.loads((Path(rewind_v1.__file__).parents[1]/"reference/spec.json").read_text())
        cls.cfg = {"teacher":PINS["root_binding"]["cfg"]}
        cls.images = torch.rand(4,2,3,128,128)
        cls.front = NS(model=PaperHeatmapFrontend(62).eval().requires_grad_(False))
        cls.features = torch.cat([predicted_features(cls.front.model,b) for b in cls.images.split(2)])
        cls.cam = DualCamera().encode(*extrinsics(4),"s",range(4))
        cls.valid = torch.ones(1,4,dtype=torch.bool)
        cls.id = "c"*64
        cls.clip = Clip(cls.features[None],cls.cam.first_midpoint[None],torch.randn(1,4,62,3),
                        torch.ones(1,4,62,dtype=torch.bool),cls.valid,torch.arange(4)[None]/25,
                        "predicted_2d",split="validation",input_sha256=cls.id)
        def frame(key,split):
            return cls.images[key[1]],{"xy":torch.zeros(2,62,2),"xy_valid":torch.ones(2,62,dtype=torch.bool)}
        data = NS(spec=spec,width=4,cache=NS(frame=frame))
        cls.pipe = NS(data=data,device="cpu",frontend_size=[128,128])
        cls.key = ("s",0)
        cls.st = CommonEvaluation(data,{"train":{cls.key:cls.clip},"validation":{cls.key:cls.clip}},cls.cfg,"cpu",0)
        cls.body = cls.st.make("body")
        cls.hand = cls.st.make("hand",PINS["selected"]["body"]["sha256"])
        for role,stage in (("body",cls.body),("hand",cls.hand)):
            stage.model.eval().requires_grad_(False)
            stage.checkpoint_sha256 = PINS["selected"][role]["sha256"]
        cls.noises = cls.st._noise(cls.key,22,100),cls.st._noise(cls.key,40,200)

    def baseline(self):
        return predict("baseline",self.st,self.body,self.hand,self.features[None],self.cam,self.valid,self.id,self.noises)

    def test_official_predict_parity_fixture(self):
        _,ref,_ = predict_clip(self.pipe,Mock(),self.front,self.st,self.body,self.hand,self.key,self.clip)
        self.assertTrue(np.array_equal(self.baseline()["whole"].numpy(),ref["predicted_3d_mm"]))
        xy = torch.from_numpy(ref["predicted_2d_px"])
        self.assertTrue(torch.equal(2*xy/127-1,self.features.reshape(4,62,2,7).transpose(1,2)[...,:2]))

    def test_candidate_routes_selected_upper_and_shared_noise(self):
        # Real V1 classes and DDIM10, with fixture weights and explicit fixture identities.
        candidate = self.st.make("body")
        candidate.model.load_state_dict(self.body.model.state_dict(),strict=True)
        candidate.model.eval().requires_grad_(False)
        candidate.checkpoint_sha256 = PINS["anchored"]
        hand = self.st.make("hand",PINS["anchored"])
        hand.model.load_state_dict(self.hand.model.state_dict(),strict=True)
        hand.model.eval().requires_grad_(False)
        hand.checkpoint_sha256 = self.hand.checkpoint_sha256
        cb,ch = Mock(wraps=candidate),Mock(wraps=hand)
        cb.checkpoint_sha256,ch.checkpoint_sha256 = candidate.checkpoint_sha256, hand.checkpoint_sha256
        res = predict("anchored",self.st,cb,ch,self.features[None],self.cam,self.valid,self.id,self.noises)
        obs,bnoise = cb.predict.call_args.args
        hobs,hnoise = ch.predict.call_args.args
        self.assertFalse(hasattr(obs,"target_mm"))
        self.assertFalse(hasattr(hobs,"target_valid"))
        self.assertTrue(torch.equal(obs.camera[0],self.cam.absolute))
        self.assertTrue(torch.equal(hobs.camera[0],self.cam.first_midpoint))
        self.assertIs(bnoise,self.noises[0])
        self.assertIs(hnoise,self.noises[1])
        self.assertTrue(torch.equal(ch.predict.call_args.kwargs["upper"].values_mm[0],res["body"][:,self.st.upper_idx]))
        self.assertFalse(torch.equal(self.baseline()["body"],res["body"]))
        wrists = [self.st.body_names.index(self.st.spec["wrists"][side]) for side in ("left","right")]
        self.assertTrue(torch.equal(res["hand_global"],res["hand_local"]+res["body"][:,np.repeat(wrists,20)]))

    def test_full_size_frozen_weights_and_noise_unchanged(self):
        from .adapter import digest
        def fingerprint():
            return {name:digest(v) for stage_name,stage in (("body",self.body),("hand",self.hand))
                    for k,v in stage.model.state_dict().items() for name in [stage_name+"."+k]}
        self.assertEqual(len(self.body.model.state_dict()),104)
        before = fingerprint()
        noise = list(map(digest,self.noises))
        self.baseline()
        self.assertEqual(before,fingerprint())
        self.assertEqual(noise,list(map(digest,self.noises)))

    def test_future_observations_leave_prefix_unchanged(self):
        from .adapter import Cameras
        changed = self.features.clone()
        changed[2:] += 0.5
        cams = Cameras(self.cam.first_midpoint.clone(),self.cam.absolute.clone(),self.cam.anchor_world_mm)
        cams.first_midpoint[2:,...,6:] += 1
        future = predict("baseline",self.st,self.body,self.hand,changed[None],cams,self.valid,self.id,self.noises)
        self.assertTrue(torch.equal(future["whole"][:2],self.baseline()["whole"][:2]))

    def test_bad_identity_shape_and_mask(self):
        for valid,features,ident in ((self.valid,self.features[None],"x"),(self.valid.float(),self.features[None],self.id),
                                     (self.valid,self.features[None]*float("nan"),self.id)):
            with self.assertRaises(ValueError):
                predict("baseline",self.st,self.body,self.hand,features,self.cam,valid,ident,self.noises)


if __name__ == "__main__":
    unittest.main()
