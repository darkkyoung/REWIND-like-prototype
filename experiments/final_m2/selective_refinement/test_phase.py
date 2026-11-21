import unittest
import numpy as np
from .policy import Signals, corrupt, fit, route, scores
from .runner import ranking, signals

class PhaseTests(unittest.TestCase):
    def setUp(self):
        self.x = np.zeros((40,22,2,7),np.float32)
        self.x[...,2] = 1
        self.x[...,0] = np.arange(40)[:,None,None]*.125
        self.x = self.x.reshape(40,-1)
        self.p = np.arange(40)[:,None,None]+np.arange(66).reshape(1,22,3)
        self.ids = np.arange(40)
        self.seq = ['a']*40
        self.edges = [(0,1),(1,2)]

    def extract(self,x,p):
        return signals(self.edges,x,p,self.seq,self.ids)

    def test_constant_velocity(self):
        f = self.extract(self.x,self.p)
        self.assertTrue((f[2:,0]>0).all())
        np.testing.assert_array_equal(f[2:,1:],0)

    def test_future_and_reset(self):
        f = self.extract(self.x,self.p)
        self.x[15:] = 777
        self.p[15:] = 888
        np.testing.assert_array_equal(f[:15],self.extract(self.x,self.p)[:15])
        for seq,idx in [('a',19),('b',2)]:
            h=Signals(self.edges)
            for i in (0,1): h.push('a',i,self.x[i],self.p[i])
            np.testing.assert_array_equal(h.push(seq,idx,self.x[2],self.p[2]),0)

    def test_corruptions(self):
        original=self.x.copy()
        for mode in ('jitter','spike','short_missing','long_missing'):
            a,y=corrupt(self.x,mode)
            b,z=corrupt(self.x,mode)
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(y,z)
            self.assertTrue(y.any())
            self.assertTrue((~y).any())
        np.testing.assert_array_equal(original,self.x)

    def test_routing_and_gt_isolation(self):
        obs,label=corrupt(self.x,'long_missing')
        train=self.extract(self.x,self.p)
        scale,cut=fit(train)
        _,score=scores(self.extract(obs,self.p),scale)
        selected=score>cut[1]
        self.assertTrue(selected[label].all())
        alt=self.p+100
        result=route(self.p,alt,selected)
        np.testing.assert_array_equal(result[selected],alt[selected])
        np.testing.assert_array_equal(result[~selected],self.p[~selected])
        with self.assertRaises(TypeError):
            Signals(self.edges).push('a',0,self.x[0],self.p[0],gt=self.p[0])

    def test_ranking_ties(self):
        y=np.array([False,True,False,True])
        self.assertEqual(ranking(y,np.ones(4)),(.5,.5))
        self.assertEqual(ranking(y,y.astype(float)),(1.,1.))
        self.assertEqual(ranking(~np.ones(4,bool),y),(None,None))
