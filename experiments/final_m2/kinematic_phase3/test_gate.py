import unittest
import numpy as np
from .gate import Gated


class GateTests(unittest.TestCase):
    def setUp(self):
        self.edges=np.array([[j,j+1] for j in range(21)])
        self.obs=np.zeros((22,2,7))
        self.obs[:,:,2]=1
        self.pose=np.zeros((22,3))
        self.pose[:,0]=np.arange(22)
        self.pose[10:,0]+=1

    def model(self,mode='gated'):
        return Gated(self.edges,np.ones(21),np.ones(21)*.01,mode=mode)

    def test_coherent_fast_noop(self):
        m=self.model()
        for f in range(6):
            obs=self.obs.copy();obs[...,:2]+=f*40
            pose=self.pose+f*50
            np.testing.assert_array_equal(m.push('s',f,pose,obs)[0],pose)

    def test_subtree_and_release(self):
        m=self.model();bad=self.obs.copy();bad[10,:,2]=0
        out,_,offset=m.push('a',0,self.pose,bad)
        self.assertEqual(np.flatnonzero(offset).tolist(),[9])
        np.testing.assert_array_equal(out[:10],self.pose[:10])
        np.testing.assert_allclose(np.diff(out[10:],axis=0),np.diff(self.pose[10:],axis=0))
        self.assertTrue(out[10,0]<self.pose[10,0])
        self.assertLessEqual(abs(offset[9]),.04)
        self.assertTrue(np.allclose(m.push('a',1,self.pose,self.obs)[0],self.pose))

    def test_future_gap_sequence_and_gt(self):
        def replay(suffix):
            m=self.model('on')
            return [m.push('a',f,self.pose*scale,self.obs)[0] for f,scale in enumerate([1,1,1,suffix])]
        np.testing.assert_array_equal(replay(2)[:3],replay(7)[:3])
        m=self.model('on');first=m.push('a',0,self.pose,self.obs)[0]
        for f in range(1,5):m.push('a',f,self.pose,self.obs)
        np.testing.assert_array_equal(first,m.push('a',9,self.pose,self.obs)[0])
        np.testing.assert_array_equal(first,m.push('b',10,self.pose,self.obs)[0])
        with self.assertRaises(TypeError):m.push('s',0,self.pose,self.obs,gt=self.pose)

    def test_prior_only_after_reliable_history(self):
        m=self.model()
        for f in range(2):m.push('a',f,self.pose,self.obs)
        np.testing.assert_array_equal(m.estimate,np.ones(21))
        m.push('a',2,self.pose,self.obs)
        self.assertGreater(m.estimate[9],1)
        previous=m.estimate.copy()
        missing=self.obs.copy();missing[:,:,2]=0
        m.push('a',3,self.pose,missing)
        np.testing.assert_array_equal(m.estimate,previous)

    def test_off_always_exact(self):
        m=self.model('off')
        for f in range(5):
            pose=self.pose*(f+1)
            np.testing.assert_array_equal(m.push('a',f,pose,self.obs*0)[0],pose)
