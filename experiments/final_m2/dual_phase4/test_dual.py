import unittest
import numpy as np
from .dual import Dual


class DualTests(unittest.TestCase):
    def setUp(self):
        self.edges=np.array([(j,j+1) for j in range(21)])
        self.prior=np.ones(21)
        self.pose=np.zeros((22,3))
        self.pose[:,0]=np.arange(22)
        self.obs=np.zeros((22,2,7))
        self.obs[:,:,2]=1

    def model(self,mode='combined'):
        return Dual(self.edges,self.prior,self.prior*.01,mode)

    def test_noop_fast(self):
        model=self.model()
        for t in range(8):
            pose=self.pose+30*t
            np.testing.assert_array_equal(model.push('s',t,pose,self.obs)[0],pose)

    def test_persistence_subtree(self):
        model=self.model('A');pose=self.pose.copy();pose[19:,0]+=1
        for t in range(4):
            np.testing.assert_array_equal(model.push('s',t,pose,self.obs)[0],pose)
        out,info=model.push('s',4,pose,self.obs)
        self.assertEqual(np.flatnonzero(info[0]).tolist(),[18])
        np.testing.assert_array_equal(out[:19],pose[:19])
        np.testing.assert_allclose(np.diff(out[19:],axis=0),np.diff(pose[19:],axis=0))
        self.assertLess(out[19,0],pose[19,0])

    def test_transient_requires_both_signals(self):
        def predict(missing,jerk):
            m=self.model('B')
            for t in range(2):m.push('s',t,self.pose,self.obs)
            x=self.obs.copy();p=self.pose.copy()
            if missing:x[-1,:,2]=0
            if jerk:p[-1,2]+=.25
            return m.push('s',2,p,x)[1][1][-1]
        self.assertFalse(predict(True,False))
        self.assertFalse(predict(False,True))
        self.assertTrue(predict(True,True))

    def test_conflict_veto(self):
        model=self.model();p=self.pose.copy();p[-1,0]+=1
        for t in range(4):model.push('s',t,p,self.obs)
        p[-1,0]-=.3
        missing=self.obs.copy();missing[-1,:,2]=0
        result,(_,_,conflict,_,_)=model.push('s',4,p,missing)
        self.assertTrue(conflict[-1])
        np.testing.assert_array_equal(result,p)

    def test_gap_sequence_future_and_gt(self):
        p=self.pose.copy();p[-1,0]+=1
        def replay(suffix):
            m=self.model();out=[m.push('a',j,p,self.obs)[0] for j in range(6)]
            m.push('a',6,p*suffix,self.obs)
            return m,out
        model,prefix=replay(7)
        np.testing.assert_array_equal(prefix,replay(2)[1])
        np.testing.assert_array_equal(model.push('a',10,p,self.obs)[0],p)
        for j in range(11,17):model.push('a',j,p,self.obs)
        np.testing.assert_array_equal(model.push('b',17,p,self.obs)[0],p)
        with self.assertRaises(TypeError):model.push('s',0,p,self.obs,gt=p)
