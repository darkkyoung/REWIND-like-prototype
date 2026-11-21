import itertools
import unittest
import numpy as np
from .oracle import paths
from .refine import Refiner, train_lengths


class Phase2Tests(unittest.TestCase):
    def setUp(self):
        self.edges=np.array([[j,j+1] for j in range(21)])
        self.pose=np.zeros((22,3));self.pose[:,0]=np.arange(22)*2
        self.x=np.zeros((22,2,7));self.x[...,2]=1

    def make(self):
        return Refiner(self.edges,np.ones(21))

    def test_direction_root_subtree(self):
        y=self.make().push('a',0,self.pose,self.x)
        np.testing.assert_array_equal(y[0],self.pose[0])
        np.testing.assert_array_equal(y[:,1:],0)
        self.assertTrue(np.all((np.diff(y[:,0])>=1.88)&(np.diff(y[:,0])<2)))
        self.assertGreater(self.pose[-1,0]-y[-1,0],self.pose[1,0]-y[1,0])

    def test_fast_translation(self):
        r=self.make();p=self.pose*.5
        for j in range(8):
            y=r.push('a',j,p+100*j,self.x)
            np.testing.assert_array_equal(y,p+100*j)

    def test_prefix_reset_and_no_gt(self):
        r=self.make();a=r.push('a',0,self.pose,self.x)
        for j in range(1,5):r.push('a',j,self.pose*j,self.x)
        np.testing.assert_array_equal(a,r.push('a',7,self.pose,self.x))
        np.testing.assert_array_equal(a,r.push('b',8,self.pose,self.x))
        with self.assertRaises(TypeError):r.push('a',8,self.pose,self.x,gt=self.pose)

    def test_unavailable_joint_local_strength(self):
        clean=self.make().push('a',0,self.pose*.62,self.x)
        self.x[20:,:,2]=0
        missing=self.make().push('a',0,self.pose*.62,self.x)
        np.testing.assert_array_equal(clean[:20],missing[:20])
        self.assertLess(missing[-1,0],clean[-1,0])

    def test_masked_train_lengths(self):
        t=np.stack([self.pose,self.pose*20])
        v=np.ones((2,22),bool);v[1]=False
        np.testing.assert_array_equal(train_lengths(t,v,self.edges),2)

    def test_dp_against_exhaustive_search(self):
        rng=np.random.default_rng(9)
        models=rng.normal(size=(2,6,3,3));gt=rng.normal(size=(6,3,3))
        result=paths(models,gt,np.ones((6,3),bool))
        def objective(path):
            residual=models[path,np.arange(6)]-gt
            return np.linalg.norm(residual,axis=-1).mean(1).sum()+np.linalg.norm(np.diff(residual,axis=0),axis=-1).mean(1).sum()+.25*np.count_nonzero(np.diff(path))
        np.testing.assert_allclose(objective(result['dp_oracle']),min(objective(np.array(p)) for p in itertools.product([0,1],repeat=6)))
