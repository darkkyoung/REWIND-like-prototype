import numpy as np


def train_lengths(target, valid, edges):
    return np.array([np.median(np.linalg.norm((target[:,b]-target[:,a])[valid[:,a]&valid[:,b]],axis=-1))
                     for a,b in edges])


class Refiner:
    def __init__(self, edges, lengths):
        self.edges = np.asarray(edges)
        self.lengths = np.asarray(lengths)
        self.last = None
        self.history = []
        self.offset = np.zeros_like(self.lengths)
        if not np.all(np.isfinite(lengths)) or not np.all(self.lengths>0):
            raise ValueError('invalid train lengths')

    def push(self, seq, frame, pose, obs):
        if self.last != (seq,frame-1):
            self.history.clear()
            self.offset.fill(0)
        self.last = (seq,frame)
        views = obs.reshape(22,2,7)
        xy,valid = views[...,:2],views[...,2]>0
        uncertainty = 1-valid.mean(1)
        if len(self.history)==2:
            (old,m0),(prev,m1) = self.history
            common = valid&m0&m1
            residual = np.linalg.norm(xy-2*prev+old,axis=-1)*32
            innovation = (residual*common).sum(1)/np.maximum(common.sum(1),1)
            uncertainty = np.maximum(uncertainty,np.clip(innovation,0,1))
        self.history = (self.history+[(xy.copy(),valid.copy())])[-2:]
        a,b = self.edges.T
        vectors = pose[b]-pose[a]
        lengths = np.linalg.norm(vectors,axis=1)
        err = self.lengths-lengths
        gain = .35+.45*np.maximum(uncertainty[a],uncertainty[b])
        desired = gain*np.sign(err)*np.maximum(abs(err)-.05*self.lengths,0)
        cap = .12*self.lengths
        self.offset = np.clip(.75*np.clip(desired,-cap,cap)+.25*self.offset,-cap,cap)
        # Preserve directions and avoid flipping a short edge after a rapid length change.
        scale = 1+np.maximum(self.offset,-.9*lengths)/np.maximum(lengths,1e-9)
        out = pose.copy()
        for (parent,child),vec in zip(self.edges,vectors*scale[:,None]):
            out[child] = out[parent]+vec
        return out
