import numpy as np
from collections import deque
from ..kinematic_phase3.gate import observation


def limit(vector, cap):
    norm = np.linalg.norm(vector, axis=-1)
    return vector * np.minimum(1, cap / np.maximum(norm, 1e-10))[:, None]


class Dual:
    def __init__(self, edges, lengths, scale, mode='combined'):
        self.edges, self.lengths, self.scale = edges, lengths, scale
        if mode not in ('A','B','combined'):
            raise ValueError(mode)
        self.mode, self.last = mode, None
        counts = np.ones(len(edges)+1, int)
        for parent,child in edges[::-1]:
            counts[parent] += counts[child]
        self.allowed = counts[edges[:,1]] < 4

    def push(self, sequence, frame, pose, features):
        if self.last != (sequence,frame-1):
            self.length_history = deque(maxlen=5)
            self.history = deque(maxlen=2)
            self.offset = np.zeros((len(self.edges),3))
        self.last = (sequence,frame)
        a,b = self.edges.T
        vec = pose[b]-pose[a]
        length = np.linalg.norm(vec,axis=1)
        obs,valid = observation(features,self.edges)
        self.length_history.append(length/self.lengths-1)
        pa = np.zeros_like(vec)
        if len(self.length_history)==5:
            ratios = np.stack(self.length_history)
            bias = np.maximum(ratios.min(0)-.10,0)+np.minimum(ratios.max(0)+.10,0)
            shift = np.clip(-.35*bias,-.10,.10)*self.lengths*self.allowed
            pa = vec*(shift/np.maximum(length,1e-8))[:,None]
        pb = np.zeros_like(vec)
        if len(self.history)==2:
            (v0,x0,m0),(v1,x1,m1) = self.history
            innovation = np.linalg.norm(obs-2*x1+x0,axis=-1)
            suspect = (~valid.all(1)) | np.any((innovation>self.scale[:,None]) & valid & m0 & m1,axis=1)
            residual = 2*v1-v0-vec
            suspect &= np.linalg.norm(residual,axis=1)>.05*self.lengths
            pb = limit(.5*residual,.08*self.lengths)*suspect[:,None]
        self.history.append((vec.copy(),obs.copy(),valid.copy()))
        has_a = np.linalg.norm(pa,axis=1)>1e-7
        has_b = np.linalg.norm(pb,axis=1)>1e-7
        conflict = has_a & has_b & ((pa*pb).sum(1)<0)
        if self.mode=='combined':
            target=np.where(has_b[:,None],pb,pa)
            target[conflict]=0
        else:
            target=pa if self.mode=='A' else pb
        self.offset += limit(target-self.offset,.04*self.lengths)
        self.offset[np.linalg.norm(self.offset,axis=1)<1e-7]=0
        shifts=np.zeros_like(pose)
        for j,(parent,child) in enumerate(self.edges):
            shifts[child]=shifts[parent]+self.offset[j]
        return pose+shifts, (has_a,has_b,conflict,self.offset.copy(),shifts)
