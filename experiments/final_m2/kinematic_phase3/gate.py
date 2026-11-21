from collections import deque
import numpy as np
from experiments.final_m2.kinematic_phase2.refine import train_lengths


def observation(obs, edges):
    views=obs.reshape(22,2,7)
    a,b=edges.T
    return views[b,:,:2]-views[a,:,:2], (views[a,:,2]>0)&(views[b,:,2]>0)


def calibrate(train, edges):
    lengths=train_lengths(train['target_raw'][:,-1],train['valid'][:,-1],edges)
    values=[[] for _ in edges]
    history=deque(maxlen=2)
    last=None
    for seq,frame,obs in zip(train['sequence_ids'],train['frame_ids'],train['features'][:,-1]):
        if last!=(seq,frame-1):history.clear()
        xy,valid=observation(obs,edges)
        if len(history)==2:
            (old,v0),(prev,v1)=history
            residual=np.linalg.norm(xy-2*prev+old,axis=-1)
            for samples,row,mask in zip(values,residual,valid&v0&v1):
                samples.extend(row[mask])
        history.append((xy,valid));last=(seq,frame)
    return lengths,np.array([max(1/1024,np.quantile(x,.95)) for x in values])


class Gated:
    def __init__(self,edges,lengths,scale,prior='sequence',mode='gated'):
        self.edges=edges
        self.lengths=lengths
        self.scale=scale
        self.prior=prior
        self.mode=mode
        self.neighbors=[np.flatnonzero(np.any(np.isin(edges,e),axis=1)&(np.arange(len(edges))!=j)) for j,e in enumerate(edges)]
        self.last=None

    def push(self,seq,frame,pose,obs):
        if self.last!=(seq,frame-1):
            self.obs=deque(maxlen=2)
            self.recent=deque(maxlen=8)
            self.estimate=self.lengths.copy()
            self.offset=np.zeros(len(self.edges))
        self.last=(seq,frame)
        a,b=self.edges.T
        vec=pose[b]-pose[a]
        length=np.linalg.norm(vec,axis=1)
        xy,valid=observation(obs,self.edges)
        available=valid.all(axis=1)
        reliable=np.zeros(len(a),bool)
        innovation=np.zeros(len(a))
        if len(self.obs)==2:
            (old,v0),(prev,v1)=self.obs
            common=valid&v0&v1
            residual=np.linalg.norm(xy-2*prev+old,axis=-1)
            innovation=(residual*common).sum(1)/np.maximum(common.sum(1),1)/self.scale
            reliable=common.all(1)&(innovation<=1)
        target=self.lengths
        if self.prior=='sequence':target=self.estimate
        if self.prior=='temporal' and self.recent:target=np.median(self.recent,axis=0)
        target=np.clip(target,.8*self.lengths,1.2*self.lengths)
        ratio=length/target
        neighbor=np.array([np.median(ratio[idx]) for idx in self.neighbors])
        structural=np.minimum(abs(ratio-1),abs(ratio-neighbor))/.05
        score=np.minimum(structural,np.maximum(innovation,2*(~available)))
        selected=score>1
        if self.mode=='on':selected[:]=True
        if self.mode=='off':selected[:]=False
        desired=.35*np.sign(1-ratio)*np.maximum(abs(1-ratio)-.05,0)*target
        desired=np.where(selected,np.clip(desired,-.12*target,.12*target),0)
        self.offset+=np.clip(desired-self.offset,-.04*target,.04*target)
        self.offset=np.where(abs(self.offset)<1e-9,0,self.offset)
        # Translate the subtree along the original edge; inactive sister branches keep their local vectors.
        shifts=np.zeros_like(pose)
        delta=np.maximum(self.offset,-.9*length)/np.maximum(length,1e-9)
        for j,(parent,child) in enumerate(self.edges):shifts[child]=shifts[parent]+delta[j]*vec[j]
        self.estimate=np.where(reliable,.9*self.estimate+.1*np.clip(length,.8*self.lengths,1.2*self.lengths),self.estimate)
        self.obs.append((xy,valid));self.recent.append(length)
        return pose+shifts,score,self.offset.copy()
