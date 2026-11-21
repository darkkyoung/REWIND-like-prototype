import numpy as np


def errors(pred, gt, valid):
    return np.array([np.linalg.norm(row[mask],axis=-1).mean() for row,mask in zip(pred-gt,valid)])


def paths(models, gt, valid):
    costs=np.array([errors(p,gt,valid) for p in models])
    rr=np.array([errors(p-p[:,:1],gt-gt[:,:1],valid & valid[:,:1]) for p in models])
    value=costs[:,0].copy()
    back=np.zeros((len(gt),2),int)
    for i in range(1,len(gt)):
        mask=valid[i]&valid[i-1]
        next_value=[]
        for current in range(2):
            transitions=[]
            for prev in range(2):
                delta=models[current,i]-models[prev,i-1]-(gt[i]-gt[i-1])
                transitions.append(value[prev]+np.linalg.norm(delta[mask],axis=-1).mean()+.25*(current!=prev))
            back[i,current]=np.argmin(transitions)
            next_value.append(min(transitions)+costs[current,i])
        value=next_value
    path=[np.argmin(value)]
    for i in range(len(gt)-1,0,-1):
        path.append(back[i,path[-1]])
    return {'jpe_oracle':costs.argmin(0),'rr_oracle':rr.argmin(0),'dp_oracle':np.array(path[::-1])}
