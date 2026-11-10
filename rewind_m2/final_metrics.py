"""2026-11-09 비교 정의 복구: raw 단위, mask 및 sequence/gap 보존."""
import numpy as np


def evaluate_motion(prediction, target, valid, identities):
    p,g=np.asarray(prediction,dtype=np.float64),np.asarray(target,dtype=np.float64)
    m=np.asarray(valid)
    if p.shape!=g.shape or p.ndim!=3 or p.shape[-1]!=3 or m.shape!=p.shape[:2] or m.dtype!=bool or len(identities)!=len(p):
        raise ValueError('metric_shape_mask')
    if not np.isfinite(p).all() or not np.isfinite(g[m]).all() or len(set(identities))!=len(identities):
        raise ValueError('metric_nonfinite_or_duplicate')
    error=np.linalg.norm(p[m]-g[m],axis=-1); dp,dg=[],[]
    for i in range(1,len(p)):
        if identities[i][0]==identities[i-1][0] and identities[i][1]==identities[i-1][1]+1:
            mask=m[i]&m[i-1];dp.extend((p[i]-p[i-1])[mask]);dg.extend((g[i]-g[i-1])[mask])
    n=len(dp); pm=float(np.linalg.norm(dp,axis=-1).mean()) if n else None
    gm=float(np.linalg.norm(dg,axis=-1).mean()) if n else None
    return {'joint_position_error_raw':float(error.mean()) if error.size else None,
            'frame_displacement_error_raw':float(np.linalg.norm(np.asarray(dp)-dg,axis=-1).mean()) if n else None,
            'motion_magnitude_ratio':pm/gm if n and gm>0 else None,
            'predicted_mean_displacement_raw':pm,'gt_mean_displacement_raw':gm,
            'valid_joint_count':int(error.size),'adjacent_valid_joint_pairs':n}


def latency_summary(milliseconds):
    x=np.asarray(milliseconds,dtype=np.float64)
    if x.ndim!=1 or not x.size or not np.isfinite(x).all() or (x<0).any():raise ValueError('invalid_latency')
    return {'mean_latency_ms_per_frame':float(x.mean()),'p95_latency_ms_per_frame':float(np.percentile(x,95,method='linear')),'latency_sample_count':len(x)}
