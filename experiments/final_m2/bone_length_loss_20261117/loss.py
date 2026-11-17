"""Body22 x0 bone-length objective in existing normalized pose units."""
import torch


def bone_length_loss(prediction, target_raw, valid, time_valid, edges, scale_raw):
    """Mean squared normalized length error over valid (frame, edge) observations."""
    if prediction.ndim != 4 or prediction.shape != target_raw.shape or prediction.shape[-1] != 3:
        raise ValueError('pose_shape')
    b, t, j, _ = prediction.shape
    if valid.shape != (b,t,j) or time_valid.shape != (b,t) or valid.dtype != torch.bool or time_valid.dtype != torch.bool:
        raise ValueError('mask_shape')
    if not scale_raw > 0 or not edges or any(len(e)!=2 or min(e)<0 or max(e)>=j or e[0]==e[1] for e in edges):
        raise ValueError('edges_or_scale')
    if not torch.isfinite(prediction).all():
        raise ValueError('nonfinite_prediction')
    mask = valid & time_valid[...,None]
    if not torch.isfinite(target_raw[mask]).all():
        raise ValueError('nonfinite_target')
    a, z = zip(*edges)
    pair = mask[...,a] & mask[...,z]
    gt = torch.where(mask[...,None], target_raw, 0.) / scale_raw
    p_len = torch.linalg.vector_norm(prediction[...,a,:]-prediction[...,z,:], dim=-1)
    g_len = torch.linalg.vector_norm(gt[...,a,:]-gt[...,z,:], dim=-1)
    if not pair.any():
        return prediction.sum()*0., 0
    result = (p_len[pair]-g_len[pair]).square().mean()
    if not torch.isfinite(result):
        raise ValueError('nonfinite_bone_loss')
    return result, int(pair.sum())


def combined_loss(existing, bone, weight):
    if weight < 0:
        raise ValueError('negative_weight')
    return existing if weight == 0 else existing + weight*bone
