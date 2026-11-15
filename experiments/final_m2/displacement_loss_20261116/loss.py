"""실험 전용 raw 이동 벡터 손실. 기존 position_losses는 변경하지 않는다."""
import numpy as np
import torch

S_DELTA = 4.1444875
LAMBDA = 0.1


def displacement_loss(pred_raw, target_raw, valid, time_valid, frame_ids, sequence_ids):
    """유효 인접 joint-pair마다 xyz 제곱합을 계산한 뒤 평균 (xyz 평균 아님)."""
    if pred_raw.shape != target_raw.shape or pred_raw.ndim != 4 or pred_raw.shape[-1] != 3:
        raise ValueError('pose_shape')
    b, t, j, _ = pred_raw.shape
    if valid.shape != (b,t,j) or time_valid.shape != (b,t) or frame_ids.shape != (b,t) or sequence_ids.shape != (b,t):
        raise ValueError('metadata_shape')
    if valid.dtype != torch.bool or time_valid.dtype != torch.bool:
        raise ValueError('boolean_masks_required')
    mask = valid & time_valid[...,None]
    if not torch.isfinite(pred_raw).all() or not torch.isfinite(target_raw[mask]).all():
        raise ValueError('nonfinite_pose')
    pair = mask[:,1:] & mask[:,:-1]
    adjacent = (frame_ids[:,1:]-frame_ids[:,:-1] == 1) & (sequence_ids[:,1:] == sequence_ids[:,:-1])
    pair = pair & adjacent[...,None]
    # 누락 target의 NaN이 backward에 섞이지 않도록 차분 이전에 정리한다.
    gt = torch.where(mask[...,None], target_raw, 0.)
    delta = ((pred_raw[:,1:]-pred_raw[:,:-1])-(gt[:,1:]-gt[:,:-1])) / S_DELTA
    if not pair.any():
        return pred_raw.sum()*0., 0
    loss = delta[pair].square().sum(-1).mean()
    if not torch.isfinite(loss):
        raise ValueError('nonfinite_displacement_loss')
    return loss, int(pair.sum())


def window_metadata(data):
    """저장 window의 모든 유효 slot을 실제 anchor 행과 대조한다.

    frame 번호 산술 추측만으로 연속성을 승인하지 않는다. gap 이전의 history,
    미래 slot, 잘못된 padding 또는 다른 sequence 값이 들어오면 거부한다.
    """
    seq, ids, time = data['sequence_ids'], data['frame_ids'], data['time']
    keys = [(str(s),int(f)) for s,f in zip(seq,ids)]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate_anchor')
    lookup = {k:i for i,k in enumerate(keys)}
    codes = {s:i for i,s in enumerate(sorted(set(map(str,seq))))}
    frames = np.full(time.shape,-1,dtype=np.int64)
    sequences = np.full(time.shape,-1,dtype=np.int64)
    width = time.shape[1]
    for row,(s,f) in enumerate(keys):
        history = []
        for old in range(f,f-width,-1):
            if (s,old) not in lookup: break
            history.append(old)
        expected = np.zeros(width,dtype=bool); expected[-len(history):] = True
        if not np.array_equal(time[row],expected):
            raise ValueError('time_padding_or_gap')
        for slot in range(width):
            if not time[row,slot]:
                if any(np.any(data[k][row,slot]) for k in ('features','target_raw','valid')):
                    raise ValueError('nonzero_padding')
                continue
            old = f-width+1+slot
            source = lookup[(s,old)]
            for k in ('features','target_raw','valid'):
                if not np.array_equal(data[k][row,slot],data[k][source,-1]):
                    raise ValueError('window_frame_content_mismatch')
            frames[row,slot], sequences[row,slot] = old,codes[s]
    return torch.from_numpy(frames),torch.from_numpy(sequences)
