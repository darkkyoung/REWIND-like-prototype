"""Frame-causal reliability signals. Only observations and Control poses enter the gate."""
import numpy as np


def mean_norm(x):
    return float(np.linalg.norm(x, axis=-1).mean()) if x.size else 0.


class Signals:
    def __init__(self, edges):
        self.edges = np.asarray(edges).T
        self.key = None
        self.history = []

    def push(self, seq, frame, features, pose):
        if self.key != (seq, frame-1):
            self.history.clear()
        self.key = (seq, frame)
        views = features.reshape(22, 2, 7)
        xy, valid = views[..., :2], views[..., 2].astype(bool)
        a, b = self.edges
        lengths = np.linalg.norm(pose[a]-pose[b], axis=1)
        speed = obs = innovation = bone = 0.
        if self.history:
            prev, mask, old_pose, old_lengths = self.history[-1]
            joint_mask = valid & mask
            delta = xy-prev
            speed = mean_norm(delta[joint_mask])
            change = pose-old_pose
            if len(self.history)==2:
                older, older_mask, older_pose, _ = self.history[0]
                obs = mean_norm((delta-(prev-older))[joint_mask & older_mask])
                innovation = mean_norm(change-(old_pose-older_pose))
            bone = float(np.abs(lengths-old_lengths).mean())
        self.history = (self.history+[(xy.copy(), valid.copy(), pose.copy(), lengths)])[-2:]
        return np.array([speed, obs, innovation, bone, 1-valid.mean()])


def scores(features, scale):
    magnitude, obs, innovation, bone, missing = (features/scale).T
    # Require joint observational and structural evidence, or unavailable observations.
    return magnitude, np.maximum(missing, np.minimum(obs, np.maximum(innovation, bone)))


def fit(train):
    scale = np.maximum(np.quantile(train,.9,axis=0), [1e-5,1e-5,.001,.001,.05])
    return scale, np.quantile(scores(train,scale),.95,axis=1)


def route(control, improved, selected):
    return np.where(selected[:,None,None], improved, control)


def corrupt(features, condition, seed=2111):
    rng = np.random.default_rng(seed)
    result = features.copy()
    views = result.reshape(-1,22,2,7)
    for i, frame in enumerate(views):
        phase = i % 20
        active = {'clean':False,'jitter':4<=phase<9,'spike':phase==9,
                  'short_missing':4<=phase<6,'long_missing':4<=phase<12}[condition]
        if not active:
            continue
        block = frame[6:14]
        if condition.endswith('missing'):
            block[:] = 0
        else:
            noise = rng.normal(size=block[...,:2].shape)
            block[...,:2] += noise * (8 if condition=='jitter' else 64)/1024 * block[...,2:3]
    return result, np.any(result != features, axis=1)
