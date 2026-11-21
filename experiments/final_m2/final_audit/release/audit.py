"""Recalculate release metrics from frozen arrays without model inference."""
import hashlib
import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np

archive = Path(sys.argv[1])
expected = '810b70d455096a9c15a422df00c6e24deaf74b148fbc498bd4a5344dc98bc526'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected
with ZipFile(archive) as z:
    manifest = json.loads(z.read('FILE_SHA256.json'))
    members = {name for name in z.namelist() if not name.endswith('/')}
    assert members == set(manifest) | {'FILE_SHA256.json'}
    for name, digest in manifest.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == digest
    evidence = json.loads(z.read('results/evaluation.json'))
    with np.load(io.BytesIO(z.read('results/poses.npz'))) as data:
        poses = dict(data)
    timing = json.loads(z.read('verification/control_timing.json'))
source = Path(__file__).resolve().parents[2] / 'dataset_scaling_300/data_selection_manifest.json'
parent, child = np.array(json.loads(source.read_text())['edges']).T
gt = poses['gt'].astype(float)
assert gt.shape == (16, 22, 3) and poses['valid'].all()
assert np.array_equal(poses['frame_ids'], np.arange(16))
norm = lambda x: np.linalg.norm(x, axis=-1)
def speed(x):
    return norm(np.diff(x, axis=0)).mean(1)
fast = speed(gt) >= np.quantile(speed(gt), .75)
assert (np.flatnonzero(fast) + 1).tolist() == [11, 12, 14, 15]
max_diff = 0.
for name, row in evidence.items():
    pose = poses[name].astype(float)
    error = pose - gt
    glob = norm(error).mean(1)
    rr = norm(error-error[:, :1]).mean(1)
    values = dict(jpe=glob.mean(), rr=rr.mean(), fde=norm(np.diff(error,axis=0)).mean(),
                  bone=100*abs(norm(pose[:,parent]-pose[:,child])/norm(gt[:,parent]-gt[:,child])-1).mean(),
                  fast_gap=abs(100*speed(pose)[fast].mean()/speed(gt)[fast].mean()-100))
    max_diff = max(max_diff, *(abs(row[k]-v) for k,v in values.items()))
    assert np.array_equal(pose[:,0],poses['control'][:,0])
    if name == 'control':
        base, global_base, rr_base = values, glob, rr
    else:
        passed = all(values[k]<base[k] for k in ('jpe','rr','fde','bone')) and values['fast_gap']<=base['fast_gap']
        assert passed == row['pass'] == False
    changed = np.any(pose != poses['control'], axis=(1,2)).sum()
    print(name, 'changed/global improved/RR worsened', changed,(glob<global_base).sum(),(rr>rr_base).sum())
assert max_diff < 1e-5
print('members',len(manifest),'max absolute difference',max_diff)
ms=timing['infer_frame_synchronized_ms']
print('first / subsequent mean (ms)',ms[0], np.mean(ms[1:]))
