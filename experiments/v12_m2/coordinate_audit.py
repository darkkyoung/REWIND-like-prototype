"""Optional reproducible raw/mm coordinate check on existing Taunt16, no calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from rewind_prepare.extracted import _pose_targets
from rewind_v1.data import _field
from .pipeline import ROOT, layout

def audit(raw, npz):
    base=ROOT/'experiments/final_m2/final_audit'
    spec=json.loads((ROOT/'configs/real_archvis62_v1.json').read_bytes())
    layout(spec)
    pins=json.loads((base/'raw_manifest.json').read_bytes())
    assert hashlib.sha256(npz.read_bytes()).hexdigest()==json.loads((base/'results/candidate_audit.json').read_bytes())['input_sha256']
    diffs=[]
    with np.load(npz) as d:
        for index,pin in enumerate(pins):
            payload=(raw/pin['title']).read_bytes()
            assert hashlib.sha256(payload).hexdigest()==pin['sha256']
            doc=json.loads(payload)
            assert int(doc['frame'])==d['frame_ids'][index]==index
            pos,mask=_field(_pose_targets(doc,spec)[2],3)
            np.testing.assert_array_equal(mask[:22],d['valid'][index,-1])
            diffs.append(np.abs(pos[:22].numpy()/spec['coordinates']['mm_per_raw_unit']-d['target_raw'][index,-1]).max())
    assert len(diffs)==16 and max(diffs)<2e-5
    print('Taunt16 matched 16 x 22 joint coordinates and masks; max abs delta (raw)',max(diffs))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('raw',type=Path);parser.add_argument('npz',type=Path)
    args=parser.parse_args();audit(args.raw,args.npz)
