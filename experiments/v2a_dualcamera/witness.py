"""Replay either verified cache extrinsics or the original pose's camera fields."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from rewind_prepare.coordinates import camera_transform
from .adapter import DualCamera, HERE

WITNESS = json.loads((HERE / "witness.json").read_text())


def check(cam):
    for x, field in ((cam.first_midpoint,"v2_camera"), (cam.absolute,"absolute_camera")):
        if not torch.equal(x[:1], torch.tensor(WITNESS[field], dtype=torch.float32)):
            raise ValueError(field)
    if cam.anchor_world_mm != WITNESS["anchor_world_mm"]:
        raise ValueError("anchor witness")
    delta = float((cam.absolute[:1,...,6:] - cam.first_midpoint[:1,...,6:]).abs().max()*100)
    if delta != WITNESS["max_translation_delta_mm"]:
        raise ValueError("translation witness")
    return delta


def replay(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != WITNESS["row"]["files"]["pose"]["sha256"]:
        raise ValueError("Gunplay frame 0 source identity")
    doc = json.loads(raw)
    cams = [camera_transform(doc["camera_"+s],10.,1e-6)[0] for s in ("left","right")]
    r,t = [torch.tensor([[c[k] for c in cams]],dtype=torch.float64) for k in ("rotation_matrix","translation")]
    encoded = DualCamera().encode(r,t,WITNESS["row"]["sequence_id"],[0])
    return {"input_translation_delta_mm":check(encoded),"anchor_world_mm":encoded.anchor_world_mm,
            "pose_sha256":hashlib.sha256(raw).hexdigest(),"both_encoded_tensors":"exact"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pose")
    print(json.dumps(replay(parser.parse_args().pose),indent=2))
