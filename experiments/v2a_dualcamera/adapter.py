"""Inference-only adapter for the frozen A portable source. Public inputs exclude targets."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from rewind_m3.cascade import UpperCondition
from rewind_v1.camera_training import CameraOrigin
from rewind_v1.data import attach_predicted_hands

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "pins.json").read_text())
ORIGINS = {"baseline": "first_midpoint", "anchored": "absolute"}


def digest(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class Cameras:
    first_midpoint: torch.Tensor
    absolute: torch.Tensor
    anchor_world_mm: list


class DualCamera:
    """Each call is one contiguous run of real frames. Reset at each official clip start."""
    def __init__(self):
        self.state = None

    def encode(self, r, t, sequence, frames, *, new_clip=False):
        frames = list(frames)
        if (not sequence or not frames or any(type(f) is not int or f < 0 for f in frames)
                or frames != list(range(frames[0], frames[0] + len(frames)))):
            raise ValueError("split the stream into contiguous runs of real frame IDs")
        if any(x.dtype != torch.float64 or x.device.type != "cpu" for x in (r, t)):
            raise ValueError("original CPU float64 camera extrinsics required")
        state = self.state
        if new_clip or state is None or state["clip"][0] != sequence or state["next_frame"] != frames[0]:
            state = None
        start = frames[0] if state is None else state["first_frame"]
        midpoint, audit = CameraOrigin("first_midpoint", state).apply(r, t, sequence, start, frames)
        absolute, _ = CameraOrigin("absolute").apply(r, t, sequence, frames[0], frames)
        self.state = audit["state"]
        return Cameras(midpoint, absolute, self.state["anchor_world_mm"][:])


@dataclass(frozen=True)
class Observations:
    features: torch.Tensor
    camera: torch.Tensor
    time_valid: torch.Tensor
    input_sha256: str
    source: str = "predicted_2d"


def inputs(features, camera, time_valid, identity, indices):
    selected = features.unflatten(-1, (62, 14))[..., indices, :].flatten(-2)
    return Observations(selected, camera[None], time_valid, identity)


@torch.inference_mode()
def predict(policy, stages, body, hand, features, cameras, time_valid, identity, noises):
    """One complete official validation clip. Both policies receive the very same noise tensors."""
    origin = ORIGINS[policy]
    body_pin = PINS["selected"]["body"]["sha256"] if policy == "baseline" else PINS["anchored"]
    if body.checkpoint_sha256 != body_pin or hand.checkpoint_sha256 != PINS["selected"]["hand"]["sha256"]:
        raise ValueError("checkpoint identity")
    if (features.ndim != 3 or features.shape != (1, time_valid.numel(), 62 * 14)
            or time_valid.shape != features.shape[:2] or time_valid.dtype != torch.bool or not time_valid.all()):
        raise ValueError("requires a complete single clip")
    n = features.shape[1]
    for x, shape in ((features, features.shape), (cameras.first_midpoint,(n,2,9)), (cameras.absolute,(n,2,9)),
                     (noises[0],(1,n,22,3)), (noises[1],(1,n,40,3))):
        if x.shape != shape or x.dtype != torch.float32 or not x.isfinite().all():
            raise ValueError("input shape, precision or finiteness")
    spec = stages.spec
    names = list(stages.body_names) + list(stages.fingers)
    if names != spec["joint_names"] or names[0] != "pelvis":
        raise ValueError("unexpected device-coordinate Whole62 layout")
    b = body.predict(inputs(features, getattr(cameras, origin), time_valid, identity, stages.body_idx), noises[0]).cpu()
    values = b[..., stages.upper_idx, :]
    upper = UpperCondition(values, time_valid[..., None].expand(values.shape[:-1]), "predicted_frozen_body", body_pin)
    local = hand.predict(inputs(features, cameras.first_midpoint, time_valid, identity, stages.finger_idx),
                         noises[1], upper=upper).cpu()
    global_hand = attach_predicted_hands(b, local, spec)
    whole = torch.cat((b, global_hand), dim=2)
    for value, joints in ((b,22), (local,40), (whole,62)):
        if value.shape != (1,n,joints,3) or not value.isfinite().all():
            raise ValueError("invalid prediction")
    return dict(body=b[0], hand_local=local[0], hand_global=global_hand[0], whole=whole[0])


def mixed_stages(stages, frozen_hand, path):
    """Separate inference instance binds the genuine candidate upper to unchanged A hand weights."""
    path = Path(path)
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != PINS["anchored"]:
            raise ValueError("anchored.pt SHA")
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
    if state["binding"] != PINS["root_binding"] or state["step"] != 24:
        raise ValueError("anchored experiment binding")
    body = stages.make("body")
    expected = body.model.state_dict()
    weights = state["body"]
    if (len(expected) != 104 or expected.keys() != weights.keys()
            or any(x.shape != weights[k].shape or x.dtype != weights[k].dtype or not weights[k].isfinite().all()
                   for k,x in expected.items())):
        raise ValueError("Body state contract")
    body.model.load_state_dict(weights, strict=True)
    body.model.eval().requires_grad_(False)
    body.checkpoint_sha256 = PINS["anchored"]
    hand = stages.make("hand", body.checkpoint_sha256)
    hand.model.load_state_dict(frozen_hand.model.state_dict(), strict=True)
    hand.model.eval().requires_grad_(False)
    hand.checkpoint_sha256 = frozen_hand.checkpoint_sha256
    return body, hand
