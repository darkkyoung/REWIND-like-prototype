"""Validate this published M2 raw-pose diagnostic checkpoint without accessing dataset."""
from pathlib import Path
import argparse, hashlib, json, sys, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rewind_m2.models import BodyDenoiser


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='experiments/m2_pose_300/checkpoints/last.pt')
    a = p.parse_args()
    path = Path(a.checkpoint)
    expected = (Path(__file__).resolve().parents[1] / 'experiments/m2_pose_300/checkpoint.sha256').read_text().split()[0]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit('Checkpoint SHA-256 mismatch')
    ck = torch.load(path, map_location='cpu', weights_only=True)
    assert ck['step'] == 300 and ck['format_version'] == 'm2_raw_pose_diagnostic_v1'
    skeleton = ck['manifest']
    model = BodyDenoiser(ck['config'], len(skeleton['joint_names']), skeleton['edges'])
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    print(json.dumps({'step': ck['step'], 'joint_count': len(skeleton['joint_names']),
        'sampling_steps': ck['config']['sampling_steps'], 'sha256': digest,
        'experiment': ck['format_version'], 'model_loaded': True}, indent=2))

if __name__ == '__main__': main()
