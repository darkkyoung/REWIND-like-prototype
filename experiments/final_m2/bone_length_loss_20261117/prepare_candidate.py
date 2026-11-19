"""사전 지정한 Joyful Jump 원본에서 Body22 NPZ와 선택 기록을 생성."""
import argparse
from pathlib import Path
import numpy as np
from experiments.final_m2.dataset_scaling_100.prepare_data import convert, coordinate_check
from experiments.final_m2.bone_length_loss_20261117.run_experiment import HERE, load, sha, write
from rewind_data.audit import read_json


def prepare(root, out):
    seq = load(HERE/'config.json')['candidate_sequence']
    manifest = load(HERE.parent/'dataset_scaling_300/data_selection_manifest.json')
    prior = {r['pose_sha256'] for r in manifest['frames'] + manifest['additional_evaluation']['gunplay47']}
    rows, hashes = [], []
    for i in range(57):
        path = root/seq/'json'/f'frame_{i}.json'
        digest = sha(path)
        if digest in prior or digest in hashes:
            raise ValueError('duplicate_source_pose')
        doc = read_json(path.read_bytes())
        coordinate_check(doc,manifest['joint_names'])
        rows.append((seq,i,doc))
        hashes.append(digest)
    data = convert(rows,manifest['joint_names'])
    out.mkdir(parents=True,exist_ok=False)
    path = out/'joyful57.npz'
    np.savez_compressed(path,**data)
    write(path.with_suffix('.selection.json'),dict(sequence_id=seq,npz_sha256=sha(path),
          source_pose_sha256=hashes,selected_before_training=True,
          selection_reason='고정된 Joyful Jump 16 frame 0..56, M2 Train300 외 추가 평가 후보'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    prepare(args.root,args.out)
