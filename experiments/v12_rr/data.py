"""Deterministic selection from the original manifest and training plan."""
import json
import re
from pathlib import Path
import torch
from rewind_v1.full_contract import load_audit, training_plan
from rewind_v1.full_cache import _frame, PreparationBudget
from experiments.v12_m2.pipeline import frontend

HERE = Path(__file__).parent
def motion(seq):
    return re.sub(r'__\d+_$','',seq.split('/')[-1]).casefold()

def selection(archive):
    manifest, coverage, _ = load_audit(archive)
    plan = training_plan(manifest, coverage)
    candidates = sorted((r['sequence'], r['start_frame']) for r in plan['train_clips'] if r['real_frames']==50)
    train, people, motions = [], set(), set()
    for seq, start in candidates:
        person=seq.split('/')[2]
        if person not in people and motion(seq) not in motions:
            train.append([seq,start]);people.add(person);motions.add(motion(seq))
        if len(train)==6:
            break
    val=plan['teacher_validation_clips']
    other=next(k for k in val if motion(k[0]) not in motions|{motion(val[0][0])})
    if len(train)!=6 or motion(val[0][0])!='standing':
        raise ValueError('selection contract')
    return dict(train=train,validation=[val[0],other],manifest=manifest['content_sha256'],plan=plan['content_sha256']),manifest

def read_clip(seq,start,split,lookup,spec,rgb,pose,models):
    images, metadata, keys = [], [], []
    budget=PreparationBudget()
    for f in range(start,start+50):
        row=lookup[seq,f]
        if row['split']!=split:
            raise ValueError('split mismatch')
        raw={role:budget.read_source(pose if role=='pose' else rgb,pin) for role,pin in row['files'].items()}
        image,meta=_frame(raw,f,spec)
        images.append(image);metadata.append(meta);keys.append((seq,f))
    return dict(features=frontend(models,torch.stack(images)),camera=torch.stack([r['camera'] for r in metadata]),
                target=torch.stack([r['target_mm'] for r in metadata]),valid=torch.stack([r['target_valid'] for r in metadata]),
                keys=keys,split=split)
