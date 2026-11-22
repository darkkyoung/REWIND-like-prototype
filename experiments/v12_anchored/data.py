import hashlib
import json
from pathlib import Path

from rewind_v1.full_contract import load_audit, training_plan
from experiments.v12_rr.data import motion, read_clip

HERE = Path(__file__).parent


def selection(archive):
    prior=json.loads((HERE.parent/'v12_rr/selection.json').read_text())
    manifest,coverage,_=load_audit(archive)
    plan=training_plan(manifest,coverage)
    if (manifest['content_sha256'],plan['content_sha256'])!=(prior['manifest'],prior['plan']):
        raise ValueError('prior split binding')
    rows={(r['sequence_id'],r['frame_index']):r for r in manifest['records']}
    def files(clip,split):
        seq,start=clip
        selected=[rows[seq,i] for i in range(start,start+50)]
        if any(r['split']!=split for r in selected):
            raise ValueError('split mismatch')
        return [f['sha256'] for r in selected for f in r['files'].values()]
    blocked={h for clip in prior['train'] for h in files(clip,'train')}
    used={motion(seq) for seq,_ in prior['train']+prior['validation']}
    people={seq.split('/')[2] for seq,_ in prior['validation']}
    train_people={seq.split('/')[2] for seq,_ in prior['train']}
    candidates=list(plan['teacher_validation_clips'])
    added=[]
    while len(added)<2:
        candidates.sort(key=lambda row:(row[0].split('/')[2] in people,row[0].split('/')[2] in train_people,row))
        candidate=next((c for c in candidates if motion(c[0]) not in used and blocked.isdisjoint(files(c,'validation'))),None)
        if candidate is None:
            raise ValueError('insufficient distinct non-overlapping validation motions')
        added.append(candidate);used.add(motion(candidate[0]));people.add(candidate[0].split('/')[2])
        blocked.update(files(candidate,'validation'))
    result=dict(prior,validation=prior['validation']+added,diagnostic=prior['validation'],new=added)
    result['selected_file_identity']=hashlib.sha256(json.dumps(
        [files(c,k) for k in ('train','validation') for c in result[k]],separators=(',',':')).encode()).hexdigest()
    return result,manifest
