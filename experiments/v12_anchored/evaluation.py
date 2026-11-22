import csv
import time

import numpy as np
import torch
from PIL import Image,ImageDraw
from rewind_v1.cross_inference import sample_features
from experiments.v12_m2.evaluate import metrics
from experiments.v12_rr import trainer as rr
from experiments.v12_rr.run import write

def decide(rows):
    d,r,c=(rows[k] for k in ('anchored','rr','control'))
    limits=[('rr','pelvis_mm',.95),('rr','rr_jpe_mm',1.01),('control','rr_jpe_mm',.99)]
    limits.extend(('control',k,1.02) for k in ('pelvis_mm','body_mpjpe_mm','wrist_mm','hand_global_mpjpe_mm','whole_mpjpe_mm'))
    checks={}
    for ref,k,factor in limits:
        x,y=d[k],rows[ref][k]
        if x is None or y is None or not np.isfinite([x,y]).all() or y<=0:
            return 'INCOMPLETE',checks
        checks[ref+'_'+k]=x<=factor*y
    return ('PASS' if all(checks.values()) else 'FAIL'),checks

def csv_file(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)

def picture(path,poses,spec):
    # Fixed first frame in device XY. One projection and bounds shared by all five panels.
    data={k:v[0,:,:2].numpy() for k,v in poses.items()}
    points=np.concatenate(list(data.values()));center=(points.min(0)+points.max(0))/2
    scale=200/max(float(np.ptp(points,axis=0).max()),1)
    im=Image.new('RGB',(230*len(data),250),'white');draw=ImageDraw.Draw(im)
    edges=[tuple(spec['joint_names'].index(n) for n in edge) for edge in spec['edges']]
    for i,(k,v) in enumerate(data.items()):
        xy=(v-center)*scale+[230*i+115,135]
        draw.text((230*i+8,5),k,fill='black')
        for a,b in edges:draw.line([tuple(xy[a]),tuple(xy[b])],fill='#21667d',width=2)
    im.save(path)

def evaluate(models,clips,selected,checkpoints,out):
    summary=[];metrics_rows=[];frame_rows=[];contrasts=[]
    for i,batch in enumerate(clips):
        seq=batch['keys'][0][0]
        group='new' if [seq,batch['keys'][0][1]] in selected['new'] else 'diagnostic'
        results={};arrays={};poses={'GT':batch['target']}
        for arm,(path,pin,binding) in checkpoints.items():
            rr.restore(path,pin,models['body'],binding)
            models['body'].eval().requires_grad_(False)
            torch.cuda.synchronize();begin=time.perf_counter()
            pred=sample_features(models,batch['features'],batch['camera'],batch['keys'],models['config'],'cuda')
            torch.cuda.synchronize();ms=1000*(time.perf_counter()-begin)/len(batch['keys'])
            if arm=='control':base=pred['body']
            row=metrics(pred,base,batch['target'],batch['valid'],models['spec'])
            row['body_hand_ms_frame']=ms;results[arm]=row
            metrics_rows.append(dict(group=group,sequence=seq,arm=arm,**row))
            poses[arm]=pred['whole']
            arrays.update({arm+'_'+k:v.numpy() for k,v in pred.items()})
            err=pred['body']-batch['target'][:,:22];valid=batch['valid'][:,:22]
            per=[]
            for j,(_,frame) in enumerate(batch['keys']):
                mask=valid[j]
                pair=mask&mask[0]
                per.append([err[j,mask].norm(dim=-1).mean().item() if mask.any() else None,
                    (err[j,pair]-err[j,:1]).norm(dim=-1).mean().item() if pair.any() else None,
                    err[j,0].norm().item() if mask[0] else None])
            if arm=='control':control=per
            for (_,frame),v,b in zip(batch['keys'],per,control):
                row=dict(group=group,sequence=seq,frame=frame,arm=arm)
                for k,x,y in zip(('global','rr','pelvis'),v,b):
                    row[k]=x;row[k+'_minus_control']=x-y if x is not None else None
                frame_rows.append(row)
        for metric in results['control']:
            a,b,c,d=(results[k][metric] for k in ('control','rr','pelvis','anchored'))
            if all(v is not None for v in (a,b,c,d)):
                contrasts.append(dict(group=group,sequence=seq,metric=metric,rr_only=b-a,pelvis_only=c-a,
                                      pelvis_given_rr=d-b,interaction=(d-b)-(c-a)))
        outcome,checks=decide(results)
        summary.append(dict(group=group,sequence=seq,results=results,decision=outcome,checks=checks))
        np.savez_compressed(out/f'clip_{i}.npz',gt=batch['target'].numpy(),valid=batch['valid'].numpy(),
                            frames=[f for _,f in batch['keys']],**arrays)
        picture(out/f'clip_{i}_frame0.png',poses,models['spec'])
    states=[r['decision'] for r in summary if r['group']=='new']
    overall='INCOMPLETE' if len(states)!=2 or 'INCOMPLETE' in states else ('PASS' if states==['PASS','PASS'] else 'FAIL')
    csv_file(out/'factorial.csv',contrasts)
    csv_file(out/'metrics.csv',metrics_rows);csv_file(out/'frames.csv',frame_rows)
    write(out/'evaluation.json',dict(clips=summary,decision=overall,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
