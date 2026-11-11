"""Saved real T4 arrays only. No torch/model load, inference, training, or alignment."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BUNDLE_SHA = 'ebf6c18a053d53d5309a73a17ed146f1c4690e92ca89f09f0b69b54fc378c2f3'
RESULT_COMMIT = 'b5d88b7368066d454c9e2b647cd7dd83508f0b19'
STEPS = (5, 10)
COLORS = ('#1975a5', '#d77524')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def average(x, mask):
    return float(np.asarray(x)[mask].mean()) if np.any(mask) else None


def ratio(a, b):
    return float(a / b) if a is not None and b is not None and b > 1e-12 else None


def rms_spread(x, mask):
    count = mask.sum(1)
    if np.any(count < 2):
        raise ValueError('spread requires at least two valid joints in each frame')
    center = np.where(mask[..., None], x, 0).sum(1) / count[:, None]
    radius2 = ((x - center[:, None]) ** 2).sum(-1)
    return np.sqrt(np.where(mask, radius2, 0).sum(1) / count)


def calculate(prediction, gt, valid, frame_ids, edges, root):
    p, g = np.asarray(prediction, float), np.asarray(gt, float)
    m, ids, e = np.asarray(valid), np.asarray(frame_ids), np.asarray(edges)
    if p.shape != g.shape or p.ndim != 3 or p.shape[-1] != 3 or m.shape != p.shape[:2] or m.dtype != bool:
        raise ValueError('shape/mask contract')
    if len(ids) != len(g) or len(set(ids)) != len(ids) or not np.isfinite(p[m]).all() or not np.isfinite(g[m]).all():
        raise ValueError('identity/finite contract')
    if not 0 <= root < p.shape[1] or e.ndim != 2 or e.shape[1] != 2 or e.min() < 0 or e.max() >= p.shape[1]:
        raise ValueError('root/edge contract')
    em = m[:, e[:, 0]] & m[:, e[:, 1]]
    gv, pv = g[:, e[:, 1]] - g[:, e[:, 0]], p[:, e[:, 1]] - p[:, e[:, 0]]
    gl, pl = np.linalg.norm(gv, axis=-1), np.linalg.norm(pv, axis=-1)
    if np.any(em & (gl <= 1e-12)):
        raise ValueError('zero GT bone length: ratio undefined')
    br = np.divide(pl, gl, out=np.full_like(pl, np.nan), where=em)
    gs, ps = rms_spread(g, m), rms_spread(p, m)
    gr, pr = g - g[:, root:root+1], p - p[:, root:root+1]
    rm = m & m[:, root:root+1]
    nonroot = rm.copy(); nonroot[:, root] = False
    global_error = np.linalg.norm(p - g, axis=-1)
    relative_error = np.linalg.norm(pr - gr, axis=-1)
    pelvis_error = np.linalg.norm(p[:, root] - g[:, root], axis=-1)
    tm = m[1:] & m[:-1] & (np.diff(ids) == 1)[:, None]
    trm = nonroot[1:] & nonroot[:-1] & (np.diff(ids) == 1)[:, None]
    dg, dp = np.diff(g, axis=0), np.diff(p, axis=0)
    drg, drp = np.diff(gr, axis=0), np.diff(pr, axis=0)
    gm = average(np.linalg.norm(dg, axis=-1), tm)
    pm = average(np.linalg.norm(dp, axis=-1), tm)
    grm = average(np.linalg.norm(drg, axis=-1), trm)
    prm = average(np.linalg.norm(drp, axis=-1), trm)
    glmean, plmean = average(gl, em), average(pl, em)
    summary = {
        'frames':len(g), 'joints':g.shape[1], 'edges':len(e),
        'gt_mean_bone_length_raw':glmean, 'pred_mean_bone_length_raw':plmean,
        'bone_length_ratio_pct':100*ratio(plmean, glmean),
        'mean_individual_bone_ratio_pct':100*average(br, em),
        'gt_joint_spread_rms_raw':float(gs.mean()), 'pred_joint_spread_rms_raw':float(ps.mean()),
        'joint_spread_ratio_pct':100*ratio(ps.mean(), gs.mean()),
        'mean_frame_spread_ratio_pct':float(100*np.mean(ps/gs)),
        'global_joint_error_raw':average(global_error, m),
        'root_relative_error_22_raw':average(relative_error, rm),
        'root_relative_error_nonroot_21_raw':average(relative_error, nonroot),
        'pelvis_position_error_raw':average(pelvis_error, m[:, root]),
        'gt_mean_displacement_raw':gm, 'pred_mean_displacement_raw':pm,
        'motion_magnitude_ratio_pct':100*ratio(pm, gm) if ratio(pm, gm) is not None else None,
        'frame_displacement_error_raw':average(np.linalg.norm(dp-dg, axis=-1), tm),
        'root_relative_motion_ratio_pct':100*ratio(prm, grm) if ratio(prm, grm) is not None else None,
        'gt_displacement_rms_raw':float(np.sqrt(average((dg**2).sum(-1), tm))) if np.any(tm) else None,
        'valid_bone_observations':int(em.sum()), 'valid_adjacent_joint_pairs':int(tm.sum()),
    }
    bone_rows = []
    for j, (a, b) in enumerate(e):
        gmj, pmj = average(gl[:, j], em[:, j]), average(pl[:, j], em[:, j])
        vals = br[:, j][em[:, j]]
        bone_rows.append({'edge_index':j, 'parent_index':int(a), 'child_index':int(b),
                         'gt_length_raw':gmj, 'pred_length_raw':pmj,
                         'ratio_pct':100*ratio(pmj, gmj) if ratio(pmj, gmj) is not None else None,
                         'mean_frame_ratio_pct':float(vals.mean()*100) if len(vals) else None,
                         'min_frame_ratio_pct':float(vals.min()*100) if len(vals) else None,
                         'max_frame_ratio_pct':float(vals.max()*100) if len(vals) else None})
    frame_rows = []
    for i, frame in enumerate(ids):
        frame_rows.append({'frame_id':int(frame), 'gt_spread_rms_raw':float(gs[i]),
            'pred_spread_rms_raw':float(ps[i]), 'spread_ratio_pct':float(100*ps[i]/gs[i]),
            'bone_ratio_pct':100*ratio(average(pl[i], em[i]), average(gl[i], em[i])),
            'global_joint_error_raw':average(global_error[i],m[i]),
            'root_relative_error_22_raw':average(relative_error[i],rm[i]),
            'pelvis_error_raw':float(pelvis_error[i]) if m[i,root] else None})
    return summary, bone_rows, frame_rows


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def draw_skeletons(data, edges, names, output, relative=False):
    frames = (0, 45, 65, 100)
    poses = [data[('holdout',5)]['gt'], data[('holdout',5)]['prediction'], data[('holdout',10)]['prediction']]
    root = names.index('pelvis')
    if relative:
        poses = [p-p[:,root:root+1] for p in poses]
    # Common scalar span across x/y/z and all shown frames/arms. No fitted alignment.
    pts = np.concatenate([p[list(frames)] for p in poses])
    lo, hi = pts.min((0,1)), pts.max((0,1))
    center, radius = (lo+hi)/2, float((hi-lo).max())*.56
    limits = [(float(c-radius),float(c+radius)) for c in center]
    fig = plt.figure(figsize=(13,16))
    titles = ['GT','DDIM 5','DDIM 10']
    panel_labels = []
    for row, frame in enumerate(frames):
        for col, p in enumerate(poses):
            ax = fig.add_subplot(4,3,row*3+col+1,projection='3d')
            pose = p[frame]
            for a,b in edges:
                side = names[b]
                color = '#1675a1' if side.endswith('_l') else '#dc7822' if side.endswith('_r') else '#3a3a3a'
                ax.plot(*pose[[a,b]].T,color=color,lw=2.5,alpha=.95)
            ax.scatter(*pose.T,color='#272727',s=13,depthshade=False)
            ax.scatter(*pose[root],color='#bd2538',s=55,depthshade=False)
            ax.set(xlim=limits[0],ylim=limits[1],zlim=limits[2],xlabel='raw x',ylabel='raw y',zlabel='raw z')
            ax.set_box_aspect((1,1,1)); ax.view_init(elev=18,azim=-70)
            panel_labels.append((ax, f'{titles[col]} | Opening frame {frame}'))
            ax.tick_params(labelsize=7); ax.locator_params(nbins=3)
    label = 'Pelvis-relative: subtract EACH skeleton pelvis only; no rotation/scaling' if relative else 'Original device coordinates: no translation/rotation/scaling'
    fig.suptitle(f'{label}\nShared limits within this figure | red: pelvis; blue: left; orange: right; gray: trunk',fontsize=12,y=.98)
    fig.subplots_adjust(top=.93,bottom=.04,left=.03,right=.97,wspace=.08,hspace=.3)
    # Figure-coordinate labels stay visible with both raw and pelvis-relative z limits.
    for ax, text in panel_labels:
        box = ax.get_position()
        fig.text((box.x0+box.x1)/2, box.y1+.012, text, ha='center', fontsize=11)
    fig.savefig(output,dpi=135);plt.close(fig)
    return limits


def plot_all(data, summaries, bones, frames, edges, names, out):
    fig, axes = plt.subplots(1,2,figsize=(15,8),layout='constrained')
    labels = [f'{names[a]} -> {names[b]}' for a,b in edges]
    for ax, split in zip(axes,('train','holdout')):
        for offset, step, color in zip((-.17,.17),STEPS,COLORS):
            rows=[r for r in bones if r['split']==split and r['ddim_steps']==step]
            ax.barh(np.arange(len(edges))+offset,[r['ratio_pct'] for r in rows],.32,color=color,label=f'DDIM {step}')
        ax.set_yticks(range(len(edges)),labels,fontsize=8);ax.invert_yaxis()
        ax.axvline(100,color='#444',ls='--');ax.axvline(80,color='#888',ls=':',label='80% screening flag')
        ax.set(xlabel='Mean predicted bone length / mean GT length (%)',title=f'{split}: original 21 edges',xlim=(0,145))
        ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('Saved T4 output | Body22 | GT stereo2D | raw lengths | no alignment')
    fig.savefig(out/'bone_length_comparison.png',dpi=140);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
    for col,split in enumerate(('train','holdout')):
        f5=[r for r in frames if r['split']==split and r['ddim_steps']==5]
        x=[r['frame_id'] for r in f5]
        axes[0,col].plot(x,[r['gt_spread_rms_raw'] for r in f5],color='#333',label='GT')
        for step,color,style in zip(STEPS,COLORS,('-', '--')):
            rows=[r for r in frames if r['split']==split and r['ddim_steps']==step]
            axes[0,col].plot(x,[r['pred_spread_rms_raw'] for r in rows],color=color,ls=style,label=f'DDIM {step}')
            axes[1,col].plot(x,[r['spread_ratio_pct'] for r in rows],color=color,ls=style,label=f'DDIM {step}')
        axes[0,col].set(title=split+' | centroid removed',ylabel='RMS joint radius (raw)',ylim=(0,55))
        axes[1,col].set(xlabel='Observed frame ID',ylabel='Predicted / GT spread (%)',ylim=(0,115))
        axes[1,col].axhline(100,color='#555',ls=':')
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Joint spread RMS = sqrt(mean squared 3D distance to joint centroid)')
    fig.savefig(out/'joint_spread_comparison.png',dpi=150);plt.close(fig)
    limits = {'global':draw_skeletons(data,edges,names,out/'skeleton_visualization.png'),
              'pelvis_relative':draw_skeletons(data,edges,names,out/'root_relative_comparison.png',True)}
    return limits


def run(bundle, output):
    raw=Path(bundle).read_bytes()
    if sha(raw)!=BUNDLE_SHA:raise ValueError('source bundle SHA mismatch')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'skeleton_diagnosis.csv').exists():raise FileExistsError('diagnosis already exists; choose a new directory')
    data, hashes = {}, {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if z.testzip() is not None:raise ValueError('source ZIP CRC')
        sums=json.loads(z.read('SHA256SUMS.json'))
        def read(name):
            b=z.read(name)
            if sha(b)!=sums[name]:raise ValueError('file checksum mismatch: '+name)
            hashes[name]=sha(b);return b
        manifest=json.loads(read('t4/data_manifest.json'));evaluation=json.loads(read('t4/evaluation.json'))
        if evaluation['status']!='completed' or evaluation['environment']['gpu']!='Tesla T4':raise ValueError('source result contract')
        if manifest['coordinate_frame']!='stored_device_pts3d' or len(manifest['joint_names'])!=22:raise ValueError('coordinate/joint contract')
        for split,n in (('train',18),('holdout',130)):
            for step in STEPS:
                with np.load(io.BytesIO(read(f't4/{split}_{step}_predictions.npz')),allow_pickle=False) as a:
                    data[(split,step)]={k:a[k].copy() for k in a.files}
                d=data[(split,step)]
                if d['prediction'].shape!=(n,22,3) or d['prediction'].dtype!=np.float32 or not np.array_equal(d['frame_ids'],np.arange(n)):raise ValueError('array contract')
            for k in ('gt','valid','frame_ids'):np.testing.assert_array_equal(data[(split,5)][k],data[(split,10)][k])
    names,edges=manifest['joint_names'],manifest['edges'];root=names.index('pelvis')
    summaries,bones,frames=[],[],[]
    for (split,step),d in data.items():
        s,b,f=calculate(d['prediction'],d['gt'],d['valid'],d['frame_ids'],edges,root)
        old=evaluation['results'][f'{split}_{step}']
        for key,oldkey,factor in [('global_joint_error_raw','joint_position_error_raw',1),('motion_magnitude_ratio_pct','motion_magnitude_ratio',100),('frame_displacement_error_raw','frame_displacement_error_raw',1)]:
            np.testing.assert_allclose(s[key],factor*old[oldkey],rtol=1e-12,atol=1e-12)
        summaries.append({'split':split,'ddim_steps':step,**s})
        bones.extend({'split':split,'ddim_steps':step,'bone':names[r['parent_index']]+' -> '+names[r['child_index']],**r} for r in b)
        frames.extend({'split':split,'ddim_steps':step,**r} for r in f)
    write_csv(out/'skeleton_diagnosis.csv',summaries)
    write_csv(out/'bone_length_by_edge.csv',bones)
    write_csv(out/'frame_diagnosis.csv',frames)
    limits=plot_all(data,summaries,bones,frames,edges,names,out)
    evidence={'status':'PASS','source_bundle_sha256':BUNDLE_SHA,'source_result_commit':RESULT_COMMIT,
        'source_file_sha256':hashes,'script_sha256':sha(Path(__file__).read_bytes()),
        'original_metrics_recomputed':'PASS','gt_mask_frame_identity_between_arms':'PASS',
        'new_model_inference_calls':0,'new_optimizer_updates':0,'root_index':root,'joint_names':names,'edges':edges,
        'coordinate_frame':'stored_device_pts3d','distance_unit':'raw; physical unit unverified',
        'figure_limits':limits,'relative_transform':'each skeleton subtracts its own pelvis; no fitted transform',
        'all_masks_valid':all(d['valid'].all().item() for d in data.values())}
    (out/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--bundle',required=True);a.add_argument('--output',required=True)
    args=a.parse_args();run(args.bundle,args.output)
