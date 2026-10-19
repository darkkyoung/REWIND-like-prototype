"""Raw-unit comparison and train-only constant-per-joint baseline; no model changes."""
import csv
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def summary(pose,gt,valid,edges):
    pose=np.asarray(pose,dtype=np.float64);gt=np.asarray(gt,dtype=np.float64)
    delta=pose[valid]-gt[valid]
    # All 22 joints valid in this pinned input; don't silently use invalid joints.
    assert valid.all()
    centered=pose-pose.mean(1,keepdims=True)
    return {'mean_joint_error_raw':float(np.linalg.norm(delta,axis=-1).mean()),
        'spatial_radius_raw':float(np.sqrt((centered**2).sum(-1).mean(1)).mean()),
        'mean_bone_length_raw':float(np.mean([np.linalg.norm(pose[:,a]-pose[:,b],axis=-1).mean() for a,b in edges])),
        'temporal_std_raw':float(np.sqrt(pose.var(0).sum(-1)).mean())}


def analyze(bundle,output,render=True):
    bundle,output=Path(bundle),Path(output);start=time.perf_counter()
    old=json.loads((bundle/'metrics.json').read_text());new=json.loads((output/'metrics.json').read_text())
    assert new['step']==new['requested_steps']==300 and new['status']=='completed', '300-step result required'
    assert old['manifest_sha256']==new['manifest_sha256'], 'comparison data mismatch'
    manifest=json.loads((output/'data_manifest.json').read_text());edges=manifest['edges']
    train=np.load(bundle/'train_motion.npz',allow_pickle=False)
    valid=train['valid']; assert valid.all()
    mean_pose=train['gt'].mean(0)  # only 18 unique train anchors; never holdout
    np.save(output/'train_mean_pose.npy',mean_pose)
    rows=[];stats={}
    for split in ('train','holdout'):
        a=np.load(bundle/f'{split}_motion.npz',allow_pickle=False);b=np.load(output/f'{split}_motion.npz',allow_pickle=False)
        for k in ('gt','noisy','valid','frame_indices'):np.testing.assert_array_equal(a[k],b[k])
        gt,mask=b['gt'],b['valid'];stats[split]={}
        candidates={'GT':gt,'Noisy':b['noisy'],'Mean pose (train only)':np.broadcast_to(mean_pose,gt.shape),
            '5 x0':a['reconstructed'],'300 x0':b['reconstructed'],'5 DDIM10':a['ddim'],'300 DDIM10':b['ddim']}
        for name,pose in candidates.items():
            stats[split][name]=summary(pose,gt,mask,edges)
            rows.append({'split':split,'method':name,**stats[split][name]})
        if render:
            frames=b['frame_indices'];assert np.all(np.diff(frames)==1),'Gap motion must not be stitched'
            def figure(keys,titles,name,gif):
                poses=[candidates[k] for k in keys];points=np.concatenate(poses)
                lo=points.min((0,1));hi=points.max((0,1));center=(lo+hi)/2;r=max((hi-lo).max()*.54,1)
                fig=plt.figure(figsize=(12,4.4));axes=[fig.add_subplot(1,3,i+1,projection='3d') for i in range(3)]
                def draw(i):
                    for ax,pose,title in zip(axes,poses,titles):
                        ax.clear();p=pose[i]
                        for j,k in edges:ax.plot(*p[[j,k]].T,c='#315da8',lw=2)
                        ax.scatter(*p.T,c='#e38a38',s=9)
                        ax.set(xlim=(center[0]-r,center[0]+r),ylim=(center[1]-r,center[1]+r),zlim=(center[2]-r,center[2]+r),title=title,xlabel='raw x',ylabel='raw y',zlabel='raw z')
                        ax.set_box_aspect((1,1,1));ax.view_init(18,-65);ax.tick_params(labelsize=6);ax.locator_params(nbins=3)
                    fig.suptitle(f'{split} frame {frames[i]} | shared axes / raw units / display 10 FPS (capture FPS unknown)')
                    return []
                fig.subplots_adjust(left=.04,right=.96,bottom=.1,top=.85,wspace=.1)
                draw(0);fig.savefig(output/f'{split}_{name}.png',dpi=150)
                if gif:FuncAnimation(fig,draw,frames=len(frames),interval=100).save(output/f'{split}_{name}.gif',writer=PillowWriter(fps=10))
                plt.close(fig)
            figure(['GT','Noisy','300 x0'],['GT','Noisy GT, t=400','300-step single x0'],'motion',True)
            figure(['GT','5 x0','300 x0'],['GT','5-step single x0','300-step single x0'],'5_vs_300',False)
            figure(['GT','5 DDIM10','300 DDIM10'],['GT','5-step DDIM10','300-step DDIM10'],'ddim',False)
    history=json.loads((output/'loss_history.json').read_text())
    fig,ax=plt.subplots(figsize=(9,3.5));ax.plot([r['step'] for r in history],[r['loss'] for r in history],lw=1)
    ax.axvline(5,c='#cb6a32',ls='--',label='previous run ends: step 5');ax.set(xlabel='Optimizer updates',ylabel='Normalized x0 + frame MSE',yscale='log');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(output/'training_loss.png',dpi=160);plt.close(fig)
    with (output/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'unit':'raw_annotation_unit_unverified','alignment':'none','constant_baseline':'per-joint mean from 18 unique training anchors',
        'pose_statistics':stats,'five_step':old['final'],'three_hundred_step':new['final'],
        'first_loss':history[0]['loss'],'step5_loss':history[4]['loss'],'last_loss':history[-1]['loss'],
        'small_holdout_not_official_validation':True,'analysis_visualization_seconds':time.perf_counter()-start}
    (output/'comparison.json').write_text(json.dumps(report,indent=2))
    lines=['# 5회 / 300회 비교','', '단위: raw annotation units, 정렬 없음. GT 2D / zero-camera 진단. 공식 검증 성능이 아님.', '',
        '| Split | 방법 | 평균 관절 오차(raw) | 공간 반경(raw) | 시간 변화 std(raw) |','|---|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['split']} | {r['method']} | {r['mean_joint_error_raw']:.5f} | {r['spatial_radius_raw']:.5f} | {r['temporal_std_raw']:.5f} |")
    lines+=['','공간 반경은 프레임별 관절 중심에서의 RMS 거리입니다. 시간 변화는 관절별 시간축 표준편차 벡터의 크기를 평균합니다. 낮은 복원 오차만으로 움직임 복원이나 일반화가 증명되지 않습니다.','',
        f"Loss: {history[0]['loss']:.8f} → step5 {history[4]['loss']:.8f} → final {history[-1]['loss']:.8f}",
        f"GPU: {new['environment']['gpu']}; 학습 {new['training_seconds']:.3f}s; 평가 {new['evaluation_seconds']:.3f}s."]
    for split in ('train','holdout'):
        s=stats[split];err=s['300 x0']['mean_joint_error_raw'];base=s['Mean pose (train only)']['mean_joint_error_raw']
        lines.append(f"{split}: 300 x0 {'우세' if err<base else '열세'} vs train 평균 자세 ({err:.4f} vs {base:.4f}).")
    (output/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    return report
