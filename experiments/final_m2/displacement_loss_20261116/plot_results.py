"""실제 저장 NPZ 네 조건, 동일 축 범위, 후처리 정렬 없음."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ARMS=('baseline','control','improved')
LABELS=('GT','Baseline 300','Control 400','Improved 400')
COLORS=('#30343b','#407db4','#df8829','#159f82')


def limits(poses):
    points=np.concatenate(poses);lo=points.min((0,1));hi=points.max((0,1))
    center=(lo+hi)/2;radius=float((hi-lo).max())*.56
    return [(float(c-radius),float(c+radius)) for c in center]


def axes_style(ax,lims):
    ax.set(xlim=lims[0],ylim=lims[1],zlim=lims[2],xlabel='raw x',ylabel='raw y',zlabel='raw z')
    ax.set_box_aspect((1,1,1));ax.view_init(elev=18,azim=-70);ax.tick_params(labelsize=7);ax.locator_params(nbins=3)


def skeleton(ax,pose,edges,names):
    lines=[]
    for a,b in edges:
        color='#407db4' if names[b].endswith('_l') else '#df8829' if names[b].endswith('_r') else '#30343b'
        lines+=ax.plot(*pose[[a,b]].T,color=color,lw=2.1,marker='o',markersize=2.5)
    root,=ax.plot(*pose[0:1].T,'o',color='#c32c48',markersize=5)
    return lines,root


def run(input_dir,output_dir):
    src=Path(input_dir);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((src/'data_manifest.json').read_text());edges=manifest['edges'];names=manifest['joint_names']
    data={s:{a:dict(np.load(src/f'{s}_{a}_predictions.npz',allow_pickle=False)) for a in ARMS} for s in ('train','holdout')}
    axis_ranges={}
    opening=data['holdout'];poses=[opening['baseline']['gt']]+[opening[a]['prediction'] for a in ARMS]
    for relative,filename in ((False,'skeleton_visualization.png'),(True,'root_relative_comparison.png')):
        ps=[p-p[:,0:1] for p in poses] if relative else poses
        lim=limits(ps);axis_ranges[filename]=lim
        fig=plt.figure(figsize=(16,16));labels=[]
        for row,frame in enumerate((0,45,65,100)):
            for col,p in enumerate(ps):
                ax=fig.add_subplot(4,4,row*4+col+1,projection='3d');axes_style(ax,lim);skeleton(ax,p[frame],edges,names)
                labels.append((ax,f'{LABELS[col]} | frame {frame}'))
        kind='Each skeleton pelvis subtracted; no rotation/scaling' if relative else 'Original device coordinates; no translation/rotation/scaling'
        fig.suptitle(f'{kind}\nShared axis ranges and 1:1:1 scale | left blue, right orange, pelvis red',y=.985,fontsize=13)
        fig.subplots_adjust(top=.93,bottom=.04,left=.02,right=.98,wspace=.06,hspace=.28)
        for ax,label in labels:
            box=ax.get_position();fig.text((box.x0+box.x1)/2,box.y1+.012,label,ha='center',fontsize=10)
        fig.savefig(out/filename,dpi=120);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(14,8),layout='constrained')
    for col,split in enumerate(('train','holdout')):
        ds=data[split];gt=ds['baseline']['gt'].astype(float);fid=ds['baseline']['frame_ids']
        movement=np.linalg.norm(np.diff(gt,axis=0),axis=-1).mean(1)
        axs[0,col].plot(fid[1:],movement,label='GT',color=COLORS[0],lw=2)
        for a,label,color in zip(ARMS,LABELS[1:],COLORS[1:]):
            p=ds[a]['prediction'].astype(float);dp=np.diff(p,axis=0)
            axs[0,col].plot(fid[1:],np.linalg.norm(dp,axis=-1).mean(1),label=label,color=color)
            axs[1,col].plot(fid[1:],np.linalg.norm(dp-np.diff(gt,axis=0),axis=-1).mean(1),label=label,color=color)
        axs[0,col].set(title=f'{split}: frame displacement magnitude',ylabel='Mean joint displacement (raw)',ylim=(0,None))
        axs[1,col].set(title=f'{split}: displacement vector error',ylabel='Mean vector error (raw)',xlabel='Frame ID (t-1 to t)',ylim=(0,None))
        for ax in axs[:,col]:ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.savefig(out/'frame_displacement_comparison.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(15,8),layout='constrained')
    e=np.array(edges)
    for ax,split in zip(axs,('train','holdout')):
        gt=data[split]['baseline']['gt'].astype(float);gl=np.linalg.norm(gt[:,e[:,1]]-gt[:,e[:,0]],axis=-1).mean(0)
        for offset,a,label,color in zip((-.24,0,.24),ARMS,LABELS[1:],COLORS[1:]):
            p=data[split][a]['prediction'].astype(float);pl=np.linalg.norm(p[:,e[:,1]]-p[:,e[:,0]],axis=-1).mean(0)
            ax.barh(np.arange(21)+offset,100*pl/gl,.23,label=label,color=color)
        ax.set_yticks(range(21),[f'{names[a]} -> {names[b]}' for a,b in edges],fontsize=8);ax.invert_yaxis()
        ax.axvline(100,color='#444',ls='--');ax.set(title=split,xlabel='Bone length / GT (%)');ax.legend(fontsize=9);ax.grid(axis='x',alpha=.2)
    fig.savefig(out/'bone_length_comparison.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for ax,split in zip(axs,('train','holdout')):
        gt=data[split]['baseline']['gt'].astype(float);spread=lambda p:np.sqrt(((p-p.mean(1,keepdims=True))**2).sum(-1).mean(1))
        for a,label,color in zip(ARMS,LABELS[1:],COLORS[1:]):
            p=data[split][a]['prediction'].astype(float);ax.plot(data[split][a]['frame_ids'],100*spread(p)/spread(gt),label=label,color=color)
        ax.axhline(100,color='#444',ls='--');ax.set(title=split,xlabel='Frame ID',ylabel='Joint spread / GT (%)',ylim=(0,115));ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'joint_spread_comparison.png',dpi=140);plt.close(fig)
    # Full130frames same limits for every arm/frame; 10fps is display rate only.
    lim=limits(poses);fig=plt.figure(figsize=(13,4));panels=[]
    for col,p in enumerate(poses):
        ax=fig.add_subplot(1,4,col+1,projection='3d');axes_style(ax,lim);ax.set_title(LABELS[col],fontsize=11)
        panels.append(skeleton(ax,p[0],edges,names))
    title=fig.suptitle('Opening frame 0 | original raw coordinates | display 10 fps (capture FPS unverified)',fontsize=11)
    fig.subplots_adjust(left=.01,right=.99,bottom=.12,top=.82,wspace=.05)
    def update(frame):
        for p,(lines,root) in zip(poses,panels):
            for line,(a,b) in zip(lines,edges):line.set_data_3d(*p[frame,[a,b]].T)
            root.set_data_3d(*p[frame,0:1].T)
        title.set_text(f'Opening frame {frame} | original raw coordinates | display 10 fps (capture FPS unverified)')
        return []
    FuncAnimation(fig,update,frames=130,interval=100,blit=False).save(out/'opening_motion.gif',writer=PillowWriter(fps=10),dpi=80)
    plt.close(fig);(out/'visualization_axes.json').write_text(json.dumps(axis_ranges,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.input,a.output)
