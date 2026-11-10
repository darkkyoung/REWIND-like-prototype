"""실제 runner 출력만 시각화. 동일 raw 축, display10 FPS(촬영 FPS 아님)."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter


def render(result,gifs=False):
    root=Path(result);r=json.loads((root/'evaluation.json').read_text())
    if r['status']!='completed' or r['data_kind']!='real_unrealego2_pose':raise ValueError('completed_real_output_required')
    out=root/'figures';out.mkdir(exist_ok=True)
    env=r['environment'];label=f"{env['gpu'] or 'CPU'} | FP32 B1 | real Pose148 | GT stereo2D | zero camera"
    colors=['#247ca6','#ea8e36'];rows=list(r['results'].values())
    for filename,keys,titles in [
        ('joint_error',['joint_position_error_raw'],['Joint position error (raw)']),
        ('motion_fidelity',['frame_displacement_error_raw','motion_magnitude_ratio'],['Displacement vector error (raw / frame pair)','Motion magnitude ratio (%)']),
        ('latency',['mean_latency_ms_per_frame','p95_latency_ms_per_frame'],['Mean streaming latency (ms / frame)','P95 streaming latency (ms / frame)'])]:
        fig,axes=plt.subplots(1,len(keys),figsize=(6*len(keys),4.5),squeeze=False,layout='constrained')
        for ax,key,title in zip(axes[0],keys,titles):
            for j,step in enumerate([5,10]):
                values=[r['results'][f'{split}_{step}'][key] for split in ['train','holdout']]
                if key=='motion_magnitude_ratio':values=[v*100 for v in values]
                bars=ax.bar(np.arange(2)+(j-.5)*.32,values,.32,label=f'DDIM {step}',color=colors[j])
                ax.bar_label(bars,labels=[f'{v:.4f}' if 'error' in key else f'{v:.2f}' for v in values],fontsize=9,padding=3)
            ax.set_xticks([0,1],['Train-fit (18)','Historical holdout (130)']);ax.set_title(title);ax.set_ylim(bottom=0);ax.margins(y=.18)
            if key=='motion_magnitude_ratio':ax.axhline(100,color='gray',linestyle='--');ax.set_ylim(0,112)
            ax.grid(axis='y',alpha=.2);ax.legend()
        fig.suptitle(label,fontsize=10);fig.savefig(out/f'{filename}.png',dpi=190);plt.close(fig)
    edges=json.loads((root/'data_manifest.json').read_text())['edges'];fast=[]
    for split in ['train','holdout']:
        a=np.load(root/f'{split}_5_predictions.npz',allow_pickle=False);b=np.load(root/f'{split}_10_predictions.npz',allow_pickle=False)
        np.testing.assert_array_equal(a['gt'],b['gt']);np.testing.assert_array_equal(a['valid'],b['valid'])
        frames=a['frame_ids'];assert np.all(np.diff(frames)==1),'gaps must not be stitched'
        poses=[a['gt'],a['prediction'],b['prediction']];titles=['GT','DDIM 5','DDIM 10'];mask=a['valid']
        displacement=[np.linalg.norm(np.diff(p.astype(float),axis=0),axis=-1).mean(1) for p in poses]
        fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
        for v,title,color in zip(displacement,titles,['#333333',*colors]):ax.plot(frames[1:],v,label=title,color=color)
        ax.set(xlabel='Observed frame ID',ylabel='Mean joint displacement (raw)',title=f'{split}: movement magnitude | capture FPS unverified');ax.legend();ax.grid(alpha=.2)
        fig.savefig(out/f'{split}_displacement.png',dpi=180);plt.close(fig)
        top=np.argsort(displacement[0])[-max(1,int(np.ceil(len(displacement[0])*.1))):]+1
        for i in top:
            row={'split':split,'frame_id':int(frames[i]),'gt_displacement_raw':float(displacement[0][i-1])}
            for step,p in zip([5,10],poses[1:]):
                row[f'ddim{step}_joint_error_raw']=float(np.linalg.norm(p[i].astype(float)-poses[0][i],axis=-1).mean())
                row[f'ddim{step}_displacement_raw']=float(np.linalg.norm(p[i].astype(float)-p[i-1],axis=-1).mean())
            fast.append(row)
        points=np.concatenate(poses);lo=points.min((0,1));hi=points.max((0,1));center=(lo+hi)/2;radius=max(float((hi-lo).max())*.56,1)
        def draw_ax(ax,p,title):
            ax.clear()
            for j,k in edges:ax.plot(*p[[j,k]].T,color='#286d9c',lw=1.6)
            ax.scatter(*p.T,color='#da7d2b',s=8)
            ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius),title=title,xlabel='raw x',ylabel='raw y',zlabel='raw z')
            ax.set_box_aspect((1,1,1));ax.view_init(18,-70);ax.tick_params(labelsize=6);ax.locator_params(nbins=3)
            ax.set_title(title, pad=18, fontsize=11)
        selected=[0,int(top[np.argmax(displacement[0][top-1])]),len(frames)-1]
        fig=plt.figure(figsize=(12,12));axes=[fig.add_subplot(3,3,j+1,projection='3d') for j in range(9)]
        for row,idx in enumerate(selected):
            for col,(p,title) in enumerate(zip(poses,titles)):draw_ax(axes[row*3+col],p[idx],f'{title} | frame {frames[idx]}')
        fig.suptitle(f'{split}: actual outputs / common axes, no alignment\n{label}',fontsize=11)
        fig.subplots_adjust(top=.90,bottom=.06,wspace=.2,hspace=.65)
        fig.savefig(out/f'{split}_motion_comparison.png',dpi=160);plt.close(fig)
        if gifs:
            fig=plt.figure(figsize=(10,3.8));axes=[fig.add_subplot(1,3,j+1,projection='3d') for j in range(3)]
            def update(i):
                for ax,p,title in zip(axes,poses,titles):draw_ax(ax,p[i],title)
                fig.suptitle(f'{split} frame {frames[i]} | shared raw axes | display10 FPS; capture FPS unknown',fontsize=10)
            fig.subplots_adjust(top=.84,bottom=.12,left=.04,right=.96,wspace=.15)
            FuncAnimation(fig,update,frames=len(frames),interval=100).save(out/f'{split}_motion.gif',writer=PillowWriter(fps=10));plt.close(fig)
    with (root/'large_movement_frames.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(fast[0]),lineterminator='\n');w.writeheader();w.writerows(fast)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('result');p.add_argument('--gifs',action='store_true');a=p.parse_args();render(a.result,a.gifs)
