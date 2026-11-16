"""회수된 Opening 실제 배열: 동일 축, 별도 pelvis-relative, 130-frame GIF."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from experiments.final_m2.displacement_loss_20261116.plot_results import limits,axes_style,skeleton
ARMS=('train18','train100');LABELS=('GT','18 frames / 400 updates','100 frames / 400 updates');COLORS=('#30343b','#407db4','#159f82')
def run(src,out,notebook):
 src,out=Path(src),Path(out);out.mkdir(parents=True,exist_ok=True);manifest=json.loads((Path(__file__).parent/'data_selection_manifest.json').read_text());edges,names=manifest['edges'],manifest['joint_names'];report=json.loads((src/'evaluation.json').read_text())
 ds={a:dict(np.load(src/f'opening130_{a}_predictions.npz',allow_pickle=False)) for a in ARMS};poses=[ds['train18']['gt']]+[ds[a]['prediction'] for a in ARMS];axis_ranges={}
 for rel,file in ((False,'skeleton_visualization.png'),(True,'root_relative_comparison.png')):
  ps=[p-p[:,:1] for p in poses] if rel else poses;lim=limits(ps);axis_ranges[file]=lim;fig=plt.figure(figsize=(12,16));labels=[]
  for row,f in enumerate((0,45,65,100)):
   for col,p in enumerate(ps):
    ax=fig.add_subplot(4,3,row*3+col+1,projection='3d');axes_style(ax,lim);skeleton(ax,p[f],edges,names);labels.append((ax,f'{LABELS[col]} | frame {f}'))
  fig.suptitle(('Each skeleton pelvis subtracted; no rotation/scaling' if rel else 'Original device coordinates; no translation/rotation/scaling')+'\nIdentical axes and 1:1:1 scale | blue left, orange right, red pelvis',y=.985,fontsize=12)
  fig.subplots_adjust(top=.93,bottom=.04,left=.02,right=.98,wspace=.06,hspace=.28)
  for ax,label in labels:
   box=ax.get_position();fig.text((box.x0+box.x1)/2,box.y1+.012,label,ha='center',fontsize=10)
  fig.savefig(out/file,dpi=120);plt.close(fig)
 g=poses[0].astype(float);dg=np.diff(g,axis=0);fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
 axs[0,0].plot(range(1,130),np.linalg.norm(dg,axis=-1).mean(1),label='GT',color=COLORS[0],lw=2)
 for a,label,col in zip(ARMS,LABELS[1:],COLORS[1:]):
  p=ds[a]['prediction'].astype(float);dp=np.diff(p,axis=0)
  axs[0,0].plot(range(1,130),np.linalg.norm(dp,axis=-1).mean(1),label=label,color=col)
  axs[0,1].plot(range(1,130),np.linalg.norm(dp-dg,axis=-1).mean(1),label=label,color=col)
  axs[1,0].plot(range(130),np.linalg.norm(p-g,axis=-1).mean(1),label=label,color=col)
  axs[1,1].plot(range(130),np.linalg.norm(p[:,0]-g[:,0],axis=-1),label=label,color=col)
 for ax,title,y in zip(axs.flat,('Joint displacement magnitude','Displacement vector error','Global joint position error','Pelvis position error'),('Raw displacement','Raw vector error','Raw joint error','Raw pelvis error')):ax.set(title=title,xlabel='Opening frame',ylabel=y,ylim=(0,None));ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.savefig(out/'frame_displacement_comparison.png',dpi=140);plt.close(fig)
 e=np.array(edges);gl=np.linalg.norm(g[:,e[:,1]]-g[:,e[:,0]],axis=-1).mean(0);fig,ax=plt.subplots(figsize=(11,8),layout='constrained')
 for off,p,label,col in zip((-.16,.16),poses[1:],LABELS[1:],COLORS[1:]):
  p=p.astype(float);pl=np.linalg.norm(p[:,e[:,1]]-p[:,e[:,0]],axis=-1).mean(0);ax.barh(np.arange(21)+off,100*pl/gl,.30,label=label,color=col)
 ax.set_yticks(range(21),[f'{names[a]} -> {names[b]}' for a,b in edges],fontsize=9);ax.invert_yaxis();ax.axvline(100,color='#444',ls='--');ax.set(title='Opening130: bone length distortion',xlabel='Mean bone length / GT (%)');ax.legend();ax.grid(axis='x',alpha=.2)
 fig.savefig(out/'bone_length_comparison.png',dpi=140);plt.close(fig)
 spread=lambda p:np.sqrt(((p-p.mean(1,keepdims=True))**2).sum(-1).mean(1));fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
 for p,label,col in zip(poses[1:],LABELS[1:],COLORS[1:]):ax.plot(range(130),100*spread(p.astype(float))/spread(g),label=label,color=col)
 ax.axhline(100,color='#444',ls='--');ax.set(xlabel='Opening frame',ylabel='Joint spread / GT (%)',title='Joint spread after centroid removal');ax.legend();ax.grid(alpha=.2);fig.savefig(out/'joint_spread_comparison.png',dpi=140);plt.close(fig)
 # Actual saved console observations at prescribed checkpoint updates only.
 nb=json.loads(Path(notebook).read_text());observations=[]
 for cell in nb['cells']:
  for o in cell.get('outputs',[]):
   if o.get('output_type')=='stream':
    for line in ''.join(o.get('text',[])).splitlines():
     try:r=json.loads(line)
     except ValueError:continue
     if set(('model','step','loss'))<=r.keys():observations.append(r)
 unique={(r['model'],r['step']):r for r in observations};observations=list(unique.values());assert len(observations)==8
 with (out/'checkpoint_training_loss.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(observations[0]),lineterminator='\n');w.writeheader();w.writerows(observations)
 fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
 for a,label,col in zip(ARMS,LABELS[1:],COLORS[1:]):
  r=sorted([r for r in observations if r['model']==a],key=lambda r:r['step']);ax.plot([r['step'] for r in r],[r['loss'] for r in r],marker='o',label=label,color=col)
 ax.set(xlabel='Optimizer update',ylabel='Normalized position loss',title='Observed checkpoint losses only (100 / 200 / 300 / 400)',yscale='log');ax.set_xticks([100,200,300,400]);ax.legend();ax.grid(alpha=.2);fig.savefig(out/'training_loss.png',dpi=140);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(13,4),layout='constrained');splits=['train18','standing50','dodging32','opening130'];labs=['Brutal18','Standing50','Dodging32','Opening130']
 for ax,key,title in zip(axs,('global_joint_error_raw','frame_displacement_error_raw'),('Joint position error','Displacement vector error')):
  for i,(a,label,col) in enumerate(zip(ARMS,LABELS[1:],COLORS[1:])):
   rr={r['split']:r for r in report['results'] if r['model']==a};ax.bar(np.arange(4)+(i-.5)*.34,[rr[s][key] for s in splits],.32,label=label,color=col)
  ax.set_xticks(range(4),labs);ax.set(title=title,ylabel='Raw error');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
 fig.suptitle('Opening: historical holdout | Other actions: training data for 100-frame model',fontsize=11);fig.savefig(out/'per_motion_comparison.png',dpi=140);plt.close(fig)
 lim=limits(poses)
 for which,file in [([0,1,2],'opening_comparison_motion.gif'),([1],'opening_train18_motion.gif'),([2],'opening_train100_motion.gif')]:
  fig=plt.figure(figsize=(4*len(which),4));panels=[]
  for k,i in enumerate(which):
   ax=fig.add_subplot(1,len(which),k+1,projection='3d');axes_style(ax,lim);ax.set_title(LABELS[i],fontsize=10);panels.append(skeleton(ax,poses[i][0],edges,names))
  title=fig.suptitle('Opening 0 | raw; no alignment | display 10 fps (capture FPS unknown)',fontsize=10);fig.subplots_adjust(left=.02,right=.98,bottom=.12,top=.82,wspace=.07)
  def update(frame):
   for i,(lines,root) in zip(which,panels):
    for line,(a,b) in zip(lines,edges):line.set_data_3d(*poses[i][frame,[a,b]].T)
    root.set_data_3d(*poses[i][frame,0:1].T)
   title.set_text(f'Opening {frame} | raw; no alignment | display 10 fps (capture FPS unknown)');return []
  FuncAnimation(fig,update,frames=130,interval=100).save(out/file,writer=PillowWriter(fps=10),dpi=85);plt.close(fig)
 (out/'visualization_axes.json').write_text(json.dumps(axis_ranges,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--notebook',required=True);a=p.parse_args();run(a.input,a.output,a.notebook)
