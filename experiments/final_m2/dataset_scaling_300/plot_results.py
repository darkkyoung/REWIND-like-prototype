"""저장된 실제 NPZ만 사용. 각 패널에 같은 축과 시점을 적용한다."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from PIL import Image
from experiments.final_m2.displacement_loss_20261116.plot_results import limits,axes_style,skeleton
ARMS=('train18','train100','train300');LABELS=('GT','Train18 / 400','Train100 / 400','Train300 / 400');COLORS=('#30343b','#407db4','#159f82','#b4539b')
def run(src,out):
 src,out=Path(src),Path(out);out.mkdir(parents=True,exist_ok=False)
 manifest=json.loads((Path(__file__).parent/'data_selection_manifest.json').read_text());edges,names=manifest['edges'],manifest['joint_names'];e=np.array(edges)
 report=json.loads((src/'evaluation.json').read_text());axes_record={};image_audit={}
 for split,chosen in [('opening130',(0,45,65,100)),('gunplay47',(0,15,30,46))]:
  ds={a:dict(np.load(src/f'{split}_{a}_predictions.npz',allow_pickle=False)) for a in ARMS}
  poses=[ds['train18']['gt']]+[ds[a]['prediction'] for a in ARMS];count=len(poses[0])
  for rel,name in [(False,'skeleton'),(True,'root_relative')]:
   ps=[p-p[:,:1] for p in poses] if rel else poses
   oldlim=limits(ps[:3]);allpts=np.concatenate(ps)
   inside=all(allpts[...,i].min()>=oldlim[i][0] and allpts[...,i].max()<=oldlim[i][1] for i in range(3))
   lim=oldlim if inside else limits(ps)
   axes_record[split+'_'+name]={'limits':lim,'uses_prior_three_arm_limits':inside,'view':'inherited axes_style elev18 azim-70','equal_xyz_scale':True,'aligned':False}
   fig=plt.figure(figsize=(16,16));labels=[]
   for row,f in enumerate(chosen):
    for col,p in enumerate(ps):
     ax=fig.add_subplot(4,4,row*4+col+1,projection='3d');axes_style(ax,lim);skeleton(ax,p[f],edges,names);labels.append((ax,f'{LABELS[col]} | frame {f}'))
   fig.suptitle(split+' | '+('Each pelvis subtracted only' if rel else 'Original device coordinates')+'\nCommon axes; no fitted translation, rotation or scaling',y=.985,fontsize=14)
   fig.subplots_adjust(top=.93,bottom=.04,left=.02,right=.98,wspace=.06,hspace=.28)
   for ax,label in labels:
    box=ax.get_position();fig.text((box.x0+box.x1)/2,box.y1+.012,label,ha='center',fontsize=10)
   fig.savefig(out/f'{split}_{name}.png',dpi=110);plt.close(fig)
  g=poses[0].astype(float);dg=np.diff(g,axis=0);fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
  axs[0,0].plot(range(1,count),np.linalg.norm(dg,axis=-1).mean(1),label='GT',color=COLORS[0],lw=2)
  for a,label,col in zip(ARMS,LABELS[1:],COLORS[1:]):
   p=ds[a]['prediction'].astype(float);dp=np.diff(p,axis=0)
   axs[0,0].plot(range(1,count),np.linalg.norm(dp,axis=-1).mean(1),label=label,color=col)
   axs[0,1].plot(range(1,count),np.linalg.norm(dp-dg,axis=-1).mean(1),label=label,color=col)
   axs[1,0].plot(range(count),np.linalg.norm(p-g,axis=-1).mean(1),label=label,color=col)
   axs[1,1].plot(range(count),np.linalg.norm(p[:,0]-g[:,0],axis=-1),label=label,color=col)
  for ax,title,y in zip(axs.flat,('Joint displacement magnitude','Displacement vector error','Global joint position error','Pelvis position error'),('Raw displacement','Raw vector error','Raw joint error','Raw pelvis error')):
   ax.set(title=title,xlabel=split+' frame',ylabel=y,ylim=(0,None));ax.grid(alpha=.2);ax.legend(fontsize=8)
  fig.savefig(out/f'{split}_frame_comparison.png',dpi=130);plt.close(fig)
  gl=np.linalg.norm(g[:,e[:,1]]-g[:,e[:,0]],axis=-1).mean(0);fig,ax=plt.subplots(figsize=(12,9),layout='constrained')
  for off,p,label,col in zip((-.24,0,.24),poses[1:],LABELS[1:],COLORS[1:]):
   p=p.astype(float);pl=np.linalg.norm(p[:,e[:,1]]-p[:,e[:,0]],axis=-1).mean(0);ax.barh(np.arange(21)+off,100*pl/gl,.23,label=label,color=col)
  ax.set_yticks(range(21),[f'{names[a]} -> {names[b]}' for a,b in edges],fontsize=9);ax.invert_yaxis();ax.axvline(100,color='#444',ls='--');ax.axvline(80,color='#777',ls=':');ax.axvline(120,color='#777',ls=':');ax.set(title=split+': individual bone length ratios',xlabel='Mean bone length / GT (%)');ax.legend();ax.grid(axis='x',alpha=.2)
  fig.savefig(out/f'{split}_bone_length.png',dpi=130);plt.close(fig)
  spread=lambda p:np.sqrt(((p-p.mean(1,keepdims=True))**2).sum(-1).mean(1));fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
  for p,label,col in zip(poses[1:],LABELS[1:],COLORS[1:]):ax.plot(range(count),100*spread(p.astype(float))/spread(g),label=label,color=col)
  ax.axhline(100,color='#444',ls='--');ax.set(xlabel=split+' frame',ylabel='Joint spread / GT (%)',title='Joint spread after centroid removal',ylim=(0,150));ax.legend();ax.grid(alpha=.2);fig.savefig(out/f'{split}_joint_spread.png',dpi=130);plt.close(fig)
  lim=axes_record[split+'_skeleton']['limits'];fig=plt.figure(figsize=(16,4));panels=[]
  for i in range(4):
   ax=fig.add_subplot(1,4,i+1,projection='3d');axes_style(ax,lim);ax.set_title(LABELS[i],fontsize=10);panels.append(skeleton(ax,poses[i][0],edges,names))
  title=fig.suptitle(split+' | raw; no alignment | display10 fps; capture FPS unknown',fontsize=10);fig.subplots_adjust(left=.02,right=.98,bottom=.12,top=.82,wspace=.07)
  def update(frame):
   for i,(lines,root) in enumerate(panels):
    for line,(a,b) in zip(lines,edges):line.set_data_3d(*poses[i][frame,[a,b]].T)
    root.set_data_3d(*poses[i][frame,0:1].T)
   title.set_text(f'{split} frame {frame} | raw; no alignment | display10 fps; capture FPS unknown');return []
  FuncAnimation(fig,update,frames=count,interval=100).save(out/f'{split}_comparison_motion.gif',writer=PillowWriter(fps=10),dpi=75);plt.close(fig)
 logs=list(csv.DictReader((src/'train300_training.csv').open()))
 fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
 for k,label in [('loss','Total position loss'),('x0_loss','x0 loss'),('aux_position_loss','Auxiliary position loss')]:ax.plot([int(r['step']) for r in logs],[float(r[k]) for r in logs],label=label,alpha=.8,lw=1)
 ax.set(xlabel='Optimizer update',ylabel='Normalized position loss',yscale='log',title='Train300: actual 400-update log, no smoothing');ax.legend();ax.grid(alpha=.2);fig.savefig(out/'training_loss.png',dpi=130);plt.close(fig)
 metrics=[('global_joint_error_raw','Joint error'),('root_relative_error_22_raw','Root-relative error'),('frame_displacement_error_raw','Displacement error'),('motion_magnitude_ratio_pct','Motion %'),('root_relative_motion_ratio_pct','RR motion %'),('direction_angle_deg','Direction angle deg'),('rr_temporal_pose_rms_ratio_pct','RR temporal RMS %'),('pelvis_position_error_raw','Pelvis error'),('bone_length_ratio_pct','Bone ratio %'),('mean_abs_bone_ratio_deviation_pp','Absolute bone distortion pp'),('joint_spread_ratio_pct','Spread %')]
 fig,axs=plt.subplots(1,2,figsize=(15,6),layout='constrained')
 for ax,split in zip(axs,('opening130','gunplay47')):
  lookup={r['model']:r for r in report['results'] if r['split']==split}
  table=ax.table(cellText=[[label]+[f"{lookup[a][k]:.4f}" for a in ARMS] for k,label in metrics],colLabels=['Metric','Train18','Train100','Train300'],loc='center',cellLoc='right',colWidths=[.49,.17,.17,.17]);table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.8)
  ax.axis('off');ax.set_title(split+' | raw units unless %/pp/deg')
 fig.savefig(out/'quantitative_comparison.png',dpi=140);plt.close(fig)
 for p in out.iterdir():
  if p.suffix in ('.png','.gif'):
   with Image.open(p) as im:
    n=getattr(im,'n_frames',1);durations=[]
    for i in range(n):im.seek(i);im.load();durations.append(im.info.get('duration'))
    image_audit[p.name]={'frames':n,'size':list(im.size),'durations_ms':sorted(set(durations),key=lambda v:-1 if v is None else v)}
    if p.suffix=='.gif':assert n==(130 if p.name.startswith('opening') else 47)
 (out/'visualization_validation.json').write_text(json.dumps({'status':'PASS','axes':axes_record,'files':image_audit,'source':'actual saved prediction arrays','rotation_scaling_alignment':'none','display_fps':10,'capture_fps':'NOT VERIFIED'},indent=2)+'\n')
 print(json.dumps({'images':len(image_audit),'axes':axes_record},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.input,a.output)
