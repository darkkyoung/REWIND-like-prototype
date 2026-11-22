"""Synthetic gradient diagnostic at full V1Body width; never use its coefficient for training."""
import json
from pathlib import Path
import torch
from rewind_v1 import teacher
from experiments.v12_m2.pipeline import ROOT,layout
from .trainer import diagnose

if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(71)
    cfg=json.loads((Path(teacher.__file__).resolve().parents[1]/'configs/rewind_v12_full_training.json').read_text())['teacher']
    spec=json.loads((ROOT/'configs/real_archvis62_v1.json').read_text())
    model=teacher.V1Body(cfg,22,layout(spec).tolist())
    batches=[dict(keys=[(str(b),i) for i in range(50)],split='train',features=torch.randn(50,868),
                  target=torch.randn(50,62,3)*100,valid=torch.ones(50,62,dtype=torch.bool),
                  camera=torch.randn(50,2,9)) for b in range(6)]
    coefficient,draws=diagnose(model,batches,cfg)
    print(json.dumps(dict(scope='synthetic CPU; random weights; not experiment calibration',hidden=cfg['hidden'],
                          coefficient=coefficient,draws=draws),indent=2))
