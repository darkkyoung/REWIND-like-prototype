"""One CPU update at the actual dimensions, with random weights and synthetic inputs."""
import json
import time
import torch
from rewind_v1.teacher import V1Body
from . import trainer as tr
from .data import HERE
from experiments.v12_m2.pipeline import layout,ROOT

def main():
    torch.set_num_threads(2);torch.manual_seed(71)
    spec=json.loads((ROOT/'configs/real_archvis62_v1.json').read_text())
    from rewind_v1 import teacher
    from pathlib import Path
    cfg=json.loads((Path(teacher.__file__).resolve().parents[1]/'configs/rewind_v12_full_training.json').read_text())['teacher']
    body=V1Body(cfg,22,layout(spec).tolist())
    batch=dict(keys=[('synthetic',i) for i in range(50)],split='train',features=torch.randn(50,868),
               target=torch.randn(50,62,3)*100,valid=torch.ones(50,62,dtype=torch.bool),camera=torch.randn(50,2,9))
    a=tr.state_hash(body);opt,sch=tr.optimizer(body);start=time.monotonic()
    result=tr.update(body,opt,sch,[batch],cfg,0,.1)
    assert tr.state_hash(body)!=a
    print(json.dumps(dict(scope='synthetic CPU, random initialization, diagnostic weight 0.1',hidden=cfg['hidden'],
          frames=50,seconds=time.monotonic()-start,finite_loss=result['total'],parameter_count=sum(p.numel() for p in body.parameters()),
          cuda_available=torch.cuda.is_available())))
if __name__=='__main__':main()
