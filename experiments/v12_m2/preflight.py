"""Real packaged weights + synthetic input only. Run from the package source."""
import argparse
import json
from pathlib import Path
import torch
from scripts.v12_share_smoke import load_models, predict, verify_manifest
from .pipeline import compare, frontend


def check(package):
    torch.set_num_threads(1)
    models = load_models(package,'cpu')
    images = torch.linspace(0,1,128).expand(2,2,3,128,128).contiguous()
    camera = torch.tensor([[[1.,0,0,0,1,0,-.6,0,0],[1.,0,0,0,1,0,.6,0,0]]]).repeat(2,1,1)
    keys = [('synthetic_package_smoke',0),('synthetic_package_smoke',1)]
    ref = predict(models,images,camera,keys)
    out,_ = compare(models,frontend(models,images),camera,keys,('baseline',))
    if any(not torch.equal(v,out['baseline'][k]) for k,v in ref.items()):
        raise ValueError('real weight baseline parity failed')
    manifest = verify_manifest(package)
    print(json.dumps(dict(source=manifest['source_revision'],scope='synthetic RGB / real frozen CPU weights',
                          baseline_max_abs_mm=0.,shapes={k:list(v.shape) for k,v in ref.items()},
                          files={k:v for k,v in manifest['files'].items() if k.startswith('models/')})))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('package',type=Path)
    check(parser.parse_args().package)
