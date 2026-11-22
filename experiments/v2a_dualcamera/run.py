"""Fixed official 3-clip paired comparison. Strict release replay or explicitly exploratory same-T4 comparison."""
import argparse
import csv
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from v2_gpu128_a_bundle_local import open_bundle
from v2_gpu128_a_bundle_format import FEATURE_IDENTITY
from v2_gpu128_a_bundle_predict import compare_predictions, predict_clip, select_clips
from rewind_m1.training import seed_run
from rewind_v1.frontend import frontend_fingerprint
from rewind_v1.full_data import FeatureStore
from scripts.v2_gpu_training_pipeline import CommonEvaluation
from experiments.v12_m2.evaluate import metrics
from .adapter import DualCamera, ORIGINS, PINS, digest, mixed_stages, predict
from .decision import MODES, decide
from .witness import WITNESS, check


def write(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")


def shared_hashes(args):
    features,cams,time_valid,_,noise = args
    return {"feature_sha256":digest(features),"camera_sha256":[digest(cams.first_midpoint),digest(cams.absolute)],
            "time_valid_sha256":digest(time_valid),"noise_sha256":list(map(digest,noise))}


def reference_stage_deltas(reference_zip, scenario, current):
    """Descriptive output differences on the already authenticated comparison member."""
    with zipfile.ZipFile(reference_zip) as archive:
        raw = archive.read(scenario+".npz")
    with np.load(io.BytesIO(raw),allow_pickle=False) as prior:
        xy = np.abs(current["predicted_2d_px"].astype(np.float64)-prior["predicted_2d_px"])
        xyz = np.abs(current["predicted_3d_mm"].astype(np.float64)-prior["predicted_3d_mm"])
    return {"left_xy_max_px":float(xy[:,0].max()),"right_xy_max_px":float(xy[:,1].max()),
            "body22_max_mm":float(xyz[:,:22].max()),"global_hand40_max_mm":float(xyz[:,22:].max())}


def timed(call):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    return value, {"seconds":time.perf_counter()-start,"peak_allocated_bytes":torch.cuda.max_memory_allocated()}


def split_audit(records, selected):
    keys = {(seq,f) for _,(seq,start) in selected for f in range(start,start+50)}
    train_hash = set()
    motions = set()
    rows = []
    for row in records:
        seq = row["sequence_id"]
        if row["split"] == "train":
            motions.add(seq.split("/")[-1])
            train_hash.update(p["sha256"] for p in row["files"].values())
        if (seq,row["frame_index"]) in keys:
            rows.append(row)
    if (len(rows) != len(keys) or {(r["sequence_id"],r["frame_index"]) for r in rows} != keys
            or any(r["split"] != "validation" for r in rows)):
        raise ValueError("validation membership or duplicate frame")
    overlap = {k:sum(r["files"][k]["sha256"] in train_hash for r in rows) for k in rows[0]["files"]}
    if any(overlap.values()):
        raise ValueError(f"exact train file overlap {overlap}")
    return {"frames":len(rows),"exact_train_file_overlap":overlap,
            "train_motion_name_overlap":sorted({seq.split("/")[-1] for seq,_ in keys} & motions),
            "original_mocap_independence":"unknown"}


@torch.inference_mode()
def evaluate(bundle, anchored, out, mode, verdict):
    p = bundle.pipeline
    pins = {role:v["sha256"] for role,v in PINS["selected"].items()}
    binding = bundle.comparison["binding"]
    if (binding["model"] != pins or binding["eval_seed"] != 71 or binding["clips"] != PINS["clips"]
            or p.cfg["teacher"]["sampling_steps"] != 10 or p.cfg["teacher"]["target_scale_mm"] != 100):
        raise ValueError("comparison binding")
    selected = select_clips(p.data,binding["clips"])
    audit = split_audit(p.data.doc["records"],selected)
    seed_run(p.cfg["seed"])
    frontend = p._make_frontend()
    p._load_best("frontend",frontend)
    identity = FeatureStore(out,p.data,frontend_fingerprint(frontend.model)).identity
    if identity != FEATURE_IDENTITY:
        raise ValueError("A frontend feature store identity")
    clips = {}
    feature_times = {}
    for name,key in selected:
        clips[key],feature_times[name] = timed(lambda: p.data.clip(key,"validation",frontend.model,p.frontend_size,identity))
    first = selected[0][1]
    stages = CommonEvaluation(p.data,{"train":{first:clips[first]},"validation":clips},p.cfg,p.device,0)
    body,hand = stages.make("body"),stages.make("hand",pins["body"])
    p._load_best("body",body)
    p._load_best("hand",hand)
    report = {"schema":"v2a_rootanchored_dualcamera_v1","split":audit,"source":PINS["source"],
              "published_binding":binding,"anchored_sha256":PINS["anchored"],"frontend_feature_identity":identity,
              "camera_origins":{"body":ORIGINS,"hand":"first_midpoint"},"scale_mm":100,
              "device":torch.cuda.get_device_name(),"torch":str(torch.__version__),"clips":{}}
    report.update(mode=mode, published_environment=bundle.diagnostic.get("environment"),
                  python=platform.python_version(), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                  deterministic=torch.are_deterministic_algorithms_enabled(), cudnn_benchmark=torch.backends.cudnn.benchmark,
                  cudnn_deterministic=torch.backends.cudnn.deterministic,
                  cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                  tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32)
    cache = {}
    for name,key in selected:
        seq,start = key
        clip = clips[key]
        frames = list(range(start,start+50))
        meta = [p.data.cache.frame((seq,f),"validation")[1] for f in frames]
        cams = DualCamera().encode(*(torch.stack([m[k] for m in meta]) for k in ("rotation64","translation64")),seq,frames)
        if not torch.equal(cams.first_midpoint,clip.camera[0]) or not torch.equal(cams.absolute,torch.stack([m["camera"] for m in meta])):
            raise ValueError("native cache/clip camera parity")
        if seq == WITNESS["row"]["sequence_id"]:
            check(cams)
        noise = (stages._noise(key,22,100),stages._noise(key,40,200))
        args = (clip.features,cams,clip.time_valid,identity,noise)
        hashes = shared_hashes(args)
        base,baseline_time = timed(lambda: predict("baseline",stages,body,hand,*args))
        local_output = []
        class ObserveLocal:
            def predict(self, *args, **kwargs):
                value = hand.predict(*args, **kwargs)
                local_output.append(value.cpu())
                return value
        _,ref,_ = predict_clip(p,bundle.ledger,frontend,stages,body,ObserveLocal(),key,clip)
        same_local = torch.equal(base["hand_local"],local_output[0][0])
        # Compare independent frontend forward with the normalized xy consumed by both paths.
        xy = torch.from_numpy(ref["predicted_2d_px"])
        official_xy = 2*xy / xy.new_tensor([127,127])-1
        feat_xy = clip.features[0].reshape(50,62,2,7).transpose(1,2)[...,:2]
        reference = compare_predictions(bundle.comparison_path,name,ref)
        stage_deltas = reference_stage_deltas(bundle.comparison_path,name,ref)
        same_xy = torch.equal(feat_xy,official_xy)
        same_xyz = np.array_equal(base["whole"].numpy(),ref["predicted_3d_mm"])
        report["clips"][name] = {"sequence":seq,"frames":frames,"camera_anchor_world_mm":cams.anchor_world_mm,
            **hashes,"evaluation_mask_sha256":digest(clip.target_valid),"published_parity":reference,"published_stage_deltas":stage_deltas,"same_environment_xy_exact":same_xy,
            "same_environment_local_hand_exact":same_local,"same_environment_whole62_exact":same_xyz,"prepare_rgb_cache_and_frontend":feature_times[name],
            "baseline_body_hand":baseline_time}
        write(out/"parity.json",report)
        if hashes != shared_hashes(args):
            raise ValueError("baseline inputs changed")
        cache[name] = args,base
    verdict.update(decide(mode,report["clips"]))
    write(out/"strict_parity.json",{"decision":dict(verdict),"evidence":report})
    write(out/"verdict.json",verdict)
    if not verdict["parity_allows_candidate"]:
        raise ValueError(verdict["reason"])
    verdict["stage"] = "candidate_strict_load"
    cand,cond_hand = mixed_stages(stages,hand,anchored)
    verdict.update(stage="paired_evaluation",candidate_strict_load="PASS")
    rows = []
    for name,key in selected:
        args,base = cache[name]
        record = report["clips"][name]
        if any(record[k] != v for k,v in shared_hashes(args).items()):
            raise ValueError("candidate inputs changed")
        result,latency = timed(lambda: predict("anchored",stages,cand,cond_hand,*args))
        if any(record[k] != v for k,v in shared_hashes(args).items()):
            raise ValueError("shared inputs changed")
        clip = clips[key]
        gt,mask = clip.target_mm[0],clip.target_valid[0]
        arr = {"gt":gt.numpy(),"valid":mask.numpy(),"frames":report["clips"][name]["frames"]}
        for policy,pred in (("baseline",base),("anchored",result)):
            scores = metrics(pred,base["body"],gt,mask,p.data.spec)
            report["clips"][name][policy] = scores
            arr.update({policy+"_"+k:v.numpy() for k,v in pred.items()})
            errors = (pred["body"]-gt[:,:22]).double().norm(dim=-1)
            referrors = (base["body"]-gt[:,:22]).double().norm(dim=-1)
            for i,m in enumerate(mask[:,:22]):
                delta = float((errors[i,m]-referrors[i,m]).mean()) if m.any() else None
                rows.append((name,key[1]+i,policy,delta,None if delta is None else delta>0))
        report["clips"][name]["anchored_body_hand"] = latency
        np.savez_compressed(out/(name+".npz"),**arr)
    with (out/"frames.csv").open("w",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scenario","frame","policy","body_mpjpe_delta_mm","harmed"))
        writer.writerows(rows)
    for pin in PINS["selected"].values():
        with (p.root/pin["name"]).open("rb") as stream:
            if hashlib.file_digest(stream,"sha256").hexdigest() != pin["sha256"]:
                raise ValueError("original checkpoint changed")
    with Path(anchored).open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest() != PINS["anchored"]:
            raise ValueError("candidate checkpoint changed")
    report["checkpoint_readback"] = "exact"
    if mode == "exploratory":
        write(out/"exploratory_comparison.json",report)
    write(out/"results.json",report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg in ("bundle","anchored","work","output"):
        parser.add_argument("--"+arg,required=True,type=Path)
    parser.add_argument("--mode",choices=MODES,default="strict")
    a = parser.parse_args()
    a.output.mkdir(exist_ok=False)
    verdict = dict(mode=a.mode,historical_release="FAIL",strict_release="NOT VERIFIED",
                   candidate_strict_load="NOT VERIFIED",stage="preflight",run_status="incomplete")
    try:
        if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name():
            raise RuntimeError("Tesla T4 required")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        verdict["stage"] = "original_models_and_parity"
        with open_bundle(a.bundle,a.work,seconds=3600,device="cuda") as bundle:
            evaluate(bundle,a.anchored,a.output,a.mode,verdict)
        verdict.update(run_status="completed",stage="done")
    except Exception as error:
        if verdict["stage"] == "candidate_strict_load":
            verdict["candidate_strict_load"] = "FAIL"
        verdict.update(run_status="stopped",error=str(error))
        raise
    finally:
        write(a.output/"verdict.json",verdict)
