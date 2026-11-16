"""Read-only handoff audit of saved files. No model inference or optimizer updates.

Run from the repository with --package <extracted VERIFIED package>
--raw <recover_full_archive.py output> --prior <extracted original input/reference>
--source <source_at_execution.zip> --output <new output directory>.
"""
import argparse, csv, hashlib, io, json, platform, subprocess, sys, zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def independent_metrics(z, edges):
    # Separate implementation of the eight headline metrics and direction angle.
    p, g = z["prediction"].astype(np.float64), z["gt"].astype(np.float64)
    assert z["valid"].all() and np.array_equal(z["frame_ids"], np.arange(len(p)))
    norm = lambda x: np.sqrt(np.sum(x*x, axis=-1))
    pr, gr = p-p[:, :1], g-g[:, :1]
    dp, dg = np.diff(p, axis=0), np.diff(g, axis=0)
    rp, rg = np.diff(pr[:, 1:], axis=0), np.diff(gr[:, 1:], axis=0)
    blp = norm(p[:, edges[:, 1]]-p[:, edges[:, 0]])
    blg = norm(g[:, edges[:, 1]]-g[:, edges[:, 0]])
    sp = np.sqrt(np.mean(norm(p-p.mean(1, keepdims=True))**2, axis=1))
    sg = np.sqrt(np.mean(norm(g-g.mean(1, keepdims=True))**2, axis=1))
    pm, gm = norm(dp), norm(dg)
    mask = (pm > 1e-6) & (gm > 1e-6)
    angle = np.degrees(np.arccos(np.clip((dp*dg).sum(-1)[mask]/(pm*gm)[mask], -1, 1))).mean()
    return {
        "global_joint_error_raw": float(norm(p-g).mean()),
        "root_relative_error_22_raw": float(norm(pr-gr).mean()),
        "frame_displacement_error_raw": float(norm(dp-dg).mean()),
        "motion_magnitude_ratio_pct": float(100*pm.mean()/gm.mean()),
        "root_relative_motion_ratio_pct": float(100*norm(rp).mean()/norm(rg).mean()),
        "bone_length_ratio_pct": float(100*blp.mean()/blg.mean()),
        "joint_spread_ratio_pct": float(100*sp.mean()/sg.mean()),
        "pelvis_position_error_raw": float(norm(p[:, 0]-g[:, 0]).mean()),
        "direction_angle_deg": float(angle),
        "mean_abs_bone_ratio_deviation_pp": float(np.abs(blp/blg-1).mean()*100),
    }

def run(package, raw, prior, source, output):
    import torch
    from experiments.final_m2.dataset_scaling_100.prepare_data import coordinate_check, convert
    from experiments.final_m2.displacement_loss_20261116.plot_results import limits
    from experiments.final_m2.dataset_scaling_100.validate_results import run as original_validator
    from experiments.final_m2.dataset_scaling_100.run_experiment import state_hash

    package, raw, prior, source, output = map(Path, (package, raw, prior, source, output))
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema":"scaling100_handoff_audit_v1",
              "audit_date_kst":"2026-11-16",
              "verification_environment":{"python":platform.python_version(), "numpy":np.__version__, "torch":str(torch.__version__), "device":"CPU"},
              "new_model_inference_calls":0, "new_optimizer_updates":0}
    checks = load(package/"PACKAGE_SHA256.json")
    for n, h in checks.items():
        assert sha(package/n) == h, n
    files = {p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()}
    assert files - {"PACKAGE_SHA256.json"} == set(checks)
    result["package_sha256_verified_files"] = len(checks)
    rawzip = package/"t4_original"/"REWIND_M2_Scaling100_T4_raw.zip"
    assert rawzip.stat().st_size == 12373875
    assert sha(rawzip) == "4ee135ed64654aa563c975fdca748eba102615ccda633b050166b9f93793c024"
    with zipfile.ZipFile(rawzip) as z:
        assert z.testzip() is None
        result["raw_archive"] = {"bytes":rawzip.stat().st_size,"sha256":sha(rawzip),"crc_entries":len(z.infolist())}
    original_checks = load(raw/"checksums.json")
    for n, h in original_checks.items():
        assert sha(raw/n) == h, n
    result["raw_checksummed_files"] = len(original_checks)
    result["missing_raw_files"] = sorted(set(original_checks)-{p.name for p in raw.iterdir()})

    identity = load(raw/"experiment_identity.json")
    with zipfile.ZipFile(source) as z:
        assert z.testzip() is None
        for n, h in identity["code_sha256"].items():
            assert hashlib.sha256(z.read(n)).hexdigest() == h, n
        assert hashlib.sha256(z.read("experiments/final_m2/dataset_scaling_100/config.json")).hexdigest() == identity["config_sha256"]
    result["source_at_execution"] = {"source_commit":identity["source_commit"], "verified_code_files":len(identity["code_sha256"]), "config_sha256":identity["config_sha256"]}
    result["original_execution_environment"] = {k:identity[k] for k in ("device","python","torch","cuda","updates_each","evaluation_steps","evaluation_noise_seed")}

    inp = package/"input"
    for n,h in load(inp/"input_checksums.json").items():
        assert sha(inp/n) == h, n
    assert sha(inp/"input_checksums.json") == identity["input_checksums_sha256"]
    manifest = load(inp/"data_selection_manifest.json")
    frames = manifest["frames"]
    assert len(frames)==100 and len({(r["sequence_id"],r["frame_id"]) for r in frames})==100
    assert len({r["pose_sha256"] for r in frames})==100
    cache = load(package/"cache_manifest.json")
    assert sha(package/"cache_manifest.json")==manifest["cache_manifest_sha256"]
    cache_records = {tuple(r["key"]):r for part in cache["parts"] for r in part["records"]}
    old_manifest = load(prior/"input"/"data_manifest.json")
    old_hashes = {r["member"]:r["sha256"] for r in old_manifest["payloads"]}
    coordinate_maxima = {}
    rows = []
    for i,r in enumerate(frames):
        p=inp/"raw_pose"/r["pose_path"]
        assert sha(p)==r["pose_sha256"] and p.stat().st_size==r["pose_size_bytes"]
        doc=load(p)
        audit=coordinate_check(doc,manifest["joint_names"])
        for k,v in audit.items(): coordinate_maxima[k]=max(coordinate_maxima.get(k,0),v)
        if i<18:
            assert old_hashes[r["pose_path"]]==sha(p)
        else:
            record=cache_records[(r["sequence_id"],r["frame_id"])]
            assert record["split"]=="train" and record["files"]["pose"]["sha256"]==sha(p)
        rows.append((r["sequence_id"],r["frame_id"],doc))
    counts=Counter(r["sequence_id"].split("/")[-1] for r in frames)
    assert sorted(counts.values())==[18,32,50]
    data={n:npz(inp/(n+".npz")) for n in ("train18","train100","standing50","dodging32","opening130")}
    rebuilt=convert(rows,manifest["joint_names"])
    for k,v in rebuilt.items():
        np.testing.assert_array_equal(v,data["train100"][k])
        np.testing.assert_array_equal(data["train18"][k],data["train100"][k][:18])
        np.testing.assert_array_equal(data["standing50"][k],data["train100"][k][18:68])
        np.testing.assert_array_equal(data["dodging32"][k],data["train100"][k][68:])
    for current,old in (("train18","train"),("opening130","holdout")):
        assert sha(inp/(current+".npz"))==sha(prior/"input"/(old+".npz"))
    hold=data["opening130"]
    hold_keys=set(zip(hold["sequence_ids"],hold["frame_ids"]))
    assert not hold_keys & set(zip(data["train100"]["sequence_ids"],data["train100"]["frame_ids"]))
    assert manifest["holdout_sequence"] not in {r["sequence_id"] for r in frames}
    assert not {r["pose_sha256"] for r in frames} & {h for n,h in old_hashes.items() if n.startswith(manifest["holdout_sequence"]+"/")}
    for key in ("features","target_raw"):
        assert not any(np.array_equal(x,y) for x in data["train100"][key][:,-1] for y in hold[key][:,-1])
    result["input_audit"]={"unique_train_frames":100,"motion_counts":dict(counts),"raw_pose_hashes":100,"coordinate_maxima":coordinate_maxima,"reconstructed_tensor_equality":"PASS","historical_train18_and_holdout_file_equality":"PASS","observable_holdout_overlap":0,"original_mocap_independence":"NOT VERIFIED"}

    # Existing validator rechecks all 36 metrics, all 8 checkpoints and full traces.
    original_validator(raw,prior,output/"existing_validator")
    validation=load(output/"existing_validator"/"result_validation.json")
    result["existing_validator"]={k:validation[k] for k in ("execution_observed","opening_array_recomputation","prior_control400_direct_array_comparison","prior_control400_opening_prediction_bitwise_identical","full_random_traces_local_check","complete_artifact_recovery","quality")}
    result["verified_prediction_groups"]=validation["verified_prediction_groups"]
    result["checkpoint_verification"]=validation["checkpoint_verification"]

    evaluation=load(raw/"evaluation.json")
    edges=np.array(manifest["edges"])
    independently_recomputed=[]
    csvrows=list(csv.DictReader((raw/"comparison.csv").open(encoding="utf-8",newline="")))
    for r in evaluation["results"]:
        z=npz(raw/f"{r['split']}_{r['model']}_predictions.npz")
        assert z["prediction"].shape==(len(z["frame_ids"]),22,3)
        for k, inputkey in (("gt","target_raw"),("valid","valid")):
            np.testing.assert_array_equal(z[k],data[r["split"]][inputkey][:,-1])
        np.testing.assert_array_equal(z["frame_ids"],data[r["split"]]["frame_ids"])
        metrics=independent_metrics(z,edges)
        cr=next(x for x in csvrows if x["split"]==r["split"] and x["model"]==r["model"])
        for k,v in metrics.items():
            np.testing.assert_allclose(v,r[k],rtol=1e-12,atol=1e-12)
            if k in cr: np.testing.assert_allclose(v,float(cr[k]),rtol=1e-12,atol=1e-12)
        independently_recomputed.append({"split":r["split"],"model":r["model"],**metrics})
    result["independent_metrics"]=independently_recomputed

    nb=load(package/"REWIND_M2_Scaling100_executed.ipynb")
    streams="\n".join("".join(o.get("text",[])) for c in nb["cells"] for o in c.get("outputs",[]) if o.get("output_type")=="stream")
    errors=[{"cell":i,"type":o.get("ename"),"traceback":"\n".join(o.get("traceback",[]))} for i,c in enumerate(nb["cells"]) for o in c.get("outputs",[]) if o.get("output_type")=="error"]
    assert len(errors)==1 and errors[0]["cell"]==2 and errors[0]["type"]=="AssertionError"
    preflight="".join("".join(o.get("text",[])) for o in nb["cells"][2].get("outputs",[]) if o.get("output_type")=="stream")
    assert "No module named 'tests.test_m2'" in preflight and "No module named 'tests.test_final_metrics'" in preflight
    completed="".join("".join(o.get("text",[])) for o in nb["cells"][3].get("outputs",[]) if o.get("output_type")=="stream")
    assert "RETURN_CODE 0" in completed and '"execution_status": "PASS"' in completed
    assert "GPU: Tesla T4" in streams
    result["notebook_execution_audit"]={"initial_preflight_import_failure_preserved":True,"failed_cell_zero_based":2,"completed_wrapper_cell_zero_based":3,"completed_wrapper_return_code":0,"interpretation":"Initial unittest import failed before training; discover wrapper later passed and completed both arms."}
    observed={}
    for line in streams.splitlines():
        try: obj=json.loads(line)
        except ValueError: continue
        if isinstance(obj,dict) and {"model","step","loss"}<=obj.keys():
            observed[(obj["model"],obj["step"])]=obj
    training=[]
    for arm in ("train18","train100"):
        log=list(csv.DictReader((raw/f"{arm}_training.csv").open(encoding="utf-8",newline="")))
        jsonl=[json.loads(x) for x in (raw/f"{arm}_training.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(log)==len(jsonl)==400
        assert [int(r["step"]) for r in log]==list(range(1,401))
        for r,j in zip(log,jsonl):
            for k,v in j.items():
                if isinstance(v,(int,float)):
                    assert np.isfinite(v)
                    assert float(r[k])==v
                else: assert r[k]==v
        for step in (100,200,300,400):
            saved=observed[(arm,step)]
            assert saved==jsonl[step-1]
            ck=torch.load(raw/f"{arm}_{step}.pt",map_location="cpu",weights_only=True)
            assert ck["samples_seen"]==step*8 and ck["arm"]==arm
            assert all(torch.isfinite(t).all() for t in ck["model"].values())
            assert ck["rng"] and ck["experiment_identity"]==identity
        arm_record=next(r for r in evaluation["arms"] if r["model"]==arm)
        trace=load(raw/f"{arm}_random_trace.json")
        counts=np.bincount([i for row in trace for i in row["anchors"]],minlength=len(data[arm]["frame_ids"]))
        assert counts.tolist()==arm_record["anchor_sample_counts"]
        assert state_hash(ck["rng"])==arm_record["end_rng_sha256"]
        training.append({"model":arm,"csv_rows":len(log),"jsonl_rows":len(jsonl),"notebook_checkpoints_matched":4})
    result["training_log_audit"]=training
    figures=package/"experiment"/"figures"
    image_audit={}
    for p in figures.iterdir():
        if p.suffix.lower() not in (".png",".gif"):continue
        with Image.open(p) as im:
            n=getattr(im,"n_frames",1)
            durations=[]
            for i in range(n):
                im.seek(i);im.load()
                if p.suffix==".gif":durations.append(im.info.get("duration"))
            if p.suffix==".gif": assert n==130 and set(durations)=={100}
            image_audit[p.name]={"frames":n,"size":list(im.size),"decoded":"PASS"}
    poses=[npz(raw/"opening130_train18_predictions.npz")["gt"]]+[npz(raw/f"opening130_{a}_predictions.npz")["prediction"] for a in ("train18","train100")]
    axes=load(package/"experiment"/"metrics"/"t4"/"visualization_axes.json")
    for name,ps in (("skeleton_visualization.png",poses),("root_relative_comparison.png",[p-p[:,:1] for p in poses])):
        np.testing.assert_allclose(limits(ps),axes[name],atol=1e-12,rtol=1e-12)
        spans=np.diff(np.asarray(axes[name]),axis=1).ravel()
        np.testing.assert_allclose(spans,spans[0],rtol=1e-6)
    result["figure_decode"]=image_audit
    result["shared_axes_recomputed"]="PASS; original plot code shares limits and 1:1:1 box aspect"
    result["limits"]=["Single seed; no statistical significance", "Historical holdout repeatedly examined", "GT Stereo2D and zero camera, not end-to-end RGB", "Physical distance unit and capture FPS unverified", "No fresh GPU training or inference; saved execution evidence audited"]
    (output/"handoff_verification.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","quality":result["existing_validator"]["quality"]["status"],"raw_files":len(original_checks),"predictions":len(independently_recomputed),"checkpoint_count":len(result["checkpoint_verification"]),"control_direct_arrays":"PASS","new_optimizer_updates":0}))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("package","raw","prior","source","output"):p.add_argument("--"+name,required=True)
    a=p.parse_args();run(a.package,a.raw,a.prior,a.source,a.output)
