"""기존 Train300 예측 오차 및 동일 noise의 stereo x 좌표 이동 진단."""
import numpy as np
import torch
from rewind_m2.models import BodyDenoiser
from rewind_m2.streaming import AnchorSampler, keyed_frame_noise
from tools.final_m2_real import infer_frame, load, sha, write
from experiments.final_m2.dataset_scaling_300.prepare_data import read_npz
from experiments.final_m2.displacement_loss_20261116.run_experiment import state_hash


def measure(ck, schedule, inputs, folder, out, hashes):
    model = BodyDenoiser(ck['config'], 22, ck['manifest']['edges']).to('cuda')
    model.load_state_dict(ck['model'])
    model.eval()
    results = {}
    for split in ('opening130', 'gunplay47'):
        name = f'{split}_train300_predictions.npz'
        if sha(folder/name) != hashes[name]:
            raise ValueError(name)
        z = read_npz(folder/name)
        d = inputs[split]
        for key in ('sequence_ids', 'frame_ids'):
            np.testing.assert_array_equal(d[key], z[key])
        p, g, v = z['prediction'], z['gt'], z['valid']
        np.testing.assert_array_equal(g, d['target_raw'][:, -1])
        np.testing.assert_array_equal(v, d['valid'][:, -1])
        errors = np.linalg.norm(p-g, axis=-1)
        rr = np.linalg.norm((p-p[:,:1])-(g-g[:,:1]), axis=-1)
        ratios = []
        for a,b in ck['manifest']['edges']:
            mask = v[:,a] & v[:,b]
            lengths = np.linalg.norm(g[:,a]-g[:,b], axis=-1)
            values = 100*np.linalg.norm(p[mask,a]-p[mask,b], axis=-1)/lengths[mask]
            ratios.append(dict(edge=[a,b], mean_pct=float(values.mean()), std_pp=float(values.std())))
        outputs = []
        traces = []
        with torch.inference_mode():
            for offset in (0., 32./1024):
                sampler = AnchorSampler(model,schedule,8,10)
                preds, noise = [], []
                for x, seq, f in zip(d['features'][:,-1],d['sequence_ids'],d['frame_ids']):
                    x = x.copy().reshape(22,14)
                    for start in (0,7):
                        x[:,start] += offset*x[:,start+2]
                    noise.append(keyed_frame_noise(str(seq),int(f),22,10043))
                    preds.append(infer_frame(sampler,x.ravel(),str(seq),int(f),10043,'cuda').cpu().numpy())
                outputs.append(np.stack(preds))
                traces.append(state_hash(noise))
        if traces[0]!=traces[1]:
            raise ValueError('diagnostic_noise')
        drift = float(np.abs(outputs[0]-p).max())
        if drift >= .001:
            raise ValueError(f'{split} baseline reproduction max raw error {drift}')
        results[split] = dict(prediction_sha256=hashes[name], joint_error_raw={
            n:float(errors[v[:,j],j].mean()) for j,n in enumerate(ck['manifest']['joint_names'])},
            pelvis_relative_error_raw=float(rr[v & v[:,:1]].mean()), edges=ratios,
            baseline_max_difference_raw=drift, noise_sha256=traces,
            shift32px_output_change_raw=float(np.linalg.norm(outputs[1]-outputs[0],axis=-1)[v].mean()))
    write(out/'parent_diagnostics.json', results)
