"""Phase 10 D2 — Layer-wise feature statistics on synth-train vs real frames.

For each model in {P7, P9-dlow3} and image set in {phase4_synth, phase9_synth,
real}, hook the 4 encoder layer_norm outputs and accumulate per-channel mean
and std. Then compute per-channel z-score of real-image activations using
that model's training-distribution synth statistics as reference.

Outputs:
  D2/<model>_<stage>_<set>.npz       per_channel_mean, per_channel_std,
                                      global_mean, global_std
  D2/feature_drift_summary.json
  D2/feature_drift_table.csv
"""
import argparse
import csv
import json
import os
import random
import sys

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE = os.path.join(PROJECT_ROOT, "data", "hf_cache")
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_rgb_only_sota_gifs import load_model  # noqa: E402
from infer_video_rgb_only import preprocess  # noqa: E402
from train_rgb_only_sota import RGB_MEAN, RGB_STD  # noqa: E402

CKPT_P7 = os.path.join(PROJECT_ROOT, "results", "segformer_b5_rgb", "full_20260430_2032", "best_model.pth")
CKPT_P9 = os.path.join(PROJECT_ROOT, "results", "segformer_b5_rgb_phase9", "full_dlow3_20260515_1330", "best_model.pth")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "phase10_investigation", "D2")
D1_DIR = os.path.join(PROJECT_ROOT, "results", "phase10_investigation", "D1")

VIDEOS = {
    "sample_1": os.path.join(PROJECT_ROOT, "data", "dlo_real_sample_1.mp4"),
    "sample_2": os.path.join(PROJECT_ROOT, "data", "dlo_real_sample_2.mp4"),
    "sample_3": os.path.join(PROJECT_ROOT, "data", "dlo_real_sample_3.mp4"),
    "sample_4": os.path.join(PROJECT_ROOT, "data", "dlo_real_sample_4.mp4"),
}

N_SYNTH_SAMPLES = 100
SEED = 12345


class StagedAccumulator:
    """Per-stage Welford-like running stats: per-channel mean and std over a
    set of images, with each image contributing H*W positions to that
    channel's accumulator.
    """

    def __init__(self):
        self.count = 0   # total spatial positions accumulated
        self.sum = None  # (C,) running sum
        self.sumsq = None  # (C,) running sum of squares

    def add(self, feat):
        """feat: (B, C, H, W) torch tensor on GPU; reduce over B,H,W."""
        with torch.no_grad():
            B, C, H, W = feat.shape
            x = feat.reshape(B, C, -1)  # (B, C, H*W)
            s = x.sum(dim=(0, 2)).double().cpu().numpy()
            sq = (x.double() ** 2).sum(dim=(0, 2)).cpu().numpy()
            n = B * H * W
            if self.sum is None:
                self.sum = s
                self.sumsq = sq
            else:
                self.sum += s
                self.sumsq += sq
            self.count += n

    def finalize(self):
        mean = self.sum / self.count
        var = self.sumsq / self.count - mean ** 2
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        return mean.astype(np.float64), std.astype(np.float64)


def attach_hooks(model, accumulators):
    """Attach forward hooks on the 4 encoder layer_norm modules. Returns the hook
    handles (caller releases them).
    """
    encoder = model.model.segformer.encoder
    handles = []
    for stage_i, ln in enumerate(encoder.layer_norm):
        def make_hook(idx):
            def hook(_module, _inp, out):
                # The HF SegFormer encoder applies layer_norm on a flattened
                # (B, H*W, C) sequence; check the rank.
                t = out
                if t.dim() == 3:
                    # (B, H*W, C) -> need H, W. The encoder caches this; for our
                    # 480x640 input the four stages are 120x160, 60x80, 30x40, 15x20.
                    expected = [(120, 160), (60, 80), (30, 40), (15, 20)][idx]
                    B = t.shape[0]
                    C = t.shape[2]
                    H, W = expected
                    assert t.shape[1] == H * W, f"stage {idx}: got HxW={t.shape[1]} expected {H*W}"
                    t = t.permute(0, 2, 1).contiguous().view(B, C, H, W)
                accumulators[idx].add(t)
            return hook
        handles.append(ln.register_forward_hook(make_hook(stage_i)))
    return handles


def detach_hooks(handles):
    for h in handles:
        h.remove()


def synth_train_frames(root, n, seed):
    """Pick `n` RGB PNGs uniformly at random from `root/<set>/rgb/*.png`."""
    rng = random.Random(seed)
    set_dirs = sorted(os.listdir(root))
    set_dirs = [s for s in set_dirs if os.path.isdir(os.path.join(root, s))]
    paths = []
    for s in set_dirs:
        rd = os.path.join(root, s, "rgb")
        if not os.path.isdir(rd):
            continue
        for f in os.listdir(rd):
            if f.endswith(".png"):
                paths.append(os.path.join(rd, f))
    rng.shuffle(paths)
    return paths[:n]


def real_frames():
    """Use the SAME 80 real frames as D1 if sample_indices.json exists, else fall
    back to 20 evenly-spaced per video."""
    si = os.path.join(D1_DIR, "sample_indices.json")
    if os.path.exists(si):
        with open(si) as f:
            indices = json.load(f)
    else:
        indices = {}
        for v in VIDEOS:
            cap = cv2.VideoCapture(VIDEOS[v])
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices[v] = np.linspace(0, n - 1, 20, dtype=int).tolist()
            cap.release()
    # Load frames
    frames = []
    for v, idxs in indices.items():
        cap = cv2.VideoCapture(VIDEOS[v])
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if not ok:
                continue
            frames.append(preprocess(f))
        cap.release()
    return frames


def normalize_to_tensor(bgr_uint8_list, device):
    """Stack a list of (H,W,3) BGR uint8 frames into (B,3,H,W) RGB float tensor."""
    arrs = np.stack(bgr_uint8_list, axis=0)  # (B, H, W, 3) BGR
    arrs = arrs[..., ::-1].copy()  # BGR -> RGB
    t = torch.from_numpy(arrs.transpose(0, 3, 1, 2)).to(device, dtype=torch.float32) / 255.0
    t = (t - RGB_MEAN.to(device)) / RGB_STD.to(device)
    return t


def run_pass(model, frames_bgr, device, batch_size=4):
    """Run model.forward in eval mode with hooks attached; returns 4 accumulators."""
    accs = [StagedAccumulator() for _ in range(4)]
    handles = attach_hooks(model, accs)
    model.eval()
    try:
        for i in range(0, len(frames_bgr), batch_size):
            chunk = frames_bgr[i:i + batch_size]
            x = normalize_to_tensor(chunk, device)
            with torch.no_grad():
                _ = model(x)
            if (i // batch_size + 1) % 10 == 0:
                print(f"    pass {i + len(chunk)}/{len(frames_bgr)} done")
    finally:
        detach_hooks(handles)
    finals = [a.finalize() for a in accs]
    return finals  # list of (mean, std) per stage


def save_npz(out_dir, model_tag, stage_i, set_tag, mean, std):
    path = os.path.join(out_dir, f"{model_tag}_stage{stage_i}_{set_tag}.npz")
    g_mean = float(mean.mean())
    g_std = float(np.sqrt(((std ** 2) + (mean - mean.mean()) ** 2).mean()))
    np.savez(path,
             per_channel_mean=mean.astype(np.float64),
             per_channel_std=std.astype(np.float64),
             global_mean=np.float64(g_mean),
             global_std=np.float64(g_std))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Sampling synth frames...")
    p4_paths = synth_train_frames(os.path.join(PROJECT_ROOT, "data", "rgbd_videos", "train"),
                                  N_SYNTH_SAMPLES, SEED)
    p9_paths = synth_train_frames(os.path.join(PROJECT_ROOT, "data", "rgbd_videos_phase9", "train"),
                                  N_SYNTH_SAMPLES, SEED + 1)
    print(f"  phase4_synth: {len(p4_paths)} paths (e.g. {os.path.relpath(p4_paths[0], PROJECT_ROOT)})")
    print(f"  phase9_synth: {len(p9_paths)} paths (e.g. {os.path.relpath(p9_paths[0], PROJECT_ROOT)})")

    p4_frames = [cv2.imread(p, cv2.IMREAD_COLOR) for p in p4_paths]
    p9_frames = [cv2.imread(p, cv2.IMREAD_COLOR) for p in p9_paths]
    assert all(f is not None and f.shape == (480, 640, 3) for f in p4_frames), \
        "p4 frames not all 480x640"
    assert all(f is not None and f.shape == (480, 640, 3) for f in p9_frames), \
        "p9 frames not all 480x640"

    print("Loading real frames...")
    real = real_frames()
    print(f"  real: {len(real)} frames")

    image_sets = {
        "phase4_synth": p4_frames,
        "phase9_synth": p9_frames,
        "real": real,
    }

    # For each model, run all 3 image sets, hook the 4 stages.
    artefacts = {}  # (model_tag, stage_i, set_tag) -> (mean, std, npz_path)
    for model_tag, ckpt in [("p7", CKPT_P7), ("p9_dlow3", CKPT_P9)]:
        print(f"\n=== Loading model {model_tag} from {ckpt} ===")
        model = load_model(ckpt, device)
        for set_tag, frames in image_sets.items():
            print(f"  forward pass on {set_tag} ({len(frames)} frames)")
            stages = run_pass(model, frames, device, batch_size=args.batch_size)
            for stage_i, (m, s) in enumerate(stages):
                path = save_npz(OUT_DIR, model_tag, stage_i, set_tag, m, s)
                artefacts[(model_tag, stage_i, set_tag)] = (m, s, path)
                print(f"    saved {os.path.relpath(path, PROJECT_ROOT)}  C={m.shape[0]}  "
                      f"mean_global={m.mean():.4f}  std_global={s.mean():.4f}")
        del model
        torch.cuda.empty_cache()

    # Compute drift: for each model, real vs training-synth z-scores per channel.
    drift_summary = {}
    csv_rows = [["model", "stage", "channels", "max_abs_z", "p95_abs_z", "mean_abs_z",
                 "frac_channels_abs_z_gt_3"]]
    for model_tag in ["p7", "p9_dlow3"]:
        ref_set = "phase4_synth" if model_tag == "p7" else "phase9_synth"
        drift_summary[model_tag] = {"reference_set": ref_set, "stages": {}}
        for stage_i in range(4):
            mu_ref, sd_ref = artefacts[(model_tag, stage_i, ref_set)][:2]
            mu_real, _ = artefacts[(model_tag, stage_i, "real")][:2]
            # Avoid division by zero
            sd_safe = np.where(sd_ref > 1e-12, sd_ref, 1.0)
            z = (mu_real - mu_ref) / sd_safe
            absz = np.abs(z)
            row = {
                "channels": int(absz.size),
                "max_abs_z": float(absz.max()),
                "p95_abs_z": float(np.percentile(absz, 95)),
                "mean_abs_z": float(absz.mean()),
                "frac_channels_abs_z_gt_3": float((absz > 3).mean()),
            }
            drift_summary[model_tag]["stages"][f"stage{stage_i}"] = row
            csv_rows.append([model_tag, stage_i, row["channels"], row["max_abs_z"],
                             row["p95_abs_z"], row["mean_abs_z"], row["frac_channels_abs_z_gt_3"]])

    # Also: compare P7 reference vs P9 reference on the same encoder? Not meaningful
    # (different encoders). What IS meaningful: P9-dlow3 real-vs-phase4_synth, to
    # show whether P9 has shifted its encoder away from "natural" synth too. Bonus.
    bonus = {}
    for model_tag in ["p7", "p9_dlow3"]:
        bonus[model_tag] = {"p4_synth_as_ref": {}}
        for stage_i in range(4):
            mu_ref, sd_ref = artefacts[(model_tag, stage_i, "phase4_synth")][:2]
            mu_real, _ = artefacts[(model_tag, stage_i, "real")][:2]
            sd_safe = np.where(sd_ref > 1e-12, sd_ref, 1.0)
            z = (mu_real - mu_ref) / sd_safe
            absz = np.abs(z)
            bonus[model_tag]["p4_synth_as_ref"][f"stage{stage_i}"] = {
                "max_abs_z": float(absz.max()),
                "p95_abs_z": float(np.percentile(absz, 95)),
                "mean_abs_z": float(absz.mean()),
                "frac_channels_abs_z_gt_3": float((absz > 3).mean()),
            }
    drift_summary["bonus_real_vs_phase4_synth"] = bonus

    with open(os.path.join(OUT_DIR, "feature_drift_summary.json"), "w") as f:
        json.dump(drift_summary, f, indent=2)
    with open(os.path.join(OUT_DIR, "feature_drift_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        for r in csv_rows:
            w.writerow(r)

    print("\nFeature drift summary (model | stage | frac(|z|>3) | max|z| | p95|z|):")
    for r in csv_rows[1:]:
        print(f"  {r[0]:10s}  stage{r[1]}  frac(|z|>3)={r[6]:.4f}  max|z|={r[3]:.3f}  p95|z|={r[4]:.3f}")
    print(f"\nAll D2 artefacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
