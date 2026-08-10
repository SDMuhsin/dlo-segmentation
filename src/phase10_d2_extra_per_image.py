"""Phase 10 D2-extra — per-image (not per-channel-mean) feature distribution
statistics, to look for per-image OOD drift that the channel-mean test misses.

For each (model, image_set, stage), compute the per-image scalar global mean
of the stage's activation tensor (averaged over all positions and channels).
Then compute Mahalanobis-distance-equivalent z over the per-image distribution.

Also: cosine similarity between per-image mean-activation vectors of synth and
real images at the final stage (stage3) — this gives a single number per
(model) summarising how alike real and synth look to that model.

This is FAST: we reuse the same hook architecture as D2 but reduce per-image
to a scalar (B, C, H, W) -> (B,) and (B, C, H, W) -> (B, C).

Outputs:
  D2/per_image_drift.json
  D2/per_image_feat_table.csv
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
D1_DIR = os.path.join(PROJECT_ROOT, "results", "phase10_investigation", "D1")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "phase10_investigation", "D2")

VIDEOS = {
    "sample_1": os.path.join(PROJECT_ROOT, "data", "dlo_real_sample_1.mp4"),
    "sample_2": os.path.join(PROJECT_ROOT, "data", "dlo_real_sample_2.mp4"),
    "sample_3": os.path.join(PROJECT_ROOT, "data", "dlo_real_sample_3.mp4"),
    "sample_4": os.path.join(PROJECT_ROOT, "data", "dlo_real_sample_4.mp4"),
}

N_SYNTH = 100
SEED = 12345


class PerImageStats:
    """Capture per-image per-channel mean activation: (B, C) for each image."""
    def __init__(self):
        self.per_image_channel_means = []  # list of (C,) np arrays

    def add(self, feat):
        with torch.no_grad():
            B, C, H, W = feat.shape
            # average over H,W -> (B, C)
            x = feat.reshape(B, C, -1).mean(dim=2)  # (B, C)
            arr = x.double().cpu().numpy()
            for i in range(B):
                self.per_image_channel_means.append(arr[i])

    def finalize(self):
        # stack into (N, C)
        return np.stack(self.per_image_channel_means, axis=0)


def attach_hooks(model, stats_list):
    encoder = model.model.segformer.encoder
    handles = []
    for stage_i, ln in enumerate(encoder.layer_norm):
        def make_hook(idx):
            def hook(_module, _inp, out):
                t = out
                if t.dim() == 3:
                    expected = [(120, 160), (60, 80), (30, 40), (15, 20)][idx]
                    B = t.shape[0]
                    C = t.shape[2]
                    H, W = expected
                    t = t.permute(0, 2, 1).contiguous().view(B, C, H, W)
                stats_list[idx].add(t)
            return hook
        handles.append(ln.register_forward_hook(make_hook(stage_i)))
    return handles


def detach_hooks(handles):
    for h in handles:
        h.remove()


def synth_train_frames(root, n, seed):
    rng = random.Random(seed)
    set_dirs = sorted([s for s in os.listdir(root) if os.path.isdir(os.path.join(root, s))])
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
    with open(os.path.join(D1_DIR, "sample_indices.json")) as f:
        si = json.load(f)
    frames = []
    for v, idxs in si.items():
        cap = cv2.VideoCapture(VIDEOS[v])
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if ok:
                frames.append(preprocess(f))
        cap.release()
    return frames


def normalize_to_tensor(bgr_list, device):
    arrs = np.stack(bgr_list, axis=0)[..., ::-1].copy()  # BGR -> RGB
    t = torch.from_numpy(arrs.transpose(0, 3, 1, 2)).to(device, dtype=torch.float32) / 255.0
    t = (t - RGB_MEAN.to(device)) / RGB_STD.to(device)
    return t


def run(model, frames, device, batch_size=4):
    stats = [PerImageStats() for _ in range(4)]
    handles = attach_hooks(model, stats)
    model.eval()
    try:
        for i in range(0, len(frames), batch_size):
            chunk = frames[i:i + batch_size]
            x = normalize_to_tensor(chunk, device)
            with torch.no_grad():
                _ = model(x)
    finally:
        detach_hooks(handles)
    return [s.finalize() for s in stats]  # list of (N, C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Sampling synth+real frames...")
    p4_paths = synth_train_frames(os.path.join(PROJECT_ROOT, "data", "rgbd_videos", "train"),
                                  N_SYNTH, SEED)
    p9_paths = synth_train_frames(os.path.join(PROJECT_ROOT, "data", "rgbd_videos_phase9", "train"),
                                  N_SYNTH, SEED + 1)
    p4_frames = [cv2.imread(p, cv2.IMREAD_COLOR) for p in p4_paths]
    p9_frames = [cv2.imread(p, cv2.IMREAD_COLOR) for p in p9_paths]
    real = real_frames()
    image_sets = {"phase4_synth": p4_frames, "phase9_synth": p9_frames, "real": real}

    artefacts = {}  # (model_tag, stage_i, set_tag) -> (N, C) array
    for model_tag, ckpt in [("p7", CKPT_P7), ("p9_dlow3", CKPT_P9)]:
        print(f"=== {model_tag} from {ckpt} ===")
        model = load_model(ckpt, device)
        for set_tag, frames in image_sets.items():
            print(f"  {set_tag} ({len(frames)} frames)")
            stages = run(model, frames, device, batch_size=args.batch_size)
            for stage_i, arr in enumerate(stages):
                artefacts[(model_tag, stage_i, set_tag)] = arr
                # Save the (N, C) array
                np.savez(os.path.join(OUT_DIR, f"{model_tag}_stage{stage_i}_{set_tag}_per_image.npz"),
                         per_image_channel_mean=arr)
                print(f"    saved {model_tag}_stage{stage_i}_{set_tag}_per_image.npz  shape={arr.shape}")
        del model
        torch.cuda.empty_cache()

    # Per-image drift: for each (model, stage), compute synth (N, C) and real (N, C),
    # then per-image global mean (M, ) = mean over C; build histogram-style:
    # how many REAL images have global mean falling in [synth_mean - 3*synth_std, synth_mean + 3*synth_std]?
    summary = {}
    csv_rows = [["model", "stage", "n_real", "n_synth_ref",
                 "real_global_mean_mean", "real_global_mean_std",
                 "synth_global_mean_mean", "synth_global_mean_std",
                 "frac_real_outside_3sigma_synth_globalmean",
                 "cosine_real_vs_synth_centroid"]]
    for model_tag in ["p7", "p9_dlow3"]:
        ref_set = "phase4_synth" if model_tag == "p7" else "phase9_synth"
        summary[model_tag] = {"reference_set": ref_set, "stages": {}}
        for stage_i in range(4):
            synth = artefacts[(model_tag, stage_i, ref_set)]  # (Nsynth, C)
            real = artefacts[(model_tag, stage_i, "real")]   # (Nreal, C)

            # Per-image global mean (= mean over channels) — coarsest summary
            synth_gm = synth.mean(axis=1)
            real_gm = real.mean(axis=1)
            sg_mu, sg_sd = float(synth_gm.mean()), float(synth_gm.std())
            rg_mu, rg_sd = float(real_gm.mean()), float(real_gm.std())
            outside_3 = float(((real_gm < sg_mu - 3 * sg_sd) | (real_gm > sg_mu + 3 * sg_sd)).mean())

            # Cosine similarity between centroids (mean over images of per-channel means)
            synth_c = synth.mean(axis=0)
            real_c = real.mean(axis=0)
            cos = float(np.dot(synth_c, real_c) /
                        (np.linalg.norm(synth_c) * np.linalg.norm(real_c) + 1e-12))

            # Per-image average activation magnitude (||z||_2 / sqrt(C)) — distribution-level OOD
            # For each real image, compute Mahalanobis-like z: ((real_i - synth_mean) / synth_std)
            synth_mu = synth.mean(axis=0)  # (C,)
            synth_sd = synth.std(axis=0)  # (C,)
            synth_sd_safe = np.where(synth_sd > 1e-12, synth_sd, 1.0)
            z_real = (real - synth_mu[None, :]) / synth_sd_safe[None, :]  # (Nreal, C)
            real_max_absz_per_image = np.abs(z_real).max(axis=1)  # (Nreal,)
            real_p95_absz_per_image = np.percentile(np.abs(z_real), 95, axis=1)  # (Nreal,)
            frac_real_images_max_absz_gt_3 = float((real_max_absz_per_image > 3).mean())
            frac_real_images_p95_absz_gt_3 = float((real_p95_absz_per_image > 3).mean())

            row = {
                "n_real": int(real.shape[0]),
                "n_synth_ref": int(synth.shape[0]),
                "real_global_mean_mean": rg_mu,
                "real_global_mean_std": rg_sd,
                "synth_global_mean_mean": sg_mu,
                "synth_global_mean_std": sg_sd,
                "frac_real_outside_3sigma_synth_globalmean": outside_3,
                "cosine_real_vs_synth_centroid": cos,
                "frac_real_images_max_absz_gt_3": frac_real_images_max_absz_gt_3,
                "frac_real_images_p95_absz_gt_3": frac_real_images_p95_absz_gt_3,
                "real_max_absz_per_image_p50": float(np.percentile(real_max_absz_per_image, 50)),
                "real_max_absz_per_image_p95": float(np.percentile(real_max_absz_per_image, 95)),
                "real_max_absz_per_image_max": float(real_max_absz_per_image.max()),
            }
            summary[model_tag]["stages"][f"stage{stage_i}"] = row
            csv_rows.append([model_tag, stage_i, row["n_real"], row["n_synth_ref"],
                             rg_mu, rg_sd, sg_mu, sg_sd,
                             outside_3, cos])

    with open(os.path.join(OUT_DIR, "per_image_drift.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT_DIR, "per_image_feat_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        for r in csv_rows:
            w.writerow(r)

    print("\nPer-image drift (per real image, max |z| across channels using synth refs):")
    for model_tag in ["p7", "p9_dlow3"]:
        ref = summary[model_tag]["reference_set"]
        for s in ["stage0", "stage1", "stage2", "stage3"]:
            r = summary[model_tag]["stages"][s]
            print(f"  {model_tag:10s}  ref={ref:14s}  {s}:  cos={r['cosine_real_vs_synth_centroid']:.4f}  "
                  f"frac_real_imgs_max|z|>3={r['frac_real_images_max_absz_gt_3']:.3f}  "
                  f"frac_p95|z|>3={r['frac_real_images_p95_absz_gt_3']:.3f}  "
                  f"max|z|_p95={r['real_max_absz_per_image_p95']:.2f}")


if __name__ == "__main__":
    main()
