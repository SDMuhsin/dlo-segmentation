"""Phase 10 ablation gate evaluator.

For each (variant, checkpoint) pair:
  - run inference on the 80 D1 frames (20 per video, 4 videos) at the indices
    saved in results/phase10_investigation/D1/sample_indices.json,
  - capture float32 P(DLO) softmax,
  - compute gate (a): mean P(DLO) on the 80 frames / Phase 7 mean,
  - compute gate (b): at pixels where Phase 7's argmax is DLO, fraction with
    variant softmax > 0.01,
  - write per-variant gate.json under results/phase10_ablation/<variant>/.

Phase 7's gate(a) = 1.00 / gate(b) = 1.00 by construction.
The reference P7 mean P(DLO) per video is taken from
results/phase10_mechanism/M4/soft_metric_comparison.json (P7.<vid>.
mean_per_frame_mean_pDLO). The P7 argmax masks are at
results/phase10_investigation/D1/<vid>/frame_<idx>_p7.png (uint8 0/255).

Usage:
    python src/phase10_ablation_gate.py --ckpt <path> --tag <name> [--video ALL]
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen_rgb_only_sota_gifs import load_model  # noqa: E402
from train_rgb_only_sota import RGB_MEAN, RGB_STD  # noqa: E402
from infer_video_rgb_only import preprocess  # noqa: E402


VIDEOS = ["sample_1", "sample_2", "sample_3", "sample_4"]
SAMPLE_INDICES_JSON = os.path.join(
    PROJECT_ROOT,
    "results", "phase10_investigation", "D1", "sample_indices.json",
)
D1_DIR = os.path.join(PROJECT_ROOT, "results", "phase10_investigation", "D1")
M4_JSON = os.path.join(
    PROJECT_ROOT, "results", "phase10_mechanism", "M4",
    "soft_metric_comparison.json",
)


def _video_path(name):
    return os.path.join(PROJECT_ROOT, "data", f"dlo_real_{name}.mp4")


@torch.no_grad()
def predict_softmax(model, bgr_uint8, device):
    """Run model on a preprocessed BGR frame and return float32 P(DLO) (H, W)."""
    rgb = bgr_uint8[:, :, ::-1].copy()
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(
        device, dtype=torch.float32) / 255.0
    t = (t - RGB_MEAN.to(device)) / RGB_STD.to(device)
    logits = model(t)
    probs = F.softmax(logits, dim=1)
    p_dlo = probs[0, 1].cpu().numpy().astype(np.float32)
    return p_dlo


def load_d1_p7_mask(video, frame_idx):
    path = os.path.join(D1_DIR, video, f"frame_{frame_idx:05d}_p7.png")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"P7 mask missing: {path}")
    if img.ndim == 3:
        img = img[..., 0]
    return (img > 0).astype(np.uint8)


def load_p7_mean_pDLO():
    with open(M4_JSON) as f:
        m = json.load(f)
    out = {}
    for v in VIDEOS:
        out[v] = float(m["per_video"]["P7"][v]["mean_per_frame_mean_pDLO"])
    return out


def evaluate_variant(ckpt_path, out_dir, backbone="nvidia/mit-b5"):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gate] device={device}, ckpt={ckpt_path}")
    model = load_model(ckpt_path, device, backbone=backbone)

    with open(SAMPLE_INDICES_JSON) as f:
        indices = json.load(f)
    p7_mean = load_p7_mean_pDLO()

    per_video = {}
    overall_p_dlo = []
    overall_frac_pos = []

    for video in VIDEOS:
        idx_list = sorted(indices[video])
        target_set = set(idx_list)
        cap = cv2.VideoCapture(_video_path(video))
        if not cap.isOpened():
            raise RuntimeError(f"could not open {_video_path(video)}")
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        per_frame_mean_pDLO = []
        per_frame_p7_pos = 0
        per_frame_p7_pos_and_var_gt01 = 0
        # IMPORTANT: D1 saved P7 masks via sequential cap.read() iteration.
        # Use the same access pattern (no seek) so the variant's softmax is
        # evaluated on exactly the same decoded frame the P7 mask was built
        # from — seeks can land on different keyframe interpolations.
        for fi in range(n_total):
            ok, frame = cap.read()
            if not ok:
                break
            if fi not in target_set:
                continue
            pre = preprocess(frame)
            p_dlo = predict_softmax(model, pre, device)
            per_frame_mean_pDLO.append(float(p_dlo.mean()))

            p7 = load_d1_p7_mask(video, fi)
            mask = (p7 == 1)
            n_pos = int(mask.sum())
            if n_pos == 0:
                continue
            n_gt01 = int(((p_dlo > 0.01) & mask).sum())
            per_frame_p7_pos += n_pos
            per_frame_p7_pos_and_var_gt01 += n_gt01
        cap.release()

        mean_pDLO = float(np.mean(per_frame_mean_pDLO)) if per_frame_mean_pDLO else 0.0
        ratio_a = mean_pDLO / max(p7_mean[video], 1e-12)
        frac_b = (per_frame_p7_pos_and_var_gt01 / max(per_frame_p7_pos, 1))

        per_video[video] = {
            "n_frames": len(idx_list),
            "mean_pDLO_variant": mean_pDLO,
            "mean_pDLO_P7": p7_mean[video],
            "gate_a_ratio_mean_pDLO": ratio_a,
            "n_P7_pos_pixels": per_frame_p7_pos,
            "n_P7_pos_and_var_gt_0.01": per_frame_p7_pos_and_var_gt01,
            "gate_b_frac_P7pos_softmax_gt_0.01": frac_b,
        }
        overall_p_dlo.append(mean_pDLO / max(p7_mean[video], 1e-12))
        overall_frac_pos.append(frac_b)
        print(
            f"  {video}: gate_a={ratio_a:.3f}  gate_b={frac_b:.3f}  "
            f"mean_pDLO={mean_pDLO:.4f} (P7 ref {p7_mean[video]:.4f})"
        )

    overall = {
        "mean_of_per_video_gate_a_ratios": float(np.mean(overall_p_dlo)),
        "mean_of_per_video_gate_b_fracs": float(np.mean(overall_frac_pos)),
        "all_pass_a_ge_0.50": all(per_video[v]["gate_a_ratio_mean_pDLO"] >= 0.50
                                  for v in VIDEOS),
        "all_pass_b_ge_0.50": all(
            per_video[v]["gate_b_frac_P7pos_softmax_gt_0.01"] >= 0.50
            for v in VIDEOS),
    }

    out = {
        "ckpt": ckpt_path,
        "per_video": per_video,
        "overall": overall,
        "decision": {
            "gate_a_threshold": 0.50,
            "gate_b_threshold": 0.50,
            "gate_a_pass_all": overall["all_pass_a_ge_0.50"],
            "gate_b_pass_all": overall["all_pass_b_ge_0.50"],
        },
    }
    with open(os.path.join(out_dir, "gate.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[gate] wrote {os.path.join(out_dir, 'gate.json')}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--backbone", default="nvidia/mit-b5")
    args = p.parse_args()
    evaluate_variant(args.ckpt, args.out_dir, backbone=args.backbone)


if __name__ == "__main__":
    main()
