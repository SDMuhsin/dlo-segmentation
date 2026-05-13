"""Independent re-evaluation of the KD student on Phase 4 val (and optional v2 val).

Mirrors the protocol in results/postmortem/cross_val_eval.py — iterates files
directly from data/rgbd_videos/val/*/{rgb,label}/* so we don't depend on the
dformer_dataset symlink staging, and produces TP/FP/FN/TN-derived metrics
matching the post-mortem.

Why a separate re-eval? In-training eval runs under AMP FP16; the
post-mortem found a ~0.005 delta between in-training and FP32 eval for
Phase 7. The report.json `best_iou_dlo` is the AMP eval; this script
provides the FP32 ground-truth for the comparison report.

Usage:
    source env/bin/activate
    CUDA_VISIBLE_DEVICES=0 python src/eval_segformer_kd_student.py \
        --ckpt results/segformer_b0_rgb_kd/full_<tag>/best_model.pth \
        --out  results/segformer_b0_rgb_kd/full_<tag>/eval_phase4_val.json
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, "data", "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(PROJECT_ROOT, "data", "hf_cache"))

from gen_rgb_only_sota_gifs import load_model, predict, gt_label_binary


def list_val_paths(root, stride=1, anim=0):
    """List (rgb_path, label_path) pairs under <root>/val/*/rgb/*.png at anim=0 + stride."""
    out = []
    val_dir = os.path.join(root, "val")
    if not os.path.isdir(val_dir):
        return out
    for set_name in sorted(os.listdir(val_dir)):
        set_path = os.path.join(val_dir, set_name)
        if not os.path.isdir(set_path):
            continue
        rgb_dir = os.path.join(set_path, "rgb")
        if not os.path.isdir(rgb_dir):
            continue
        for fn in sorted(os.listdir(rgb_dir)):
            if not fn.endswith(".png"):
                continue
            parts = fn.split("_")
            if len(parts) < 3:
                continue
            src_id = int(parts[0])
            anim_id = int(parts[1])
            if anim_id != anim:
                continue
            if (src_id % stride) != 0:
                continue
            rgb_p = os.path.join(rgb_dir, fn)
            lbl_p = os.path.join(set_path, "label", fn)
            if not os.path.isfile(lbl_p):
                continue
            out.append((rgb_p, lbl_p))
    return out


def iou_dlo_stats(model, paths, device, max_n=None):
    tp = fp = fn = tn = 0
    n = 0
    t0 = time.time()
    ious_per_image = []
    if max_n is not None:
        paths = paths[:max_n]
    for rgb_p, lbl_p in paths:
        rgb = cv2.imread(rgb_p, cv2.IMREAD_COLOR)
        gt = gt_label_binary(lbl_p)
        pred = predict(model, rgb, device)
        ip_tp = int(((pred == 1) & (gt == 1)).sum())
        ip_fp = int(((pred == 1) & (gt == 0)).sum())
        ip_fn = int(((pred == 0) & (gt == 1)).sum())
        ip_tn = int(((pred == 0) & (gt == 0)).sum())
        tp += ip_tp
        fp += ip_fp
        fn += ip_fn
        tn += ip_tn
        denom = ip_tp + ip_fp + ip_fn
        if denom > 0:
            ious_per_image.append(ip_tp / denom)
        n += 1
        if n % 200 == 0:
            print(f"    {n}/{len(paths)} ... {n / (time.time() - t0):.1f} img/s")
    iou_dlo = tp / max(tp + fp + fn, 1)
    iou_bg = tn / max(tn + fp + fn, 1)
    miou = (iou_dlo + iou_bg) / 2.0
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "n_images": n,
        "iou_dlo_pooled": float(iou_dlo),
        "iou_bg_pooled": float(iou_bg),
        "miou_pooled": float(miou),
        "mean_iou_per_image": float(np.mean(ious_per_image)) if ious_per_image else float("nan"),
        "min_iou_per_image": float(min(ious_per_image)) if ious_per_image else float("nan"),
        "max_iou_per_image": float(max(ious_per_image)) if ious_per_image else float("nan"),
        "precision_dlo": float(prec),
        "recall_dlo": float(rec),
        "pixel_acc": float(acc),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "wall_seconds": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--p4-stride", type=int, default=5)
    ap.add_argument("--anim", type=int, default=0)
    ap.add_argument("--eval-v2", action="store_true",
                    help="also evaluate on data/rgbd_videos_v2/val (optional sanity, post-mortem §0.14.7).")
    ap.add_argument("--v2-stride", type=int, default=5)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Loading student from {args.ckpt}")
    model = load_model(args.ckpt, device)

    out = {}

    p4_val = list_val_paths(os.path.join(PROJECT_ROOT, "data", "rgbd_videos"),
                            stride=args.p4_stride, anim=args.anim)
    print(f"\nPhase 4 val (stride={args.p4_stride}, anim={args.anim}): {len(p4_val)} imgs")
    stats_p4 = iou_dlo_stats(model, p4_val, device)
    out["phase4_val"] = stats_p4
    print(f"  Phase 4 val: iou_dlo_pooled={stats_p4['iou_dlo_pooled']:.4f}  "
          f"mean_per_image={stats_p4['mean_iou_per_image']:.4f}  "
          f"prec={stats_p4['precision_dlo']:.4f}  rec={stats_p4['recall_dlo']:.4f}")

    if args.eval_v2:
        v2_val = list_val_paths(os.path.join(PROJECT_ROOT, "data", "rgbd_videos_v2"),
                                stride=args.v2_stride, anim=args.anim)
        if v2_val:
            print(f"\nv2 val (stride={args.v2_stride}, anim={args.anim}): {len(v2_val)} imgs")
            stats_v2 = iou_dlo_stats(model, v2_val, device)
            out["v2_val"] = stats_v2
            print(f"  v2 val: iou_dlo_pooled={stats_v2['iou_dlo_pooled']:.4f}  "
                  f"mean_per_image={stats_v2['mean_iou_per_image']:.4f}")

    out["ckpt"] = args.ckpt
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
