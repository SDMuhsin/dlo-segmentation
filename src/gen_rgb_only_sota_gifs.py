"""Generate side-by-side animation GIFs from the trained SegFormer-B5 (RGB-only)
binary DLO model.

Phase 7 deliverable. Mirrors src/gen_dformer_v2_dlo_gifs.py so the outputs are
directly comparable to the Phase 5 RGB-D GIFs at:
    results/dformer_v2_dlo/dev_20260429_2317/gifs/

For each picked sample (val source frame + view):
    Left:   RGB animation across 20 anim frames (the wire deforming over time)
    Middle: Depth (8-bit colourised) — INPUT DEPTH (not used by this model)
    Right:  Predicted DLO mask overlay (green) on the same RGB animation

Two modes:
    --reuse-samples PATH/summary.json   reuse the SAME 6 val samples picked by
                                        the Phase 5 model (direct side-by-side
                                        comparison). Default behaviour.
    --pick-fresh                         re-score the full val set with this
                                        RGB-only model and pick its own
                                        N high / N low samples.

The full-val mean IoU(DLO) is always computed and saved to summary.json
regardless of which mode is chosen.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE = os.path.join(PROJECT_ROOT, "data", "hf_cache")
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)

# Reuse the model wrapper from train_rgb_only_sota.py so the load is faithful
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rgb_only_sota import (  # noqa: E402
    SegFormerSegmenter,
    NUM_CLASSES,
    IMAGE_H,
    IMAGE_W,
    RGB_MEAN,
    RGB_STD,
    BACKBONE_DEFAULT,
)

DEPTH_MIN_MM = 500
DEPTH_MAX_MM = 1500
NUM_ANIM_FRAMES = 20

DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "dformer_dataset")
RGBD_VIDEOS = os.path.join(PROJECT_ROOT, "data", "rgbd_videos")


def load_model(ckpt_path, device, backbone=BACKBONE_DEFAULT):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = state.get("args") or {}
    cfg = state.get("config") or {}
    backbone_name = cfg.get("backbone") or saved_args.get("backbone") or backbone

    model = SegFormerSegmenter(
        backbone_name=backbone_name, num_classes=NUM_CLASSES, criterion=None
    )
    sd = state.get("model_state_dict", state)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (first 3): {missing[:3]}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys (first 3): {unexpected[:3]}")
    return model.eval().to(device)


def convert_depth_16to8(depth_16):
    out = np.zeros(depth_16.shape, dtype=np.uint8)
    mask = depth_16 > 0
    clipped = np.clip(depth_16[mask].astype(np.float32), DEPTH_MIN_MM, DEPTH_MAX_MM)
    out[mask] = ((clipped - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * 254 + 1).astype(np.uint8)
    return out


@torch.no_grad()
def predict(model, rgb_bgr, device):
    """rgb_bgr is BGR uint8 (H, W, 3); the SegFormer wrapper expects RGB."""
    rgb_rgb = rgb_bgr[:, :, ::-1].copy()
    rgb = torch.from_numpy(rgb_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device, dtype=torch.float32) / 255.0
    rgb = (rgb - RGB_MEAN.to(device)) / RGB_STD.to(device)
    logits = model(rgb)  # already upsampled to (1, 2, H, W) inside the wrapper
    return logits.argmax(dim=1).squeeze(0).cpu().numpy()


def parse_test_paths(test_txt_path):
    items = []
    with open(test_txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            base = os.path.basename(line).replace(".png", "")
            parts = base.split("_")
            if len(parts) >= 4:
                set_id = int(parts[0])
                src_id = int(parts[1])
                anim_id = int(parts[2])
                view = "_".join(parts[3:])
                items.append((set_id, src_id, anim_id, view))
    return items


def find_split(set_id):
    for split in ("val", "test", "train"):
        if os.path.isdir(os.path.join(RGBD_VIDEOS, split, f"{set_id:03d}")):
            return split
    return None


def gt_label_binary(label_path, include_noise=False):
    lbl = np.array(Image.open(label_path), dtype=np.int64)
    if include_noise:
        return ((lbl >= 1) & (lbl <= 5)).astype(np.int64)
    return ((lbl >= 1) & (lbl <= 4)).astype(np.int64)


def iou_dlo(pred, gt):
    p = (pred == 1)
    g = (gt == 1)
    inter = (p & g).sum()
    union = (p | g).sum()
    if union == 0:
        return float("nan")
    return float(inter) / float(union)


def overlay_mask(rgb_bgr, mask, color_bgr=(0, 255, 0), alpha=0.5):
    out = rgb_bgr.copy()
    color = np.array(color_bgr, dtype=np.uint8)
    out[mask == 1] = (out[mask == 1] * (1 - alpha) + color * alpha).astype(np.uint8)
    return out


def add_label_bar(img, text, height=32):
    bar = np.full((height, img.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(bar, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def make_frame(rgb_bgr, depth_uint8, pred):
    overlay = overlay_mask(rgb_bgr, pred, color_bgr=(0, 255, 0), alpha=0.5)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_VIRIDIS)
    left = add_label_bar(rgb_bgr, "Input RGB")
    mid = add_label_bar(depth_color, "Input Depth (unused)")
    right = add_label_bar(overlay, "Predicted DLO mask (RGB-only)")
    sep = np.full((left.shape[0], 4, 3), 200, dtype=np.uint8)
    return np.hstack([left, sep, mid, sep, right])


def generate_gif(model, set_id, src_id, view, out_path, device, iou_anim0):
    split = find_split(set_id)
    if split is None:
        print(f"  skip — set {set_id:03d} not in train/val/test on disk")
        return False
    base = os.path.join(RGBD_VIDEOS, split, f"{set_id:03d}")
    frames = []
    for anim in range(NUM_ANIM_FRAMES):
        rgb_p = os.path.join(base, "rgb", f"{src_id:04d}_{anim:02d}_{view}.png")
        d_p = os.path.join(base, "depth", f"{src_id:04d}_{anim:02d}_{view}.png")
        if not (os.path.isfile(rgb_p) and os.path.isfile(d_p)):
            continue
        rgb = cv2.imread(rgb_p, cv2.IMREAD_COLOR)
        d16 = cv2.imread(d_p, cv2.IMREAD_UNCHANGED)
        d8 = convert_depth_16to8(d16)
        pred = predict(model, rgb, device)
        frames.append(make_frame(rgb, d8, pred))
    if not frames:
        print(f"  skip — no frames found for set {set_id:03d} src {src_id:04d} view {view}")
        return False
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=120,
        loop=0,
        optimize=True,
    )
    print(f"  saved {os.path.basename(out_path)}  ({len(frames)} frames, IoU(anim0)={iou_anim0:.3f})")
    return True


def pick_unique_sources(sorted_iter, n):
    picked = []
    seen = set()
    for entry in sorted_iter:
        key = (entry[1], entry[2])
        if key in seen:
            continue
        seen.add(key)
        picked.append(entry)
        if len(picked) >= n:
            break
    return picked


def score_full_val(model, items, device):
    """Return list of (iou, set_id, src_id, anim_id, view) with NaN-IoU rows dropped,
    plus the mean IoU(DLO) over the val set."""
    scored = []
    for i, (set_id, src_id, anim_id, view) in enumerate(items):
        split = find_split(set_id)
        if split is None:
            continue
        base = os.path.join(RGBD_VIDEOS, split, f"{set_id:03d}")
        rgb_p = os.path.join(base, "rgb", f"{src_id:04d}_{anim_id:02d}_{view}.png")
        l_p = os.path.join(base, "label", f"{src_id:04d}_{anim_id:02d}_{view}.png")
        if not all(os.path.isfile(p) for p in (rgb_p, l_p)):
            continue
        rgb = cv2.imread(rgb_p, cv2.IMREAD_COLOR)
        pred = predict(model, rgb, device)
        gt = gt_label_binary(l_p)
        iou = iou_dlo(pred, gt)
        if not np.isnan(iou):
            scored.append((iou, set_id, src_id, anim_id, view))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(items)} ...")
    scored.sort(key=lambda x: x[0])
    mean_iou = float(np.mean([s[0] for s in scored])) if scored else float("nan")
    return scored, mean_iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-high", type=int, default=3)
    ap.add_argument("--n-low", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--scoring-stride", type=int, default=1)
    ap.add_argument("--reuse-samples", default=None,
                    help="path to a Phase-5 summary.json; reuse the SAME high/low picks "
                         "so GIFs are directly comparable side-by-side. "
                         "Mean IoU is still scored over the full val set independently.")
    ap.add_argument("--pick-fresh", action="store_true",
                    help="ignore --reuse-samples (if any) and re-pick fresh high/low samples "
                         "based on this model's own per-image IoUs.")
    ap.add_argument("--backbone", default=BACKBONE_DEFAULT,
                    help="HF backbone id; only used as fallback if the ckpt has no config.")
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model from {args.ckpt} on {device} ...")
    model = load_model(args.ckpt, device, backbone=args.backbone)

    items = parse_test_paths(os.path.join(DATASET_DIR, "test.txt"))
    items = items[::args.scoring_stride]
    print(f"Scoring {len(items)} val images (stride={args.scoring_stride}) ...")

    scored, mean_iou = score_full_val(model, items, device)
    if not scored:
        print("ERROR: no scored items.")
        return
    print(f"\nScored {len(scored)} images. IoU range: {scored[0][0]:.3f} ... {scored[-1][0]:.3f}")
    print(f"Mean IoU(DLO) over val: {mean_iou:.4f}")

    # Determine picks
    use_reuse = (args.reuse_samples is not None) and (not args.pick_fresh)
    if use_reuse:
        with open(args.reuse_samples) as f:
            ref = json.load(f)
        # Build (set, src, view) -> iou map from our scoring (anim 0 is the val anim)
        score_lookup = {(s[1], s[2], s[4]): s[0] for s in scored}
        ref_high = ref.get("high", [])
        ref_low = ref.get("low", [])

        def resolve(refs, label):
            out = []
            for r in refs:
                key = (int(r["set"]), int(r["src"]), r["view"])
                iou = score_lookup.get(key)
                if iou is None:
                    print(f"  WARN: reuse {label} {key} not found in val score; skipping")
                    continue
                out.append((iou, key[0], key[1], 0, key[2]))
            return out

        high_picks = resolve(ref_high, "high")
        low_picks = resolve(ref_low, "low")
        print(f"\nReusing samples from {args.reuse_samples}")
    else:
        high_picks = pick_unique_sources(reversed(scored), args.n_high)
        low_picks = pick_unique_sources(scored, args.n_low)

    print(f"\nHigh IoU picks ({len(high_picks)}):")
    for iou, set_id, src_id, _, view in high_picks:
        print(f"  set={set_id:03d} src={src_id:04d} view={view:6s} IoU={iou:.3f}")
    print(f"\nLow IoU picks ({len(low_picks)}):")
    for iou, set_id, src_id, _, view in low_picks:
        print(f"  set={set_id:03d} src={src_id:04d} view={view:6s} IoU={iou:.3f}")

    print(f"\nGenerating GIFs to {args.out_dir} ...")
    summary = {
        "mean_val_iou": mean_iou,
        "ckpt": args.ckpt,
        "reused_samples_from": args.reuse_samples if use_reuse else None,
        "high": [],
        "low": [],
    }
    for label, picks in (("high", high_picks), ("low", low_picks)):
        for iou, set_id, src_id, _, view in picks:
            name = f"{label}_iou{iou:.3f}_set{set_id:03d}_src{src_id:04d}_{view}.gif"
            path = os.path.join(args.out_dir, name)
            if generate_gif(model, set_id, src_id, view, path, device, iou):
                summary[label].append(
                    {"iou": iou, "set": set_id, "src": src_id, "view": view, "file": name}
                )

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Summary at {args.out_dir}/summary.json")
    print(f"Mean val IoU(DLO) = {mean_iou:.4f}")


if __name__ == "__main__":
    main()
