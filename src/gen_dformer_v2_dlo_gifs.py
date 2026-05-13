"""Generate side-by-side animation GIFs from the trained DFormer-v2-Large
binary DLO model.

For each picked sample (val source frame + view):
  Left:  RGB animation across 20 anim frames (the wire deforming over time)
  Right: Predicted DLO mask overlay (green) on the same RGB animation

Picks N high-IoU and N low-IoU samples from the val set. IoU is scored at
anim 0 (which is what the val set actually contains label PNGs for); the
animation extends the picked source/view across all 20 anim frames using
data/rgbd_videos/<split>/<set>/{rgb,depth}/.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DFORMER_DIR = os.path.join(PROJECT_ROOT, "src", "dformer")
sys.path.insert(0, DFORMER_DIR)
from models.builder import EncoderDecoder as DFormerModel  # noqa: E402

# Constants copied verbatim from src/train_dformer_v2_dlo.py
NUM_CLASSES = 2
IGNORE_INDEX = -1
RGB_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
DEPTH_MEAN = torch.tensor([0.48, 0.48, 0.48], dtype=torch.float32).view(1, 3, 1, 1)
DEPTH_STD = torch.tensor([0.28, 0.28, 0.28], dtype=torch.float32).view(1, 3, 1, 1)
DEPTH_MIN_MM = 500
DEPTH_MAX_MM = 1500
NUM_ANIM_FRAMES = 20

DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "dformer_dataset")
RGBD_VIDEOS = os.path.join(PROJECT_ROOT, "data", "rgbd_videos")


class ModelConfig:
    backbone = "DFormerv2_L"
    pretrained_model = None
    decoder = "ham"
    decoder_embed_dim = 1024
    num_classes = NUM_CLASSES
    background = IGNORE_INDEX
    bn_eps = 1e-3
    bn_momentum = 0.1
    drop_path_rate = 0.3
    aux_rate = 0.0
    fix_bias = True


def load_model(ckpt_path, device):
    cfg = ModelConfig()
    model = DFormerModel(cfg=cfg, criterion=None, norm_layer=nn.BatchNorm2d, syncbn=False)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
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
def predict(model, rgb_bgr, depth_uint8, device):
    rgb = torch.from_numpy(rgb_bgr.transpose(2, 0, 1)).unsqueeze(0).to(device, dtype=torch.float32) / 255.0
    rgb = (rgb - RGB_MEAN.to(device)) / RGB_STD.to(device)
    depth = torch.from_numpy(depth_uint8).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32) / 255.0
    depth = (depth - DEPTH_MEAN.to(device)) / DEPTH_STD.to(device)
    logits = model(rgb, depth)
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
    mid = add_label_bar(depth_color, "Input Depth")
    right = add_label_bar(overlay, "Predicted DLO mask")
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
        pred = predict(model, rgb, d8, device)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-high", type=int, default=3)
    ap.add_argument("--n-low", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--scoring-stride", type=int, default=1,
                    help="evaluate every Nth val image when picking (1 = all)")
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model from {args.ckpt} on {device} ...")
    model = load_model(args.ckpt, device)

    items = parse_test_paths(os.path.join(DATASET_DIR, "test.txt"))
    items = items[::args.scoring_stride]
    print(f"Scoring {len(items)} val images (stride={args.scoring_stride}) ...")

    scored = []
    for i, (set_id, src_id, anim_id, view) in enumerate(items):
        split = find_split(set_id)
        if split is None:
            continue
        rgb_p = os.path.join(RGBD_VIDEOS, split, f"{set_id:03d}", "rgb", f"{src_id:04d}_{anim_id:02d}_{view}.png")
        d_p = os.path.join(RGBD_VIDEOS, split, f"{set_id:03d}", "depth", f"{src_id:04d}_{anim_id:02d}_{view}.png")
        l_p = os.path.join(RGBD_VIDEOS, split, f"{set_id:03d}", "label", f"{src_id:04d}_{anim_id:02d}_{view}.png")
        if not all(os.path.isfile(p) for p in (rgb_p, d_p, l_p)):
            continue
        rgb = cv2.imread(rgb_p, cv2.IMREAD_COLOR)
        d16 = cv2.imread(d_p, cv2.IMREAD_UNCHANGED)
        d8 = convert_depth_16to8(d16)
        pred = predict(model, rgb, d8, device)
        gt = gt_label_binary(l_p)
        iou = iou_dlo(pred, gt)
        if not np.isnan(iou):
            scored.append((iou, set_id, src_id, anim_id, view))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(items)} ...")

    scored.sort(key=lambda x: x[0])
    print(f"\nScored {len(scored)} images. IoU range: {scored[0][0]:.3f} ... {scored[-1][0]:.3f}")
    mean_iou = float(np.mean([s[0] for s in scored]))
    print(f"Mean IoU(DLO) over val: {mean_iou:.4f}")

    high_picks = pick_unique_sources(reversed(scored), args.n_high)
    low_picks = pick_unique_sources(scored, args.n_low)

    print(f"\nHigh IoU picks ({len(high_picks)}):")
    for iou, set_id, src_id, _, view in high_picks:
        print(f"  set={set_id:03d} src={src_id:04d} view={view:6s} IoU={iou:.3f}")
    print(f"\nLow IoU picks ({len(low_picks)}):")
    for iou, set_id, src_id, _, view in low_picks:
        print(f"  set={set_id:03d} src={src_id:04d} view={view:6s} IoU={iou:.3f}")

    print(f"\nGenerating GIFs to {args.out_dir} ...")
    summary = {"mean_val_iou": mean_iou, "high": [], "low": []}
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


if __name__ == "__main__":
    main()
