#!/usr/bin/env python
"""
Render a single real-wires sample as a clean 3-panel side-by-side at the image's full
native resolution: [ Input RGB | Ground-truth mask | Our segmentation ].

GT and prediction are shown as translucent green overlays on the RGB (most legible for
judging alignment); a thin cyan rectangle marks the 4:3 region the model actually saw
(the 16:9 -> 4:3 center-crop the eval uses). Prediction is produced by the EXACT 4-video
path (preprocess -> predict) and mapped back onto the full image.

Usage:
  python src/show_real_wire_sample.py --basename c4_3
"""
import os
import sys
import argparse

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_video_rgb_only import preprocess  # noqa: E402  (sets HF_HOME on import)
from gen_rgb_only_sota_gifs import load_model, predict, overlay_mask, add_label_bar  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data", "real_wires_valset")
DEFAULT_CKPT = os.path.join(
    PROJECT_ROOT,
    "results/segformer_b5_rgb_phase15_wirefree_ft/full_20260605_0409/best_model.pth",
)
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "results", "real_wires_valset_eval")
MASK_THRESH = 127
TARGET_AR = 640.0 / 480.0  # model input is 4:3


def crop_box_4to3(h, w):
    """The exact center-crop box infer_video_rgb_only.center_crop_to_aspect would take."""
    if w / h > TARGET_AR:                # too wide -> crop horizontally
        new_w = int(round(h * TARGET_AR))
        return (w - new_w) // 2, 0, new_w, h
    new_h = int(round(w / TARGET_AR))    # too tall -> crop vertically
    return 0, (h - new_h) // 2, w, new_h


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basename", default="c4_3", help="e.g. c4_3, c1_11, c2_2")
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--panel-width", type=int, default=900, help="display px per panel")
    args = ap.parse_args()

    bn = args.basename
    rgb_path = os.path.join(args.data_dir, "imgs", f"{bn}.jpg")
    mask_path = os.path.join(args.data_dir, "masks", f"{bn}.jpg")
    for p in (rgb_path, mask_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)            # full-res BGR
    h, w = rgb.shape[:2]
    gt = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) >= MASK_THRESH).astype(np.uint8)

    # prediction via the exact 4-video path, mapped back to full resolution
    pred_small = (predict(model, preprocess(rgb), device) >= 1).astype(np.uint8)  # 480x640
    x0, y0, cw, ch = crop_box_4to3(h, w)
    pred_full = np.zeros((h, w), np.uint8)
    pred_full[y0:y0 + ch, x0:x0 + cw] = cv2.resize(pred_small, (cw, ch),
                                                   interpolation=cv2.INTER_NEAREST)

    # IoU within the 4:3 region the model actually saw (matches the eval)
    g = gt[y0:y0 + ch, x0:x0 + cw].astype(bool)
    p = pred_full[y0:y0 + ch, x0:x0 + cw].astype(bool)
    inter = int((p & g).sum()); union = int((p | g).sum())
    iou = inter / union if union else float("nan")

    panels = {
        "Input RGB (full native res)": rgb.copy(),
        "Ground-truth mask (green)": overlay_mask(rgb, gt, (0, 255, 0), 0.45),
        f"Our segmentation (green)   IoU={iou:.3f}": overlay_mask(rgb, pred_full, (0, 255, 0), 0.45),
    }
    thick = max(2, w // 500)
    out = []
    for title, img in panels.items():
        cv2.rectangle(img, (x0, y0), (x0 + cw, y0 + ch), (255, 255, 0), thick)  # 4:3 FOV
        dh = int(h * args.panel_width / w)
        img = cv2.resize(img, (args.panel_width, dh), interpolation=cv2.INTER_AREA)
        out.append(add_label_bar(img, title))
    sep = np.full((out[0].shape[0], 6, 3), 200, dtype=np.uint8)
    sheet = out[0]
    for pnl in out[1:]:
        sheet = np.hstack([sheet, sep, pnl])

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"sample_{bn}_3panel.png")
    cv2.imwrite(out_path, sheet)
    print(f"basename={bn}  res={w}x{h}  IoU(4:3 region)={iou:.4f}")
    print(f"cyan box = the 4:3 region the model saw (16:9 -> 4:3 center-crop)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
