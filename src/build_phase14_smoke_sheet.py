"""Phase 14 negatives smoke contact sheet.

Side-by-side CONTROL (Phase 4 + lighting) vs NEGATIVES (+ dark confuser
clutter: black cylinders / sharp edges / hands) on the same source frames,
plus a wire-label overlay column proving the confusers are labeled BACKGROUND
(green = the DLO/harness label; the dark cylinders / hands must NOT be green).

Usage:
    python src/build_phase14_smoke_sheet.py --out results/phase14_smoke/smoke_sheet.png
"""
import argparse
import os

import cv2
import numpy as np

CTRL = "data/rgbd_videos_phase14_smoke_ctrl/train/000"
NEG = "data/rgbd_videos_phase14_smoke_neg/train/000"
COLS = [
    ("CONTROL  (Phase4 + lighting)", CTRL, False),
    ("NEGATIVES (+ dark cyl/edge/hands)", NEG, False),
    ("NEG + wire label (green = DLO)", NEG, True),
]
# (src, view) rows — front shows the whole scene; bottom maximises floor area
# where the confuser clutter sits.
ROWS = [(0, "front"), (100, "front"), (250, "front"),
        (50, "bottom"), (150, "bottom"), (200, "bottom")]
PAD, HDR, LBL = 6, 34, 78


def _bar(text, h, w, sub=None):
    bar = np.full((h, w, 3), 30, np.uint8)
    cv2.putText(bar, text, (8, int(h * 0.62) if sub else int(h * 0.66)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(bar, sub, (8, int(h * 0.92)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (180, 220, 255), 1, cv2.LINE_AA)
    return bar


def _wire_overlay(rgb, label):
    """Tint DLO/harness label pixels (label > 0) green so they're visible."""
    out = rgb.copy()
    mask = (label > 0).astype(np.uint8)
    if mask.any():
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        green = np.zeros_like(out)
        green[..., 1] = 255
        out[mask > 0] = (0.45 * out[mask > 0] + 0.55 * green[mask > 0]).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase14_smoke/smoke_sheet.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    sample = cv2.imread(f"{CTRL}/rgb/0000_00_front.png")
    h, w = sample.shape[:2]
    ncol, nrow = len(COLS), len(ROWS)
    W = LBL + ncol * (w + PAD) + PAD
    H = HDR + nrow * (h + PAD) + PAD
    canvas = np.full((H, W, 3), 245, np.uint8)

    for ci, (name, _, _) in enumerate(COLS):
        x0 = LBL + ci * (w + PAD) + PAD
        canvas[0:HDR, x0:x0 + w] = _bar(name, HDR, w)

    for ri, (src, view) in enumerate(ROWS):
        y0 = HDR + ri * (h + PAD) + PAD
        canvas[y0:y0 + h, 0:LBL] = _bar(f"src{src:04d}", h, LBL, sub=view)
        for ci, (_, root, overlay) in enumerate(COLS):
            x0 = LBL + ci * (w + PAD) + PAD
            img = cv2.imread(f"{root}/rgb/{src:04d}_00_{view}.png")
            if img is None:
                img = np.full((h, w, 3), 60, np.uint8)
            elif overlay:
                lab = cv2.imread(f"{root}/label/{src:04d}_00_{view}.png",
                                 cv2.IMREAD_UNCHANGED)
                if lab is not None:
                    img = _wire_overlay(img, lab)
            canvas[y0:y0 + h, x0:x0 + w] = img

    cv2.imwrite(args.out, canvas)
    print(f"wrote {args.out}  ({W}x{H})")


if __name__ == "__main__":
    main()
