"""Phase 13 smoke contact sheet.

Side-by-side CONTROL vs LIGHTING vs OBJ-GRADIENT on the same source frames, so
the levers can be sanity-checked against the strict-Phase-4 base before any
multi-hour train. CONTROL is the byte-identical lever-OFF render (== H0).

Usage:
    python src/build_phase13_contact_sheet.py --out results/phase13_smoke/contact_sheet.png
"""
import argparse
import os

import cv2
import numpy as np

CTRL = "data/rgbd_videos_phase13_ctrl/train/000"
LIGHT = "data/rgbd_videos_phase13_smoke_light/train/000"
GRAD = "data/rgbd_videos_phase13_smoke_objgrad/train/000"
BOTH = "data/rgbd_videos_phase13_smoke_both/train/000"
COLS = [("CONTROL (= Phase 4 / H0)", CTRL),
        ("LIGHTING (lever a)", LIGHT),
        ("OBJ-GRADIENT (lever b)", GRAD),
        ("BOTH (compose)", BOTH)]
# (src_id, view) rows — mix of front (harness+floor+backdrop) and bottom
# (max floor/object area, best shows the 3D light gradient).
ROWS = [(0, "front"), (50, "front"), (120, "front"),
        (200, "front"), (0, "bottom"), (120, "bottom")]
PAD = 6
HDR = 34
LBL = 90


def _label(img, text, h, w):
    bar = np.full((h, w, 3), 30, np.uint8)
    cv2.putText(bar, text, (8, int(h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase13_smoke/contact_sheet.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    sample = cv2.imread(f"{CTRL}/rgb/0000_00_front.png")
    h, w = sample.shape[:2]
    ncol, nrow = len(COLS), len(ROWS)
    W = LBL + ncol * (w + PAD) + PAD
    H = HDR + nrow * (h + PAD) + PAD
    canvas = np.full((H, W, 3), 245, np.uint8)

    # Column headers
    for ci, (name, _) in enumerate(COLS):
        x0 = LBL + ci * (w + PAD) + PAD
        canvas[0:HDR, x0:x0 + w] = _label(None, name, HDR, w)

    for ri, (src, view) in enumerate(ROWS):
        y0 = HDR + ri * (h + PAD) + PAD
        rl = _label(None, f"src {src:04d}\n{view}", h, LBL)
        cv2.putText(rl, f"src {src:04d}", (6, int(h * 0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(rl, view, (6, int(h * 0.55)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA)
        canvas[y0:y0 + h, 0:LBL] = rl
        for ci, (_, root) in enumerate(COLS):
            x0 = LBL + ci * (w + PAD) + PAD
            img = cv2.imread(f"{root}/rgb/{src:04d}_00_{view}.png")
            if img is None:
                img = np.full((h, w, 3), 60, np.uint8)
            canvas[y0:y0 + h, x0:x0 + w] = img

    cv2.imwrite(args.out, canvas)
    print(f"wrote {args.out}  ({W}x{H})")


if __name__ == "__main__":
    main()
