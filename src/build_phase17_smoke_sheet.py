"""Phase 17 background-clutter smoke contact sheet.

Side-by-side Phase 15 (lever OFF: 3-6 floor clutter objects/scene) vs Phase 17
(lever ON: 5-9 objects/scene, ~+50%) on the SAME source frames, each with a
wire-label overlay column proving (a) Phase 17 has more background clutter and
(b) the added clutter is labeled BACKGROUND — ONLY the wire/harness is the wire
class (green = the DLO/harness label; all the floor clutter must NOT be green).

The lever-OFF "before" frames are REUSED from the shipped Phase 15 render; only
the lever-ON frames were re-rendered (data/rgbd_videos_phase17_smoke).

Usage:
    python src/build_phase17_smoke_sheet.py --out results/phase17_smoke/contact_sheet.png
"""
import argparse
import os

import cv2
import numpy as np

OFF = "data/rgbd_videos_phase15_wirefree/train/000"   # lever OFF: 3-6 objects
ON = "data/rgbd_videos_phase17_smoke/train/000"        # lever ON : 5-9 objects
COLS = [
    ("Phase15 OFF  (3-6 obj)  rgb", OFF, False),
    ("Phase15 OFF  label (green=wire)", OFF, True),
    ("Phase17 ON  (5-9 obj)  rgb", ON, False),
    ("Phase17 ON  label (green=wire)", ON, True),
]
# (src, view) rows. front shows the whole scene incl. the wire; top maximises
# the floor area where the added clutter sits. src0200 is wire-free at p=0.2 so
# we lead with the 5 wired sources (label-correctness checkable) and show the
# floor-rich top view on a couple of them.
ROWS = [(0, "front"), (50, "front"), (100, "top"),
        (150, "front"), (250, "top"), (200, "front")]
PAD, HDR, LBL = 6, 34, 92


def _bar(text, h, w, sub=None):
    bar = np.full((h, w, 3), 30, np.uint8)
    cv2.putText(bar, text, (8, int(h * 0.62) if sub else int(h * 0.66)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(bar, sub, (8, int(h * 0.92)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (180, 220, 255), 1, cv2.LINE_AA)
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
    ap.add_argument("--out", default="results/phase17_smoke/contact_sheet.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    sample = cv2.imread(f"{OFF}/rgb/0000_00_front.png")
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
