"""Phase 18 wire-appearance smoke contact sheet.

Side-by-side Phase 15 (lever OFF: 11-photo wire texture pool, k=1 thickness)
vs Phase 18 (lever ON: +16 solid-cable swatches via KIAT_P18_WIRETEX_DIR and
per-harness thickness k~U[0.8,2.8] via KIAT_P18_THICK_LO/HI) on the SAME
source frames, each with a wire-label overlay column proving (a) only the
wire's APPEARANCE changed (colour/width), (b) the label still covers exactly
the wire (green = the DLO/harness label; backdrop/clutter/hands must NOT be
green), and (c) the scene composition (backdrop, lighting, clutter, hands)
is untouched.

Both renders are the stride-12 smoke twins under
results/realism_campaign/p18_smoke/ (the OFF twin is the same as the
shipped Phase 15 render on shared sources — verified).

Usage:
    python src/build_phase18_smoke_sheet.py \
        --out results/realism_campaign/p18_smoke/contact_sheet.png
"""
import argparse
import os

import cv2
import numpy as np

OFF = "results/realism_campaign/p18_smoke/render_off/train/000"  # Phase 15 baseline
ON = "results/realism_campaign/p18_smoke/render/train/000"       # Phase 18 levers ON
COLS = [
    ("Phase15 OFF  rgb", OFF, False),
    ("Phase15 OFF  label (green=wire)", OFF, True),
    ("Phase18 ON  rgb (swatch+thick)", ON, False),
    ("Phase18 ON  label (green=wire)", ON, True),
]
# (src, view, note) rows — wired sources spanning the thickness draw k
# (0.87 → 2.63) so both ends of the multiplier and a spread of swatch colours
# are visible; last row is a wire-free source (must stay wire-free, empty
# label, under the lever).
ROWS = [(0, "front", "k=0.98"), (12, "front", "k=1.76"),
        (36, "front", "k=2.36"), (60, "front", "k=0.87"),
        (96, "top", "k=2.40"), (120, "front", "k=2.12"),
        (132, "bottom", "k=2.57"), (276, "front", "k=2.63"),
        (72, "front", "wirefree")]
PAD, HDR, LBL = 6, 34, 110


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
    ap.add_argument("--out",
                    default="results/realism_campaign/p18_smoke/contact_sheet.png")
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

    for ri, (src, view, note) in enumerate(ROWS):
        y0 = HDR + ri * (h + PAD) + PAD
        canvas[y0:y0 + h, 0:LBL] = _bar(f"src{src:04d} {note}", h, LBL, sub=view)
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
