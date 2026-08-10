#!/usr/bin/env python3
"""Phase 15 wire-free smoke contact sheet.

Demonstrates the default-OFF ``KIAT_P15_WIREFREE_P`` renderer lever: with the
lever ON some SOURCES are rendered WIRE-FREE (the full strict-Phase-4 + lighting
+ hard-negative-confuser scene composed exactly as normal, but the harness/wire
itself omitted ⇒ the frame shows the scene with NO wire and an all-background
label), while the rest are normal wired frames.

Each row is one source rendered at the front + bottom views; for each view we
show the RGB and a wire-label overlay (green = label>0 = harness/wire). Rows are
chosen so BOTH kinds are visible:
  * WIRED rows     → green wire present in the overlay, wire-label pixels > 0.
  * WIRE-FREE rows → scene + confusers but ZERO green / zero wire-label pixels.

The per-row banner prints the source's total wire-label pixel count (summed over
all 6 views) so the wired/wire-free split is verifiable numerically.

Usage:
    python src/build_phase15_smoke_sheet.py \
        --out results/phase15_smoke/contact_sheet.png
"""
import argparse
import os

import cv2
import numpy as np

SMOKE = "data/rgbd_videos_phase15_smoke/train/000"
VIEWS_FOR_COUNT = ("front", "back", "right", "left", "top", "bottom")
COLS = [
    ("FRONT  RGB", "front", False),
    ("FRONT  wire label (green = wire)", "front", True),
    ("BOTTOM  RGB", "bottom", False),
    ("BOTTOM  wire label (green = wire)", "bottom", True),
]
# Rows chosen to interleave WIRED and WIRE-FREE sources (rendered with
# src-stride 25, KIAT_P15_WIREFREE_P=0.5):
#   wired:     0, 25, 50, 150, 275      wire-free: 100, 125, 200, 250
ROWS = [0, 100, 25, 125, 150, 200, 275, 250]
PAD, HDR, LBL = 6, 34, 118


def _bar(text, h, w, sub=None, sub2=None, color=(255, 255, 255)):
    bar = np.full((h, w, 3), 30, np.uint8)
    cv2.putText(bar, text, (8, int(h * 0.30) if sub else int(h * 0.62)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
    if sub:
        cv2.putText(bar, sub, (8, int(h * 0.60)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (180, 220, 255), 1, cv2.LINE_AA)
    if sub2:
        cv2.putText(bar, sub2, (8, int(h * 0.88)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (180, 255, 180), 1, cv2.LINE_AA)
    return bar


def _wire_overlay(rgb, label):
    """Tint wire/harness label pixels (label > 0) green so they're visible."""
    out = rgb.copy()
    mask = (label > 0).astype(np.uint8)
    if mask.any():
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        green = np.zeros_like(out)
        green[..., 1] = 255
        out[mask > 0] = (0.40 * out[mask > 0]
                         + 0.60 * green[mask > 0]).astype(np.uint8)
    return out


def _wire_px_total(src):
    """Total wire-label pixels for a source, summed over all 6 views."""
    tot = 0
    for v in VIEWS_FOR_COUNT:
        lab = cv2.imread(f"{SMOKE}/label/{src:04d}_00_{v}.png",
                         cv2.IMREAD_UNCHANGED)
        if lab is not None:
            tot += int((lab > 0).sum())
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase15_smoke/contact_sheet.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    sample = cv2.imread(f"{SMOKE}/rgb/{ROWS[0]:04d}_00_front.png")
    h, w = sample.shape[:2]
    ncol, nrow = len(COLS), len(ROWS)
    W = LBL + ncol * (w + PAD) + PAD
    H = HDR + nrow * (h + PAD) + PAD
    canvas = np.full((H, W, 3), 245, np.uint8)

    # Title strip over the row-label gutter.
    canvas[0:HDR, 0:LBL] = _bar("P15 wire-free", HDR, LBL,
                                sub="0.5 prob smoke")
    for ci, (name, _, _) in enumerate(COLS):
        x0 = LBL + ci * (w + PAD) + PAD
        canvas[0:HDR, x0:x0 + w] = _bar(name, HDR, w)

    for ri, src in enumerate(ROWS):
        y0 = HDR + ri * (h + PAD) + PAD
        wpx = _wire_px_total(src)
        is_wf = (wpx == 0)
        kind = "WIRE-FREE" if is_wf else "WIRED"
        kcol = (140, 140, 255) if is_wf else (140, 255, 140)
        canvas[y0:y0 + h, 0:LBL] = _bar(
            f"src{src:04d}", h, LBL,
            sub=kind, sub2=f"wirepx={wpx}", color=kcol)
        for ci, (_, view, overlay) in enumerate(COLS):
            x0 = LBL + ci * (w + PAD) + PAD
            img = cv2.imread(f"{SMOKE}/rgb/{src:04d}_00_{view}.png")
            if img is None:
                img = np.full((h, w, 3), 60, np.uint8)
            elif overlay:
                lab = cv2.imread(f"{SMOKE}/label/{src:04d}_00_{view}.png",
                                 cv2.IMREAD_UNCHANGED)
                if lab is not None:
                    img = _wire_overlay(img, lab)
            canvas[y0:y0 + h, x0:x0 + w] = img

    cv2.imwrite(args.out, canvas)
    print(f"wrote {args.out}  ({W}x{H})")

    # Print the numeric split so the wired/wire-free decision is verifiable.
    print("\nPer-source wire-label pixel counts (sum over 6 views):")
    n_wf = 0
    for src in sorted({int(os.path.basename(f)[:4])
                       for f in os.listdir(f"{SMOKE}/label")}):
        wpx = _wire_px_total(src)
        wf = (wpx == 0)
        n_wf += int(wf)
        print(f"  src{src:04d}: wirepx={wpx:>7}  -> "
              f"{'WIRE-FREE' if wf else 'wired'}")
    print(f"\nwire-free sources: {n_wf}  (wirepx == 0 ⇒ zero wire pixels)")


if __name__ == "__main__":
    main()
