"""Phase 16 OPEN/SPLAYED-hand smoke contact sheet.

Builds a contact sheet from a Phase-16 smoke render (lever ON,
``KIAT_P16_OPENHAND_P`` high). Rows = several source frames; columns =
front RGB, front wire-label overlay (green = label > 0), bottom RGB, bottom
overlay. The sheet should show an OPEN / SPLAYED hand clearly in the scene and
the wire-label overlay confirming the open hand is BACKGROUND (NOT green) while
the real wire (if present) IS green.

The per-row annotation states whether an open hand was placed for that source
(replaying the Phase-16 lever rng) and whether the frame is wire-free.

Usage:
    python src/build_phase16_smoke_sheet.py \
        --out results/phase16_smoke/contact_sheet.png
"""
import argparse
import os

import cv2
import numpy as np

ROOT = "data/rgbd_videos_phase16_smoke/train/000"
SET_ID = 0
# Lever probabilities used for the smoke render (to replay the per-source
# open-hand / wire-free decisions for the row annotations).
P16_OPENHAND_P = 0.8
P15_WIREFREE_P = 0.2
# Rng offsets must match convert_to_video_dataset.py.
_OH_OFFSET = 1217
_WF_OFFSET = 401

# Source rows: a spread of sources, chosen to include wired+open-hand,
# wire-free+open-hand, and a no-open-hand control.
ROWS = [0, 75, 175, 250, 125, 225, 25]

COLS = [
    ("front RGB", "front", False),
    ("front + wire label (green=DLO)", "front", True),
    ("bottom RGB", "bottom", False),
    ("bottom + wire label", "bottom", True),
]
PAD, HDR, LBL = 6, 34, 150


def _placed_open_hand(src: int) -> bool:
    rng = np.random.RandomState(SET_ID * 1000 + src + _OH_OFFSET)
    return bool(rng.uniform(0.0, 1.0) < P16_OPENHAND_P)


def _is_wirefree(src: int) -> bool:
    rng = np.random.RandomState(SET_ID * 1000 + src + _WF_OFFSET)
    return bool(rng.uniform(0.0, 1.0) < P15_WIREFREE_P)


def _bar(text, h, w, sub=None, sub2=None):
    bar = np.full((h, w, 3), 30, np.uint8)
    cv2.putText(bar, text, (8, int(h * 0.34) if (sub or sub2) else int(h * 0.6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(bar, sub, (8, int(h * 0.64)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (140, 255, 140), 1, cv2.LINE_AA)
    if sub2:
        cv2.putText(bar, sub2, (8, int(h * 0.92)), cv2.FONT_HERSHEY_SIMPLEX,
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
        out[mask > 0] = (0.40 * out[mask > 0] + 0.60 * green[mask > 0]).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase16_smoke/contact_sheet.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    sample = cv2.imread(f"{ROOT}/rgb/{ROWS[0]:04d}_00_front.png")
    h, w = sample.shape[:2]
    ncol, nrow = len(COLS), len(ROWS)
    W = LBL + ncol * (w + PAD) + PAD
    H = HDR + nrow * (h + PAD) + PAD
    canvas = np.full((H, W, 3), 245, np.uint8)

    # Column headers.
    for ci, (name, _, _) in enumerate(COLS):
        x0 = LBL + ci * (w + PAD) + PAD
        canvas[0:HDR, x0:x0 + w] = _bar(name, HDR, w)

    print("Per-source open-hand placement (KIAT_P16_OPENHAND_P=%.2f):"
          % P16_OPENHAND_P)
    for ri, src in enumerate(ROWS):
        oh = _placed_open_hand(src)
        wf = _is_wirefree(src)
        oh_s = "open hand" if oh else "no hand"
        wf_s = "WIRE-FREE" if wf else "has wire"
        print(f"  src{src:04d}: open_hand={oh}  wire_free={wf}")
        y0 = HDR + ri * (h + PAD) + PAD
        canvas[y0:y0 + h, 0:LBL] = _bar(
            f"src{src:04d}", h, LBL, sub=oh_s, sub2=wf_s)
        for ci, (_, view, overlay) in enumerate(COLS):
            x0 = LBL + ci * (w + PAD) + PAD
            img = cv2.imread(f"{ROOT}/rgb/{src:04d}_00_{view}.png")
            if img is None:
                img = np.full((h, w, 3), 60, np.uint8)
            elif overlay:
                lab = cv2.imread(f"{ROOT}/label/{src:04d}_00_{view}.png",
                                 cv2.IMREAD_UNCHANGED)
                if lab is not None:
                    img = _wire_overlay(img, lab)
            canvas[y0:y0 + h, x0:x0 + w] = img

    cv2.imwrite(args.out, canvas)
    print(f"wrote {args.out}  ({W}x{H})")


if __name__ == "__main__":
    main()
