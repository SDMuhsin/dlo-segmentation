"""Smoke-visualise the --aug-wirecolor label-aware recolouring.

Builds a grid PNG: rows = real training frames (with wires), columns =
  [original RGB] [->pale-white] [->light-grey] [->warm-tan] [->bright-yellow]
each column has the WIRE LABEL OUTLINE overlaid (thin green contour) so we can
confirm (a) the mask is unchanged across columns and (b) only wire pixels are
recoloured (background identical). Strength fixed near max so the target look is
unambiguous in the visualisation.

Run from project root with env activated:
    source env/bin/activate
    export HF_HOME=data/hf_home TORCH_HOME=data/torch_home
    python src/smoke_wirecolor_grid.py
"""

import importlib.util
import os
import random

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "dformer_dataset_phase15_wirefree")
OUT_DIR = os.path.join(ROOT, "results", "realism_campaign", "p24_aug_smoke")
OUT_PNG = os.path.join(OUT_DIR, "wirecolor_grid.png")

COLS = ["pale-white", "light-grey", "warm-tan", "bright-yellow"]
COL_TITLES = ["original"] + COLS


def load_module():
    spec = importlib.util.spec_from_file_location(
        "trainmod", os.path.join(ROOT, "src", "train_rgb_only_sota.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def overlay_outline(bgr, wire_mask, color=(0, 255, 0)):
    """Draw a thin contour of the wire mask onto a COPY of bgr (BGR uint8)."""
    out = bgr.copy()
    mask_u8 = (wire_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 1)
    return out


def label_strip(width, text, h=22):
    strip = np.full((h, width, 3), 30, dtype=np.uint8)
    cv2.putText(strip, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return strip


def main():
    m = load_module()
    from train_rgbd_seg import build_cache  # noqa
    train_rgb, _, train_label = build_cache(DATA, "train")

    # Read train.txt so we can label each row with its source frame name.
    with open(os.path.join(DATA, "train.txt")) as f:
        names = [ln.strip().split("/")[-1].replace(".png", "")
                 for ln in f if ln.strip()]

    # Pick 6 frames that contain a healthy amount of wire, spread across sets.
    candidates = []
    for i in range(0, train_label.shape[0], 7):
        wpx = int((train_label[i] <= 3).sum())
        if 1500 < wpx < 15000:
            candidates.append(i)
        if len(candidates) >= 600:
            break
    random.seed(7)
    random.shuffle(candidates)
    frames = sorted(candidates[:6])
    print(f"  rows (frames): {[(i, names[i]) for i in frames]}")

    aug = m.WireColorAugmentation(p=1.0, num_classes=2)

    H, W = train_rgb.shape[1], train_rgb.shape[2]
    GAP = 6
    rows_imgs = []
    for idx in frames:
        rgb_in = train_rgb[idx].copy()       # BGR uint8
        lbl_in = train_label[idx].copy()
        wire_mask = (lbl_in <= 3)

        cells = [overlay_outline(rgb_in, wire_mask)]
        for look in COLS:
            # strength near max so the target look reads clearly in the smoke
            recol = aug.recolor(rgb_in.copy(), wire_mask, name=look, strength=0.9)
            # sanity within the smoke too: background must be identical
            assert np.array_equal(recol[~wire_mask], rgb_in[~wire_mask]), \
                f"background changed for look={look} frame={idx}"
            cells.append(overlay_outline(recol, wire_mask))

        # horizontal concat with gaps
        sep = np.full((H, GAP, 3), 60, dtype=np.uint8)
        row = cells[0]
        for c in cells[1:]:
            row = np.concatenate([row, sep, c], axis=1)
        rows_imgs.append(row)

    row_w = rows_imgs[0].shape[1]

    # column header strip aligned to the cells
    header_cells = [label_strip(W, t) for t in COL_TITLES]
    sep_h = np.full((22, GAP, 3), 60, dtype=np.uint8)
    header = header_cells[0]
    for hc in header_cells[1:]:
        header = np.concatenate([header, sep_h, hc], axis=1)

    vgap = np.full((GAP, row_w, 3), 60, dtype=np.uint8)
    grid = header
    for ri, row in enumerate(rows_imgs):
        grid = np.concatenate([grid, vgap, row], axis=0)
        # per-row label (frame name) under the row would shift alignment; skip.

    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(OUT_PNG, grid)
    print(f"  grid shape: {grid.shape}")
    print(f"  saved: {OUT_PNG}")
    # also dump the frame->name mapping so the report can name the rows
    with open(os.path.join(OUT_DIR, "frames.txt"), "w") as f:
        for r, idx in enumerate(frames):
            f.write(f"row{r}\tframe_idx={idx}\t{names[idx]}\t"
                    f"wire_px={int((train_label[idx] <= 3).sum())}\n")
    print(f"  frame map: {os.path.join(OUT_DIR, 'frames.txt')}")


if __name__ == "__main__":
    main()
