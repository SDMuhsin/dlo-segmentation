"""Smoke-visualise the --aug-bgclutter label-aware BACKGROUND clutter aug.

Builds a grid PNG for visual review:
  rows    = real training frames (that contain wires)
  columns = [ original RGB ]
            [ bg-clutter variant 1 ] [ variant 2 ] [ variant 3 ] [ variant 4 ]
            [ background-mask overlay ]
Every column has the WIRE LABEL OUTLINE overlaid (thin green contour) so you
can confirm at a glance that (a) the wires are UNTOUCHED across all variants
(the green outline still sits on un-perturbed cable pixels) and (b) only the
BACKGROUND gets diverse busy texture, with nothing that looks like an added wire.
The final column tints the perturbed BACKGROUND region red so it is obvious which
pixels the aug is (and is not) allowed to touch.

Run from project root with env activated:
    source env/bin/activate
    export HF_HOME=data/hf_home TORCH_HOME=data/torch_home
    python src/smoke_bgclutter_grid.py
"""

import importlib.util
import os
import random

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "dformer_dataset_phase15_wirefree")
OUT_DIR = os.path.join(ROOT, "results", "realism_campaign", "p25_aug_smoke")
OUT_PNG = os.path.join(OUT_DIR, "bgclutter_grid.png")

N_VARIANTS = 4
COL_TITLES = (["original"]
              + [f"bgclutter v{i + 1}" for i in range(N_VARIANTS)]
              + ["bg-mask (red=perturbed)"])


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


def tint_bg(bgr, bg_mask, color=(0, 0, 255), alpha=0.35):
    """Tint the BACKGROUND region (the pixels the aug may perturb) so it is
    obvious which pixels are foreground (protected) vs background."""
    out = bgr.copy().astype(np.float32)
    tint = np.array(color, np.float32)
    out[bg_mask] = (1 - alpha) * out[bg_mask] + alpha * tint
    return np.clip(out, 0, 255).astype(np.uint8)


def label_strip(width, text, h=22):
    strip = np.full((h, width, 3), 30, dtype=np.uint8)
    cv2.putText(strip, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return strip


def main():
    m = load_module()
    from train_rgbd_seg import build_cache  # noqa
    train_rgb, _, train_label = build_cache(DATA, "train")

    with open(os.path.join(DATA, "train.txt")) as f:
        names = [ln.strip().split("/")[-1].replace(".png", "")
                 for ln in f if ln.strip()]

    # Pick 6 frames with a healthy amount of wire, spread across sets.
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

    aug = m.BgClutterAugmentation(p=1.0, num_classes=2)  # always fires in smoke

    H, W = train_rgb.shape[1], train_rgb.shape[2]
    GAP = 6
    rows_imgs = []
    # fixed seed list so the grid is reproducible across runs
    variant_seeds = [101, 202, 303, 404]
    for idx in frames:
        rgb_in = train_rgb[idx].copy()       # BGR uint8
        lbl_in = train_label[idx].copy()
        wire_mask = (lbl_in <= 3)            # foreground / wire (PROTECTED)
        bg_mask = ~wire_mask                  # background (perturbed)

        cells = [overlay_outline(rgb_in, wire_mask)]
        for vi in range(N_VARIANTS):
            # force every op on so the variant is maximally busy for review,
            # with an isolated seeded RNG so the grid is deterministic.
            rng = np.random.default_rng(variant_seeds[vi] + idx)
            clut = aug.perturb(rgb_in.copy(), bg_mask, rng=rng,
                               force_ops=["texture", "patches", "photometric"])
            # safety assertion INSIDE the smoke too: foreground identical.
            assert np.array_equal(clut[wire_mask], rgb_in[wire_mask]), \
                f"FOREGROUND changed for variant={vi} frame={idx}"
            cells.append(overlay_outline(clut, wire_mask))

        # background-mask overlay column
        cells.append(overlay_outline(tint_bg(rgb_in, bg_mask), wire_mask))

        sep = np.full((H, GAP, 3), 60, dtype=np.uint8)
        row = cells[0]
        for c in cells[1:]:
            row = np.concatenate([row, sep, c], axis=1)
        rows_imgs.append(row)

    row_w = rows_imgs[0].shape[1]

    header_cells = [label_strip(W, t) for t in COL_TITLES]
    sep_h = np.full((22, GAP, 3), 60, dtype=np.uint8)
    header = header_cells[0]
    for hc in header_cells[1:]:
        header = np.concatenate([header, sep_h, hc], axis=1)

    vgap = np.full((GAP, row_w, 3), 60, dtype=np.uint8)
    grid = header
    for row in rows_imgs:
        grid = np.concatenate([grid, vgap, row], axis=0)

    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(OUT_PNG, grid)
    print(f"  grid shape: {grid.shape}")
    print(f"  saved: {OUT_PNG}")
    with open(os.path.join(OUT_DIR, "bgclutter_frames.txt"), "w") as f:
        for r, idx in enumerate(frames):
            f.write(f"row{r}\tframe_idx={idx}\t{names[idx]}\t"
                    f"wire_px={int((train_label[idx] <= 3).sum())}\n")
    print(f"  frame map: {os.path.join(OUT_DIR, 'bgclutter_frames.txt')}")


if __name__ == "__main__":
    main()
