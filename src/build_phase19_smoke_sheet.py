"""Phase 19 busy-negative-texture smoke contact sheet + texture montage.

Side-by-side Phase 18 base (P19 OFF) vs Phase 19 (busy backdrop p=0.35 via
KIAT_P19_BUSYBG_DIR/_P + busy floor p=0.35 via KIAT_P19_BUSYFLOOR_P) on the
SAME source frames, with a wire-label overlay column proving (a) busy
backdrops/floors appear at roughly the knob rate, (b) wires are still
correctly labelled ON busy backgrounds, (c) no painted stroke is ever
labelled wire (the busy pixels live only in the 2D backdrop / floor texture,
which carry no label), and (d) the P18 wire colours/thickness and the rest
of the scene composition are untouched.

Both renders are the stride-12 smoke twins under
results/realism_campaign/p19_smoke/ (depth/label/npys of the twins verified
identical; rgb differs only on busy frames).

Also writes a 12-texture montage (2 per family, darkest + lightest base) of
the generated busy pool.

Usage:
    python src/build_phase19_smoke_sheet.py \
        --out results/realism_campaign/p19_smoke/contact_sheet.png \
        --montage results/realism_campaign/p19_smoke/texture_montage.png
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np

OFF = "results/realism_campaign/p19_smoke/render_off/train/000"  # P18 base
ON = "results/realism_campaign/p19_smoke/render/train/000"       # P19 ON
BUSY_DIR = "data/textures/busy_negatives_p19"
SET_ID = 0
BUSY_P = 0.35
COLS = [
    ("P18 base OFF  rgb", OFF, False),
    ("P19 ON  rgb (busy bg/floor)", ON, False),
    ("P19 ON  label (green=wire)", ON, True),
]
# (src, view) rows — chosen to span: busy-backdrop frames across texture
# families, busy-floor-only frames (bottom view shows the floor face-on),
# busy backdrop+floor, busy WIRE-FREE, and calm controls. The busy/wire-free
# annotation is re-derived from the rng streams below, never hardcoded.
ROWS = [(0, "front"), (12, "front"), (24, "bottom"), (48, "front"),
        (60, "front"), (72, "front"), (84, "bottom"), (96, "front"),
        (120, "front"), (144, "front"), (156, "bottom"), (168, "front"),
        (204, "front"), (228, "front"), (264, "front"), (276, "front")]
PAD, HDR, LBL = 6, 34, 150


def _flags(src):
    """Re-derive the per-source P19 (+601/+602) and wire-free (+401) draws."""
    names = sorted(Path(BUSY_DIR).glob("*.png"))
    r1 = np.random.RandomState(SET_ID * 1000 + src + 601)
    bg = r1.uniform(0.0, 1.0) < BUSY_P
    bg_tex = names[r1.randint(0, len(names))].stem[7:] if bg else None
    r2 = np.random.RandomState(SET_ID * 1000 + src + 602)
    fl = r2.uniform(0.0, 1.0) < BUSY_P
    fl_tex = names[r2.randint(0, len(names))].stem[7:] if fl else None
    wf = np.random.RandomState(SET_ID * 1000 + src + 401).uniform() < 0.2
    return bg_tex, fl_tex, wf


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


def build_sheet(out_path):
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
        bg_tex, fl_tex, wf = _flags(src)
        note = []
        if bg_tex: note.append(f"BG:{bg_tex}")
        if fl_tex: note.append(f"FL:{fl_tex}")
        if not note: note.append("calm")
        if wf: note.append("WIRE-FREE")
        y0 = HDR + ri * (h + PAD) + PAD
        canvas[y0:y0 + h, 0:LBL] = _bar(f"src{src:04d} {view}", h, LBL,
                                        sub=" ".join(note))
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

    cv2.imwrite(out_path, canvas)
    print(f"wrote {out_path}  ({W}x{H})")


def build_montage(out_path, tile=256):
    """12 generated textures (darkest j=0 + lightest j=5 of each family)."""
    files = sorted(Path(BUSY_DIR).glob("*.png"))
    picks = [f for f in files if f.stem.endswith(("_0", "_5"))]
    ncol, hdr = 4, 22
    nrow = int(np.ceil(len(picks) / ncol))
    W = ncol * (tile + PAD) + PAD
    H = nrow * (tile + hdr + PAD) + PAD
    canvas = np.full((H, W, 3), 245, np.uint8)
    for i, f in enumerate(picks):
        r, c = divmod(i, ncol)
        x0 = PAD + c * (tile + PAD)
        y0 = PAD + r * (tile + hdr + PAD)
        canvas[y0:y0 + hdr, x0:x0 + tile] = _bar(f.stem[4:], hdr, tile)
        img = cv2.resize(cv2.imread(str(f)), (tile, tile),
                         interpolation=cv2.INTER_AREA)
        canvas[y0 + hdr:y0 + hdr + tile, x0:x0 + tile] = img
    cv2.imwrite(out_path, canvas)
    print(f"wrote {out_path}  ({W}x{H}, {len(picks)} textures)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default="results/realism_campaign/p19_smoke/contact_sheet.png")
    ap.add_argument("--montage",
                    default="results/realism_campaign/p19_smoke/texture_montage.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    build_sheet(args.out)
    build_montage(args.montage)


if __name__ == "__main__":
    main()
