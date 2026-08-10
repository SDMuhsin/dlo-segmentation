#!/usr/bin/env python3
"""Restage FULL-PRECISION depth for the 3-way de-cheat dataset so the
connector-vs-wire LOCAL depth contrast survives normalization.

WHY THIS EXISTS
---------------
The staged 8-bit depth (``data/dformer_dataset_3way_decheat/Depth/*.png``) is
normalized GLOBALLY over the whole frame range (~0..1400 mm). The DLO body only
spans ~42 mm in depth, and a connector sits only ~6-28 mm proud of the adjacent
cable, so after a global 8-bit normalize that contrast collapses to ~1-2 of 255
units (≈2 % of dynamic range) -- WASHED OUT. A model can never learn connector
from that.

The RAW render keeps full-precision depth:
    data/render_3way_decheat/{train,val}/<sid>/depth/<frame>_<anim>_<view>.png
as uint16 millimetres. This script re-normalizes each frame PER-FRAME over a
ROBUST window of just the DLO (wire ∪ connector) region -- so the 42 mm DLO span
fills the full 0..255 range and the connector's local lift becomes ~12-15 % of
dynamic range (clearly learnable). The window is derived from the staged
``Label`` PNG ({0=bg,1=wire,2=connector}); it is deterministic and IDENTICAL for
train and val (no augmentation, no randomness).

It produces a uint8 mmap cache ``cache/{split}_depth_fp.npy`` aligned 1:1 (same
order) with the existing ``cache/{split}_{rgb,label}.npy`` built by
``train_rgbd_seg.build_cache`` -- so the DFormer trainer can swap depth in by
index with no other change.

Staged filename <-> raw mapping:
    staged ``032_0000_00_front``  (sid 032, src-frame 0000, anim 00, view front)
    train split  -> data/render_3way_decheat/train/032/depth/0000_00_front.png
    val   split  -> data/render_3way_decheat/val/032/depth/0000_00_front.png
(train.txt ids live under render/train/, test.txt ids under render/val/.)

USAGE
-----
    # Build both depth caches (train + val):
    ./env/bin/python src/restage_depth_fullprecision.py --build

    # Run the make-or-break depth-contrast GATE on N val frames:
    ./env/bin/python src/restage_depth_fullprecision.py --gate --gate-frames 30
"""

import argparse
import os

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "data", "dformer_dataset_3way_decheat")
RENDER_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "data", "render_3way_decheat")

IMAGE_H, IMAGE_W = 480, 640

# Robust DLO-window percentiles + symmetric padding (fraction of the p2..p98
# span) so the nearest connector pixels do NOT clip onto the unrendered-zero
# floor (which would alias connector with background). Pad keeps connector ~>=0.08
# while the unrendered background stays exactly 0.
_PCT_LO, _PCT_HI = 2.0, 98.0
_PAD_FRAC = 0.10


# ─────────────────────── core normalization ───────────────────────


def normalize_depth_frame(raw_u16, dlo_mask):
    """Per-frame normalize a uint16-mm depth map to uint8, windowed over the
    DLO region so the connector-vs-wire LOCAL contrast survives.

    raw_u16  : (H, W) uint16 depth in mm; 0 == unrendered/background.
    dlo_mask : (H, W) bool, True on DLO pixels (label wire ∪ connector). The
               robust [p2,p98] window is computed over rendered DLO depths only.

    Deterministic (no RNG). Returns (H, W) uint8. Unrendered (raw==0) -> 0.
    Falls back to the whole nonzero frame if the DLO region is too small, and
    to all-zeros if the frame has no usable depth (degenerate; never NaN).
    """
    d = raw_u16.astype(np.float32)
    rendered = raw_u16 > 0
    dlo_vals = d[dlo_mask & rendered]
    if dlo_vals.size >= 50:
        lo, hi = np.percentile(dlo_vals, [_PCT_LO, _PCT_HI])
    else:
        nz = d[rendered]
        if nz.size < 50:
            return np.zeros(d.shape, np.uint8)
        lo, hi = np.percentile(nz, [_PCT_LO, _PCT_HI])
    span = hi - lo
    if span < 1e-3:
        return np.zeros(d.shape, np.uint8)
    lo -= _PAD_FRAC * span
    hi += _PAD_FRAC * span
    nd = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    nd[~rendered] = 0.0  # keep unrendered background pinned at the 0 floor
    return (nd * 255.0).round().astype(np.uint8)


# ─────────────────────── filename <-> raw mapping ───────────────────────


def _render_split(split):
    # build_cache uses split in {"train","val"}; train.txt ids live under
    # render/train/, test.txt (val) ids under render/val/.
    return "train" if split == "train" else "val"


def raw_depth_path(render_dir, split, base):
    """base == '<sid>_<frame>_<anim>_<view>' -> raw uint16 depth png path."""
    sid, frame, anim, view = base.split("_")
    return os.path.join(render_dir, _render_split(split), sid, "depth",
                        f"{frame}_{anim}_{view}.png")


def file_list(data_dir, split):
    txt = os.path.join(data_dir, "train.txt" if split == "train" else "test.txt")
    with open(txt) as f:
        return [line.strip() for line in f if line.strip()]


def _base_of(entry):
    # entry like 'RGB/032_0000_00_front.png' -> '032_0000_00_front'
    return os.path.basename(entry).replace(".png", "")


# ─────────────────────── cache builder ───────────────────────


def build_depth_fp_cache(data_dir=DATASET_DIR_DEFAULT, render_dir=RENDER_DIR_DEFAULT,
                         split="train", rebuild=False, verbose=True):
    """Build (or load) the full-precision uint8 depth mmap for a split.

    Returns a read-only (N, H, W) uint8 mmap aligned 1:1 with the existing
    cache/{split}_{rgb,label}.npy ordering (same train.txt/test.txt order).
    """
    cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"{split}_depth_fp.npy")

    entries = file_list(data_dir, split)
    n = len(entries)

    if os.path.exists(out_path) and not rebuild:
        arr = np.load(out_path, mmap_mode="r")
        if arr.shape[0] == n:
            if verbose:
                print(f"  [depth_fp] loaded cached {split}: {arr.shape}")
            return arr
        if verbose:
            print(f"  [depth_fp] cache size {arr.shape[0]} != {n}; rebuilding")

    label_dir = os.path.join(data_dir, "Label")
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8,
                                    shape=(n, IMAGE_H, IMAGE_W))
    if verbose:
        print(f"  [depth_fp] building {split} ({n} frames) -> {out_path}")
    n_missing = 0
    for i, entry in enumerate(entries):
        base = _base_of(entry)
        rp = raw_depth_path(render_dir, split, base)
        raw = cv2.imread(rp, cv2.IMREAD_UNCHANGED)
        lbl = cv2.imread(os.path.join(label_dir, base + ".png"), cv2.IMREAD_GRAYSCALE)
        if raw is None or lbl is None:
            n_missing += 1
            arr[i] = 0
            continue
        if raw.dtype != np.uint16:
            raw = raw.astype(np.uint16)
        arr[i] = normalize_depth_frame(raw, lbl >= 1)
        if verbose and (i + 1) % 2000 == 0:
            print(f"    {i + 1}/{n}")
    arr.flush()
    if n_missing and verbose:
        print(f"  [depth_fp] WARNING: {n_missing} frame(s) missing raw/label -> zeroed")
    if verbose:
        print(f"  [depth_fp] done {split}: {arr.shape}")
    return np.load(out_path, mmap_mode="r")


# ─────────────────────── depth-contrast GATE ───────────────────────


def run_gate(data_dir=DATASET_DIR_DEFAULT, render_dir=RENDER_DIR_DEFAULT,
             n_frames=30, ring=15):
    """Measure the median |depth(connector) - depth(adjacent-wire ring)| margin
    on N val frames, in the FINAL normalized uint8 the model receives. Reports
    the margin as a fraction of dynamic range and contrasts it with the broken
    global 8-bit wash. This is the make-or-break feasibility gate."""
    entries = file_list(data_dir, "val")
    bases = [_base_of(e) for e in entries]
    step = max(len(bases) // n_frames, 1)
    sample = bases[::step][:n_frames]

    raw_mm, fp_frac, global_frac = [], [], []
    used = 0
    label_dir = os.path.join(data_dir, "Label")
    for b in sample:
        rp = raw_depth_path(render_dir, "val", b)
        if not os.path.isfile(rp):
            continue
        raw = cv2.imread(rp, cv2.IMREAD_UNCHANGED)
        lbl = cv2.imread(os.path.join(label_dir, b + ".png"), cv2.IMREAD_GRAYSCALE)
        if raw is None or lbl is None:
            continue
        d = raw.astype(np.float32)
        con = (lbl == 2) & (raw > 0)
        wire = (lbl == 1) & (raw > 0)
        if con.sum() < 30 or wire.sum() < 30:
            continue
        cm = con.astype(np.uint8)
        ring_band = cv2.dilate(cm, np.ones((ring, ring), np.uint8)) - cm
        ring_wire = (ring_band > 0) & wire
        if ring_wire.sum() < 20:
            ring_wire = wire
        # raw millimetre margin
        raw_mm.append(abs(float(np.median(d[con])) - float(np.median(d[ring_wire]))))
        # FINAL normalized (the exact uint8 the trainer feeds, /255)
        fp = normalize_depth_frame(raw if raw.dtype == np.uint16 else raw.astype(np.uint16),
                                   lbl >= 1).astype(np.float32)
        fp_frac.append(abs(float(np.median(fp[con])) - float(np.median(fp[ring_wire]))) / 255.0)
        # broken global 8-bit wash (max-normalized whole frame) for contrast
        mx = max(float(d.max()), 1.0)
        g = (np.clip(d / mx, 0, 1) * 255).round()
        global_frac.append(abs(float(np.median(g[con])) - float(np.median(g[ring_wire]))) / 255.0)
        used += 1

    raw_mm = np.array(raw_mm); fp_frac = np.array(fp_frac); global_frac = np.array(global_frac)
    print("=" * 64)
    print(f"DEPTH-CONTRAST GATE  ({used} val frames, adjacent-ring={ring}px)")
    print("=" * 64)
    print(f"  RAW |connector - adjacent wire|        : median {np.median(raw_mm):6.2f} mm "
          f"(>=5mm in {100*(raw_mm>=5).mean():.0f}% of frames)")
    print(f"  FINAL DLO-window norm (model input)    : median {100*np.median(fp_frac):6.2f}% of range "
          f"(>=5%% in {100*(fp_frac>=0.05).mean():.0f}% of frames)")
    print(f"  (broken) global 8-bit wash, for contrast: median {100*np.median(global_frac):6.2f}% of range")
    passed = np.median(fp_frac) >= 0.05
    print(f"  GATE {'PASS' if passed else 'FAIL'}: connector vs adjacent wire is "
          f"{'a clear, learnable margin' if passed else 'WASHED OUT'} "
          f"({100*np.median(fp_frac):.1f}% >= 5% target)")
    print("=" * 64)
    return passed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DATASET_DIR_DEFAULT)
    p.add_argument("--render-dir", default=RENDER_DIR_DEFAULT)
    p.add_argument("--build", action="store_true", help="build train+val depth_fp caches")
    p.add_argument("--rebuild", action="store_true", help="force rebuild even if cache exists")
    p.add_argument("--splits", default="train,val", help="comma list of splits to build")
    p.add_argument("--gate", action="store_true", help="run the depth-contrast gate")
    p.add_argument("--gate-frames", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    if args.gate:
        run_gate(args.data_dir, args.render_dir, n_frames=args.gate_frames)
    if args.build:
        for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
            build_depth_fp_cache(args.data_dir, args.render_dir, split,
                                 rebuild=args.rebuild)
    if not args.gate and not args.build:
        print("nothing to do; pass --gate and/or --build")


if __name__ == "__main__":
    main()
