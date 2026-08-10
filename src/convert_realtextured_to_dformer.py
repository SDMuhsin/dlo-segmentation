#!/usr/bin/env python
"""Convert the assembled REAL THIN-CABLE-ON-TEXTURED-BACKGROUND sources into the
binary wire-segmentation training layout consumed by ``src/train_rgb_only_sota.py``
(P28 real-textured pool).

Mirrors ``src/convert_movingcables_to_dformer.py`` and
``src/convert_rmdlo_to_dformer.py`` exactly:
  RGB/<base>.png    BGR, letterboxed to 480x640 (INTER_AREA)
  Label/<base>.png  {0,4} grayscale, letterboxed 480x640 (INTER_NEAREST)
  Depth/<base>.png  zeros uint16, 480x640 (RGB-only co-train; carried for RGB-D layout)
  train.txt / test.txt   lines "RGB/<base>.png"

Label encoding D (legacy): wire -> 4, bg -> 0. build_cache auto-detects mode from the
FIRST label's max: max<=2 => three_way, else legacy. With wire=4 every frame has max>=3
=> LEGACY forced, gt maps 4->3 and 0->255, binary collapse cache<=3 => wire. Mode-stable.

Sources (each = one SOURCE group with its own set-id block; split BY SOURCE/SCENE so the
~15-20% val split does not leak near-duplicate frames):

  bwh        CVF-DLO/DATASETS/BWH        imgs/<b>.jpg  + masks/<b>.png (binary 0/255)
  labd_real  CVF-DLO/DATASETS/LABD_Real  imgs/<b>.jpg  + masks/<b>.png (binary 0/255)
  m4nh       cables_dataset/cable_dataset_{simple,hard}  <b>.jpg + <b>_mask_all.png (0/255)
  putvision  putvision_extract/data      img/<b>.jpg   + mask/<b>.jpg  (lossy-JPEG binary)

EWD / SBHC (black-bg) / RT-DLO are DELIBERATELY ABSENT (EWD = forbidden ElectricWires;
RT-DLO overlaps the held-out valset; SBHC = plain black background, recall-only).

Provided masks are used AS-IS (never re-thresholded by colour guesswork). Binary masks
are thresholded at >0 (clean 0/255) or >=128 (lossy-JPEG masks: PUTvision, like the
valset's own lossy-JPEG masks).

Basename convention: ``{setid:03d}_{frameidx:04d}_00_rt.png``. setid is a contiguous
per-source-image integer so ``filter_indices_by_set`` (int(basename.split('_')[0]))
groups frames; here each kept image is its own set (one frame per set, no temporal
duplication) and the train/val split is assigned per-source.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

WIRE_VAL = 4
BG_VAL = 0
IMAGE_H, IMAGE_W = 480, 640


def letterbox(img, target_h, target_w, interp, pad_value=0):
    """Aspect-preserving resize + center pad to (target_h, target_w)."""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    if img.ndim == 3:
        canvas = np.full((target_h, target_w, img.shape[2]), pad_value, img.dtype)
    else:
        canvas = np.full((target_h, target_w), pad_value, img.dtype)
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def load_binary_mask(path, thr):
    """Read a provided mask and binarize wire vs bg. thr: 0 -> >0, else >=thr."""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if thr <= 0:
        return (m > 0).astype(np.uint8)
    return (m >= thr).astype(np.uint8)


def discover_pairs(args):
    """Return a list of source groups. Each group is
        {"name": str, "pairs": [(img_path, mask_path, mask_thr), ...]}.
    Only image/mask pairs that both exist are kept."""
    R = args.raw_root
    groups = []

    # --- BWH (CVF-DLO) : imgs/<b>.jpg + masks/<b>.png, clean 0/255 ---
    bwh_img = os.path.join(R, "CVF-DLO/DATASETS/BWH/imgs")
    bwh_msk = os.path.join(R, "CVF-DLO/DATASETS/BWH/masks")
    pairs = []
    for ip in sorted(glob.glob(os.path.join(bwh_img, "*.jpg"))):
        b = os.path.basename(ip)[:-4]
        mp = os.path.join(bwh_msk, b + ".png")
        if os.path.exists(mp):
            pairs.append((ip, mp, 0))
    if pairs:
        groups.append({"name": "bwh", "pairs": pairs})

    # --- LABD_Real (CVF-DLO) : imgs/<b>.jpg + masks/<b>.png, clean 0/255 ---
    lab_img = os.path.join(R, "CVF-DLO/DATASETS/LABD_Real/imgs")
    lab_msk = os.path.join(R, "CVF-DLO/DATASETS/LABD_Real/masks")
    pairs = []
    for ip in sorted(glob.glob(os.path.join(lab_img, "*.jpg"))):
        b = os.path.basename(ip)[:-4]
        mp = os.path.join(lab_msk, b + ".png")
        if os.path.exists(mp):
            pairs.append((ip, mp, 0))
    if pairs:
        groups.append({"name": "labd_real", "pairs": pairs})

    # --- m4nh : cable_dataset_{simple,hard}/<b>.jpg + <b>_mask_all.png, 0/255 ---
    pairs = []
    for sub in ("cable_dataset_simple", "cable_dataset_hard"):
        d = os.path.join(R, "cables_dataset", sub)
        for ip in sorted(glob.glob(os.path.join(d, "*.jpg"))):
            b = os.path.basename(ip)[:-4]
            mp = os.path.join(d, b + "_mask_all.png")
            if os.path.exists(mp):
                pairs.append((ip, mp, 0))
    if pairs:
        groups.append({"name": "m4nh", "pairs": pairs})

    # --- PUTvision : data/img/<b>.jpg + data/mask/<b>.jpg, lossy-JPEG binary (>=128) ---
    pv_img = os.path.join(R, "putvision_extract/data/img")
    pv_msk = os.path.join(R, "putvision_extract/data/mask")
    pairs = []
    for ip in sorted(glob.glob(os.path.join(pv_img, "*.jpg"))):
        b = os.path.basename(ip)[:-4]
        mp = os.path.join(pv_msk, b + ".jpg")
        if os.path.exists(mp):
            pairs.append((ip, mp, 128))
    if pairs:
        groups.append({"name": "putvision", "pairs": pairs})

    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default="data/realtextured_raw",
                    help="dir containing CVF-DLO/, cables_dataset/, putvision_extract/")
    ap.add_argument("--out-dir", default="data/dformer_dataset_realtextured")
    ap.add_argument("--val-frac", type=float, default=0.18,
                    help="fraction of images per source held out for val (default 0.18)")
    ap.add_argument("--min-fg-px", type=int, default=80,
                    help="drop frames whose binarized mask has fewer fg px")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    groups = discover_pairs(args)
    if not groups:
        raise SystemExit(f"no source pairs found under {args.raw_root}")
    print("source groups:")
    for g in groups:
        print(f"  {g['name']:12s} {len(g['pairs'])} img/mask pairs")

    rgb_out = os.path.join(args.out_dir, "RGB")
    depth_out = os.path.join(args.out_dir, "Depth")
    label_out = os.path.join(args.out_dir, "Label")
    for d in (rgb_out, depth_out, label_out):
        os.makedirs(d, exist_ok=True)

    zero_depth = np.zeros((IMAGE_H, IMAGE_W), np.uint16)
    rng = np.random.default_rng(args.seed)

    train_lines, val_lines = [], []
    fg_fracs = []
    n_written = 0
    n_dropped = 0
    setid = 0
    per_source = []

    for g in groups:
        pairs = g["pairs"]
        n = len(pairs)
        # split BY SCENE within this source: shuffle then hold out the tail val-frac.
        perm = rng.permutation(n)
        n_val = max(1, int(round(n * args.val_frac))) if n > 1 else 0
        val_idx = set(perm[:n_val].tolist())
        src_written = 0
        src_val = 0
        for i, (ip, mp, thr) in enumerate(pairs):
            rgb = cv2.imread(ip, cv2.IMREAD_COLOR)
            wire = load_binary_mask(mp, thr)
            if rgb is None or wire is None:
                n_dropped += 1
                continue
            if wire.shape[:2] != rgb.shape[:2]:
                wire = cv2.resize(wire, (rgb.shape[1], rgb.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            if int(wire.sum()) < args.min_fg_px:
                n_dropped += 1
                continue

            rgb_lb = letterbox(rgb, IMAGE_H, IMAGE_W, cv2.INTER_AREA, pad_value=0)
            wire_lb = letterbox(wire, IMAGE_H, IMAGE_W, cv2.INTER_NEAREST, pad_value=0)
            lbl = np.where(wire_lb > 0, WIRE_VAL, BG_VAL).astype(np.uint8)
            fg_fracs.append(float((wire_lb > 0).mean()))

            base = f"{setid:03d}_{0:04d}_00_rt"
            cv2.imwrite(os.path.join(rgb_out, base + ".png"), rgb_lb)
            cv2.imwrite(os.path.join(depth_out, base + ".png"), zero_depth)
            cv2.imwrite(os.path.join(label_out, base + ".png"), lbl)
            line = f"RGB/{base}.png"
            if i in val_idx:
                val_lines.append(line)
                src_val += 1
            else:
                train_lines.append(line)
            n_written += 1
            src_written += 1
            setid += 1
        per_source.append({"source": g["name"], "pairs": n,
                           "written": src_written, "val": src_val,
                           "train": src_written - src_val})
        print(f"  [{g['name']}] {src_written} written "
              f"(train {src_written - src_val} / val {src_val})")

    with open(os.path.join(args.out_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_lines) + ("\n" if train_lines else ""))
    with open(os.path.join(args.out_dir, "test.txt"), "w") as f:
        f.write("\n".join(val_lines) + ("\n" if val_lines else ""))

    fg = np.array(fg_fracs) if fg_fracs else np.zeros(1)
    meta = {
        "out_dir": args.out_dir,
        "n_written": n_written,
        "n_dropped": n_dropped,
        "train": len(train_lines), "val": len(val_lines),
        "val_frac": args.val_frac,
        "fg_fraction_mean": float(fg.mean()),
        "fg_fraction_median": float(np.median(fg)),
        "fg_fraction_min": float(fg.min()),
        "fg_fraction_max": float(fg.max()),
        "n_all_zero": int((fg == 0).sum()),
        "n_all_one": int((fg > 0.99).sum()),
        "per_source": per_source,
    }
    with open(os.path.join(args.out_dir, "convert_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== REAL-TEXTURED CONVERSION DONE ===")
    print(f"frames written: {n_written}  (train {len(train_lines)} / val {len(val_lines)})")
    print(f"dropped: {n_dropped}")
    print(f"fg-fraction: mean={fg.mean():.4f} median={np.median(fg):.4f} "
          f"min={fg.min():.4f} max={fg.max():.4f}")
    print(f"all-zero: {(fg == 0).sum()}  all-one(>0.99): {(fg > 0.99).sum()}")
    print(f"out-dir: {args.out_dir}")


if __name__ == "__main__":
    main()
