#!/usr/bin/env python
"""Hard validation gate for the converted real-textured binary wire dataset (P28).

Adapted from src/validate_rmdlo_dataset.py. Saves to
results/realism_campaign/p28_realtextured/convert_validation/:
  overlay_montage.png    random RGB+mask overlays (mask alignment check)
  overlay_sidebyside.png [rgb | mask | overlay] strips
  rgbd_alignment.png     [rgb | label | depth-vis | overlay]
  fg_fraction.json       fg distribution + checks
  summary.txt            human verdict

Acceptance: 0 all-zero, 0 all-one, 0 three-way-misdetect (legacy path triggers),
label encoding {0,4}, depth dtype uint16, RGB/Label/Depth aligned (same shape).
"""
import os
import sys
import json
import glob
import random

import cv2
import numpy as np

OUT = "results/realism_campaign/p28_realtextured/convert_validation"


def load_label_binary(label_path):
    """Replicate trainer build_cache+collapse: label{0,4} -> legacy -> wire=1/bg=0."""
    l = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    three_way = int(l.max()) <= 2
    if three_way:
        cache = l.astype(np.uint8)
    else:
        c = l.astype(np.int16) - 1
        c[c < 0] = 255
        cache = c.astype(np.uint8)
    return (cache <= 3).astype(np.uint8), three_way


def depth_vis(depth_path):
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None, False, 0
    is16 = (d.dtype == np.uint16)
    dv = np.clip(d.astype(np.float32) / max(int(d.max()), 1) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(dv, cv2.COLORMAP_JET), is16, int(d.max())


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/dformer_dataset_realtextured"
    os.makedirs(OUT, exist_ok=True)
    rgb_dir = os.path.join(data_dir, "RGB")
    label_dir = os.path.join(data_dir, "Label")
    depth_dir = os.path.join(data_dir, "Depth")

    bases = sorted(os.path.basename(p)[:-4]
                   for p in glob.glob(os.path.join(rgb_dir, "*.png")))
    print(f"dataset {data_dir}: {len(bases)} frames")

    fgs, three_way_flags = [], []
    label_vals = set()
    depth_u16_ok = 0
    depth_present = 0
    n_aligned = 0
    n_total = 0
    for b in bases:
        n_total += 1
        lp = os.path.join(label_dir, b + ".png")
        wire, tw = load_label_binary(lp)
        fgs.append(float(wire.mean()))
        three_way_flags.append(tw)
        lraw = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        label_vals.update(np.unique(lraw).tolist())

        rgb = cv2.imread(os.path.join(rgb_dir, b + ".png"), cv2.IMREAD_COLOR)
        dp = os.path.join(depth_dir, b + ".png")
        d = cv2.imread(dp, cv2.IMREAD_UNCHANGED) if os.path.exists(dp) else None
        if os.path.exists(dp):
            depth_present += 1
            depth_u16_ok += int(d is not None and d.dtype == np.uint16)
        # alignment: rgb, label, depth all same HxW
        shapes = {rgb.shape[:2], lraw.shape[:2]}
        if d is not None:
            shapes.add(d.shape[:2])
        if len(shapes) == 1:
            n_aligned += 1

    fgs = np.array(fgs)
    stats = {
        "n_frames": len(bases),
        "fg_fraction_mean": float(fgs.mean()),
        "fg_fraction_median": float(np.median(fgs)),
        "fg_fraction_min": float(fgs.min()),
        "fg_fraction_max": float(fgs.max()),
        "fg_fraction_p05": float(np.percentile(fgs, 5)),
        "fg_fraction_p95": float(np.percentile(fgs, 95)),
        "n_all_zero_masks": int((fgs == 0).sum()),
        "n_all_one_masks": int((fgs > 0.99).sum()),
        "n_label_three_way_misdetect": int(sum(three_way_flags)),
        "label_unique_values": sorted(int(x) for x in label_vals),
        "label_encoding_ok_wire4_bg0": sorted(int(x) for x in label_vals) == [0, 4],
        "depth_present": depth_present,
        "depth_uint16_frames": depth_u16_ok,
        "depth_all_uint16": depth_u16_ok == len(bases),
        "n_rgb_label_depth_aligned": n_aligned,
        "all_aligned": n_aligned == n_total,
    }
    with open(os.path.join(OUT, "fg_fraction.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("fg-fraction + checks:", json.dumps(stats, indent=2))

    random.seed(0)
    sample = random.sample(bases, min(20, len(bases)))
    tiles = []
    for b in sample:
        rgb = cv2.imread(os.path.join(rgb_dir, b + ".png"), cv2.IMREAD_COLOR)
        wire, _ = load_label_binary(os.path.join(label_dir, b + ".png"))
        ov = rgb.copy(); ov[wire > 0] = (0, 0, 255)
        blend = cv2.addWeighted(rgb, 0.5, ov, 0.5, 0)
        tiles.append(cv2.resize(blend, (240, 180)))
    while len(tiles) < 20:
        tiles.append(np.zeros((180, 240, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + 5]) for i in range(0, 20, 5)]
    cv2.imwrite(os.path.join(OUT, "overlay_montage.png"), np.vstack(rows))

    sbs = []
    for b in sample[:6]:
        rgb = cv2.imread(os.path.join(rgb_dir, b + ".png"), cv2.IMREAD_COLOR)
        wire, _ = load_label_binary(os.path.join(label_dir, b + ".png"))
        ov = rgb.copy(); ov[wire > 0] = (0, 0, 255)
        blend = cv2.addWeighted(rgb, 0.5, ov, 0.5, 0)
        m = cv2.cvtColor(wire * 255, cv2.COLOR_GRAY2BGR)
        sbs.append(cv2.resize(np.hstack([rgb, m, blend]), (720, 180)))
    cv2.imwrite(os.path.join(OUT, "overlay_sidebyside.png"), np.vstack(sbs))

    rows = []
    for b in sample[:6]:
        rgb = cv2.imread(os.path.join(rgb_dir, b + ".png"), cv2.IMREAD_COLOR)
        wire, _ = load_label_binary(os.path.join(label_dir, b + ".png"))
        m = cv2.cvtColor(wire * 255, cv2.COLOR_GRAY2BGR)
        dv, _, _ = depth_vis(os.path.join(depth_dir, b + ".png"))
        if dv is None:
            dv = np.zeros_like(rgb)
        ov = rgb.copy(); ov[wire > 0] = (0, 0, 255)
        blend = cv2.addWeighted(rgb, 0.5, ov, 0.5, 0)
        rows.append(cv2.resize(np.hstack([rgb, m, dv, blend]), (960, 180)))
    cv2.imwrite(os.path.join(OUT, "rgbd_alignment.png"), np.vstack(rows))

    verdict = (stats["n_all_zero_masks"] == 0 and stats["n_all_one_masks"] == 0
               and stats["n_label_three_way_misdetect"] == 0
               and stats["label_encoding_ok_wire4_bg0"]
               and stats["depth_all_uint16"]
               and stats["all_aligned"])
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write(json.dumps(stats, indent=2))
        f.write(f"\n\nACCEPTANCE PASS: {verdict}\n")
    print(f"\nACCEPTANCE PASS: {verdict}")
    print("artifacts written to", OUT)


if __name__ == "__main__":
    main()
