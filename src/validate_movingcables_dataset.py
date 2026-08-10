#!/usr/bin/env python
"""Hard validation gate for the converted MovingCables binary wire dataset.

Saves artifacts to results/realism_campaign/p28_movingcables/validation/:
  overlay_montage_*.png   24 random RGB+mask overlays (mask alignment check)
  raw_rgb_crops.png       12 raw RGB crops (UV/chroma/domain check)
  fg_fraction.json        fg-pixel-fraction distribution + all-zero/all-one flags
  summary.txt             human-readable verdict block
"""
import os
import sys
import json
import glob
import random

import cv2
import numpy as np

OUT = "results/realism_campaign/p28_movingcables/validation"


def load_label_binary(label_path):
    """Replicate trainer build_cache+collapse: label{0,4} -> legacy -> wire=1/bg=0."""
    l = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    three_way = l.max() <= 2
    if three_way:
        cache = l.astype(np.uint8)
    else:
        c = l.astype(np.int16) - 1
        c[c < 0] = 255
        cache = c.astype(np.uint8)
    return (cache <= 3).astype(np.uint8), three_way


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/dformer_dataset_movingcables"
    os.makedirs(OUT, exist_ok=True)
    rgb_dir = os.path.join(data_dir, "RGB")
    label_dir = os.path.join(data_dir, "Label")

    bases = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(rgb_dir, "*.png")))
    print(f"dataset {data_dir}: {len(bases)} frames")

    # ---- (b) fg-fraction distribution over ALL frames ----
    fgs = []
    three_way_flags = []
    for b in bases:
        wire, tw = load_label_binary(os.path.join(label_dir, b + ".png"))
        fgs.append(float(wire.mean()))
        three_way_flags.append(tw)
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
        "n_near_all_one_masks": int((fgs > 0.9).sum()),
        "n_label_three_way_misdetect": int(sum(three_way_flags)),
    }
    with open(os.path.join(OUT, "fg_fraction.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("fg-fraction:", json.dumps(stats, indent=2))

    # ---- (a) 24 random overlay montages (mask alignment) ----
    random.seed(0)
    sample = random.sample(bases, min(24, len(bases)))
    tiles = []
    for b in sample:
        rgb = cv2.imread(os.path.join(rgb_dir, b + ".png"), cv2.IMREAD_COLOR)
        wire, _ = load_label_binary(os.path.join(label_dir, b + ".png"))
        ov = rgb.copy()
        ov[wire > 0] = (0, 0, 255)
        blend = cv2.addWeighted(rgb, 0.5, ov, 0.5, 0)
        tile = cv2.resize(blend, (240, 180))
        tiles.append(tile)
    # 6x4 grid
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, 24, 6)]
    montage = np.vstack(rows)
    cv2.imwrite(os.path.join(OUT, "overlay_montage.png"), montage)

    # also a side-by-side [rgb | mask | overlay] for 6 frames for crisp alignment view
    sbs = []
    for b in sample[:6]:
        rgb = cv2.imread(os.path.join(rgb_dir, b + ".png"), cv2.IMREAD_COLOR)
        wire, _ = load_label_binary(os.path.join(label_dir, b + ".png"))
        ov = rgb.copy(); ov[wire > 0] = (0, 0, 255)
        blend = cv2.addWeighted(rgb, 0.5, ov, 0.5, 0)
        m = cv2.cvtColor(wire * 255, cv2.COLOR_GRAY2BGR)
        sbs.append(cv2.resize(np.hstack([rgb, m, blend]), (720, 180)))
    cv2.imwrite(os.path.join(OUT, "overlay_sidebyside.png"), np.vstack(sbs))

    # ---- (c) 12 raw RGB crops (UV/chroma/domain) ----
    crops = random.sample(bases, min(12, len(bases)))
    ct = [cv2.resize(cv2.imread(os.path.join(rgb_dir, b + ".png"), cv2.IMREAD_COLOR), (320, 240))
          for b in crops]
    rows = [np.hstack(ct[i:i + 4]) for i in range(0, 12, 4)]
    cv2.imwrite(os.path.join(OUT, "raw_rgb_crops.png"), np.vstack(rows))

    # colour stats for domain judgement
    bgr_means = []
    sat_means = []
    for b in crops:
        im = cv2.imread(os.path.join(rgb_dir, b + ".png"), cv2.IMREAD_COLOR)
        bgr_means.append(im.reshape(-1, 3).mean(0))
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        sat_means.append(hsv[:, :, 1].mean())
    bgr_means = np.array(bgr_means).mean(0)
    color = {
        "mean_B": float(bgr_means[0]), "mean_G": float(bgr_means[1]), "mean_R": float(bgr_means[2]),
        "mean_saturation": float(np.mean(sat_means)),
    }
    with open(os.path.join(OUT, "color_stats.json"), "w") as f:
        json.dump(color, f, indent=2)
    print("color stats:", color)
    print("artifacts written to", OUT)


if __name__ == "__main__":
    main()
