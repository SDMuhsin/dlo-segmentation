#!/usr/bin/env python
"""P29-v2 hard-negative ACCEPTANCE VALIDATION (post-build, GPU-free).

Gates (all MANDATORY):
  1. VALSET-OVERLAP: md5 of every v2 negative PNG vs all 62 real_wires_valset images -> 0
     exact hits. Perceptual hash (pHash, 64-bit DCT) of every v2 negative vs every valset
     image -> report the MINIMUM Hamming distance over all pairs; require min > 10.
  2. ALL-BACKGROUND: every Label PNG max == 0.
  3. APPEARANCE REPORT: accepted count, edge width of drawn seams (where present), color
     stats (HSV V/S, warm R-B), local surround texstd recap (vs target 25-35 / real-FP 28.9).

Writes results/realism_campaign/p29v2_coplanar_hardneg/validation_report.json and prints a
PASS/FAIL summary.
"""
import argparse
import glob
import hashlib
import json
import os

import cv2
import numpy as np


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(bgr, hash_size=8, highfreq=4):
    """64-bit DCT perceptual hash (same construction as imagehash.phash)."""
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (hash_size * highfreq, hash_size * highfreq),
                     interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(img)
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:, 1:])   # exclude DC term from the median (imagehash convention)
    bits = (low > med).flatten()
    return bits


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def local_texstd(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.boxFilter(gray, -1, (9, 9), normalize=True)
    sq = cv2.boxFilter(gray * gray, -1, (9, 9), normalize=True)
    return np.sqrt(np.clip(sq - mean * mean, 0, None))


def edge_width_px(bgr):
    """Median 10-90% rise width across the strongest gradient edges (proxy for the drawn
    seam edge softness). Sobel magnitude -> sample edge profiles; cheap global estimate:
    width ~ |grad| normalized so a step over k px has width k. We use the ratio of total
    edge mass to peak gradient as a coarse width; report median over high-grad pixels."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    # second derivative magnitude; width ~ |grad| / |laplacian| at edge ridges
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)) + 1e-3
    ridge = mag > np.percentile(mag, 99.0)
    if ridge.sum() < 50:
        return float("nan")
    w = (mag[ridge] / lap[ridge])
    return float(np.median(np.clip(w, 0.3, 5.0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neg-dir", default="data/dformer_dataset_coplanar_hardneg_v2")
    ap.add_argument("--valset", default="data/real_wires_valset")
    ap.add_argument("--report-dir", default="results/realism_campaign/p29v2_coplanar_hardneg")
    args = ap.parse_args()

    neg_rgb = sorted(glob.glob(os.path.join(args.neg_dir, "RGB", "*.png")))
    neg_lab = sorted(glob.glob(os.path.join(args.neg_dir, "Label", "*.png")))
    val_imgs = sorted(glob.glob(os.path.join(args.valset, "imgs", "*")))
    print(f"v2 negatives: {len(neg_rgb)} RGB / {len(neg_lab)} Label   valset: {len(val_imgs)}")

    # --- 1a. md5 exact-dup ---
    val_md5 = set(md5_file(p) for p in val_imgs)
    neg_md5 = {p: md5_file(p) for p in neg_rgb}
    md5_hits = [p for p, m in neg_md5.items() if m in val_md5]

    # --- 1b. pHash min Hamming over all pairs ---
    val_ph = [phash(cv2.imread(p, cv2.IMREAD_COLOR)) for p in val_imgs]
    min_ph = 64
    argmin = None
    for p in neg_rgb:
        nh = phash(cv2.imread(p, cv2.IMREAD_COLOR))
        for vp, vh in zip(val_imgs, val_ph):
            d = hamming(nh, vh)
            if d < min_ph:
                min_ph = d
                argmin = (os.path.basename(p), os.path.basename(vp))

    # --- 2. all-background ---
    label_maxes = []
    nonzero_labels = []
    for p in neg_lab:
        m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        mx = int(m.max())
        label_maxes.append(mx)
        if mx != 0:
            nonzero_labels.append(os.path.basename(p))

    # --- 3. appearance ---
    ts_med, fstd, warm, vmean, smean, ewid = [], [], [], [], [], []
    for p in neg_rgb:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        ts = local_texstd(im)
        ts_med.append(float(np.median(ts)))
        fstd.append(float(np.mean([im[..., c].std() for c in range(3)])))
        warm.append(float(im[..., 2].mean() - im[..., 0].mean()))
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        vmean.append(float(hsv[..., 2].mean()))
        smean.append(float(hsv[..., 1].mean()))
        ew = edge_width_px(im)
        if not np.isnan(ew):
            ewid.append(ew)

    report = {
        "neg_dir": args.neg_dir,
        "n_negatives": len(neg_rgb),
        "valset_overlap": {
            "md5_exact_hits": len(md5_hits),
            "md5_hit_files": md5_hits[:10],
            "phash_min_hamming": int(min_ph),
            "phash_min_pair": argmin,
            "phash_gate_min_gt_10": bool(min_ph > 10),
            "md5_gate_zero_hits": bool(len(md5_hits) == 0),
        },
        "all_background": {
            "label_max_overall": int(max(label_maxes)) if label_maxes else None,
            "n_nonzero_labels": len(nonzero_labels),
            "nonzero_files": nonzero_labels[:10],
            "gate_all_zero": bool(max(label_maxes) == 0) if label_maxes else False,
        },
        "appearance": {
            "surround_texstd_median": float(np.median(ts_med)),
            "surround_texstd_p25_p75": [float(np.percentile(ts_med, 25)),
                                        float(np.percentile(ts_med, 75))],
            "frame_perchan_std_median": float(np.median(fstd)),
            "warm_R_minus_B_median": float(np.median(warm)),
            "hsv_v_median": float(np.median(vmean)),
            "hsv_s_median": float(np.median(smean)),
            "edge_width_px_median": float(np.median(ewid)) if ewid else None,
            "targets": {"surround_texstd": [25, 35], "frame_std": [45, 65],
                        "edge_width_px": [1.4, 2.5], "warm": ">0",
                        "real_FP_surround_texstd": 28.86},
        },
    }
    os.makedirs(args.report_dir, exist_ok=True)
    with open(os.path.join(args.report_dir, "validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    ov = report["valset_overlap"]
    bg = report["all_background"]
    ap_ = report["appearance"]
    print("\n=== P29-v2 VALIDATION ===")
    print(f"VALSET-OVERLAP: md5 hits={ov['md5_exact_hits']} (gate 0: "
          f"{'PASS' if ov['md5_gate_zero_hits'] else 'FAIL'})   "
          f"min pHash={ov['phash_min_hamming']} (gate >10: "
          f"{'PASS' if ov['phash_gate_min_gt_10'] else 'FAIL'})")
    print(f"ALL-BACKGROUND: label max={bg['label_max_overall']} nonzero={bg['n_nonzero_labels']} "
          f"({'PASS' if bg['gate_all_zero'] else 'FAIL'})")
    print(f"APPEARANCE: surround texstd median={ap_['surround_texstd_median']:.2f} "
          f"(target 25-35, real-FP 28.9); frame std={ap_['frame_perchan_std_median']:.2f}; "
          f"edge width px={ap_['edge_width_px_median']}; warm R-B={ap_['warm_R_minus_B_median']:.1f}; "
          f"HSV V/S={ap_['hsv_v_median']:.0f}/{ap_['hsv_s_median']:.0f}")
    overall_pass = (ov["md5_gate_zero_hits"] and ov["phash_gate_min_gt_10"]
                    and bg["gate_all_zero"])
    print(f"OVERALL GATES: {'PASS' if overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
