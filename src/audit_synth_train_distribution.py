#!/usr/bin/env python3
"""Audit the colour / thickness / density / background statistics of a staged
synthetic training set (default: data/dformer_dataset_phase15_wirefree).

Read-only diagnostic for the realism campaign (2026-06). Computes, on a
seeded random sample of >=800 TRAIN frames, the SAME metric definitions used
by the real-set audit so the synth and real distributions are directly
comparable:

1. Wire-pixel colour histogram (HSV):
     chromatic   = S >= 0.25 -> 8 equal 45-degree hue bins centred at
                   0/45/90/135/180/225/270/315 deg, named
                   red / orange / yellow / green / cyan / blue / purple /
                   magenta (red bin = [337.5, 360) U [0, 22.5)).
     achromatic  = S < 0.25  -> black (V < 0.25), grey (0.25 <= V < 0.7),
                   white (V >= 0.7).
   S, V are OpenCV S/255, V/255. Wire px = Label > 0 (NOTE: PNG label 5 =
   Noise is trained as BACKGROUND by the binary trainer; both definitions
   are reported).
2. Wire local thickness at 640x480: distance transform (L2) of the wire
   mask; widths = 2 * dist sampled on the ridge (medial-axis proxy = local
   maxima of dist within a 3x3 window); bins <2 / 2-4 / 4-8 / 8-16 / >16 px.
3. Wires per frame: 8-connected components of Label>0, ignoring components
   with < 20 px.
4. Wire coverage % of the frame.
5. Background (non-wire) stats: Canny(50,150) edge density (% of bg px),
   CIELAB L* mean/std, Hasler-Suesstrunk colourfulness.

Outputs
-------
results/realism_campaign/diag_synth/synth_train_stats.csv   per-frame rows
stdout                                                       aggregate tables

Usage:
    ./env/bin/python src/audit_synth_train_distribution.py \
        [--data-dir data/dformer_dataset_phase15_wirefree] \
        [--n-frames 1000] [--seed 0]
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import maximum_filter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HUE_BIN_NAMES = ["red", "orange", "yellow", "green",
                 "cyan", "blue", "purple", "magenta"]
ACHROMA_NAMES = ["black", "grey", "white"]
ALL_BIN_NAMES = HUE_BIN_NAMES + ACHROMA_NAMES
THICK_EDGES = [0, 2, 4, 8, 16, np.inf]
THICK_NAMES = ["<2", "2-4", "4-8", "8-16", ">16"]
SAT_THRESH = 0.25          # chromatic vs achromatic
V_BLACK, V_WHITE = 0.25, 0.70
MIN_CC_PX = 20             # ignore tiny components


def colour_bins_for_pixels(bgr_px: np.ndarray) -> np.ndarray:
    """Map (N,3) uint8 BGR pixels to bin indices 0..10 (8 hue + 3 achroma).

    Hue bins: equal 45-deg bins centred at 0,45,...,315 deg, i.e. bin k covers
    [45k - 22.5, 45k + 22.5) deg. OpenCV hue is [0,180) in half-degrees.
    """
    if bgr_px.size == 0:
        return np.empty((0,), dtype=np.int64)
    hsv = cv2.cvtColor(bgr_px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h_deg = hsv[:, 0].astype(np.float64) * 2.0          # [0, 360)
    s = hsv[:, 1].astype(np.float64) / 255.0
    v = hsv[:, 2].astype(np.float64) / 255.0

    out = np.empty(h_deg.shape[0], dtype=np.int64)
    chrom = s >= SAT_THRESH
    # hue bin: shift +22.5 so bin = floor(h/45) after centering.
    hue_bin = (np.floor(((h_deg + 22.5) % 360.0) / 45.0)).astype(np.int64)
    out[chrom] = hue_bin[chrom]
    ach = ~chrom
    out[ach & (v < V_BLACK)] = 8                          # black
    out[ach & (v >= V_BLACK) & (v < V_WHITE)] = 9         # grey
    out[ach & (v >= V_WHITE)] = 10                        # white
    return out


def ridge_widths_px(wire_mask: np.ndarray) -> np.ndarray:
    """Local wire width (2 x distance transform) sampled on the ridge."""
    if not wire_mask.any():
        return np.empty((0,), dtype=np.float64)
    dist = cv2.distanceTransform(wire_mask.astype(np.uint8),
                                 cv2.DIST_L2, 5)
    ridge = (dist > 0) & (dist >= maximum_filter(dist, size=3) - 1e-6)
    return (2.0 * dist[ridge]).astype(np.float64)


def colourfulness_hs(bgr_px: np.ndarray) -> float:
    """Hasler-Suesstrunk colourfulness metric on (N,3) uint8 BGR pixels."""
    if bgr_px.shape[0] == 0:
        return 0.0
    b = bgr_px[:, 0].astype(np.float64)
    g = bgr_px[:, 1].astype(np.float64)
    r = bgr_px[:, 2].astype(np.float64)
    rg = r - g
    yb = 0.5 * (r + g) - b
    sigma = np.hypot(rg.std(), yb.std())
    mu = np.hypot(rg.mean(), yb.mean())
    return float(sigma + 0.3 * mu)


def analyse_frame(rgb_path: Path, lbl_path: Path) -> dict | None:
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    lbl = cv2.imread(str(lbl_path), cv2.IMREAD_UNCHANGED)
    if bgr is None or lbl is None:
        return None
    if bgr.shape[:2] != (480, 640):
        bgr = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)
    if lbl.shape[:2] != (480, 640):
        lbl = cv2.resize(lbl, (640, 480), interpolation=cv2.INTER_NEAREST)

    wire = lbl > 0
    wire_trained = (lbl >= 1) & (lbl <= 4)   # PNG 5 = Noise -> bg in training
    n_px = lbl.size
    row: dict = {
        "frame": rgb_path.name,
        "coverage_pct": 100.0 * wire.sum() / n_px,
        "coverage_trained_pct": 100.0 * wire_trained.sum() / n_px,
        "noise5_px": int((lbl == 5).sum()),
    }

    # 1. wire colour bins
    bins = colour_bins_for_pixels(bgr[wire])
    counts = np.bincount(bins, minlength=11)
    total = max(counts.sum(), 1)
    for i, name in enumerate(ALL_BIN_NAMES):
        row[f"wirecol_{name}_pct"] = 100.0 * counts[i] / total
    row["dominant_bin"] = (ALL_BIN_NAMES[int(np.argmax(counts))]
                           if counts.sum() > 0 else "none")

    # 2. thickness
    widths = ridge_widths_px(wire)
    if widths.size:
        hist, _ = np.histogram(widths, bins=THICK_EDGES)
        wt = max(hist.sum(), 1)
        for name, h in zip(THICK_NAMES, hist):
            row[f"thick_{name}_pct"] = 100.0 * h / wt
        row["thick_median_px"] = float(np.median(widths))
        row["thick_p90_px"] = float(np.percentile(widths, 90))
    else:
        for name in THICK_NAMES:
            row[f"thick_{name}_pct"] = 0.0
        row["thick_median_px"] = 0.0
        row["thick_p90_px"] = 0.0

    # 3. connected components (8-conn, >= MIN_CC_PX)
    n_cc, _, cc_stats, _ = cv2.connectedComponentsWithStats(
        wire.astype(np.uint8), connectivity=8)
    sizes = cc_stats[1:, cv2.CC_STAT_AREA]
    row["n_components"] = int((sizes >= MIN_CC_PX).sum())

    # 5. background stats
    bg = ~wire
    bg_px = bgr[bg]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150) > 0
    row["bg_edge_density_pct"] = 100.0 * (edges & bg).sum() / max(bg.sum(), 1)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0][bg].astype(np.float64) * (100.0 / 255.0)
    row["bg_L_mean"] = float(L.mean()) if L.size else 0.0
    row["bg_L_std"] = float(L.std()) if L.size else 0.0
    row["bg_colourfulness"] = colourfulness_hs(bg_px)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", type=Path,
                    default=PROJECT_ROOT / "data" / "dformer_dataset_phase15_wirefree")
    ap.add_argument("--n-frames", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "results" / "realism_campaign" / "diag_synth")
    args = ap.parse_args()

    train_txt = args.data_dir / "train.txt"
    names = [l.strip().split("/")[-1] for l in train_txt.read_text().splitlines()
             if l.strip()]
    rng = random.Random(args.seed)
    sample = rng.sample(names, min(args.n_frames, len(names)))
    print(f"Dataset: {args.data_dir}  ({len(names)} train frames; "
          f"sampling {len(sample)}, seed={args.seed})")

    rows: list[dict] = []
    for i, n in enumerate(sample):
        r = analyse_frame(args.data_dir / "RGB" / n, args.data_dir / "Label" / n)
        if r is not None:
            rows.append(r)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(sample)} frames done")
    if not rows:
        print("No frames analysed.")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "synth_train_stats.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows)")

    # ── aggregates ─────────────────────────────────────────────────────
    def col(k):
        return np.array([r[k] for r in rows], dtype=np.float64)

    wired = [r for r in rows if r["coverage_pct"] > 0]
    print(f"\nFrames: {len(rows)} total; {len(wired)} with wire px "
          f"({100.0 * len(wired) / len(rows):.1f}%), "
          f"{len(rows) - len(wired)} wire-free "
          f"({100.0 * (len(rows) - len(wired)) / len(rows):.1f}%)")

    # pooled colour distribution (weight frames by their wire px share via
    # per-frame percentage x coverage; simpler: average per-frame percentages
    # over wired frames AND pooled-by-pixel recount)
    print("\n[1] Wire-pixel colour bins — mean of per-frame % (wired frames):")
    for name in ALL_BIN_NAMES:
        vals = np.array([r[f"wirecol_{name}_pct"] for r in wired])
        print(f"    {name:8s} {vals.mean():6.2f}%")
    print("    Dominant-bin distribution (per wired frame):")
    from collections import Counter
    dom = Counter(r["dominant_bin"] for r in wired)
    for name, c in dom.most_common():
        print(f"    {name:8s} {c:5d} frames ({100.0 * c / len(wired):5.1f}%)")

    print("\n[2] Thickness bins — mean of per-frame % (wired frames):")
    for name in THICK_NAMES:
        vals = np.array([r[f"thick_{name}_pct"] for r in wired])
        print(f"    {name:5s} {vals.mean():6.2f}%")
    med = np.array([r["thick_median_px"] for r in wired])
    p90 = np.array([r["thick_p90_px"] for r in wired])
    print(f"    median width: mean={med.mean():.2f}px  "
          f"p10={np.percentile(med, 10):.2f}  p50={np.percentile(med, 50):.2f}  "
          f"p90={np.percentile(med, 90):.2f}")
    print(f"    p90 width:    mean={p90.mean():.2f}px")

    print("\n[3] Components per frame (>=20px, 8-conn):")
    nc = col("n_components")
    for k in sorted(set(nc.astype(int))):
        c = int((nc == k).sum())
        print(f"    {k:3d} comp: {c:5d} frames ({100.0 * c / len(rows):5.1f}%)")
    print(f"    mean={nc.mean():.2f}  median={np.median(nc):.0f}")

    print("\n[4] Wire coverage % (all frames):")
    cov = col("coverage_pct")
    print(f"    mean={cov.mean():.2f}  p10={np.percentile(cov, 10):.2f}  "
          f"p50={np.percentile(cov, 50):.2f}  p90={np.percentile(cov, 90):.2f}  "
          f"max={cov.max():.2f}")
    covt = col("coverage_trained_pct")
    print(f"    trained (label 1-4 only): mean={covt.mean():.2f}")

    print("\n[5] Background stats:")
    for k in ["bg_edge_density_pct", "bg_L_mean", "bg_L_std", "bg_colourfulness"]:
        v = col(k)
        print(f"    {k:22s} mean={v.mean():7.2f}  p10={np.percentile(v, 10):7.2f}  "
              f"p50={np.percentile(v, 50):7.2f}  p90={np.percentile(v, 90):7.2f}")

    print("\n[6] Depth: the RGB-only trainer (src/train_rgb_only_sota.py:393) "
          "discards the depth cache array — depth never reaches the model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
