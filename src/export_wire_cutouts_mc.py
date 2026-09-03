#!/usr/bin/env python
"""Export REAL cable strand cutouts from MovingCables as an RGBA cutout library.

P44.3. The P44 diagnosis showed that both prior arms taught the model
P(wire | pixel came from a real photograph) == 0, so real footage - where every pixel is
photographic - loses its wire. `probe_p44_wire_cue.py` confirmed it directly: on real
cable pasted into a synthetic frame the baseline predicts WIRE on 20.5 % of the pixels,
P44 on 2.2 % (89.9 % connector) and P44.2 on 2.4 %, while all three sit at ~70-80 % on
RENDERED wire pasted identically. The fix is to give class 1 photographic pixels too.

Source: data/dformer_dataset_movingcables - 9,371 real cable frames with per-pixel GT
(Holesovsky/Skoviera/Hlavac 2024, CC-BY-SA; label value 4 = cable). It is NOT the
evaluation data: cleared against both real-world arbiters (4 PI videos + real_wires_valset)
by md5 + pHash, min Hamming 12 vs a <=6 overlap threshold, and cleared previously in P28.

Extraction: a random WINDOW crop, bbox-tightened, alpha = the window's own GT mask.
Connected components do NOT work on this data - MovingCables shows a dense tangle in which
every cable merges into one component (measured: median component fill 0.39 of its bbox,
median elongation 1.33, i.e. blobs), and pasting a blob as class 1 would write exactly the
kind of label noise that forced P44's connector positives through a four-stage gate. A
window with fill inside --fill is a few crossing strands: thin, cable-shaped, all-cable
pixels. Windows are rejected if their apparent stroke width or fill says "blob".

Cutouts are drawn ONLY from clips selected by --clip-parity, so the probe
(`probe_p44_wire_cue.py`, which reads the complementary parity) is scored on real cable the
trained model has never been shown - no asset leakage between the lever and its own test.

The library is written as <out>/strand/*.png so the compositor's existing load_cutouts()
(which globs <root>/*/*.png and skips _rejected/) reads it unchanged.

Usage:
  python src/export_wire_cutouts_mc.py --out data/wire_cutouts_mc --n 800
"""
from __future__ import annotations
import argparse, glob, json, os, random
import cv2
import numpy as np

MC_WIRE = 4


def stroke_width(mask):
    """2 x the 95th-pct distance-to-edge = a robust apparent cable width in px."""
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    v = d[mask]
    return float(2 * np.percentile(v, 95)) if v.size else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-dir", default="data/dformer_dataset_movingcables")
    ap.add_argument("--out", default="data/wire_cutouts_mc")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--area", default="600,40000", help="min,max cable-pixel area")
    ap.add_argument("--window", default="120,300", help="min,max crop window side px")
    ap.add_argument("--fill", default="0.06,0.32",
                    help="min,max cable-pixel share of the tightened bbox (drops tangles)")
    ap.add_argument("--tries-per-frame", type=int, default=6)
    ap.add_argument("--min-stroke", type=float, default=8.0)
    ap.add_argument("--max-stroke", type=float, default=45.0)
    ap.add_argument("--min-elong", type=float, default=1.15,
                    help="min bbox aspect ratio (long side / short side)")
    ap.add_argument("--clip-parity", default="even", choices=["even", "odd", "all"],
                    help="use only clips with this id parity; the probe uses the other")
    ap.add_argument("--per-clip-max", type=int, default=6,
                    help="cap per source clip so the library is not 20 frames of one cable")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    amin, amax = (int(v) for v in a.area.split(","))
    rng = random.Random(a.seed)
    outdir = os.path.join(a.out, "strand")
    os.makedirs(outdir, exist_ok=True)

    labs = sorted(glob.glob(os.path.join(a.mc_dir, "Label", "*.png")))
    if a.clip_parity in ("even", "odd"):
        want = 0 if a.clip_parity == "even" else 1
        labs = [p for p in labs if int(os.path.basename(p).split("_")[0]) % 2 == want]
    rng.shuffle(labs)
    print(f"{len(labs)} candidate frames (clip parity: {a.clip_parity})")

    fmin, fmax = (float(v) for v in a.fill.split(","))
    wmin, wmax = (int(v) for v in a.window.split(","))
    per_clip, recs = {}, []
    for lp in labs:
        if len(recs) >= a.n:
            break
        clip = os.path.basename(lp).split("_")[0]
        if per_clip.get(clip, 0) >= a.per_clip_max:
            continue
        lab = cv2.imread(lp, cv2.IMREAD_UNCHANGED)
        if lab is None:
            continue
        m_full = (lab == MC_WIRE)
        if not m_full.any():
            continue
        rgb = cv2.imread(os.path.join(a.mc_dir, "RGB", os.path.basename(lp)), cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        H, W = m_full.shape
        for _try in range(a.tries_per_frame):
            if len(recs) >= a.n or per_clip.get(clip, 0) >= a.per_clip_max:
                break
            hh, ww = rng.randrange(wmin, wmax), rng.randrange(wmin, wmax)
            if hh >= H or ww >= W:
                continue
            y, x = rng.randrange(0, H - hh), rng.randrange(0, W - ww)
            m = m_full[y:y + hh, x:x + ww]
            if m.sum() < amin:
                continue
            ys, xs = np.where(m)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            m = m[y0:y1, x0:x1]
            ar = int(m.sum())
            if not (amin <= ar <= amax) or not (fmin <= m.mean() <= fmax):
                continue
            bh, bw = m.shape
            if min(bh, bw) < 8 or max(bh, bw) / max(min(bh, bw), 1) < a.min_elong:
                continue
            sw_ = stroke_width(m)
            if not (a.min_stroke <= sw_ <= a.max_stroke):
                continue
            name = (f"{os.path.splitext(os.path.basename(lp))[0]}"
                    f"_w{x + x0:03d}_{y + y0:03d}.png")
            cv2.imwrite(os.path.join(outdir, name),
                        np.dstack([rgb[y + y0:y + y1, x + x0:x + x1],
                                   m.astype(np.uint8) * 255]))
            recs.append({"file": name, "src": os.path.basename(lp), "clip": clip,
                         "area": ar, "bbox": [int(bw), int(bh)],
                         "stroke_px": round(sw_, 2), "fill": round(float(m.mean()), 3)})
            per_clip[clip] = per_clip.get(clip, 0) + 1

    sw = np.array([r["stroke_px"] for r in recs])
    json.dump({"args": vars(a), "n": len(recs),
               "n_clips": len(per_clip),
               "fill_p5_50_95": np.percentile([r["fill"] for r in recs],
                                              [5, 50, 95]).round(3).tolist(),
               "stroke_px_p5_50_95": np.percentile(sw, [5, 50, 95]).round(2).tolist(),
               "cutouts": recs},
              open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    with open(os.path.join(a.out, "ATTRIBUTION.md"), "w") as f:
        f.write("# Wire cutout library — source attribution\n\n"
                "All cutouts are crops of the **MovingCables** dataset\n"
                "(Holesovsky, Skoviera, Hlavac, 2024), used under **CC BY-SA 4.0**.\n"
                "Cable masks are the dataset's own ground truth (label value 4).\n\n"
                "Any redistribution of a derived dataset must carry this attribution and\n"
                "the CC BY-SA share-alike terms. Per-cutout source frames are listed in\n"
                "`manifest.json`.\n")
    print(f"wrote {len(recs)} cable strand cutouts from {len(per_clip)} clips -> {outdir}")
    print(f"native stroke width px  p5/p50/p95 = "
          f"{np.percentile(sw, [5, 50, 95]).round(1).tolist()}")


if __name__ == "__main__":
    main()
