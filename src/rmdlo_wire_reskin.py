#!/usr/bin/env python3
"""RMDLO wire re-skin — realistic-CABLE appearance on REAL RMDLO geometry (P28).

The RMDLO probe (results/realism_campaign/p28_rmdlo/) proved the RMDLO DLO has
everything REAL we want — true ~15.8 mm cable-on-surface relief, real ~2.75 px
anti-aliased optical edges, clean single-component HSV wire masks, real metric
depth — BUT the DLO's *appearance* is a vivid blue thick rope (BGR~[92,47,9]),
categorically wrong for a real electrical cable (valset cable BGR~[119,114,104]).
A "wire=blue" appearance would inject a spurious cue that real cables lack.

This module does an **image-space GUIDED re-skin**: it KEEPS the wire's
luminance / cylindrical shading gradient and its real soft-boundary anti-aliased
alpha, and replaces ONLY the chroma/material with a realistic, VARIED cable
appearance (colours sampled per frame & per along-wire segment so we do not swap
the blue shortcut for a single-warm-colour shortcut). It also produces a variant
that composites the re-skinned wire cut-out (with its real AA alpha) onto a
varied background, because RMDLO's own backdrop is a near-uniform white table.

NON-NEGOTIABLE invariants (verified by validate.py / the report):
  * the depth map and the wire LABEL/mask are unchanged — ONLY RGB wire
    pixels change;
  * the real ~2.75 px anti-aliased edge is preserved (not hardened, not blurred);
  * the cylindrical luminance shading of the tube is preserved (not flat-filled).

Appearance palette source: ``data/dformer_dataset_movingcables`` real cable crops
(MANY colours). The valset and ElectricWires/EWD datasets are FORBIDDEN as
appearance sources (they are the eval target / a poisoning source).

Reuses the probe extractor ``src/rmdlo_bag_to_pcl.py`` (hsv_wire_mask) so the
wire region is found identically to how the probe / GT labels are produced. The
module is parameterised and scales to the full RMDLO dataset.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from rmdlo_bag_to_pcl import hsv_wire_mask, HSV_LO_DEFAULT, HSV_HI_DEFAULT  # noqa: E402

# Appearance palette source (real cable crops, many colours). FORBIDDEN to use
# real_wires_valset or ElectricWires/EWD here.
MOVINGCABLES_DIR = "data/dformer_dataset_movingcables"


# ── 1. soft anti-aliased alpha from the real wire edge ────────────────────────

def coverage_alpha(bgr, hard, table_lum):
    """Recover the rope's REAL anti-aliased coverage alpha from luminance.

    At the wire boundary the optical edge is anti-aliased over ~2.75 px (probe-
    measured), so an edge pixel's value is a PARTIAL mix of rope and table:
    ``Y_obs = a*Y_rope + (1-a)*Y_table``. Solving for the coverage ``a`` from the
    observed luminance (with the rope's interior luminance and the local table
    luminance as endpoints) recovers the rope's OWN sub-pixel coverage profile.
    Compositing the re-skinned cable back with THIS exact ``a`` reproduces the
    original anti-aliased edge geometry pixel-for-pixel — the edge is neither
    hardened nor blurred, it is the rope's real AA carried over verbatim.

    Returns alpha: HxW float32 in [0,1], non-zero only within a thin boundary band
    of the hard mask (interior = 1, exterior table = 0).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    band = cv2.dilate(hard.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    rope_lum = np.median(gray[hard]) if hard.sum() > 20 else 50.0
    denom = (table_lum - rope_lum)
    denom = denom if abs(denom) > 5 else (5.0 if denom >= 0 else -5.0)
    cov = np.clip((table_lum - gray) / denom, 0.0, 1.0)
    cov = np.where(band, cov, 0.0).astype(np.float32)
    cov = np.maximum(cov, hard.astype(np.float32))     # interior fully covered
    return cov


def local_table_luminance(bgr, hard):
    """Median luminance of the table ring just outside the wire (the AA endpoint)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    outer = cv2.dilate(hard.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    inner = cv2.dilate(hard.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ring = outer & (~inner)
    return float(np.median(gray[ring])) if ring.sum() > 50 else 200.0


# ── 2. realistic varied cable palette (from movingcables real cable crops) ────

def sample_cable_colors(n, rng, mc_dir=MOVINGCABLES_DIR, max_frames=300):
    """Return ``n`` realistic, VARIED cable mean-colours (BGR) + a few crops.

    Samples real movingcables frames, reads each frame's cable mask, and takes
    the per-frame mean cable colour. These span the real cable colour gamut
    (warm/neutral greys, browns, reds, whites) — exactly the variety we want so
    the re-skin does not become a new single-colour shortcut.

    Returns:
        colors: (n,3) float32 BGR cable colours
        crops:  list of (HxWx3 uint8) small real cable texture crops (for texture
                modulation) — may be shorter than n.
    """
    rgbs = sorted(glob.glob(os.path.join(mc_dir, "RGB", "*.png")))
    if not rgbs:
        raise FileNotFoundError(f"no movingcables RGB under {mc_dir}/RGB")
    step = max(1, len(rgbs) // max_frames)
    cand = rgbs[::step]
    rng.shuffle(cand)
    colors, crops = [], []
    for rp in cand:
        if len(colors) >= n and len(crops) >= min(n, 24):
            break
        lp = rp.replace(os.sep + "RGB" + os.sep, os.sep + "Label" + os.sep)
        lab = cv2.imread(lp, cv2.IMREAD_UNCHANGED)
        img = cv2.imread(rp)
        if lab is None or img is None:
            continue
        mb = (lab > 0) if lab.ndim == 2 else (lab[..., 0] > 0)
        if mb.sum() < 400:
            continue
        if len(colors) < n:
            colors.append(img[mb].mean(0).astype(np.float32))
        if len(crops) < min(n, 24):
            ys, xs = np.where(mb)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            crop = img[y0:y1 + 1, x0:x1 + 1]
            cm = mb[y0:y1 + 1, x0:x1 + 1]
            if cm.sum() > 200 and min(crop.shape[:2]) > 8:
                crops.append((crop, cm))
    while len(colors) < n:                       # pad if dataset tiny
        colors.append(colors[rng.integers(len(colors))])
    return np.array(colors, np.float32), crops


# ── 3. along-wire segmentation so colour varies ALONG the cable ───────────────

def wire_segments(hard, n_seg):
    """Partition the wire mask into ``n_seg`` along-axis bands.

    Cheap proxy for arclength: project wire pixels on their dominant PCA axis and
    bin. Lets us paint different cable colours on different along-wire stretches
    so a single frame already shows colour VARIETY (not one flat colour).

    Returns:
        seg_id: HxW int32, -1 off-wire, else 0..n_seg-1
    """
    ys, xs = np.where(hard)
    seg_id = np.full(hard.shape, -1, np.int32)
    if len(xs) < 10:
        seg_id[hard] = 0
        return seg_id
    pts = np.column_stack([xs, ys]).astype(np.float32)
    pts -= pts.mean(0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    t = pts @ vt[0]                              # coordinate along main axis
    t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
    bins = np.clip((t * n_seg).astype(np.int32), 0, n_seg - 1)
    seg_id[ys, xs] = bins
    return seg_id


# ── 4. the guided re-skin (luminance-preserving chroma swap) ──────────────────

def reskin_frame(bgr, rng, n_seg=4, hsv_lo=HSV_LO_DEFAULT, hsv_hi=HSV_HI_DEFAULT,
                 sat_scale=0.6, texture=True, texture_amp=0.03,
                 shade_clip=(0.55, 1.55)):
    """Re-skin one RMDLO RGB frame's wire with realistic cable appearance.

    GUIDED, edge- and shading-preserving. Two real signals must survive:

      (a) the ~2.75 px anti-aliased EDGE — recovered exactly via the rope's own
          luminance COVERAGE alpha (``coverage_alpha``) and re-applied as the
          composite alpha, so the boundary geometry is the rope's verbatim AA;
      (b) the cylindrical SHADING — the rope's interior luminance fluctuation
          (its bright specular top / dark sides) carried over ADDITIVELY onto the
          cable colour, so the tube's 3-D form is preserved, not flat-filled.

    Only chroma + absolute brightness are replaced: per along-wire segment we
    sample a real cable colour (varied so we don't introduce a single-colour
    shortcut), anchor its luminance to the cable's natural brightness (NOT the
    dim blue rope's, which would go black), and modulate it by the rope's interior
    shading. The cable is then composited with the coverage alpha so ONLY wire
    pixels + their real AA skirt change; depth and the hard label are untouched.

    Returns dict with: reskinned (HxWx3 uint8), alpha (HxW f32 coverage),
    hard (HxW bool label), seg (HxW i32), colors_used (list of BGR).
    """
    hard = hsv_wire_mask(bgr, hsv_lo, hsv_hi)
    seg = wire_segments(hard, n_seg)
    colors, crops = sample_cable_colors(n_seg, rng)

    bgr_f = bgr.astype(np.float32)
    out = bgr_f.copy()
    table_lum = local_table_luminance(bgr, hard)
    alpha = coverage_alpha(bgr, hard, table_lum)            # real AA coverage
    # YCrCb luminance of the rope (the shading lives here).
    Y_rope = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)[..., 0]

    colors_used = []
    target = bgr_f.copy()
    for s in range(n_seg):
        seg_m = seg == s
        if seg_m.sum() == 0:
            continue
        c = colors[s % len(colors)].copy()
        # de-saturate slightly toward neutral (electrical look) + per-segment
        # jitter so adjacent frames/segments differ (variety, not a shortcut).
        c = c * (1.0 + rng.uniform(-0.12, 0.12, size=3).astype(np.float32))
        c = np.clip(c, 8, 245)
        cmean = c.mean()
        c = cmean + (c - cmean) * sat_scale
        colors_used.append([round(float(x), 1) for x in c])
        # cable's own luminance (anchor brightness here, NOT the dim rope's).
        Yc = float(cv2.cvtColor(np.clip(c, 0, 255).reshape(1, 1, 3).astype(np.uint8),
                                cv2.COLOR_BGR2YCrCb)[0, 0, 0])
        yref = np.median(Y_rope[seg_m])
        # cable target luminance = cable brightness + rope's interior shading
        # (additive => preserves the gradient MAGNITUDE = true cylindrical form),
        # expressed as a gentle multiplicative shade so chroma stays the cable's.
        lum_target = Yc + (Y_rope[seg_m] - yref)
        shade = np.clip(lum_target / max(Yc, 1e-3), shade_clip[0], shade_clip[1])[:, None]
        seg_target = c[None, :] * shade
        # optional TINY real-cable texture ripple for surface micro-detail.
        if texture and crops:
            crop, cm = crops[rng.integers(len(crops))]
            cg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)[cm]
            if cg.size > 50:
                ripple = (cg - cg.mean()) / (cg.std() + 1e-3)
                idx = rng.integers(0, ripple.size, size=seg_m.sum())
                seg_target = seg_target * (1.0 + texture_amp * ripple[idx][:, None])
        target[seg_m] = np.clip(seg_target, 0, 255)

    # Composite with the rope's real coverage alpha — reproduces its AA edge.
    a = alpha[..., None]
    out = target * a + out * (1.0 - a)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return {"reskinned": out, "alpha": alpha, "hard": hard, "seg": seg,
            "colors_used": colors_used}


# ── 5. composite onto a varied background (real AA alpha preserved) ───────────

def varied_background(shape, rng, mc_dir=MOVINGCABLES_DIR):
    """A varied background image (HxWx3 uint8) for the composited variant.

    Uses a CABLE-FREE region of a real movingcables frame (its own backdrop,
    NOT the cable), so the backdrop is a real cluttered scene rather than the
    near-uniform RMDLO white table. Never samples the valset / ElectricWires.
    """
    h, w = shape[:2]
    rgbs = sorted(glob.glob(os.path.join(mc_dir, "RGB", "*.png")))
    for _ in range(12):
        rp = rgbs[rng.integers(len(rgbs))]
        img = cv2.imread(rp)
        if img is None:
            continue
        # mask out the cable so we tile from backdrop only
        lp = rp.replace(os.sep + "RGB" + os.sep, os.sep + "Label" + os.sep)
        lab = cv2.imread(lp, cv2.IMREAD_UNCHANGED)
        if lab is not None:
            mb = (lab > 0) if lab.ndim == 2 else (lab[..., 0] > 0)
            img = cv2.inpaint(img, (mb * 255).astype(np.uint8), 3, cv2.INPAINT_TELEA)
        bg = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        # mild brightness jitter for variety
        bg = np.clip(bg.astype(np.float32) * rng.uniform(0.8, 1.15), 0, 255).astype(np.uint8)
        return bg
    return np.full((h, w, 3), 200, np.uint8)


def composite_on_bg(reskinned, alpha, bg):
    """Composite re-skinned wire (with real AA alpha) over a new background.

    ONLY the wire (alpha>0) replaces the backdrop; the wire's own pixels and its
    soft AA boundary are carried over unchanged from ``reskinned``.
    """
    a = alpha[..., None]
    out = reskinned.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


# ── CLI: re-skin a directory of extracted RMDLO frames ────────────────────────

def process_dir(in_dir, out_dir, n_seg=4, seed=0, swap_bg=True, save_panels=True):
    """Re-skin every rgb_*.png in an extracted-frames dir.

    Writes, per frame:
      reskin_NNNN.png       — re-skinned wire on the REAL RMDLO background
      reskinbg_NNNN.png     — re-skinned wire composited on a VARIED background
      alpha_NNNN.png        — the soft AA wire alpha (8-bit vis)
      panel_NNNN.png        — side-by-side [orig | mask | reskin | reskin+bg]
    Depth/label are NOT copied or modified here (identical re-use upstream).
    """
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgbs = sorted(glob.glob(str(in_dir / "rgb_*.png")))
    rng = np.random.default_rng(seed)
    summary = []
    for rp in rgbs:
        idx = Path(rp).stem.split("_")[-1]
        bgr = cv2.imread(rp)
        r = reskin_frame(bgr, rng, n_seg=n_seg)
        cv2.imwrite(str(out_dir / f"reskin_{idx}.png"), r["reskinned"])
        cv2.imwrite(str(out_dir / f"alpha_{idx}.png"),
                    (r["alpha"] * 255).astype(np.uint8))
        # CRITICAL: export the GT label from the ORIGINAL frame. The re-skinned
        # cable is no longer blue, so re-thresholding it with the HSV-blue rule
        # would FAIL (IoU~0.15). The canonical wire label is the original mask;
        # the training bundle MUST consume THIS file, never re-segment the reskin.
        cv2.imwrite(str(out_dir / f"label_{idx}.png"),
                    (r["hard"] * 255).astype(np.uint8))
        # Copy the original aligned 16-bit depth through the same if present
        # (geometry is untouched by the re-skin).
        depth_src = in_dir / f"depth_{idx}.png"
        if depth_src.exists():
            d = cv2.imread(str(depth_src), cv2.IMREAD_UNCHANGED)
            cv2.imwrite(str(out_dir / f"depth_{idx}.png"), d)
        comp = None
        if swap_bg:
            bg = varied_background(bgr.shape, rng)
            comp = composite_on_bg(r["reskinned"], r["alpha"], bg)
            cv2.imwrite(str(out_dir / f"reskinbg_{idx}.png"), comp)
        if save_panels:
            mvis = cv2.cvtColor((r["hard"] * 255).astype(np.uint8),
                                cv2.COLOR_GRAY2BGR)
            tiles = [bgr, mvis, r["reskinned"]]
            if comp is not None:
                tiles.append(comp)
            panel = np.hstack([cv2.resize(t, (480, 270)) for t in tiles])
            cv2.imwrite(str(out_dir / f"panel_{idx}.png"), panel)
        summary.append({"frame": idx, "colors_used": r["colors_used"],
                        "wire_px": int(r["hard"].sum())})
    with open(out_dir / "reskin_manifest.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description="RMDLO wire re-skin (P28).")
    ap.add_argument("--in-dir", default="data/rmdlo_probe/extracted")
    ap.add_argument("--out-dir", default="results/realism_campaign/p28_rmdlo/reskin")
    ap.add_argument("--n-seg", type=int, default=4,
                    help="along-wire colour segments per frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-swap-bg", action="store_true")
    args = ap.parse_args()
    s = process_dir(args.in_dir, args.out_dir, n_seg=args.n_seg, seed=args.seed,
                    swap_bg=not args.no_swap_bg)
    print(f"re-skinned {len(s)} frames -> {args.out_dir}")


if __name__ == "__main__":
    main()
