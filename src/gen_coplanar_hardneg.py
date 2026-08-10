#!/usr/bin/env python
"""Generate the P29 TARGETED COPLANAR HARD-NEGATIVE dataset — wire-FREE frames of the
REAL FP-trigger surfaces, labelled ALL-BACKGROUND, to teach the segmenter that a
"textured surface feature / dark thin line on a textured surface" is NOT wire.

WHY (locked diagnostic, results/realism_campaign/p28_realtextured/fp_characterization.md):
the entire remaining gap to target is c3/c4 COPLANAR-SURFACE FALSE POSITIVES. The model
hallucinates "wire" on dark mid-saturation linear/grain features (HSV V~108, S~99) that
OVERLAP true-wire colour and are wire-SHAPED, so a geometry post-filter cannot help —
only context/texture learning (these hard-negatives) can.

DESIGN (v1, deliberately scoped to minimise the dark-cable RECALL risk):
  Cover ONLY the three lowest-recall-risk, highest-FP-mass trigger families and SKIP
  connectors/clips (~13%) and specular (~4%) THIS round (documented choice):
    ~55% linear surface seams/edges/trim   (#1 FP trigger, ~45% of real FP mass)
    ~30% textured-flat grain/stone/metal    (~25% of FP mass)
    ~15% shadow grooves on a surface        (~13% of FP mass)
  All frames wire-FREE, label all-zero. NEVER darken a positive wire albedo and NEVER
  derive a feature from wire geometry (co-located-cue poison rule) — every seam/groove
  here is a genuine SURFACE feature, wire-independent.

SOURCES: CC0 Poly Haven flat surface-texture photos already in the repo
  (data/textures/backgrounds/, entries 01-11 + 42-80; the HDRI SCENE panoramas 12-41 are
  EXCLUDED — those are the Phase-12 "scene-like backdrop" catastrophe pool). HARD
  CONSTRAINT: nothing here comes from data/real_wires_valset/ or ElectricWires/EWD.

APPEARANCE TARGETS (match measured FP cluster):
  dark linear/grain features  HSV V in [100,115], S in [90,110]
  composed background per-channel std ~ 50-65
  feature edge widths ~ real surface seams (1.4-2.5 px), NOT razor-thin 1.1px synth wires
  WARM tone bias (valset is warm).

FORMAT (mirrors src/convert_realtextured_to_dformer.py EXACTLY):
  RGB/<base>.png    BGR 480x640 (resize INTER_AREA)
  Label/<base>.png  all-ZERO uint8 480x640 (bg=0)
  Depth/<base>.png  zeros uint16 480x640
  train.txt / test.txt   lines "RGB/<base>.png"  (split BY DONOR TEXTURE — no leak)

Basename convention: ``{setid:03d}_{0:04d}_00_hn.png`` (one frame per set-id, matches the
realtextured per-image set convention so filter_indices_by_set groups them cleanly).
"""
import argparse
import json
import os

import cv2
import numpy as np

IMAGE_H, IMAGE_W = 480, 640
BG_VAL = 0

# ---- donor-texture pools (Poly Haven CC0; flat textures only, NO HDRIs) ----
# Linear-seam donors: warm wood / plank / parquet / laminate / panel with seams or grain
# ridges that are themselves wire-width linear features (the #1 real FP trigger).
LINEAR_DONORS = [
    "01_workbench_plywood", "02_workbench_wood_planks", "03_wood_planks_dry",
    "62_dark_wood", "66_distressed_painted_planks", "67_herringbone_parquet",
    "68_kitchen_wood", "69_laminate_floor", "70_laminate_floor_03",
    "72_oriented_strand_board", "75_plank_flooring", "76_plank_flooring_02",
    "77_plank_flooring_03", "78_plank_flooring_04", "80_raw_plank_wall",
    "44_black_painted_planks", "79_plaster_brick_pattern",
    "50_concrete_block_wall", "51_concrete_block_wall_02", "54_container_side",
    "73_painted_metal_shutter",
]
# Textured-flat donors: busy mid-frequency grain/stone/brushed-metal/fabric (no dominant
# single seam; the c4_14 terrazzo / c4_11 grain "flat busy texture" class).
FLAT_DONORS = [
    "05_concrete_weathered", "06_concrete_granular", "11_tile_floor_06",
    "42_anti_slip_concrete", "46_brushed_concrete", "47_brushed_concrete_2",
    "52_concrete_debris", "53_concrete_floor_damaged_01", "55_corrugated_iron_02",
    "56_corrugated_iron_03", "71_metal_plate_02", "74_patterned_plaster_wall",
    "10_tile_marble_cream", "09_tile_white_long", "49_climbing_wall_02",
    "07_fabric_rough_linen", "08_fabric_denim", "57_cotton_jersey",
    "60_curly_teddy_checkered", "61_curly_teddy_natural", "64_denim_fabric_03",
    "65_denim_fabric_04",
]
# Shadow-groove base surfaces: a clean-ish warm wood/stone over which we draw a soft,
# wire-INDEPENDENT cast shadow line (raised batten/trim). Warm wood preferred.
SHADOW_BASES = [
    "01_workbench_plywood", "02_workbench_wood_planks", "03_wood_planks_dry",
    "68_kitchen_wood", "69_laminate_floor", "70_laminate_floor_03",
    "75_plank_flooring", "10_tile_marble_cream", "46_brushed_concrete",
    "42_anti_slip_concrete", "74_patterned_plaster_wall", "80_raw_plank_wall",
]


def load_donor(tex_dir, name):
    p = os.path.join(tex_dir, name + ".jpg")
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"donor texture missing: {p}")
    return img


def _rot_scale_crop(img, rng, scale_lo, scale_hi, max_rot_deg):
    """Pick a random rotation + scale, then crop a 480x640 window. Scaling UP magnifies
    the surface grain/seam so a 480x640 crop is filled with fewer, larger features
    (raising the composed per-channel std toward the 50-65 target)."""
    h, w = img.shape[:2]
    scale = rng.uniform(scale_lo, scale_hi)
    nh, nw = max(IMAGE_H + 8, int(h * scale)), max(IMAGE_W + 8, int(w * scale))
    big = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    rot = rng.uniform(-max_rot_deg, max_rot_deg)
    if abs(rot) > 0.2:
        M = cv2.getRotationMatrix2D((nw / 2, nh / 2), rot, 1.0)
        big = cv2.warpAffine(big, M, (nw, nh), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    y0 = rng.integers(0, max(1, big.shape[0] - IMAGE_H))
    x0 = rng.integers(0, max(1, big.shape[1] - IMAGE_W))
    crop = big[y0:y0 + IMAGE_H, x0:x0 + IMAGE_W]
    if crop.shape[:2] != (IMAGE_H, IMAGE_W):
        crop = cv2.resize(crop, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)
    return crop


def _fp_seam_color(local_bgr, rng):
    """Target BGR for a drawn seam/groove pixel: pull the LOCAL surface toward the measured
    FP cluster (HSV V~100-115, S~90-110), keeping the surface hue so it reads as a genuine
    darker streak of THAT surface (warm wood seam stays warm). Returns a BGR triple."""
    hsv = cv2.cvtColor(np.uint8([[local_bgr]]), cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)
    hsv[2] = rng.uniform(104, 118)    # V toward FP cluster (slightly high to survive shade)
    hsv[1] = rng.uniform(82, 102)     # S toward FP cluster (absolute, not additive)
    bgr = cv2.cvtColor(np.uint8([[np.clip(hsv, 0, 255)]]), cv2.COLOR_HSV2BGR)[0, 0]
    return bgr.astype(np.float32)


def _two_region(donor, rng, scale_lo, scale_hi, max_rot_deg):
    """Compose TWO scaled/rotated crops of the donor joined by a soft diagonal boundary,
    each under a different brightness — like two planks / two surface patches under uneven
    scene light. This lifts whole-frame per-channel std toward the busy-real FP range
    (50-65) realistically, without inventing wire cues."""
    a = _rot_scale_crop(donor, rng, scale_lo, scale_hi, max_rot_deg).astype(np.float32)
    b = _rot_scale_crop(donor, rng, scale_lo, scale_hi, max_rot_deg).astype(np.float32)
    # strongly different brightness/contrast per region (two surface patches under very
    # uneven scene light — the dominant std source in the real busy FP frames). One patch
    # is pushed into deep shade so the frame spans a wide tonal range like a real workbench.
    a = (a - a.mean()) * rng.uniform(0.9, 1.35) + a.mean() + rng.uniform(-6, 26)
    b = (b - b.mean()) * rng.uniform(0.9, 1.35) + b.mean() - rng.uniform(18, 52)
    # soft boundary: a diagonal ramp
    Y, X = np.mgrid[0:IMAGE_H, 0:IMAGE_W].astype(np.float32)
    nx, ny = rng.uniform(-1, 1), rng.uniform(-1, 1)
    nrm = np.hypot(nx, ny) + 1e-6
    proj = (X * nx + Y * ny) / nrm
    thr = rng.uniform(proj.min() + 0.25 * (proj.max() - proj.min()),
                      proj.min() + 0.75 * (proj.max() - proj.min()))
    soft = rng.uniform(25, 70)
    alpha = 1.0 / (1.0 + np.exp(-(proj - thr) / soft))
    alpha = alpha[..., None]
    return np.clip(a * (1 - alpha) + b * alpha, 0, 255)


def _photometric(f, rng, warm_bias=True):
    """Moderate desaturation toward the FP chroma + a moderate illumination gradient + soft
    vignette. Operates on a float frame in place of the donor's native chroma so the SURFACE
    sits in the dark-mid-sat warm band the real FP frames occupy. Kept moderate so the
    surface stays a believable photo (the aggressive version crushed darks)."""
    f = f.astype(np.float32).copy()
    luma = (0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2])[..., None]
    desat = rng.uniform(0.18, 0.40)
    f = f * (1 - desat) + luma * desat
    f += rng.uniform(-14, 8)
    yy = np.linspace(rng.uniform(-1, 1), rng.uniform(-1, 1), IMAGE_H)[:, None]
    xx = np.linspace(rng.uniform(-1, 1), rng.uniform(-1, 1), IMAGE_W)[None, :]
    f += ((yy + xx) * rng.uniform(14, 34))[..., None]
    cy, cx = rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75)
    Y, X = np.mgrid[0:IMAGE_H, 0:IMAGE_W].astype(np.float32)
    rr = np.sqrt(((Y / IMAGE_H) - cy) ** 2 + ((X / IMAGE_W) - cx) ** 2)
    f -= ((rr / rr.max()) * rng.uniform(14, 40))[..., None]
    if warm_bias:
        f[..., 2] += rng.uniform(3, 14)
        f[..., 0] -= rng.uniform(3, 12)
    return np.clip(f, 0, 255)


def _draw_seam(img, rng, n_seams, soft_lo=1.4, soft_hi=2.5, return_mask=False):
    """Draw subtle DARK mid-sat linear seams/trim lines as genuine surface features, painted
    toward the measured FP colour cluster (V~108, S~99) so the drawn seams reproduce exactly
    the pixels the model currently mislabels as wire. Edge width softened to the REAL
    surface-seam range (~1.4-2.5px), NOT the razor 1.1px synthetic-wire width. Varied
    orientation incl. ~parallel to where a wire would lie."""
    h, w = img.shape[:2]
    over = img.astype(np.float32)
    feat = np.zeros((h, w), np.float32)
    for _ in range(n_seams):
        if rng.random() < 0.5:
            theta = rng.uniform(-12, 12)
        else:
            theta = rng.uniform(78, 102)
        theta_r = np.deg2rad(theta)
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        L = max(h, w) * 1.5
        dx, dy = np.cos(theta_r) * L, np.sin(theta_r) * L
        p1 = (int(cx - dx), int(cy - dy))
        p2 = (int(cx + dx), int(cy + dy))
        width = int(rng.integers(1, 3))
        line = np.zeros((h, w), np.float32)
        cv2.line(line, p1, p2, 1.0, width, lineType=cv2.LINE_AA)
        k = float(rng.uniform(soft_lo, soft_hi))
        line = cv2.GaussianBlur(line, (0, 0), k / 2.0)
        line /= max(line.max(), 1e-6)
        # local surface colour along the seam -> FP-cluster target colour
        ys, xs = np.where(line > 0.3)
        if len(ys) == 0:
            continue
        local = over[ys, xs].mean(0)
        tgt = _fp_seam_color(local, rng)
        a = (line * rng.uniform(0.7, 1.0))[..., None]
        over = over * (1 - a) + tgt[None, None, :] * a
        feat = np.maximum(feat, line)
    out = np.clip(over, 0, 255)
    return (out, feat) if return_mask else out


def _draw_shadow_groove(img, rng, return_mask=False):
    """One or two SOFT dark cast-shadow lines from a raised batten/trim — wire-INDEPENDENT
    scene geometry (a raised strip on the surface), broader & softer than a seam. The 13%
    shadow-groove FP class. Painted toward the FP dark cluster."""
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    feat = np.zeros((h, w), np.float32)
    n = int(rng.integers(1, 3))
    for _ in range(n):
        if rng.random() < 0.55:
            theta = rng.uniform(-18, 18)
        else:
            theta = rng.uniform(72, 108)
        theta_r = np.deg2rad(theta)
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        L = max(h, w) * 1.6
        dx, dy = np.cos(theta_r) * L, np.sin(theta_r) * L
        p1 = (int(cx - dx), int(cy - dy))
        p2 = (int(cx + dx), int(cy + dy))
        groove = np.zeros((h, w), np.float32)
        cv2.line(groove, p1, p2, 1.0, int(rng.integers(2, 5)), lineType=cv2.LINE_AA)
        groove = cv2.GaussianBlur(groove, (0, 0), rng.uniform(1.6, 3.2))
        groove /= max(groove.max(), 1e-6)
        darken = rng.uniform(0.55, 0.78)            # softer shade -> less over-saturation
        f = f * (1.0 - groove[..., None] * (1.0 - darken))
        feat = np.maximum(feat, groove)
    # multiplicative shade over-saturates; pull grooved pixels toward the FP S target
    g3 = feat[..., None]
    luma = (0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2])[..., None]
    f = f * (1 - g3 * 0.35) + luma * (g3 * 0.35)
    out = np.clip(f, 0, 255)
    return (out, feat) if return_mask else out


def _post_grain(img, rng):
    """Tiny sensor-like grain + optional mild blur so synth edges aren't razor-sharp."""
    f = img.astype(np.float32)
    f += rng.normal(0, rng.uniform(1.5, 4.5), f.shape)
    if rng.random() < 0.5:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.5, 1.0))
    return np.clip(f, 0, 255).astype(np.uint8)


def make_linear(donor, rng, return_mask=False):
    crop = _two_region(donor, rng, 1.2, 3.2, max_rot_deg=22)
    crop = _photometric(crop, rng, warm_bias=True)
    feat = np.zeros((IMAGE_H, IMAGE_W), np.float32)
    if rng.random() < 0.85:
        crop, feat = _draw_seam(crop, rng, n_seams=int(rng.integers(1, 4)), return_mask=True)
    out = _post_grain(crop, rng)
    return (out, feat) if return_mask else out


def make_flat(donor, rng, return_mask=False):
    crop = _two_region(donor, rng, 1.4, 3.8, max_rot_deg=35)
    crop = _photometric(crop, rng, warm_bias=(rng.random() < 0.75))
    feat = np.zeros((IMAGE_H, IMAGE_W), np.float32)
    # occasional faint seam so the model still sees a thin line ON busy texture
    if rng.random() < 0.30:
        crop, feat = _draw_seam(crop, rng, n_seams=1, return_mask=True)
    out = _post_grain(crop, rng)
    return (out, feat) if return_mask else out


def make_shadow(donor, rng, return_mask=False):
    crop = _two_region(donor, rng, 1.1, 2.6, max_rot_deg=18)
    crop = _photometric(crop, rng, warm_bias=True)
    crop, feat = _draw_shadow_groove(crop, rng, return_mask=True)
    out = _post_grain(crop, rng)
    return (out, feat) if return_mask else out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex-dir", default="data/textures/backgrounds")
    ap.add_argument("--out-dir", default="data/dformer_dataset_coplanar_hardneg")
    ap.add_argument("--n-total", type=int, default=720,
                    help="target unique frame count (~600-800)")
    ap.add_argument("--frac-linear", type=float, default=0.55)
    ap.add_argument("--frac-flat", type=float, default=0.30)
    ap.add_argument("--frac-shadow", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    n_linear = int(round(args.n_total * args.frac_linear))
    n_flat = int(round(args.n_total * args.frac_flat))
    n_shadow = args.n_total - n_linear - n_flat

    rgb_out = os.path.join(args.out_dir, "RGB")
    depth_out = os.path.join(args.out_dir, "Depth")
    label_out = os.path.join(args.out_dir, "Label")
    for d in (rgb_out, depth_out, label_out):
        os.makedirs(d, exist_ok=True)

    zero_depth = np.zeros((IMAGE_H, IMAGE_W), np.uint16)
    zero_label = np.zeros((IMAGE_H, IMAGE_W), np.uint8)

    # preload donors
    lin = [(n, load_donor(args.tex_dir, n)) for n in LINEAR_DONORS]
    flt = [(n, load_donor(args.tex_dir, n)) for n in FLAT_DONORS]
    shd = [(n, load_donor(args.tex_dir, n)) for n in SHADOW_BASES]

    # build a flat job list: (category, maker, donor_name, donor_img)
    jobs = []

    def add(cat, pool, maker, count):
        for i in range(count):
            name, img = pool[i % len(pool)]
            jobs.append((cat, maker, name, img))

    add("linear", lin, make_linear, n_linear)
    add("flat", flt, make_flat, n_flat)
    add("shadow", shd, make_shadow, n_shadow)

    # split BY DONOR TEXTURE so val/train share no donor (no near-dup leak). Hold out a
    # set of donor names per category for val.
    def pick_val_donors(pool, frac):
        names = [n for n, _ in pool]
        k = max(1, int(round(len(names) * frac)))
        idx = rng.permutation(len(names))[:k]
        return set(names[i] for i in idx)

    val_donor = (pick_val_donors(lin, args.val_frac) |
                 pick_val_donors(flt, args.val_frac) |
                 pick_val_donors(shd, args.val_frac))

    train_lines, val_lines = [], []
    cat_counts = {"linear": 0, "flat": 0, "shadow": 0}
    # embedded appearance audit on the DRAWN-FEATURE pixels (the synthetic FP analogue)
    feat_V, feat_S, frame_std, frame_warm = [], [], [], []
    setid = 0
    for cat, maker, donor_name, donor_img in jobs:
        frame, feat = maker(donor_img, rng, return_mask=True)
        base = f"{setid:03d}_{0:04d}_00_hn"
        cv2.imwrite(os.path.join(rgb_out, base + ".png"), frame)
        cv2.imwrite(os.path.join(depth_out, base + ".png"), zero_depth)
        cv2.imwrite(os.path.join(label_out, base + ".png"), zero_label)
        # audit
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        fm = feat > 0.4
        if fm.sum() > 5:
            feat_V.append(float(hsv[..., 2][fm].mean()))
            feat_S.append(float(hsv[..., 1][fm].mean()))
        frame_std.append(float(np.mean([frame[..., c].std() for c in range(3)])))
        frame_warm.append(float(frame[..., 2].mean() - frame[..., 0].mean()))
        line = f"RGB/{base}.png"
        if donor_name in val_donor:
            val_lines.append(line)
        else:
            train_lines.append(line)
        cat_counts[cat] += 1
        setid += 1

    audit = {
        "feat_pixel_HSV_V_mean": float(np.mean(feat_V)) if feat_V else None,
        "feat_pixel_HSV_V_median": float(np.median(feat_V)) if feat_V else None,
        "feat_pixel_HSV_S_mean": float(np.mean(feat_S)) if feat_S else None,
        "feat_pixel_HSV_S_median": float(np.median(feat_S)) if feat_S else None,
        "frame_perchan_std_mean": float(np.mean(frame_std)),
        "frame_perchan_std_median": float(np.median(frame_std)),
        "frame_warm_R_minus_B_mean": float(np.mean(frame_warm)),
        "n_frames_with_drawn_feature": len(feat_V),
        "targets": {"feat_V": [100, 115], "feat_S": [90, 110],
                    "frame_std": [50, 65], "warm": ">0"},
    }

    with open(os.path.join(args.out_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_lines) + ("\n" if train_lines else ""))
    with open(os.path.join(args.out_dir, "test.txt"), "w") as f:
        f.write("\n".join(val_lines) + ("\n" if val_lines else ""))

    meta = {
        "out_dir": args.out_dir,
        "n_total": setid,
        "n_train": len(train_lines),
        "n_val": len(val_lines),
        "category_counts": cat_counts,
        "val_frac": args.val_frac,
        "val_donor_textures": sorted(val_donor),
        "linear_donors": LINEAR_DONORS,
        "flat_donors": FLAT_DONORS,
        "shadow_bases": SHADOW_BASES,
        "seed": args.seed,
        "note": "wire-FREE coplanar FP hard-negatives; connectors+specular SKIPPED v1 "
                "(recall-risk minimisation). Sources: CC0 Poly Haven flat textures "
                "(HDRIs excluded). All labels all-zero.",
        "appearance_audit": audit,
    }
    with open(os.path.join(args.out_dir, "gen_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("=== P29 COPLANAR HARD-NEGATIVE GENERATION DONE ===")
    print(f"out-dir: {args.out_dir}")
    print(f"total: {setid}  train {len(train_lines)} / val {len(val_lines)}")
    print(f"categories: {cat_counts}")
    print(f"val donor textures held out: {len(val_donor)}")
    print("APPEARANCE (drawn-feature pixels = synthetic FP analogue):")
    print(f"  feat HSV V mean/median: {audit['feat_pixel_HSV_V_mean']:.1f} / "
          f"{audit['feat_pixel_HSV_V_median']:.1f}  (target 100-115)")
    print(f"  feat HSV S mean/median: {audit['feat_pixel_HSV_S_mean']:.1f} / "
          f"{audit['feat_pixel_HSV_S_median']:.1f}  (target 90-110)")
    print(f"  frame per-chan std mean/median: {audit['frame_perchan_std_mean']:.1f} / "
          f"{audit['frame_perchan_std_median']:.1f}  (target 50-65)")
    print(f"  frame warm (R-B) mean: {audit['frame_warm_R_minus_B_mean']:.1f}  (target >0)")


if __name__ == "__main__":
    main()
