#!/usr/bin/env python
"""Generate the P29-v2 TARGETED COPLANAR HARD-NEGATIVE dataset — wire-FREE frames of
BUSY textured surfaces, labelled ALL-BACKGROUND, that are *genuinely hard* for the current
model (it mis-fires "wire" on them) so fine-tuning on them ROTATES the wire-vs-busy-surface
decision boundary. v1 (data/dformer_dataset_coplanar_hardneg) failed the
results/realism_campaign/p29_neg_feature_audit audit: its negatives were CLEAN smooth
surfaces with a thin painted line (surround texstd ~14.6, frame std ~22, median wire-logit
−17.67) — they sat in the "already-ignored" TN_surface basin (real seam-FP is logit +1.7..+3.2
on surfaces with surround texstd ~28.9), so the model already got them right and the FT was a
near-no-op.

WHAT v2 CHANGES (the two operative deficits, from the audit's v2 spec):
  1. BUSY SURROUNDS are the cue. Generate coarse-grain / brushed-metal / terrazzo-stone /
     rough-fabric-mesh / concrete surfaces whose LOCAL 9×9 surround texture-energy (patch
     std) lands in the ~25-35 band and frame per-channel std ~45-65 — *not* smooth planks
     with one painted streak. The dark thin linear seam/trim/groove features of v1 are kept
     (same warm dark mid-sat colour cluster, 1.4-2.5 px soft edges) but they are SECONDARY;
     the busy surface texture is what trips the model. Wire-FREE, label all-background.
  2. A build-time HARDNESS GATE (model-confusability, valset-FREE). Each candidate is scored
     by SegFormer-B5 epoch_15 (the current best). KEEP only candidates the model actually
     mis-fires on: wire-logit > 0 on >= a tuned fraction of surface/seam pixels. The
     threshold is auto-tuned so ~700-800 survive; we report it + retention. This recruits the
     FULL learned separating axis (not just the named surround-texture component) and never
     touches the valset.

WIRE-INDEPENDENCE (co-located-cue poison rule, [[p26_2_colocated_render_cue]]): every seam /
groove here is a genuine wire-INDEPENDENT SURFACE feature; nothing is derived from wire
geometry, no positive wire albedo is darkened.

SOURCES: CC0 Poly Haven flat surface-texture photos already in the repo
  (data/textures/backgrounds/, the flat-texture entries 01-11 + 42-80; the HDRI SCENE
  panoramas 12-41 are EXCLUDED — Phase-12 "scene-like backdrop" catastrophe pool). HARD
  CONSTRAINT: nothing comes from data/real_wires_valset/ or ElectricWires/EWD.

This script PHASE 1 generates a LARGE candidate pool (RGB + zero Label + zero Depth) under
--cand-dir, recording per-candidate the local SURROUND texstd. PHASE 2 (the hardness gate +
final dataset assembly) is src/gate_coplanar_hardneg_v2.py.

FORMAT (mirrors src/gen_coplanar_hardneg.py / convert_realtextured_to_dformer.py EXACTLY):
  RGB/<base>.png    BGR 480x640 (resize INTER_AREA)
  Label/<base>.png  all-ZERO uint8 480x640
  Depth/<base>.png  zeros uint16 480x640
Basename: ``{setid:03d}_{0:04d}_00_hn2.png`` (one frame per set-id).
"""
import argparse
import json
import os

import cv2
import numpy as np

IMAGE_H, IMAGE_W = 480, 640

# ---- donor-texture pools: BUSY mid-frequency surfaces only (flat textures, NO HDRIs). ----
# These are the donors whose native grain/weave/aggregate carries the mid-frequency energy
# that, magnified into a 480x640 crop, yields surround texstd ~25-35. Smooth tiles / clean
# planks (which made v1 too easy) are DEMOTED to seam-line bases only.
BUSY_DONORS = [
    # coarse aggregate / concrete / stone / terrazzo
    "05_concrete_weathered", "06_concrete_granular", "42_anti_slip_concrete",
    "46_brushed_concrete", "47_brushed_concrete_2", "52_concrete_debris",
    "53_concrete_floor_damaged_01", "50_concrete_block_wall", "51_concrete_block_wall_02",
    "04_concrete_floor_worn", "11_tile_floor_06", "49_climbing_wall_02",
    "74_patterned_plaster_wall", "79_plaster_brick_pattern",
    # brushed / corrugated / patterned metal
    "55_corrugated_iron_02", "56_corrugated_iron_03", "71_metal_plate_02",
    "73_painted_metal_shutter", "54_container_side",
    # rough / coarse fabric & mesh & teddy
    "07_fabric_rough_linen", "08_fabric_denim", "57_cotton_jersey",
    "60_curly_teddy_checkered", "61_curly_teddy_natural", "64_denim_fabric_03",
    "65_denim_fabric_04", "63_denim_fabric", "43_bi_stretch",
    # coarse / distressed wood grain & OSB (busy grain, not smooth plank)
    "72_oriented_strand_board", "66_distressed_painted_planks", "80_raw_plank_wall",
    "62_dark_wood", "44_black_painted_planks", "67_herringbone_parquet",
    "01_workbench_plywood", "02_workbench_wood_planks", "03_wood_planks_dry",
]


def load_donor(tex_dir, name):
    p = os.path.join(tex_dir, name + ".jpg")
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"donor texture missing: {p}")
    return img


def local_texstd(bgr):
    """9x9 local std of luma (the audit's surround texture-energy cue). Returns a HxW map.
    IDENTICAL formula to results/realism_campaign/p29_feature_separability/extract_features.py
    compute_props (boxFilter mean/sq -> sqrt(sq-mean^2))."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.boxFilter(gray, -1, (9, 9), normalize=True)
    sq = cv2.boxFilter(gray * gray, -1, (9, 9), normalize=True)
    return np.sqrt(np.clip(sq - mean * mean, 0, None))


def _rot_scale_crop(img, rng, scale_lo, scale_hi, max_rot_deg):
    """Random rotation+scale then a 480x640 crop. To raise mid-frequency energy we scale the
    donor so its native grain fills the crop at a coarse pitch (a few px per grain cell), which
    is exactly the surround-texstd-25-35 regime — NOT so large the texture goes smooth-blobby,
    NOT so small it aliases into noise."""
    h, w = img.shape[:2]
    scale = rng.uniform(scale_lo, scale_hi)
    nh, nw = max(IMAGE_H + 8, int(h * scale)), max(IMAGE_W + 8, int(w * scale))
    big = cv2.resize(img, (nw, nh),
                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
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


def _two_region(donor, rng, scale_lo, scale_hi, max_rot_deg):
    """TWO scaled/rotated crops joined by a soft diagonal boundary, each under different
    brightness (two surface patches under uneven scene light) — raises whole-frame per-channel
    std toward 45-65 realistically."""
    a = _rot_scale_crop(donor, rng, scale_lo, scale_hi, max_rot_deg).astype(np.float32)
    b = _rot_scale_crop(donor, rng, scale_lo, scale_hi, max_rot_deg).astype(np.float32)
    a = (a - a.mean()) * rng.uniform(1.05, 1.55) + a.mean() + rng.uniform(-4, 32)
    b = (b - b.mean()) * rng.uniform(1.05, 1.55) + b.mean() - rng.uniform(18, 58)
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


def _enhance_busy(f, rng):
    """Push LOCAL mid-frequency energy UP into the surround-texstd 25-35 band: a local
    contrast (unsharp on the band-pass) boost + a touch of structured value-noise riding the
    existing grain. This is the operative v2 lever — it makes the SURROUND busy, the actual
    cue the real FP surfaces carry, WITHOUT inventing any wire-like structure (isotropic /
    grain-following, never elongated)."""
    f = f.astype(np.float32)
    gray = (0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2])
    # band-pass (detail) at the ~9px scale we measure surround-texstd over; amplify it HARD
    # to raise local std into the 25-35 band (a tight blur sigma so the boosted detail lands
    # at the 9px window scale, not coarse blobs).
    blur = cv2.GaussianBlur(gray, (0, 0), rng.uniform(1.6, 2.8))
    detail = gray - blur
    gain = rng.uniform(2.8, 4.6)
    f += (detail * gain)[..., None]
    # fine structured noise that FOLLOWS the surface (multiplicative on detail magnitude) so
    # busy areas get busier and flat areas stay flatter (no uniform film-grain wash). A second
    # broader noise octave adds mid-frequency busyness everywhere (concrete-aggregate look).
    fine = rng.normal(0, 1, gray.shape).astype(np.float32)
    fine = cv2.GaussianBlur(fine, (0, 0), rng.uniform(0.6, 1.1))
    amp = rng.uniform(9.0, 15.0) * (0.45 + 0.55 * np.clip(np.abs(detail) / 12.0, 0, 1.6))
    f += (fine * amp)[..., None]
    mid = rng.normal(0, 1, gray.shape).astype(np.float32)
    mid = cv2.GaussianBlur(mid, (0, 0), rng.uniform(1.5, 2.4))
    f += (mid * rng.uniform(8.0, 13.0))[..., None]
    return np.clip(f, 0, 255)


def _calibrate_surround(f, rng, target_lo=22.0, target_hi=29.0):
    """Deterministically nudge the frame's median 9×9 surround texstd into [target_lo,
    target_hi] by adding a single mid-frequency structured-noise octave scaled to close the
    gap (or, if already too busy, a mild blur). Eliminates the costly accept/reject loop under
    CPU contention while keeping the surround in the real-FP busy band. The added noise is
    isotropic / non-elongated (never a wire cue)."""
    cur = float(np.median(local_texstd(np.clip(f, 0, 255).astype(np.uint8))))
    target = float(rng.uniform(target_lo, target_hi))
    if cur < target - 0.5:
        # add a mid-frequency noise octave; energy ~ sqrt(target^2 - cur^2) in the 9px window.
        need = np.sqrt(max(target * target - cur * cur, 0.0))
        oct_ = rng.normal(0, 1, f.shape[:2]).astype(np.float32)
        oct_ = cv2.GaussianBlur(oct_, (0, 0), rng.uniform(1.2, 1.8))
        oct_ /= (oct_.std() + 1e-6)
        # gentle iterative scale (one extra check) — the 9px-window std of blurred unit noise
        # is < 1, so scale up by an empirical factor and verify once.
        for scale in (need * 1.6, need * 2.4, need * 3.4):
            cand = np.clip(f + (oct_ * scale)[..., None], 0, 255)
            m = float(np.median(local_texstd(cand.astype(np.uint8))))
            if m >= target_lo:
                return cand
        return cand
    elif cur > target_hi + 4.0:
        return cv2.GaussianBlur(np.clip(f, 0, 255), (0, 0), 0.6)
    return np.clip(f, 0, 255)


def _photometric(f, rng, warm_bias=True):
    """Moderate desaturation toward the FP chroma + illumination gradient + soft vignette,
    keeping the surface in the dark-mid-sat WARM band the real FP frames occupy."""
    f = f.astype(np.float32).copy()
    luma = (0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2])[..., None]
    desat = rng.uniform(0.15, 0.38)
    f = f * (1 - desat) + luma * desat
    f += rng.uniform(-12, 8)
    yy = np.linspace(rng.uniform(-1, 1), rng.uniform(-1, 1), IMAGE_H)[:, None]
    xx = np.linspace(rng.uniform(-1, 1), rng.uniform(-1, 1), IMAGE_W)[None, :]
    f += ((yy + xx) * rng.uniform(12, 30))[..., None]
    cy, cx = rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75)
    Y, X = np.mgrid[0:IMAGE_H, 0:IMAGE_W].astype(np.float32)
    rr = np.sqrt(((Y / IMAGE_H) - cy) ** 2 + ((X / IMAGE_W) - cx) ** 2)
    f -= ((rr / rr.max()) * rng.uniform(12, 34))[..., None]
    if warm_bias:
        f[..., 2] += rng.uniform(3, 14)
        f[..., 0] -= rng.uniform(3, 12)
    return np.clip(f, 0, 255)


def _dark_seam_color(rng):
    """Target BGR for a drawn seam pixel: a DARK warm-leaning streak (HSV V~10-75) — a genuine
    dark continuous linear surface feature (deep groove / dark trim / cable-shadow seam). The
    v1 generator painted toward V~104 (BRIGHTER than the dark surround) so the line wasn't dark
    relative to the surface and the model ignored it; the hardness-calibration probe showed the
    model only mis-fires "wire" on DARK continuous lines (V~10-75) of width>=3 ON a busy
    surface. Warm-leaning (R>=B) to keep the valset warm bias. Returns a BGR triple."""
    v = rng.uniform(10, 75)
    # mild warm tint: red channel slightly above blue, low saturation streak
    r = np.clip(v + rng.uniform(0, 18), 0, 255)
    g = np.clip(v + rng.uniform(-6, 8), 0, 255)
    b = np.clip(v - rng.uniform(0, 14), 0, 255)
    return np.array([b, g, r], np.float32)


def _draw_seam(img, rng, n_seams, soft_lo=1.4, soft_hi=2.5):
    """Draw DARK continuous linear seams/trim/grooves as the PRIMARY hard feature ON the busy
    surround: long, continuous, DARK (V~10-75), core width 3-6px, soft 1.4-2.5px edge. This is
    the genuine wire-INDEPENDENT surface feature (deep seam / dark trim) that the current model
    mis-reads as wire when it sits on a busy mid-frequency surface — the exact real c3/c4 FP
    trigger (a dark continuous line on a textured coplanar surface). Returns (img, seam_mask)
    where seam_mask marks the line vicinity (the gate scores fired-fraction over THESE px)."""
    h, w = img.shape[:2]
    over = img.astype(np.float32)
    seam = np.zeros((h, w), np.float32)
    for _ in range(n_seams):
        theta = rng.uniform(-15, 15) if rng.random() < 0.5 else rng.uniform(75, 105)
        theta_r = np.deg2rad(theta)
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        L = max(h, w) * 1.6
        dx, dy = np.cos(theta_r) * L, np.sin(theta_r) * L
        p1 = (int(cx - dx), int(cy - dy))
        p2 = (int(cx + dx), int(cy + dy))
        width = int(rng.integers(3, 7))   # core width 3-6 px (calibrated hardness band)
        line = np.zeros((h, w), np.float32)
        cv2.line(line, p1, p2, 1.0, width, lineType=cv2.LINE_AA)
        # soft edge in the real-seam range; sigma chosen so the 10-90% rise is ~1.4-2.5px
        k = float(rng.uniform(soft_lo, soft_hi)) / 2.0
        line = cv2.GaussianBlur(line, (0, 0), k)
        line /= max(line.max(), 1e-6)
        tgt = _dark_seam_color(rng)
        a = (line * rng.uniform(0.85, 1.0))[..., None]
        over = over * (1 - a) + tgt[None, None, :] * a
        seam = np.maximum(seam, line)
    return np.clip(over, 0, 255), (seam > 0.3)


def _post_grain(img, rng):
    f = img.astype(np.float32)
    f += rng.normal(0, rng.uniform(1.5, 3.5), f.shape)
    if rng.random() < 0.35:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.4, 0.8))
    return np.clip(f, 0, 255).astype(np.uint8)


def make_busy(donor, rng):
    """A busy textured surface (the v2 workhorse). Keep the donor grain near/below native
    pitch so the 9px-window surround energy is HIGH (magnifying smooths it — the v1->surround
    deficit), enhance local mid-frequency energy hard, photometric warm/dark, then 0-2
    secondary dark seams. The scale band (0.5-1.4) keeps grain cells a few px wide = the
    texstd-25-35 regime; _enhance_busy then lifts it the rest of the way."""
    crop = _two_region(donor, rng, 0.4, 1.1, max_rot_deg=35)
    crop = _enhance_busy(crop, rng)
    crop = _photometric(crop, rng, warm_bias=(rng.random() < 0.8))
    crop = _calibrate_surround(crop, rng)   # land surround texstd in [26,34] deterministically
    n = int(rng.integers(2, 5))   # 2-4 DARK seams (PRIMARY hardness feature, not optional)
    crop, seam = _draw_seam(crop, rng, n_seams=n)
    out = _post_grain(crop, rng)
    return out, seam


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex-dir", default="data/textures/backgrounds")
    ap.add_argument("--cand-dir", default="results/realism_campaign/p29v2_coplanar_hardneg/candidates")
    ap.add_argument("--n-cand", type=int, default=2600,
                    help="candidate pool size before the busy/hardness filters")
    ap.add_argument("--surround-lo", type=float, default=24.0,
                    help="reject candidates whose median local surround texstd < this "
                         "(pre-filter so we don't waste model passes on flat frames)")
    ap.add_argument("--surround-hi", type=float, default=40.0,
                    help="reject candidates whose median local surround texstd > this "
                         "(keep it a real busy SURFACE, not pure noise)")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rgb_out = os.path.join(args.cand_dir, "RGB")
    depth_out = os.path.join(args.cand_dir, "Depth")
    label_out = os.path.join(args.cand_dir, "Label")
    for d in (rgb_out, depth_out, label_out):
        os.makedirs(d, exist_ok=True)

    seam_out = os.path.join(args.cand_dir, "Seam")   # seam-vicinity mask (gate scores fired-% here)
    os.makedirs(seam_out, exist_ok=True)

    zero_depth = np.zeros((IMAGE_H, IMAGE_W), np.uint16)
    zero_label = np.zeros((IMAGE_H, IMAGE_W), np.uint8)
    donors = [(n, load_donor(args.tex_dir, n)) for n in BUSY_DONORS]

    records = []   # one per ACCEPTED candidate (passed surround pre-filter)
    surround_all = []
    setid = 0
    # RESUME: if RGB frames already exist (a prior run was killed before writing meta),
    # re-derive their records from disk and continue from the next set-id (crash-resilient).
    existing = sorted(int(os.path.splitext(f)[0].split("_")[0])
                      for f in os.listdir(rgb_out) if f.endswith(".png"))
    if existing:
        setid = existing[-1] + 1
        for sid in existing:
            base = f"{sid:03d}_{0:04d}_00_hn2"
            fr = cv2.imread(os.path.join(rgb_out, base + ".png"), cv2.IMREAD_COLOR)
            if fr is None:
                continue
            ts = local_texstd(fr)
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
            records.append({
                "base": base, "donor": "resumed",
                "surround_texstd_median": float(np.median(ts)),
                "surround_texstd_mean": float(ts.mean()),
                "frame_perchan_std": float(np.mean([fr[..., c].std() for c in range(3)])),
                "frame_warm_R_minus_B": float(fr[..., 2].mean() - fr[..., 0].mean()),
                "hsv_v_mean": float(hsv[..., 2].mean()),
                "hsv_s_mean": float(hsv[..., 1].mean()),
            })
        print(f"RESUME: found {len(existing)} existing candidates, continuing from set-id {setid}")
    attempts = 0
    max_attempts = args.n_cand * 4
    while setid < args.n_cand and attempts < max_attempts:
        attempts += 1
        name, img = donors[attempts % len(donors)]
        frame, seam = make_busy(img, rng)
        ts = local_texstd(frame)
        med_ts = float(np.median(ts))
        surround_all.append(med_ts)
        if not (args.surround_lo <= med_ts <= args.surround_hi):
            continue
        base = f"{setid:03d}_{0:04d}_00_hn2"
        cv2.imwrite(os.path.join(rgb_out, base + ".png"), frame)
        cv2.imwrite(os.path.join(depth_out, base + ".png"), zero_depth)
        cv2.imwrite(os.path.join(label_out, base + ".png"), zero_label)
        cv2.imwrite(os.path.join(seam_out, base + ".png"),
                    (seam.astype(np.uint8) * 255))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        records.append({
            "base": base,
            "donor": name,
            "surround_texstd_median": med_ts,
            "surround_texstd_mean": float(ts.mean()),
            "frame_perchan_std": float(np.mean([frame[..., c].std() for c in range(3)])),
            "frame_warm_R_minus_B": float(frame[..., 2].mean() - frame[..., 0].mean()),
            "hsv_v_mean": float(hsv[..., 2].mean()),
            "hsv_s_mean": float(hsv[..., 1].mean()),
        })
        setid += 1

    meta = {
        "cand_dir": args.cand_dir,
        "n_candidates_written": setid,
        "n_attempts": attempts,
        "surround_prefilter": [args.surround_lo, args.surround_hi],
        "surround_texstd_median_of_written": float(np.median([r["surround_texstd_median"]
                                                              for r in records])),
        "frame_perchan_std_median_of_written": float(np.median([r["frame_perchan_std"]
                                                               for r in records])),
        "warm_R_minus_B_median": float(np.median([r["frame_warm_R_minus_B"] for r in records])),
        "busy_donors": BUSY_DONORS,
        "seed": args.seed,
        "note": "v2 BUSY-surround coplanar hard-neg candidates (PHASE 1). Hardness gate + "
                "final dataset assembly = src/gate_coplanar_hardneg_v2.py.",
        "records": records,
    }
    with open(os.path.join(args.cand_dir, "candidates_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("=== P29-v2 BUSY-SURROUND CANDIDATE GENERATION DONE (PHASE 1) ===")
    print(f"cand-dir: {args.cand_dir}")
    print(f"written: {setid}  (attempts {attempts}, surround pre-filter "
          f"[{args.surround_lo},{args.surround_hi}])")
    print(f"surround texstd median of written: {meta['surround_texstd_median_of_written']:.2f} "
          f"(target 25-35)")
    print(f"frame per-chan std median: {meta['frame_perchan_std_median_of_written']:.2f} "
          f"(target 45-65)")
    print(f"warm R-B median: {meta['warm_R_minus_B_median']:.2f} (target >0)")


if __name__ == "__main__":
    main()
