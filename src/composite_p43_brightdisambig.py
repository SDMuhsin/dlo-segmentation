#!/usr/bin/env python
"""P43 compositing: "bright/pale disambiguation negatives".

IDENTICAL to P39 (``src/composite_p39_cutoutnegs.py``) EXCEPT a fraction
(``--p-bright``, default 0.55) of the pasted real photographic object cutouts are
RECOLORED toward a bright / pale / warm target BEFORE pasting (still labeled
BACKGROUND). This injects bright/pale REAL-TEXTURED non-wire objects into the
labeled-background distribution, so that at train time (with --aug-wirecolor
making WIRES bright) brightness is NON-DIAGNOSTIC and the model must use the
thin-elongated STRUCTURE to find wire (not "bright pixels = wire").

Everything else is unchanged from P39:
  * SAME 1800 synth train bases (read from --bases-list, NOT resampled).
  * SAME cutout pool (data/neg_cutouts_p39/), SAME EWD gate.
  * SAME seed (39), SAME wire-independent placement (choose_bg_position).
  * SAME offset contact shadow, SAME wire protection + the same wire GT
    assert, SAME sharp ~1px edge contract.

Recolor METHOD (preserve texture, do NOT flatten):
  * Convert opaque RGB -> HSV float (OpenCV H in [0,180], S,V in [0,255]).
  * Blend H toward target hue (for colored targets), S toward target S, both
    with ~0.7-0.85 weight (retain a little original so texture/chroma variance
    survives).
  * V_new = V_target_mean + (V_orig - V_orig.mean()) * keep, keep ~ U(0.6,0.85)
    so highlights/shadows/texture survive (NOT a flat fill).
  * Convert back to BGR. Alpha left UNTOUCHED -> edges stay sharp.
  * ANTI-FLAT GATE: if the recolored opaque-luma std collapses <= 22, fall back
    to the original (un-brightened) cutout and count the fallback.

CPU only. No torch / no GPU.

Outputs:
  <out-dir>/{RGB,Label,Depth}/<base>_p43neg.png
  <out-dir>/p43_substituted_basenames.txt   (the chosen synth train bases, original names)
  <out-dir>/composite_meta.json             (per-frame paste records, for validation)
"""
import argparse, os, glob, json, random
import numpy as np
import cv2

WIRE = 4
SYNTH = "data/dformer_dataset_phase15_wirefree"
CUTDIR = "data/neg_cutouts_p39"
DEFAULT_BASES = "data/dformer_dataset_phase15_wirefree_p39neg/p39_substituted_basenames.txt"

# ---- bright/pale recolor targets (OpenCV HSV: H in [0,180), S,V in [0,255]) ----
# Mirrors src/train_rgb_only_sota.py WireColorAugmentation TARGETS + bright-red
# (counters the red device bodies that fire; H~0/175 in OpenCV hue).
BRIGHT_TARGETS = {
    # name           : (H_opencv,  S_target,  V_target_mean)
    "pale-white":    (0.0,          12.0,      225.0),   # hue irrelevant at S~0
    "light-grey":    (0.0,          10.0,      180.0),   # hue irrelevant at S~0
    "warm-tan":      (27.0 / 2.0,   60.0,      195.0),   # H 27deg/2 (OpenCV)
    "bright-yellow": (33.0 / 2.0,   150.0,     205.0),   # H 33deg/2
    "pale-pastel":   (27.0 / 2.0,   40.0,      210.0),   # warm-ish low-S high-V
    "bright-red":    (175.0,        150.0,     170.0),   # red wraps; H~175 (OpenCV)
}
# Spread the 5 wirecolor targets ~equally (0.16 each = 0.80) plus bright-red ~0.20.
BRIGHT_TARGET_NAMES = ["pale-white", "light-grey", "warm-tan", "bright-yellow",
                       "pale-pastel", "bright-red"]
BRIGHT_TARGET_WEIGHTS = [0.16, 0.16, 0.16, 0.16, 0.16, 0.20]

ANTI_FLAT_STD = 22.0   # opaque-luma std must stay > this after recolor


def load_cutouts():
    paths = sorted(glob.glob(os.path.join(CUTDIR, "*", "*.png")))
    # honor EWD gate: drop any cutout flagged ewd_dropped in manifest
    man_path = os.path.join(CUTDIR, "manifest.json")
    dropped = set()
    if os.path.exists(man_path):
        with open(man_path) as f:
            m = json.load(f)
        cutlist = m.get("cutouts", m) if isinstance(m, dict) else m
        for e in cutlist:
            if e.get("ewd_dropped"):
                dropped.add(os.path.abspath(e["file"]))
    cuts = []
    for p in paths:
        if os.path.abspath(p) in dropped:
            continue
        rgba = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            continue
        cuts.append((p, rgba))
    return cuts


def rotate_rgba(rgba, deg):
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += (nw - w) / 2
    M[1, 2] += (nh - h) / 2
    out = cv2.warpAffine(rgba, M, (nw, nh), flags=cv2.INTER_LINEAR,
                         borderValue=(0, 0, 0, 0))
    return out


def prep_cutout(rgba, target_px, flip, rot):
    bgr, a = rgba[..., :3], rgba[..., 3]
    h, w = a.shape
    scale = target_px / max(h, w)
    nw, nh = max(8, int(w * scale)), max(8, int(h * scale))
    bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
    rgba2 = np.dstack([bgr, a])
    if flip:
        rgba2 = rgba2[:, ::-1]
    if abs(rot) > 0.5:
        rgba2 = rotate_rgba(rgba2, rot)
    # Re-tighten 1px AA boundary: hard-ish alpha with a single feather step.
    a2 = rgba2[..., 3].astype(np.float32)
    a2 = cv2.GaussianBlur(a2, (0, 0), 0.7)        # ~1px boundary AA, no global blur
    rgba2 = np.dstack([rgba2[..., :3], np.clip(a2, 0, 255).astype(np.uint8)])
    return rgba2


def _opaque_luma_std(rgba):
    """Std of luma over opaque (alpha>30) pixels of an RGBA cutout."""
    a = rgba[..., 3]
    m = a > 30
    if m.sum() < 20:
        return 0.0
    luma = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2GRAY)
    return float(luma[m].std())


def brighten_cutout(rgba, rng):
    """Recolor ONLY the RGB of opaque pixels of an RGBA cutout toward a random
    bright/pale/warm target, preserving texture (NOT a flat fill) and leaving the
    alpha channel UNTOUCHED (sharp edges).

    Returns (new_rgba, target_name, anti_flat_fallback_bool, brightened_luma_std).
    On anti-flat-gate fallback, returns the ORIGINAL rgba unchanged with the
    fallback flag set True.
    """
    name = rng.choices(BRIGHT_TARGET_NAMES, weights=BRIGHT_TARGET_WEIGHTS, k=1)[0]
    h_t, s_t, v_t = BRIGHT_TARGETS[name]

    bgr = rgba[..., :3]
    a = rgba[..., 3]
    m = a > 30                            # opaque pixels only
    if m.sum() < 20:
        # too few opaque pixels to recolor meaningfully -> treat as fallback
        return rgba, name, True, _opaque_luma_std(rgba)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H = hsv[..., 0]
    S = hsv[..., 1]
    V = hsv[..., 2]

    # blend weights (retain a little original so texture/chroma variance remains)
    w_hs = rng.uniform(0.70, 0.85)       # H,S blend toward target
    keep = rng.uniform(0.60, 0.85)       # V relative-variation keep factor

    Hm, Sm, Vm = H[m], S[m], V[m]
    v_mean = float(Vm.mean())

    # VALUE: target mean + scaled original deviation (texture survives)
    V_new = v_t + (Vm - v_mean) * keep

    # SATURATION: blend toward target S
    S_new = (1.0 - w_hs) * Sm + w_hs * s_t

    # HUE: blend toward target hue on the circle (handles red wrap at 180).
    #   meaningful mainly for colored targets; harmless for pale-white/light-grey
    #   since S collapses there.
    th = np.deg2rad(Hm * 2.0)            # OpenCV H (0..180) -> degrees -> radians
    tt = np.deg2rad(h_t * 2.0)
    cx = (1.0 - w_hs) * np.cos(th) + w_hs * np.cos(tt)
    cy = (1.0 - w_hs) * np.sin(th) + w_hs * np.sin(tt)
    H_new = (np.rad2deg(np.arctan2(cy, cx)) % 360.0) / 2.0   # back to OpenCV H

    hsv2 = hsv.copy()
    h2 = hsv2[..., 0]; s2 = hsv2[..., 1]; v2 = hsv2[..., 2]
    h2[m] = np.clip(H_new, 0, 179)
    s2[m] = np.clip(S_new, 0, 255)
    v2[m] = np.clip(V_new, 0, 255)
    hsv2[..., 0] = h2; hsv2[..., 1] = s2; hsv2[..., 2] = v2

    new_bgr = cv2.cvtColor(hsv2.astype(np.uint8), cv2.COLOR_HSV2BGR)
    new_rgba = np.dstack([new_bgr, a])    # alpha untouched -> edges sharp

    std = _opaque_luma_std(new_rgba)
    if std <= ANTI_FLAT_STD:
        # recolor flattened the texture -> fall back to the original cutout
        return rgba, name, True, std
    return new_rgba, name, False, std


def local_luma_match(patch_bgr, alpha, frame_region):
    """Mild brightness match of object to local frame luma (keep speculars)."""
    m = alpha > 30
    if m.sum() < 20 or frame_region.size == 0:
        return patch_bgr
    obj_l = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)[m].mean()
    loc_l = cv2.cvtColor(frame_region, cv2.COLOR_BGR2GRAY).mean()
    if obj_l < 1:
        return patch_bgr
    gain = np.clip(0.5 + 0.5 * (loc_l / obj_l), 0.75, 1.25)  # mild
    return np.clip(patch_bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def paste(frame, label, rgba, x, y, rng):
    """Alpha-composite rgba at top-left (x,y). Returns paste alpha-mask in frame coords
    (after wire-protection) or None if nothing pasted. Adds offset contact shadow."""
    H, W = frame.shape[:2]
    h, w = rgba.shape[:2]
    if x < 0 or y < 0 or x + w > W or y + h > H:
        return None
    obj_bgr = rgba[..., :3].copy()
    obj_a = rgba[..., 3].astype(np.float32) / 255.0

    region = frame[y:y + h, x:x + w]
    obj_bgr = local_luma_match(obj_bgr, rgba[..., 3], region)

    # --- wire protection: never cover label==4 ---
    lab_region = label[y:y + h, x:x + w]
    wire_here = (lab_region == WIRE)
    if wire_here.any():
        obj_a[wire_here] = 0.0

    if (obj_a > 0.04).sum() < 30:
        return None  # essentially nothing left after protection

    # --- soft offset contact shadow (scene-generic, NOT wire-derived) ---
    sh_dx = rng.choice([-1, 1]) * rng.randint(3, 8)
    sh_dy = rng.randint(4, 10)
    shadow_a = cv2.GaussianBlur((obj_a * 255).astype(np.uint8), (0, 0), 5.0).astype(np.float32) / 255.0
    sa = np.zeros_like(obj_a)
    ys, xs = np.where(shadow_a > 0.02)
    for_y = np.clip(ys + sh_dy, 0, h - 1)
    for_x = np.clip(xs + sh_dx, 0, w - 1)
    sa[for_y, for_x] = shadow_a[ys, xs]
    sa = sa * (1 - obj_a) * 0.28  # low opacity, only where object isn't
    region_f = region.astype(np.float32)
    region_f = region_f * (1 - sa[..., None])  # darken for shadow

    # --- alpha composite object over (shadowed) region ---
    a3 = obj_a[..., None]
    comp = obj_bgr.astype(np.float32) * a3 + region_f * (1 - a3)
    frame[y:y + h, x:x + w] = np.clip(comp, 0, 255).astype(np.uint8)

    # label: pasted object pixels -> 0 (background). Use a firm threshold so the
    # labeled-bg region matches the visible object (not the soft AA fringe/shadow).
    paste_mask = (obj_a > 0.5)
    lab_region[paste_mask] = 0          # background (NOT wire)
    label[y:y + h, x:x + w] = lab_region

    full = np.zeros((H, W), np.uint8)
    full[y:y + h, x:x + w] = (obj_a > 0.5).astype(np.uint8) * 255
    return full


def choose_bg_position(label, w, h, rng, tries=40):
    """Pick a top-left so the object box lies mostly on background (Label!=4).
    Wire-independent: random candidate boxes, accept if wire-overlap is small."""
    H, W = label.shape
    if w >= W or h >= H:
        return None
    for _ in range(tries):
        x = rng.randint(0, W - w)
        y = rng.randint(0, H - h)
        box = label[y:y + h, x:x + w]
        wire_frac = (box == WIRE).mean()
        if wire_frac < 0.10:    # mostly background; small chance-overlap allowed (protected)
            return x, y
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases-list", default=DEFAULT_BASES,
                    help="exact synth train bases to composite (1800, no resampling)")
    ap.add_argument("--p-bright", type=float, default=0.55,
                    help="prob a pasted cutout is recolored toward bright/pale target")
    ap.add_argument("--out-dir", default="data/dformer_dataset_phase15_wirefree_p43neg")
    ap.add_argument("--synth", default=SYNTH)
    ap.add_argument("--seed", type=int, default=39)
    ap.add_argument("--min-objs", type=int, default=1)
    ap.add_argument("--max-objs", type=int, default=3)
    ap.add_argument("--scale-min", type=int, default=60)
    ap.add_argument("--scale-max", type=int, default=220)
    ap.add_argument("--samples-dir", default="results/realism_campaign/p43_brightdisambig/samples")
    ap.add_argument("--n-samples", type=int, default=16)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cuts = load_cutouts()
    if not cuts:
        raise SystemExit("no cutouts found / all gated out")
    print(f"loaded {len(cuts)} EWD-passed cutouts")

    out = args.out_dir
    for sub in ("RGB", "Label", "Depth"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)

    # EXACT bases (no resampling): read / strip / sort the P39 substitution list.
    with open(args.bases_list) as f:
        chosen = sorted(l.strip() for l in f if l.strip())
    if not chosen:
        raise SystemExit(f"no bases read from {args.bases_list}")
    print(f"compositing EXACTLY {len(chosen)} bases from {args.bases_list} (no resampling)")
    print(f"p-bright = {args.p_bright}")

    meta = []
    n_sample_saved = 0
    sample_idxs = set(range(0, len(chosen), max(1, len(chosen) // args.n_samples)))

    # bright-recolor bookkeeping
    n_objects = 0
    n_brightened = 0          # successfully recolored (passed anti-flat gate)
    n_natural = 0             # not selected for brightening
    n_antiflat_fallback = 0   # selected for brightening but gate fired -> reverted
    per_target = {k: 0 for k in BRIGHT_TARGET_NAMES}   # successful brightens only
    bright_luma_stds = []     # opaque-luma std of successfully brightened cutouts

    for i, base in enumerate(chosen):
        rgb = cv2.imread(os.path.join(args.synth, "RGB", base + ".png"))
        lab = cv2.imread(os.path.join(args.synth, "Label", base + ".png"), cv2.IMREAD_UNCHANGED)
        dep = cv2.imread(os.path.join(args.synth, "Depth", base + ".png"), cv2.IMREAD_UNCHANGED)
        if rgb is None or lab is None:
            continue
        orig_rgb = rgb.copy()
        orig_wire = (lab == WIRE)
        n_obj = rng.randint(args.min_objs, args.max_objs)
        pastes = []
        composite_mask = np.zeros(lab.shape, np.uint8)
        for _ in range(n_obj):
            _, rgba = cuts[rng.randrange(len(cuts))]
            tgt = rng.randint(args.scale_min, args.scale_max)
            flip = rng.random() < 0.5
            rot = rng.uniform(-25, 25)
            cut = prep_cutout(rgba, tgt, flip, rot)
            ch, cw = cut.shape[:2]
            if ch >= rgb.shape[0] or cw >= rgb.shape[1]:
                continue
            pos = choose_bg_position(lab, cw, ch, rng)
            if pos is None:
                continue
            x, y = pos

            # ---- bright/pale recolor (AFTER prep, BEFORE paste) ----
            do_bright = rng.random() < args.p_bright
            tgt_name = None
            antiflat = False
            if do_bright:
                cut, tgt_name, antiflat, bstd = brighten_cutout(cut, rng)
                if antiflat:
                    n_antiflat_fallback += 1
                else:
                    n_brightened += 1
                    per_target[tgt_name] += 1
                    bright_luma_stds.append(bstd)
            else:
                n_natural += 1

            pm = paste(rgb, lab, cut, x, y, rng)
            if pm is None:
                # paste rejected (off-frame / fully wire-protected); the object
                # never lands, but the brighten bookkeeping above already counted
                # the recolor attempt (it's a per-attempt statistic). Skip record.
                continue
            n_objects += 1
            composite_mask |= pm
            pastes.append({"x": int(x), "y": int(y), "w": int(cw), "h": int(ch),
                           "scale_px": int(tgt),
                           "brightened": bool(do_bright and not antiflat),
                           "bright_target": (tgt_name if (do_bright and not antiflat) else None),
                           "antiflat_fallback": bool(antiflat)})
        if not pastes:
            pass
        # GUARANTEE wire GT identical
        assert np.array_equal((lab == WIRE), orig_wire), f"wire GT changed on {base}"

        nb = base + "_p43neg"
        cv2.imwrite(os.path.join(out, "RGB", nb + ".png"), rgb)
        cv2.imwrite(os.path.join(out, "Label", nb + ".png"), lab)
        if dep is not None:
            cv2.imwrite(os.path.join(out, "Depth", nb + ".png"), dep)
        meta.append({"base": base, "new_base": nb, "n_pasted": len(pastes),
                     "pastes": pastes})

        # save sample montage
        if i in sample_idxs and n_sample_saved < args.n_samples and pastes:
            overlay = rgb.copy()
            overlay[composite_mask > 0] = (0, 255, 0)  # pasted=green (labeled bg)
            overlay[lab == WIRE] = (0, 0, 255)         # wire=red (untouched)
            vis = np.hstack([orig_rgb, rgb, cv2.addWeighted(rgb, 0.55, overlay, 0.45, 0)])
            cv2.imwrite(os.path.join(args.samples_dir, f"montage_{n_sample_saved:02d}_{base}.png"), vis)
            n_sample_saved += 1

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(chosen)} composited")

    with open(os.path.join(out, "p43_substituted_basenames.txt"), "w") as f:
        f.write("\n".join(sorted(chosen)) + "\n")

    mean_bright_std = float(np.mean(bright_luma_stds)) if bright_luma_stds else 0.0
    summary = {
        "p_bright": args.p_bright,
        "seed": args.seed,
        "n_frames": len(meta),
        "n_objects": n_objects,
        "n_brightened": n_brightened,
        "n_natural": n_natural,
        "n_antiflat_fallback": n_antiflat_fallback,
        "per_target": per_target,
        "mean_brightened_luma_std": mean_bright_std,
        "anti_flat_std_threshold": ANTI_FLAT_STD,
    }
    with open(os.path.join(out, "composite_meta.json"), "w") as f:
        json.dump({"summary": summary, "frames": meta}, f, indent=2)

    print(f"\n=== P43 BRIGHT-DISAMBIG COMPOSITE DONE ===")
    print(f"out: {out}")
    print(f"frames written: {len(meta)}  total objects pasted: {n_objects}")
    print(f"brightened (passed anti-flat): {n_brightened}")
    print(f"natural (not selected):        {n_natural}")
    print(f"anti-flat-gate fallbacks:      {n_antiflat_fallback}  (threshold std>{ANTI_FLAT_STD})")
    print(f"per-target (successful brightens):")
    for k in BRIGHT_TARGET_NAMES:
        print(f"    {k:14s}: {per_target[k]}")
    print(f"mean brightened opaque-luma std: {mean_bright_std:.2f}  (must be > {ANTI_FLAT_STD})")
    print(f"samples -> {args.samples_dir} ({n_sample_saved})")
    print(f"substituted list -> {os.path.join(out, 'p43_substituted_basenames.txt')}")


if __name__ == "__main__":
    main()
