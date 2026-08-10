#!/usr/bin/env python
"""P39 compositing: paste REAL photographic object cutouts (data/neg_cutouts_p39/)
onto BACKGROUND regions of randomly-chosen synthetic phase15_wirefree TRAIN frames,
labeling every pasted pixel as BACKGROUND (0). Produces drift-controlled hard-negative
frames with basename suffix ``_p39neg`` under a new dataset dir.

KEY CONTRACTS (learned the hard way):
  * Composite at NATIVE 640x480, NO global blur. Object boundary AA ~1px only, so the
    model cannot learn "soft edge = negative". Internal speculars/seams stay sharp/real.
  * Placement is WIRE-INDEPENDENT: positions chosen on Label!=4 pixels, but the choice of
    where is NOT derived from the wire mask. Objects may land near/far from wires by chance.
  * Wire GT (label==4) is NEVER covered: any pasted pixel that would overlap a wire pixel
    is masked out of the paste (alpha->0 there) so wire GT stays the same.
  * Pasted pixels -> Label 0 (background), NOT 4. Depth left as-is (RGB-only path).
  * Soft, low-opacity contact/drop shadow OFFSET from the object (scene-generic, NOT
    derived from wire geometry).

CPU only. No torch / no GPU.

Outputs:
  <out-dir>/{RGB,Label,Depth}/<base>_p39neg.png
  <out-dir>/p39_substituted_basenames.txt   (the chosen synth train bases, original names)
  <out-dir>/composite_meta.json             (per-frame paste records, for validation)
"""
import argparse, os, glob, json, random
import numpy as np
import cv2

WIRE = 4
SYNTH = "data/dformer_dataset_phase15_wirefree"
CUTDIR = "data/neg_cutouts_p39"


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
    ap.add_argument("--n", type=int, default=1800, help="# train frames to composite")
    ap.add_argument("--out-dir", default="data/dformer_dataset_phase15_wirefree_p39neg")
    ap.add_argument("--synth", default=SYNTH)
    ap.add_argument("--seed", type=int, default=39)
    ap.add_argument("--min-objs", type=int, default=1)
    ap.add_argument("--max-objs", type=int, default=3)
    ap.add_argument("--scale-min", type=int, default=60)
    ap.add_argument("--scale-max", type=int, default=220)
    ap.add_argument("--samples-dir", default="results/realism_campaign/p39_cutoutneg/samples")
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

    # train bases (original synth names, e.g. 000_0000_00_front)
    with open(os.path.join(args.synth, "train.txt")) as f:
        train_bases = [l.strip().split("/", 1)[1][:-4] for l in f if l.strip()]
    chosen = rng.sample(train_bases, min(args.n, len(train_bases)))
    chosen_set = set(chosen)
    print(f"chosen {len(chosen)} / {len(train_bases)} train frames to composite")

    meta = []
    n_sample_saved = 0
    sample_idxs = set(range(0, len(chosen), max(1, len(chosen) // args.n_samples)))

    for i, base in enumerate(sorted(chosen)):
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
            pm = paste(rgb, lab, cut, x, y, rng)
            if pm is None:
                continue
            composite_mask |= pm
            pastes.append({"x": int(x), "y": int(y), "w": int(cw), "h": int(ch),
                           "scale_px": int(tgt)})
        if not pastes:
            # nothing pasted (rare) -> still write the frame unchanged-ish? skip to keep
            # the substitution meaningful; we just re-pick is overkill. Write as-is.
            pass
        # GUARANTEE wire GT identical
        assert np.array_equal((lab == WIRE), orig_wire), f"wire GT changed on {base}"

        nb = base + "_p39neg"
        cv2.imwrite(os.path.join(out, "RGB", nb + ".png"), rgb)
        cv2.imwrite(os.path.join(out, "Label", nb + ".png"), lab)
        if dep is not None:
            cv2.imwrite(os.path.join(out, "Depth", nb + ".png"), dep)
        meta.append({"base": base, "new_base": nb, "n_pasted": len(pastes),
                     "pastes": pastes})

        # save sample montage
        if i in sample_idxs and n_sample_saved < args.n_samples and pastes:
            overlay = rgb.copy()
            overlay[composite_mask > 0] = (0, 255, 0)  # pasted=green
            overlay[lab == WIRE] = (0, 0, 255)         # wire=red
            vis = np.hstack([orig_rgb, rgb, cv2.addWeighted(rgb, 0.55, overlay, 0.45, 0)])
            cv2.imwrite(os.path.join(args.samples_dir, f"montage_{n_sample_saved:02d}_{base}.png"), vis)
            n_sample_saved += 1

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(chosen)} composited")

    with open(os.path.join(out, "p39_substituted_basenames.txt"), "w") as f:
        f.write("\n".join(sorted(chosen)) + "\n")
    with open(os.path.join(out, "composite_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    npasted = sum(m["n_pasted"] for m in meta)
    print(f"\n=== P39 COMPOSITE DONE ===")
    print(f"out: {out}")
    print(f"frames written: {len(meta)}  total objects pasted: {npasted}")
    print(f"samples -> {args.samples_dir} ({n_sample_saved})")
    print(f"substituted list -> {os.path.join(out, 'p39_substituted_basenames.txt')}")


if __name__ == "__main__":
    main()
