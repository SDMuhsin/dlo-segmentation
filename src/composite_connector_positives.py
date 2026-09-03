#!/usr/bin/env python
"""Graft REAL photographic connector cutouts onto synthetic 3-way frames as
LABELLED CONNECTOR (class 2).

Motivation: wire transfers to real footage, connectors do not. The renders only ever
show PointWire's procedural connector geometry, so the model has seen one connector
"look". Pasting real connector photographs adds genuine shape/colour diversity without
touching the renderer.

Contracts (each one earned by a past failure in this project):
  * Composite at NATIVE 640x480 with INTER_AREA downscale and NO global blur. Rendered
    connectors have ~0.97 px edges; a pasted cutout with systematically softer or
    sharper edges becomes an "is this a paste?" shortcut.
  * Placement is WIRE-INDEPENDENT: positions are sampled uniformly and rejected only if
    they would overlap existing GT. P42 showed placing negatives AT wire ends made
    things worse - do not co-locate pasted content with wire geometry.
  * Existing GT is NEVER overwritten. A paste that would land on label 1 (wire) or
    label 2 (connector) is re-sampled, so wire GT and rendered-connector GT are
    byte-identical to the source frame.
  * Pasted pixels -> Label 2. Depth is left untouched (RGB-only path).
  * Frames are SUBSTITUTED 1:1, not added, so the arm differs from its baseline in
    content only and not in dataset size. P37's "permissive drift" confound came from
    adding 2000 frames.
  * Cutouts carrying an attached cable are already rejected upstream (s3 tail gate):
    cable pixels under label 2 would teach "cable == connector", which is the exact
    confusion that costs this model connector IoU.

Scale is the interesting knob. Rendered connectors are TINY - median bbox 22x22 px,
95th pct ~51 px - so a 1000 px photo pasted at render scale loses nearly all its
detail. --scale-mode render matches the render distribution; --scale-mode wide also
covers the larger apparent sizes a real close-up camera produces, which the model has
never seen (P3W found this model scale-brittle: 0.745 @1x -> 0.621 @2x).

Usage:
  python src/composite_connector_positives.py \
      --synth-dir data/dformer_dataset_3way_connscale3 \
      --cutouts   data/connector_cutouts \
      --out-dir   data/dformer_dataset_p44_connpaste \
      --n-frames  1500 --scale-mode wide
"""
from __future__ import annotations

import argparse, glob, json, os, random
import cv2
import numpy as np

BG, WIRE, CONN = 0, 1, 2

# P44.2 -- REAL-TEXTURE BALANCE.
# P44 pasted real connector photos and nothing else, so in training
# P(connector | pixel came from a real photograph) == 1.0. The model duly learned
# "photographic texture => connector": it labelled 94.8% of a pasted photo of a WRENCH
# as connector, while a synthetic-texture control stayed at 1.9%, and stage ablation
# showed the cue distributed across the encoder (no single stage removes it). On real
# video every pixel is photographic, hence the overshoot.
# The fix is to paste real photos of NON-connector objects as labelled BACKGROUND in the
# SAME frames, so photographic texture stops being diagnostic and only connector SHAPE
# is left to carry the class.
#
# P44.3 -- THE MISSING THIRD CLASS.
# P44.2 balanced photographic texture across BACKGROUND and CONNECTOR but left WIRE out,
# so P(wire | photographic) stayed 0 and the shortcut only MOVED: real-video wire coverage
# fell further, 6.26% -> 1.95% (P44) -> 0.82% (P44.2). probe_p44_wire_cue.py measured the
# wire half of it directly - on real cable pasted into a synthetic frame the baseline
# predicts WIRE on 20.5% of pixels, P44 on 2.2% (89.9% CONNECTOR) and P44.2 on 2.4%, while
# all three sit at 70-80% on RENDERED wire pasted the same way.
# --wire-cutouts pastes real cable strands (MovingCables GT masks) as labelled WIRE in the
# same frames, so every class carries photographic texture and P(class | photographic) is
# driven to ~1/3 each BY PIXEL AREA (reported at the end of the run).
# Empirical rendered-connector bbox stats (400 frames, data/..._3way_connscale3):
# p5 5 px, p25 14, p50 22, p75 30, p95 51.
RENDER_SCALE = (8, 55)
WIDE_SCALE = (18, 190)
# Wire pastes are scaled by STROKE WIDTH, not by max-dimension: a cable's max-dimension is
# its length, which says nothing about how thick it looks, and thickness is what has to
# match the render. Measured: rendered wire stroke ~10 px, MovingCables ~29 px.
WIRE_STROKE = (6.0, 16.0)


def stroke_width(mask):
    """2 x the 95th-pct distance-to-edge = a robust apparent cable width in px."""
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    v = d[mask]
    return float(2 * np.percentile(v, 95)) if v.size else 0.0


def load_cutouts(root):
    """Load RGBA cutouts from <root>/<family>/*.png (skipping the curation reject bin)."""
    out = []
    for p in sorted(glob.glob(os.path.join(root, "*", "*.png"))):
        if os.sep + "_rejected" + os.sep in p:   # curation pass moves rejects here
            continue
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None or im.ndim != 3 or im.shape[2] != 4:
            continue
        out.append((p, im))
    return out


def prep(cut, target, rng):
    """Rotate / flip / resize one RGBA cutout to a target max-dimension."""
    rgb, a = cut[:, :, :3], cut[:, :, 3]
    if rng.random() < 0.5:
        rgb, a = rgb[:, ::-1], a[:, ::-1]
    ang = rng.uniform(0, 360)
    h, w = a.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    rgb = cv2.warpAffine(rgb, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    a = cv2.warpAffine(a, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    ys, xs = np.where(a > 8)
    if not len(ys):
        return None, None
    rgb = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    s = target / max(h, w)
    nw, nh = max(2, int(round(w * s))), max(2, int(round(h * s)))
    # INTER_AREA gives the ~1 px antialiased edge the renderer produces; no extra blur.
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return (cv2.resize(rgb, (nw, nh), interpolation=interp),
            cv2.resize(a, (nw, nh), interpolation=interp))


def photometric(rgb, frame, rng):
    """Nudge the cutout toward the frame's exposure. Scene-generic (frame-wide
    statistics only) - never derived from wire or label geometry, which past render
    experiments showed becomes a spurious cue that real footage lacks."""
    fl = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
    cl = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).mean() + 1e-6
    g = np.clip((fl / cl) ** 0.5, 0.65, 1.5) * rng.uniform(0.92, 1.08)
    return np.clip(rgb.astype(np.float32) * g, 0, 255).astype(np.uint8)


def prep_strand(cut, target_stroke, rng):
    """Rotate / flip one RGBA cable strand and rescale it to a target STROKE WIDTH."""
    rgb, a = cut[:, :, :3], cut[:, :, 3]
    if rng.random() < 0.5:
        rgb, a = rgb[:, ::-1], a[:, ::-1]
    ang = rng.uniform(0, 360)
    h, w = a.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    rgb = cv2.warpAffine(rgb, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    a = cv2.warpAffine(a, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    ys, xs = np.where(a > 8)
    if not len(ys):
        return None, None
    rgb = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sw = stroke_width(a > 127)
    if sw < 1.0:
        return None, None
    scale = target_stroke / sw
    h, w = a.shape
    nw, nh = max(3, int(round(w * scale))), max(3, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return (cv2.resize(rgb, (nw, nh), interpolation=interp),
            cv2.resize(a, (nw, nh), interpolation=interp))


def paste_wire(frame, label, cuts, area_budget, stroke_rng, rng, jitter, max_n=8):
    """Paste real cable strands as labelled WIRE until `area_budget` px are covered.

    Area-driven rather than count-driven because the P44.3 balance target is
    P(class | photographic) by PIXEL AREA, and a thin strand covers far fewer pixels than
    a compact connector cutout - matching counts would leave wire badly under-represented,
    which is the exact asymmetry that made P44.2 fail. All other contracts are the ones
    `paste` obeys: wire-independent uniform placement, existing GT never overwritten,
    INTER_AREA and no global blur, no contact shadow.
    """
    H, W = label.shape
    recs, covered = [], 0
    for _ in range(max_n):
        if covered >= area_budget:
            break
        placed = False
        for _try in range(40):
            path, cut = cuts[rng.randrange(len(cuts))]
            target = rng.uniform(*stroke_rng)
            rgb, a = prep_strand(cut, target, rng)
            if rgb is None:
                continue
            ch, cw = a.shape
            if ch >= H or cw >= W:
                continue
            y = rng.randrange(0, H - ch)
            x = rng.randrange(0, W - cw)
            roi_lab = label[y:y + ch, x:x + cw]
            solid = a > 127
            if solid.sum() < 60:
                continue
            if (roi_lab[solid] != BG).any():
                continue
            if jitter:
                rgb = photometric(rgb, frame, rng)
            af = (a.astype(np.float32) / 255.0)[..., None]
            roi = frame[y:y + ch, x:x + cw].astype(np.float32)
            frame[y:y + ch, x:x + cw] = (rgb * af + roi * (1 - af)).astype(np.uint8)
            roi_lab[solid] = WIRE
            covered += int(solid.sum())
            recs.append({"cutout": path, "x": x, "y": y, "w": cw, "h": ch,
                         "stroke_px": round(target, 2), "area": int(solid.sum()),
                         "cls": int(WIRE)})
            placed = True
            break
        if not placed:
            break
    return recs


def paste(frame, label, cuts, n, scale_rng, rng, jitter, as_class=CONN):
    H, W = label.shape
    recs = []
    for _ in range(n):
        placed = False
        for _try in range(40):
            path, cut = cuts[rng.randrange(len(cuts))]
            target = int(round(np.exp(rng.uniform(np.log(scale_rng[0]),
                                                  np.log(scale_rng[1])))))
            rgb, a = prep(cut, target, rng)
            if rgb is None:
                continue
            ch, cw = a.shape
            if ch >= H or cw >= W:
                continue
            y = rng.randrange(0, H - ch)
            x = rng.randrange(0, W - cw)
            roi_lab = label[y:y + ch, x:x + cw]
            solid = a > 127
            # never overwrite existing wire / connector GT
            if (roi_lab[solid] != BG).any():
                continue
            if jitter:
                rgb = photometric(rgb, frame, rng)
            af = (a.astype(np.float32) / 255.0)[..., None]
            roi = frame[y:y + ch, x:x + cw].astype(np.float32)
            frame[y:y + ch, x:x + cw] = (rgb * af + roi * (1 - af)).astype(np.uint8)
            roi_lab[solid] = as_class
            recs.append({"cutout": path, "x": x, "y": y, "w": cw, "h": ch,
                         "target_px": target, "area": int(solid.sum()),
                         "cls": int(as_class)})
            placed = True
            break
        if not placed:
            break
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", default="data/dformer_dataset_3way_connscale3")
    ap.add_argument("--cutouts", default="data/connector_cutouts")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-frames", type=int, default=1500)
    ap.add_argument("--per-frame", default="1,3", help="min,max pastes per frame")
    ap.add_argument("--scale-mode", choices=["render", "wide"], default="wide")
    ap.add_argument("--neg-cutouts", default=None,
                    help="dir of REAL NON-connector cutouts pasted as labelled "
                         "BACKGROUND, to decorrelate photographic texture from class 2")
    ap.add_argument("--neg-per-frame", default="1,3",
                    help="min,max background-labelled real-object pastes per frame")
    ap.add_argument("--wire-cutouts", default=None,
                    help="dir of REAL cable strand cutouts (see export_wire_cutouts_mc.py) "
                         "pasted as labelled WIRE, so class 1 also carries photographic "
                         "texture. Without this, P(wire | photographic) = 0 and the model "
                         "learns 'photographic => not wire' (P44 / P44.2 both failed here)")
    ap.add_argument("--wire-stroke", default="6,16",
                    help="min,max apparent cable stroke width in px for wire pastes "
                         "(rendered wire measures ~10 px)")
    ap.add_argument("--wire-max", type=int, default=8,
                    help="max wire strand pastes per frame (the count is area-driven)")
    ap.add_argument("--photometric", action="store_true",
                    help="match cutout exposure to the frame (scene-generic)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cuts = load_cutouts(args.cutouts)
    if not cuts:
        raise SystemExit(f"no RGBA cutouts under {args.cutouts}")
    print(f"loaded {len(cuts)} connector cutouts")
    negs = load_cutouts(args.neg_cutouts) if args.neg_cutouts else []
    if args.neg_cutouts:
        if not negs:
            raise SystemExit(f"no RGBA cutouts under {args.neg_cutouts}")
        print(f"loaded {len(negs)} NON-connector cutouts (pasted as background)")
    wires = load_cutouts(args.wire_cutouts) if args.wire_cutouts else []
    if args.wire_cutouts:
        if not wires:
            raise SystemExit(f"no RGBA cutouts under {args.wire_cutouts}")
        print(f"loaded {len(wires)} real cable strands (pasted as WIRE)")

    train = [l.strip() for l in open(os.path.join(args.synth_dir, "train.txt")) if l.strip()]
    bases = [t.split()[0] if " " in t else t for t in train]
    bases = [os.path.splitext(os.path.basename(b))[0] for b in bases]
    chosen = rng.sample(bases, min(args.n_frames, len(bases)))
    lo, hi = (int(v) for v in args.per_frame.split(","))
    nlo, nhi = (int(v) for v in args.neg_per_frame.split(","))
    smin, smax = RENDER_SCALE if args.scale_mode == "render" else WIDE_SCALE
    wlo, whi = (float(v) for v in args.wire_stroke.split(","))

    for sub in ("RGB", "Label", "Depth"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    meta, n_paste, n_neg, n_wire = [], 0, 0, 0
    area = {BG: 0, WIRE: 0, CONN: 0}   # pasted PHOTOGRAPHIC pixels per class
    for i, b in enumerate(chosen):
        rp = os.path.join(args.synth_dir, "RGB", b + ".png")
        lp = os.path.join(args.synth_dir, "Label", b + ".png")
        dp = os.path.join(args.synth_dir, "Depth", b + ".png")
        frame = cv2.imread(rp, cv2.IMREAD_COLOR)
        label = cv2.imread(lp, cv2.IMREAD_UNCHANGED)
        if frame is None or label is None:
            continue
        before = label.copy()
        recs = paste(frame, label, cuts, rng.randint(lo, hi), (smin, smax), rng,
                     args.photometric)
        if not recs:
            continue
        # Same real-photo provenance, same size range, same wire-independent placement --
        # the ONLY difference from the positives is the label. That is what makes
        # photographic texture uninformative about the class.
        if negs:
            recs += paste(frame, label, negs, rng.randint(nlo, nhi), (smin, smax), rng,
                          args.photometric, as_class=BG)
        # WIRE last, with an area budget equal to the mean of what the other two classes
        # actually covered -> P(class | photographic) ~ 1/3 each by pixel area.
        if wires:
            other = [sum(r["area"] for r in recs if r["cls"] == c) for c in (CONN, BG)]
            budget = max(200, int(round(sum(other) / 2.0)))
            recs += paste_wire(frame, label, wires, budget, (wlo, whi), rng,
                               args.photometric, max_n=args.wire_max)
        # contract check: existing GT untouched
        keep = before != BG
        assert (label[keep] == before[keep]).all(), f"GT overwritten in {b}"
        out_b = b + "_p44conn"
        cv2.imwrite(os.path.join(args.out_dir, "RGB", out_b + ".png"), frame)
        cv2.imwrite(os.path.join(args.out_dir, "Label", out_b + ".png"), label)
        if os.path.exists(dp):
            d = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(os.path.join(args.out_dir, "Depth", out_b + ".png"), d)
        meta.append({"base": b, "out": out_b, "pastes": recs})
        n_paste += sum(1 for r in recs if r["cls"] == CONN)
        n_neg += sum(1 for r in recs if r["cls"] == BG)
        n_wire += sum(1 for r in recs if r["cls"] == WIRE)
        for r in recs:
            area[r["cls"]] += r.get("area", 0)
        if i % 200 == 0:
            print(f"  {i}/{len(chosen)} frames, {n_paste} pastes", flush=True)

    with open(os.path.join(args.out_dir, "substituted_basenames.txt"), "w") as f:
        f.write("\n".join(m["base"] for m in meta))
    tot = sum(area.values()) or 1
    pclass = {"background": area[BG] / tot, "wire": area[WIRE] / tot,
              "connector": area[CONN] / tot}
    json.dump({"args": vars(args), "n_frames": len(meta), "n_pastes": n_paste,
               "n_neg_pastes": n_neg, "n_wire_pastes": n_wire,
               "n_cutouts": len(cuts), "n_negs": len(negs), "n_wires": len(wires),
               "pasted_area_px": {str(k): v for k, v in area.items()},
               "p_class_given_photographic": pclass,
               "frames": meta},
              open(os.path.join(args.out_dir, "composite_meta.json"), "w"), indent=2)
    print(f"\nwrote {len(meta)} composited frames, {n_paste} connector pastes, "
          f"{n_neg} background pastes, {n_wire} wire pastes -> {args.out_dir}")
    print("P(class | pixel came from a real photograph), by pasted pixel AREA:")
    for k, v in pclass.items():
        print(f"    {k:11s} {v:6.3f}")
    print("    (target ~0.333 each; P44 was conn=1.000, P44.2 conn=0.513 wire=0.000)")


if __name__ == "__main__":
    main()
