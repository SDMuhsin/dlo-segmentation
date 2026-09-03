#!/usr/bin/env python
"""P44.3 STEP 1 (go/no-go): does a P44-trained model still call REAL cable pixels WIRE?

`probe_p44_texture_cue.py` proved the CONNECTOR half of the shortcut: P44 labels 94.8 %
of a pasted photograph of a wrench "connector". The unified explanation in the P44
diagnosis says the same fact has a WIRE half - in both P44 arms

    P(wire | pixel came from a real photograph) == 0

because no photographic pixel is ever labelled WIRE. If that is what the model learned,
then real cable texture pasted into a synthetic frame should stop firing WIRE for P44 /
P44.2 while the baseline still fires.

Test (mirror image of the connector probe): cut REAL cable regions out of
data/dformer_dataset_movingcables (9,371 real frames with binary cable GT; label value 4)
using their own masks, paste them wire-independently onto held-out SYNTHETIC frames with
the synthetic labels left untouched, and measure over the pasted cable pixels only the
fraction predicted WIRE by each model.

  baseline fires WIRE, P44/P44.2 do not  -> mechanism confirmed, build P44.3.
  all models behave the same             -> premise refuted, STOP and re-diagnose.

Control arm: the identical cable SILHOUETTE filled with a crop of the frame's OWN
rendered backdrop. That separates "real cable texture" from "a thin cable-shaped thing",
which is the confound that would otherwise let a shape-driven model pass for a
texture-driven one.

Scale is matched, not arbitrary: MovingCables is shot much closer than these renders
(cable stroke ~32 px vs ~10 px here), so each crop is rescaled to put its stroke width
inside the synthetic wire width distribution. An unmatched scale would test scale
brittleness (P3W: 0.745 @1x -> 0.621 @2x) instead of texture.
"""
from __future__ import annotations
import argparse, glob, json, os, random, sys
import cv2, numpy as np, torch

BG, WIRE, CONN = 0, 1, 2
MC_WIRE = 4                      # MovingCables encoding D (convert_movingcables_to_dformer)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_model(ckpt, device):
    from transformers import SegformerForSemanticSegmentation
    m = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b5", num_labels=3, ignore_mismatched_sizes=True)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, _ = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "decode_head" in k or "encoder" in k], missing[:5]
    return m.to(device).eval()


@torch.no_grad()
def predict(model, bgr, device):
    x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
    lo = model(pixel_values=t).logits
    lo = torch.nn.functional.interpolate(lo, size=bgr.shape[:2], mode="bilinear",
                                         align_corners=False)[0]
    return lo.argmax(0).cpu().numpy().astype(np.uint8)


def stroke_width(mask):
    """2 x the 95th-pct distance-to-edge = a robust apparent cable width in px."""
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    v = d[mask]
    return float(2 * np.percentile(v, 95)) if v.size else 0.0


def build_cable_cutouts(mc_dir, n, rng, area=(600, 40000), window=(120, 300),
                        fill=(0.06, 0.32), stroke=(8.0, 45.0), clip_parity="odd"):
    """RGBA cable cutouts cut from real MovingCables frames with their own GT masks.

    Same window-crop extraction as `export_wire_cutouts_mc.py` (connected components do
    not work on this data - every cable merges into one blob-shaped component), but drawn
    from the COMPLEMENTARY clip parity, so a model trained on the exported library is
    probed on real cable from clips it has never seen.
    """
    labs = sorted(glob.glob(os.path.join(mc_dir, "Label", "*.png")))
    if clip_parity in ("even", "odd"):
        want = 0 if clip_parity == "even" else 1
        labs = [p for p in labs if int(os.path.basename(p).split("_")[0]) % 2 == want]
    rng.shuffle(labs)
    out = []
    for lp in labs:
        if len(out) >= n:
            break
        lab = cv2.imread(lp, cv2.IMREAD_UNCHANGED)
        if lab is None:
            continue
        m_full = (lab == MC_WIRE)
        if not m_full.any():
            continue
        rgb = cv2.imread(os.path.join(mc_dir, "RGB", os.path.basename(lp)), cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        H, W = m_full.shape
        for _try in range(6):
            if len(out) >= n:
                break
            hh, ww = rng.randrange(*window), rng.randrange(*window)
            if hh >= H or ww >= W:
                continue
            y, x = rng.randrange(0, H - hh), rng.randrange(0, W - ww)
            m = m_full[y:y + hh, x:x + ww]
            if m.sum() < area[0]:
                continue
            ys, xs = np.where(m)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            m = m[y0:y1, x0:x1]
            if not (area[0] <= m.sum() <= area[1]) or not (fill[0] <= m.mean() <= fill[1]):
                continue
            if min(m.shape) < 8:
                continue
            sw = stroke_width(m)
            if not (stroke[0] <= sw <= stroke[1]):
                continue
            out.append((np.dstack([rgb[y + y0:y + y1, x + x0:x + x1],
                                   m.astype(np.uint8) * 255]), sw,
                        os.path.basename(lp)))
    return out


def build_synth_wire_cutouts(data_dir, n, rng, area=(400, 40000), window=(120, 400),
                             fill=(0.03, 0.32), stroke=(3.0, 45.0)):
    """Positive control: the SAME window-crop extraction applied to RENDERED wire (label 1).

    All models must fire WIRE strongly on these. That is what separates "this model has
    stopped responding to real cable" from "this probe's pasting mechanics break wire
    prediction", and it calibrates how much of the real-cable number is domain gap.
    """
    labs = sorted(glob.glob(os.path.join(data_dir, "Label", "*.png")))
    rng.shuffle(labs)
    out = []
    for lp in labs:
        if len(out) >= n:
            break
        lab = cv2.imread(lp, cv2.IMREAD_UNCHANGED)
        if lab is None:
            continue
        m_full = (lab == WIRE)
        if m_full.sum() < area[0]:
            continue
        rgb = cv2.imread(os.path.join(data_dir, "RGB", os.path.basename(lp)), cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        H, W = m_full.shape
        for _try in range(8):
            if len(out) >= n:
                break
            hh, ww = rng.randrange(*window), rng.randrange(*window)
            if hh >= H or ww >= W:
                continue
            y, x = rng.randrange(0, H - hh), rng.randrange(0, W - ww)
            m = m_full[y:y + hh, x:x + ww]
            if m.sum() < area[0]:
                continue
            ys, xs = np.where(m)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            m = m[y0:y1, x0:x1]
            if not (area[0] <= m.sum() <= area[1]) or not (fill[0] <= m.mean() <= fill[1]):
                continue
            if min(m.shape) < 8:
                continue
            sw = stroke_width(m)
            if not (stroke[0] <= sw <= stroke[1]):
                continue
            out.append((np.dstack([rgb[y + y0:y + y1, x + x0:x + x1],
                                   m.astype(np.uint8) * 255]), sw,
                        os.path.basename(lp)))
    return out


def place(cut, sw, target_w, rng):
    """Rescale a cable cutout so its stroke width becomes ~target_w px, then flip."""
    s = target_w / max(sw, 1e-6)
    rgb, a = cut[:, :, :3], cut[:, :, 3]
    if rng.random() < 0.5:
        rgb, a = rgb[:, ::-1], a[:, ::-1]
    h, w = a.shape
    nw, nh = max(4, int(round(w * s))), max(4, int(round(h * s)))
    it = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    # INTER_AREA / no global blur: same edge contract as the compositor.
    return cv2.resize(rgb, (nw, nh), interpolation=it), cv2.resize(a, (nw, nh), interpolation=it)


def sample_placements(cuts, lab, n, twlo, twhi, rng):
    """Wire-independent placement: uniform positions, rejected only if they touch GT."""
    H, W = lab.shape
    out = []
    for _ in range(n):
        for _try in range(60):
            cut, sw, _src = cuts[rng.randrange(len(cuts))]
            cr, ca = place(cut, sw, rng.uniform(twlo, twhi), rng)
            ch, cw = ca.shape
            if ch >= H or cw >= W:
                continue
            y, x = rng.randrange(0, H - ch), rng.randrange(0, W - cw)
            solid = ca > 127
            if solid.sum() < 150:
                continue
            if (lab[y:y + ch, x:x + cw][solid] != BG).any():
                continue
            out.append((y, x, cr, ca))
            break
    return out


def composite(rgb, placements, fill_from_backdrop, rng):
    H, W = rgb.shape[:2]
    frame = rgb.copy()
    m_all = np.zeros((H, W), bool)
    for (y, x, cr, ca) in placements:
        ch, cw = ca.shape
        if fill_from_backdrop:
            sy, sx = rng.randrange(0, H - ch), rng.randrange(0, W - cw)
            src = rgb[sy:sy + ch, sx:sx + cw]
        else:
            src = cr
        af = (ca.astype(np.float32) / 255.0)[..., None]
        roi = frame[y:y + ch, x:x + cw].astype(np.float32)
        frame[y:y + ch, x:x + cw] = (src * af + roi * (1 - af)).astype(np.uint8)
        m_all[y:y + ch, x:x + cw] |= ca > 127
    return frame, m_all


ARMS = ("real_cable", "synth_wire", "backdrop_ctrl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--per-frame", type=int, default=2)
    ap.add_argument("--data", default="data/dformer_dataset_3way_connscale3")
    ap.add_argument("--mc-dir", default="data/dformer_dataset_movingcables")
    ap.add_argument("--n-cutouts", type=int, default=400)
    ap.add_argument("--target-width", default="6,16",
                    help="px stroke-width range to rescale cables into "
                         "(the synthetic wire width distribution)")
    ap.add_argument("--out", default="results/p44_diagnosis/wire_cue")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dump", type=int, default=8, help="save N qualitative panels")
    ap.add_argument("--extra-model", action="append", default=[], metavar="NAME=CKPT")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(0)
    twlo, twhi = (float(v) for v in a.target_width.split(","))

    real_cuts = build_cable_cutouts(a.mc_dir, a.n_cutouts, rng)
    synth_cuts = build_synth_wire_cutouts(a.data, a.n_cutouts, rng)
    print(f"{len(real_cuts)} REAL cable strands (MovingCables, native stroke "
          f"{np.median([c[1] for c in real_cuts]):.1f} px) | "
          f"{len(synth_cuts)} RENDERED wire strands (native stroke "
          f"{np.median([c[1] for c in synth_cuts]):.1f} px)")

    bases = [os.path.splitext(os.path.basename(l.split()[0]))[0]
             for l in open(os.path.join(a.data, "test.txt")) if l.strip()]
    bases = rng.sample(bases, min(a.frames, len(bases)))

    specs = {
        "baseline": "results/realism_campaign/p3w_lovasz/lovasz050_zoom/best_model.pth",
        "p44": "results/realism_campaign/p44_connpaste/p44_wide/best_model.pth",
        "p442": "results/realism_campaign/p44_connpaste/p442_balanced/best_model.pth",
    }
    for e in a.extra_model:
        name, _, ck = e.partition("=")
        specs[name] = ck
    models = {k: load_model(v, a.device) for k, v in specs.items()}

    res = {m: {arm: {"wire": [], "conn": []} for arm in ARMS} for m in models}
    n_dumped = 0
    for i, b in enumerate(bases):
        rgb = cv2.imread(os.path.join(a.data, "RGB", b + ".png"))
        lab = cv2.imread(os.path.join(a.data, "Label", b + ".png"), cv2.IMREAD_UNCHANGED)
        if rgb is None or lab is None:
            continue
        pl_real = sample_placements(real_cuts, lab, a.per_frame, twlo, twhi, rng)
        pl_syn = sample_placements(synth_cuts, lab, a.per_frame, twlo, twhi, rng)
        if not pl_real or not pl_syn:
            continue

        frames = {
            "real_cable": composite(rgb, pl_real, False, rng),
            "synth_wire": composite(rgb, pl_syn, False, rng),
            # matched to real_cable: identical silhouettes and positions, filled with a
            # crop of the frame's OWN rendered backdrop
            "backdrop_ctrl": composite(rgb, pl_real, True, rng),
        }
        panels = {}
        for arm, (frame, m_all) in frames.items():
            if m_all.sum() < 150:
                continue
            for name, mdl in models.items():
                pred = predict(mdl, frame, a.device)
                res[name][arm]["wire"].append(float((pred[m_all] == WIRE).mean()))
                res[name][arm]["conn"].append(float((pred[m_all] == CONN).mean()))
                if arm == "real_cable" and n_dumped < a.dump:
                    ov = frame.copy()
                    ov[pred == WIRE] = (0.35 * ov[pred == WIRE] +
                                        0.65 * np.array([0, 255, 0])).astype(np.uint8)
                    ov[pred == CONN] = (0.35 * ov[pred == CONN] +
                                        0.65 * np.array([0, 0, 255])).astype(np.uint8)
                    panels[name] = ov
        if panels and n_dumped < a.dump:
            cv2.imwrite(os.path.join(a.out, f"panel_{n_dumped:02d}.png"),
                        np.hstack([frames["real_cable"][0]] +
                                  [panels[n] for n in models if n in panels]))
            n_dumped += 1
        if i % 30 == 0:
            print(f"  {i}/{len(bases)}", flush=True)

    print("\n=== fraction of PASTED pixels predicted WIRE / CONNECTOR ===")
    hdr = f"{'model':10s}"
    for arm in ARMS:
        hdr += f" | {arm:>14s} W    C"
    print(hdr)
    summary = {}
    for name in models:
        row = f"{name:10s}"
        summary[name] = {}
        for arm in ARMS:
            v = res[name][arm]
            e = lambda z: float(np.mean(z)) if z else float("nan")
            summary[name][arm] = {"wire": e(v["wire"]), "conn": e(v["conn"]),
                                  "n": len(v["wire"])}
            row += f" | {e(v['wire']):>13.1%} {e(v['conn']):>6.1%}"
        print(row)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(f"\n-> {a.out}/summary.json  ({n_dumped} panels: real | " +
          " | ".join(models) + ")")


if __name__ == "__main__":
    main()
