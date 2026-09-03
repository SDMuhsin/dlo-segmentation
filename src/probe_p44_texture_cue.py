#!/usr/bin/env python
"""Ground-level probe: does P44 key on CONNECTOR SHAPE, or on REAL PHOTOGRAPHIC TEXTURE?

In the P44 training set the ONLY pixels sourced from a real photograph are the pasted
connectors; everything else is synthetic splat render. "Looks photographic" therefore
predicts class 2 perfectly in training. On real video every pixel is photographic - which
would explain why the overshoot appears ONLY on real footage (synthetic-val connector
precision/recall are unchanged vs baseline).

Test: paste REAL photos of NON-connector objects (P39 tool cutouts: pliers, screwdrivers,
wrenches, hex keys...) onto held-out SYNTHETIC frames, wire-independently, leaving labels
untouched (they stay background). Then measure, over the pasted pixels only, the fraction
predicted CONNECTOR by each model.

  If P44 >> baseline  -> the cue is photographic texture, NOT connector shape.
  If P44 ~= baseline  -> the overshoot is something else (shape, prior, scale).

Control arm: paste a SYNTHETIC patch (a crop of the frame's own rendered backdrop) at the
same places and sizes. A texture-cue model should stay quiet on that, which separates
"real texture" from "any pasted object / any edge discontinuity".
"""
from __future__ import annotations
import argparse, glob, json, os, random, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BG, WIRE, CONN = 0, 1, 2
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_model(ckpt, device):
    from transformers import SegformerForSemanticSegmentation
    m = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b5", num_labels=3, ignore_mismatched_sizes=True)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
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
    return lo.argmax(0).cpu().numpy().astype(np.uint8), lo.cpu().numpy()


def prep(cut, target, rng):
    rgb, a = cut[:, :, :3], cut[:, :, 3]
    if rng.random() < 0.5:
        rgb, a = rgb[:, ::-1], a[:, ::-1]
    h, w = a.shape
    s = target / max(h, w)
    nw, nh = max(2, int(w * s)), max(2, int(h * s))
    it = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.resize(rgb, (nw, nh), interpolation=it), cv2.resize(a, (nw, nh), interpolation=it)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--per-frame", type=int, default=2)
    ap.add_argument("--data", default="data/dformer_dataset_3way_connscale3")
    ap.add_argument("--negs", default="data/neg_cutouts_p39")
    ap.add_argument("--out", default="results/p44_diagnosis/texture_cue")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--extra-model", action="append", default=[],
                    metavar="NAME=CKPT",
                    help="additional model to score, e.g. p442=path/to/best_model.pth")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(0)

    negs = []
    for p in sorted(glob.glob(os.path.join(a.negs, "*", "*.png"))):
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is not None and im.ndim == 3 and im.shape[2] == 4:
            negs.append(im)
    print(f"{len(negs)} real NON-connector cutouts")

    bases = [os.path.splitext(os.path.basename(l.split()[0]))[0]
             for l in open(os.path.join(a.data, "test.txt")) if l.strip()]
    bases = rng.sample(bases, min(a.frames, len(bases)))

    specs = {
        "baseline": "results/realism_campaign/p3w_lovasz/lovasz050_zoom/best_model.pth",
        "p44": "results/realism_campaign/p44_connpaste/p44_wide/best_model.pth",
    }
    for e in a.extra_model:
        name, _, ck = e.partition("=")
        specs[name] = ck
    models = {k: load_model(v, a.device) for k, v in specs.items()}

    res = {m: {"real": [], "synth": []} for m in models}
    for i, b in enumerate(bases):
        rgb = cv2.imread(os.path.join(a.data, "RGB", b + ".png"))
        lab = cv2.imread(os.path.join(a.data, "Label", b + ".png"), cv2.IMREAD_UNCHANGED)
        if rgb is None or lab is None:
            continue
        H, W = lab.shape
        placements = []
        for _ in range(a.per_frame):
            for _try in range(40):
                cut = negs[rng.randrange(len(negs))]
                tgt = int(np.exp(rng.uniform(np.log(30), np.log(120))))
                cr, ca = prep(cut, tgt, rng)
                ch, cw = ca.shape
                if ch >= H or cw >= W:
                    continue
                y, x = rng.randrange(0, H - ch), rng.randrange(0, W - cw)
                if (lab[y:y + ch, x:x + cw][ca > 127] != BG).any():
                    continue
                placements.append((y, x, cr, ca))
                break
        if not placements:
            continue

        for kind in ("real", "synth"):
            frame = rgb.copy()
            m_all = np.zeros((H, W), bool)
            for (y, x, cr, ca) in placements:
                ch, cw = ca.shape
                if kind == "synth":
                    # same silhouette, but filled with the frame's OWN rendered backdrop
                    sy = rng.randrange(0, H - ch); sx = rng.randrange(0, W - cw)
                    src = rgb[sy:sy + ch, sx:sx + cw]
                else:
                    src = cr
                af = (ca.astype(np.float32) / 255.0)[..., None]
                roi = frame[y:y + ch, x:x + cw].astype(np.float32)
                frame[y:y + ch, x:x + cw] = (src * af + roi * (1 - af)).astype(np.uint8)
                m_all[y:y + ch, x:x + cw] |= ca > 127
            if m_all.sum() < 50:
                continue
            for name, mdl in models.items():
                pred, _ = predict(mdl, frame, a.device)
                res[name][kind].append(float((pred[m_all] == CONN).mean()))
        if i % 30 == 0:
            print(f"  {i}/{len(bases)}", flush=True)

    print("\n=== fraction of PASTED-OBJECT pixels predicted CONNECTOR ===")
    print(f"{'model':10s} {'REAL photo objects':>20s} {'SYNTH-texture control':>24s}")
    summary = {}
    for name in models:
        r = float(np.mean(res[name]["real"])) if res[name]["real"] else float("nan")
        s = float(np.mean(res[name]["synth"])) if res[name]["synth"] else float("nan")
        summary[name] = {"real": r, "synth": s, "n": len(res[name]["real"])}
        print(f"{name:10s} {r:>19.1%} {s:>23.1%}")
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(f"\n-> {a.out}/summary.json")


if __name__ == "__main__":
    main()
