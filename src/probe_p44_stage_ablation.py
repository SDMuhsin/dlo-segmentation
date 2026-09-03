#!/usr/bin/env python
"""Localise the 'photographic texture => connector' cue inside SegFormer.

SegFormer's decode head projects all four encoder stages (linear_c.0..3), concatenates
them and fuses. Zeroing one projected stage before the fuse removes that stage's
contribution to the logits, so the drop in connector firing on pasted REAL-photo regions
says which stage carries the cue.

  stage 1 (1/4 res, low-level texture/grain)  -> a texture statistic
  stage 4 (1/32 res, semantic/shape context)  -> an object-identity cue

Also reports the classifier head's class-2 bias for both models, to confirm the effect is
not a global prior shift (synthetic-val precision/recall already say it is not).
"""
from __future__ import annotations
import argparse, glob, os, random, sys, json
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_p44_texture_cue import load_model, prep, MEAN, STD, BG, CONN


@torch.no_grad()
def logits_with_ablation(model, bgr, device, ablate=None):
    x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
    enc = model.segformer(pixel_values=t, output_hidden_states=True)
    hs = enc.hidden_states
    dh = model.decode_head
    feats = []
    for i, (h, proj) in enumerate(zip(hs, dh.linear_c)):
        # SegformerMLP flattens internally - pass the 4D stage tensor straight in
        f = proj(h).permute(0, 2, 1)
        f = f.reshape(f.shape[0], -1, h.shape[2], h.shape[3])
        f = torch.nn.functional.interpolate(f, size=hs[0].shape[2:], mode="bilinear",
                                            align_corners=False)
        if ablate is not None and i == ablate:
            f = torch.zeros_like(f)
        feats.append(f)
    fused = dh.linear_fuse(torch.cat(feats[::-1], dim=1))
    fused = dh.batch_norm(fused)
    fused = dh.activation(fused)
    lo = dh.classifier(dh.dropout(fused))
    lo = torch.nn.functional.interpolate(lo, size=bgr.shape[:2], mode="bilinear",
                                         align_corners=False)[0]
    return lo.argmax(0).cpu().numpy().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--data", default="data/dformer_dataset_3way_connscale3")
    ap.add_argument("--negs", default="data/neg_cutouts_p39")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/p44_diagnosis/stage_ablation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(0)

    negs = [im for im in (cv2.imread(p, cv2.IMREAD_UNCHANGED)
            for p in sorted(glob.glob(os.path.join(a.negs, "*", "*.png"))))
            if im is not None and im.ndim == 3 and im.shape[2] == 4]
    bases = [os.path.splitext(os.path.basename(l.split()[0]))[0]
             for l in open(os.path.join(a.data, "test.txt")) if l.strip()]
    bases = rng.sample(bases, a.frames)

    ck = {"baseline": "results/realism_campaign/p3w_lovasz/lovasz050_zoom/best_model.pth",
          "p44": "results/realism_campaign/p44_connpaste/p44_wide/best_model.pth"}
    models = {k: load_model(v, a.device) for k, v in ck.items()}

    print("=== classifier head, class-2 (connector) bias ===")
    for k, m in models.items():
        b = m.decode_head.classifier.bias.detach().cpu().numpy()
        w = m.decode_head.classifier.weight.detach().cpu().numpy()
        print(f"  {k:9s} bias={np.round(b,3)}  |W[conn]|={np.linalg.norm(w[2]):.3f} "
              f"|W[wire]|={np.linalg.norm(w[1]):.3f}")

    scores = {k: {v: [] for v in ["none", 0, 1, 2, 3]} for k in models}
    for n, b in enumerate(bases):
        rgb = cv2.imread(os.path.join(a.data, "RGB", b + ".png"))
        lab = cv2.imread(os.path.join(a.data, "Label", b + ".png"), cv2.IMREAD_UNCHANGED)
        if rgb is None or lab is None:
            continue
        H, W = lab.shape
        frame = rgb.copy(); m_all = np.zeros((H, W), bool); placed = 0
        for _ in range(2):
            for _try in range(40):
                cr, ca = prep(negs[rng.randrange(len(negs))],
                              int(np.exp(rng.uniform(np.log(40), np.log(120)))), rng)
                ch, cw = ca.shape
                if ch >= H or cw >= W:
                    continue
                y, x = rng.randrange(0, H - ch), rng.randrange(0, W - cw)
                if (lab[y:y+ch, x:x+cw][ca > 127] != BG).any():
                    continue
                af = (ca.astype(np.float32) / 255.0)[..., None]
                roi = frame[y:y+ch, x:x+cw].astype(np.float32)
                frame[y:y+ch, x:x+cw] = (cr * af + roi * (1 - af)).astype(np.uint8)
                m_all[y:y+ch, x:x+cw] |= ca > 127
                placed += 1
                break
        if placed == 0 or m_all.sum() < 50:
            continue
        for k, m in models.items():
            for abl in ["none", 0, 1, 2, 3]:
                p = logits_with_ablation(m, frame, a.device,
                                         None if abl == "none" else abl)
                scores[k][abl].append(float((p[m_all] == CONN).mean()))
        if n % 20 == 0:
            print(f"  {n}/{len(bases)}", flush=True)

    print("\n=== connector firing on pasted REAL-photo pixels, by ablated stage ===")
    print(f"{'model':10s} {'intact':>8s} {'-stage1':>9s} {'-stage2':>9s} "
          f"{'-stage3':>9s} {'-stage4':>9s}")
    out = {}
    for k in models:
        row = [np.mean(scores[k][v]) if scores[k][v] else float('nan')
               for v in ["none", 0, 1, 2, 3]]
        out[k] = {n2: float(v) for n2, v in zip(["intact", "s1", "s2", "s3", "s4"], row)}
        print(f"{k:10s} " + " ".join(f"{v:>8.1%}" for v in row))
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(f"\n-> {a.out}/summary.json")


if __name__ == "__main__":
    main()
