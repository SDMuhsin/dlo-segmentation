#!/usr/bin/env python
"""FAIR real-wire benchmark (neither QN nor P3W lineage trained on it) — see
results/qn_backbone_3way/fairbench/README.md. Lists built by src/fairbench_overlap_gate.py.

Per dataset list + pooled over the FAIR sets: pixel-pooled IoU/P/R of predicted WIRE (class 1;
= DLO for 2-class) and of WIRE u CONNECTOR (class >= 1) vs GT foreground, plus the fraction of GT
fg px predicted connector and the per-image mean IoU. `p3wseen_*` sets (in the P3W lineage's
realtextured co-train pool) are reported separately and never pooled into the fair score.

Preprocessing: portrait images (h > w) are rotated 90 deg CW (mBEST is 672x896 = exact 3:4, so it
becomes an exact 4:3 frame, nothing cropped); then infer_video_rgb_only.preprocess
(centre-crop 4:3 -> 640x480, the video/suite protocol; MC-test frames are native 640x480 so this is
a no-op for them). The GT mask gets the identical crop + INTER_NEAREST resize. Inference as in
src/eval_qn3way_suite.py (fp32, argmax, ImageNet norm, batch 8, one GPU via --gpu, default 1).

  ./env/bin/python src/eval_fairbench.py qn=... p3w=... [--overlay qn,p3w,a0_ep5,a1_ep10]
"""
import argparse
import json
import os
import sys

import eval_qn3way_suite as S  # sets CUDA_VISIBLE_DEVICES from --gpu before torch import
import cv2
import numpy as np
import torch
from infer_video_rgb_only import center_crop_to_aspect, preprocess, TARGET_AR

ROOT = S.ROOT
FB = os.path.join(ROOT, "results/qn_backbone_3way/fairbench")
FAIR = ["mbest_sbhc", "rmdlo_fs2"]
SEEN = ["p3wseen_bwh", "p3wseen_labd_real", "p3wseen_m4nh_cables", "p3wseen_putvision"]
N_OVERLAY = 4


def read_pair(ip, mp):
    im = cv2.imread(os.path.join(ROOT, ip), cv2.IMREAD_COLOR)
    if mp.endswith(".npy"):
        gt = np.load(os.path.join(ROOT, mp)) > 0
    else:
        m = cv2.imread(os.path.join(ROOT, mp), cv2.IMREAD_GRAYSCALE)
        gt = m >= 128 if mp.endswith(".jpg") else m > 0
    if gt.shape != im.shape[:2]:
        gt = cv2.resize(gt.astype(np.uint8), (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    if im.shape[0] > im.shape[1]:
        im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        gt = cv2.rotate(gt.astype(np.uint8), cv2.ROTATE_90_CLOCKWISE) > 0
    x = preprocess(im)
    g = center_crop_to_aspect(gt.astype(np.uint8)[:, :, None], TARGET_AR)[:, :, 0]
    g = cv2.resize(g, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return x, g


def load_lists():
    D = {}
    for n in FAIR + SEEN:
        p = os.path.join(FB, "lists", n + ".txt")
        D[n] = [l.rstrip("\n").split("\t") for l in open(p) if l.strip()]
    return D


def overlay_idx(n):
    return np.linspace(0, n - 1, N_OVERLAY, dtype=int).tolist()


def eval_model(m, nc, D):
    R, ov = {}, {}
    pooled = {k: dict(tw=0, fw=0, nw=0, tf=0, ff=0, nf=0, gt=0, gc=0) for k in ("fair", "p3wseen")}
    for dn, pairs in D.items():
        c = dict(tw=0, fw=0, nw=0, tf=0, ff=0, nf=0, gt=0, gc=0)
        per_img = []
        oi = set(overlay_idx(len(pairs)))
        for i in range(0, len(pairs), S.BS):
            chunk = [read_pair(ip, mp) for ip, mp in pairs[i:i + S.BS]]
            p = S.predict_batch(m, [x for x, _ in chunk])
            for j, (x, g) in enumerate(chunk):
                pw, pf = p[j] == 1, p[j] >= 1
                d = dict(tw=int((pw & g).sum()), fw=int((pw & ~g).sum()), nw=int((~pw & g).sum()),
                         tf=int((pf & g).sum()), ff=int((pf & ~g).sum()), nf=int((~pf & g).sum()),
                         gt=int(g.sum()), gc=int(((p[j] == 2) & g).sum()))
                for k in c:
                    c[k] += d[k]
                per_img.append(d["tf"] / max(d["tf"] + d["ff"] + d["nf"], 1))
                if i + j in oi:
                    ov[f"{dn}__{i + j}"] = p[j]
        R[dn] = {"n_images": len(pairs), "wire": S.prf(c["tw"], c["fw"], c["nw"]),
                 "wire_or_conn": S.prf(c["tf"], c["ff"], c["nf"]),
                 "gt_pred_conn_frac": c["gc"] / max(c["gt"], 1),
                 "mean_img_iou_wire_or_conn": float(np.mean(per_img))}
        grp = "fair" if dn in FAIR else "p3wseen"
        for k in c:
            pooled[grp][k] += c[k]
    for grp, c in pooled.items():
        names = FAIR if grp == "fair" else SEEN
        R[f"POOLED_{grp}"] = {"n_images": sum(len(D[n]) for n in names),
                              "wire": S.prf(c["tw"], c["fw"], c["nw"]),
                              "wire_or_conn": S.prf(c["tf"], c["ff"], c["nf"]),
                              "gt_pred_conn_frac": c["gc"] / max(c["gt"], 1),
                              "macro_iou_wire_or_conn": float(np.mean([R[n]["wire_or_conn"]["iou"] for n in names])),
                              "macro_iou_wire": float(np.mean([R[n]["wire"]["iou"] for n in names]))}
    return R, ov


def overlays(names, D):
    od = os.path.join(FB, "overlays")
    os.makedirs(od, exist_ok=True)
    preds = {n: np.load(os.path.join(FB, n, "overlay_preds.npz")) for n in names}
    for dn, pairs in D.items():
        for i in overlay_idx(len(pairs)):
            x, g = read_pair(*pairs[i])
            gv = x.copy(); gv[g] = (gv[g] * 0.4 + np.array((0, 200, 255)) * 0.6).astype(np.uint8)
            panels = [S.bar(x, os.path.basename(pairs[i][0])), S.bar(gv, "GT fg (orange)")]
            for n in names:
                panels.append(S.bar(S.overlay(x, preds[n][f"{dn}__{i}"]), f"{n} (wire=green conn=magenta)"))
            top = np.hstack(panels[:3]); bot = np.hstack(panels[3:] + [np.zeros_like(panels[0])] * (3 - len(panels[3:])))
            cv2.imwrite(os.path.join(od, f"{dn}_{i:03d}.jpg"), np.vstack([top, bot]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def fmt_row(name, R, keys):
    cells = []
    for k in keys:
        r = R[k]
        cells.append(f"{r['wire']['iou']:.3f} / **{r['wire_or_conn']['iou']:.3f}** "
                     f"({r['wire_or_conn']['precision']:.2f}/{r['wire_or_conn']['recall']:.2f})")
    return f"| {name} | " + " | ".join(cells) + " |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", metavar="NAME=CKPT")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--overlay", default="", help="comma list of model names for overlay PNGs")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    D = load_lists()
    allR = {}
    for spec in a.models:
        name, _, ckpt = spec.partition("=")
        md = os.path.join(FB, name)
        mj = os.path.join(md, "metrics.json")
        os.makedirs(md, exist_ok=True)
        if os.path.exists(mj) and not a.force:
            allR[name] = json.load(open(mj)); print(f"[skip] {name} cached"); continue
        m, nc = S.load(ckpt)
        print(f"[{name}] {ckpt} nc={nc}", flush=True)
        R, ov = eval_model(m, nc, D)
        R["_meta"] = {"ckpt": ckpt, "num_classes": nc}
        del m; torch.cuda.empty_cache()
        json.dump(R, open(mj, "w"), indent=2)
        np.savez_compressed(os.path.join(md, "overlay_preds.npz"), **ov)
        allR[name] = R
        print(f"  fair pooled w+c IoU {R['POOLED_fair']['wire_or_conn']['iou']:.3f}", flush=True)
    keys = ["POOLED_fair"] + FAIR + ["POOLED_p3wseen"] + SEEN
    hdr = "| model | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys)
    lines = [hdr] + [fmt_row(n, R, keys) for n, R in allR.items()]
    table = "\n".join(lines)
    open(os.path.join(FB, "table.md"), "w").write(
        "cell = IoU(wire) / **IoU(wire u conn)** (P/R of wire u conn), pixel-pooled\n\n" + table + "\n")
    print(table)
    if a.overlay:
        overlays([n for n in a.overlay.split(",") if n], D)


if __name__ == "__main__":
    main()
