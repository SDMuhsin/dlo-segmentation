"""Diagnose what caps IoU(connector) at ~0.76 on the 3-way synthetic val set.

Answers four questions with numbers, not narrative:

  Q1  Confusion structure   -- where do connector FN/FP pixels go (wire vs bg)?
  Q2  Boundary geometry     -- how much of the connector error sits in a k-px
                               band around the GT connector boundary?  A small
                               class whose error is boundary-dominated is capped
                               by label/decode resolution, not by semantics.
  Q3  Per-blob detection    -- are whole connectors missed, or are detected
                               connectors just eroded/dilated at the rim?
  Q4  Decode bottleneck     -- the model emits logits at H/4 x W/4 and bilinear
                               upsamples 4x.  Re-running the SAME weights at 2x
                               input resolution (logits then land at H/2 x W/2)
                               isolates how much IoU the 4x upsample is costing.

Usage:
  python src/diag_3way_connector_ceiling.py \
      --ckpt results/realism_campaign/p_3way_decheat/seg_b5_3way_connscale3/best_model.pth \
      --data-dir data/dformer_dataset_3way_connscale3 \
      --out results/realism_campaign/p_3way_diag
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_rgb_only_sota import (  # noqa: E402
    IMAGE_H, IMAGE_W, RGB_MEAN, RGB_STD,
)
from gen_rgb_only_sota_gifs import load_model  # noqa: E402  (canonical loader)

BG, WIRE, CON = 0, 1, 2


@torch.no_grad()
def infer(model, rgb_bgr_batch, device, scale=1):
    """rgb_bgr_batch: (B,H,W,3) uint8 BGR.  Returns argmax (B,H,W) at IMAGE_H/W."""
    rgb = rgb_bgr_batch[:, :, :, ::-1].copy()                    # BGR -> RGB
    t = torch.from_numpy(rgb.transpose(0, 3, 1, 2)).to(device)
    t = t.float() / 255.0
    t = (t - RGB_MEAN.to(device)) / RGB_STD.to(device)
    if scale != 1:
        t = F.interpolate(t, scale_factor=scale, mode="bilinear",
                          align_corners=False)
    with torch.autocast("cuda", dtype=torch.float16):
        out = model.model(pixel_values=t)
    # Always score at native label resolution.
    logits = F.interpolate(out.logits.float(), size=(IMAGE_H, IMAGE_W),
                           mode="bilinear", align_corners=False)
    return logits.argmax(1).cpu().numpy().astype(np.uint8)


def boundary_band(mask_bool, k):
    """Pixels within k px of the mask's boundary (inside OR outside)."""
    m = mask_bool.astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    dil = cv2.dilate(m, ker)
    ero = cv2.erode(m, ker)
    return (dil > 0) & (ero == 0)


def iou_from_cm(cm, c):
    tp = cm[c, c]
    fn = cm[c, :].sum() - tp
    fp = cm[:, c].sum() - tp
    return tp / max(tp + fp + fn, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all val images")
    ap.add_argument("--scales", default="1,2",
                    help="input scale factors to evaluate (Q4)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    model = load_model(args.ckpt, device)

    cache = os.path.join(args.data_dir, "cache")
    rgb = np.load(os.path.join(cache, "val_rgb.npy"), mmap_mode="r")
    lab = np.load(os.path.join(cache, "val_label.npy"), mmap_mode="r")
    n = len(rgb) if args.limit == 0 else min(args.limit, len(rgb))
    print(f"[diag] val images: {n}  shape={rgb.shape[1:]}")

    scales = [int(s) for s in args.scales.split(",")]
    report = {"n_images": n, "ckpt": args.ckpt}

    # ---- Q4 first: it needs a full pass per scale, and it reuses the same CM. --
    for scale in scales:
        cm = np.zeros((3, 3), dtype=np.int64)
        # boundary-band accumulators (only for scale 1, the shipped config)
        bands = {k: {"err_in": 0, "err_out": 0, "gt_in": 0, "gt_out": 0}
                 for k in (1, 2, 3, 4)}
        blob_stats = []
        pix_gt_con = 0

        for i0 in range(0, n, args.batch):
            i1 = min(i0 + args.batch, n)
            rb = np.asarray(rgb[i0:i1])
            lb = np.asarray(lab[i0:i1]).astype(np.int64)
            pr = infer(model, rb, device, scale=scale)

            idx = lb.ravel() * 3 + pr.ravel().astype(np.int64)
            cm += np.bincount(idx, minlength=9).reshape(3, 3)

            if scale != 1:
                continue

            for b in range(i1 - i0):
                g, p = lb[b], pr[b]
                gcon = (g == CON)
                if not gcon.any():
                    continue
                pix_gt_con += int(gcon.sum())
                err = (g != p) & (gcon | (p == CON))   # connector-relevant errors
                for k in bands:
                    band = boundary_band(gcon, k)
                    bands[k]["err_in"] += int((err & band).sum())
                    bands[k]["err_out"] += int((err & ~band).sum())
                    bands[k]["gt_in"] += int((gcon & band).sum())
                    bands[k]["gt_out"] += int((gcon & ~band).sum())

                # Q3: per connected component of GT connector
                ncc, cc = cv2.connectedComponents(gcon.astype(np.uint8))
                for c in range(1, ncc):
                    m = cc == c
                    area = int(m.sum())
                    hit = int((m & (p == CON)).sum())
                    as_wire = int((m & (p == WIRE)).sum())
                    as_bg = int((m & (p == BG)).sum())
                    blob_stats.append((area, hit, as_wire, as_bg))

            if i0 % (args.batch * 25) == 0:
                print(f"  scale={scale} {i1}/{n}", flush=True)

        res = {
            "confusion_gt_rows_pred_cols": cm.tolist(),
            "iou_bg": iou_from_cm(cm, BG),
            "iou_wire": iou_from_cm(cm, WIRE),
            "iou_con": iou_from_cm(cm, CON),
        }
        res["miou"] = (res["iou_bg"] + res["iou_wire"] + res["iou_con"]) / 3.0
        tp = int(cm[CON, CON])
        res["con_precision"] = tp / max(int(cm[:, CON].sum()), 1)
        res["con_recall"] = tp / max(int(cm[CON, :].sum()), 1)
        res["con_fn_to_wire"] = int(cm[CON, WIRE])
        res["con_fn_to_bg"] = int(cm[CON, BG])
        res["con_fp_from_wire"] = int(cm[WIRE, CON])
        res["con_fp_from_bg"] = int(cm[BG, CON])
        report[f"scale_{scale}"] = res
        print(f"[diag] scale={scale}: IoU(con)={res['iou_con']:.4f} "
              f"IoU(wire)={res['iou_wire']:.4f} mIoU={res['miou']:.4f}")

        if scale == 1:
            for k, v in bands.items():
                tot = v["err_in"] + v["err_out"]
                v["frac_err_in_band"] = v["err_in"] / max(tot, 1)
                v["frac_gt_in_band"] = v["gt_in"] / max(v["gt_in"] + v["gt_out"], 1)
            report["boundary_bands"] = bands

            bs = np.array(blob_stats, dtype=np.int64) if blob_stats else np.zeros((0, 4), np.int64)
            if len(bs):
                area, hit, as_wire, as_bg = bs[:, 0], bs[:, 1], bs[:, 2], bs[:, 3]
                rec = hit / np.maximum(area, 1)
                report["blobs"] = {
                    "n_blobs": int(len(bs)),
                    "area_percentiles": {
                        str(q): float(np.percentile(area, q))
                        for q in (5, 25, 50, 75, 95)},
                    "n_fully_missed(rec<0.05)": int((rec < 0.05).sum()),
                    "n_mostly_missed(rec<0.50)": int((rec < 0.50).sum()),
                    "n_well_detected(rec>=0.80)": int((rec >= 0.80).sum()),
                    "mean_blob_recall": float(rec.mean()),
                    "missed_mass_to_wire": int(as_wire.sum()),
                    "missed_mass_to_bg": int(as_bg.sum()),
                    "recall_by_area_quartile": {},
                }
                qs = np.percentile(area, [25, 50, 75])
                for name, sel in [
                    ("q1_smallest", area <= qs[0]),
                    ("q2", (area > qs[0]) & (area <= qs[1])),
                    ("q3", (area > qs[1]) & (area <= qs[2])),
                    ("q4_largest", area > qs[2]),
                ]:
                    if sel.sum():
                        report["blobs"]["recall_by_area_quartile"][name] = {
                            "n": int(sel.sum()),
                            "mean_area": float(area[sel].mean()),
                            "mean_recall": float(rec[sel].mean()),
                        }
            report["gt_connector_pixels"] = pix_gt_con

    out = os.path.join(args.out, "connector_ceiling_diag.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[diag] wrote {out}")

    # ---- human-readable summary --------------------------------------------
    s1 = report["scale_1"]
    print("\n===== SUMMARY =====")
    print(f"IoU(con)={s1['iou_con']:.4f}  P={s1['con_precision']:.3f}  "
          f"R={s1['con_recall']:.3f}")
    print(f"  FN -> wire: {s1['con_fn_to_wire']:,}   FN -> bg: {s1['con_fn_to_bg']:,}")
    print(f"  FP <- wire: {s1['con_fp_from_wire']:,}   FP <- bg: {s1['con_fp_from_bg']:,}")
    for k, v in report.get("boundary_bands", {}).items():
        print(f"  band k={k}: {v['frac_err_in_band']*100:.1f}% of connector error "
              f"is within {k}px of the GT boundary "
              f"({v['frac_gt_in_band']*100:.1f}% of GT connector px live there)")
    if "blobs" in report:
        b = report["blobs"]
        print(f"  blobs: n={b['n_blobs']}  median area={b['area_percentiles']['50']:.0f}px  "
              f"fully-missed={b['n_fully_missed(rec<0.05)']}  "
              f"well-detected={b['n_well_detected(rec>=0.80)']}")
        for kk, vv in b["recall_by_area_quartile"].items():
            print(f"    {kk}: n={vv['n']} area~{vv['mean_area']:.0f}px "
                  f"recall={vv['mean_recall']:.3f}")
    for scale in scales:
        r = report[f"scale_{scale}"]
        print(f"  scale={scale}: IoU(con)={r['iou_con']:.4f} "
              f"IoU(wire)={r['iou_wire']:.4f} mIoU={r['miou']:.4f}")


if __name__ == "__main__":
    main()
