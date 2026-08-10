#!/usr/bin/env python
"""
BOUNDED diagnostic: characterise the wire pixels the B2 model MISSES (false
negatives) vs the wire pixels it DETECTS (true positives) on the partner-team
real-world validation set (data/real_wires_valset, 62 frames).

We are RECALL-LIMITED (pooled IoU=0.612, P=0.70, R=0.82) and want R~0.90.
This script answers: what distinguishes FN (GT=wire, pred=bg) from TP (GT=wire,
pred=wire) wire pixels across 5 properties:
  1. wire thickness   (cv2.distanceTransform L2 of GT mask at the pixel)
  2. local contrast   (|mean gray of GT-wire pixels - mean gray of GT-bg ring| in 11x11)
  3. brightness       (luma 0.299R+0.587G+0.114B)
  4. edge sharpness   (Sobel gradient magnitude, mean over 3x3)
  5. per-camera FN rate
  (+ optional wire hue, HSV H)

CRITICAL: the prediction pipeline is identical to src/eval_real_wires_valset.py
-- we import its exact functions (load_model_auto, preprocess_size, predict_size,
crop_mask_to_aspect, load_gt_binary) so the reproduced pooled IoU/P/R must match
0.612/0.70/0.82. A SANITY GATE enforces this before any FN analysis.

All measurements are taken on the SAME geometry the evaluator scores on:
the center-cropped, GT-resolution space (pred upsampled NEAREST to GT crop; the
preprocessed-then-resized RGB is mapped to that crop the same way, so a luma /
gradient / DT value sits exactly at the pixel the IoU counts).
Outputs: results/realism_campaign/b2_recall_fn/fn_analysis.json + 4 worst-recall
montages.
"""
import os
import sys
import glob
import json

import numpy as np
import cv2
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
PROJECT_ROOT = os.path.dirname(SRC)

# Import the EXACT evaluator logic. Importing eval_real_wires_valset pulls in
# infer_video_rgb_only, which sets HF_HOME/TRANSFORMERS_CACHE -> data/hf_cache
# via os.environ.setdefault on import, so we must NOT pre-set HF_HOME elsewhere.
from eval_real_wires_valset import (  # noqa: E402
    load_model_auto,
    preprocess_size,
    predict_size,
    crop_mask_to_aspect,
    load_gt_binary,
)
from train_rgb_only_sota import IMAGE_H, IMAGE_W  # noqa: E402

DATA = os.path.join(PROJECT_ROOT, "data", "real_wires_valset")
CKPT = os.path.join(
    PROJECT_ROOT,
    "results/segformer_b5_rgb_phase18_aug/full_20260612_0033/best_model.pth",
)
OUT = os.path.join(PROJECT_ROOT, "results", "realism_campaign", "b2_recall_fn")
INFER_HW = (IMAGE_H, IMAGE_W)  # 480x640
WIN = 11  # local-contrast window (odd)


def list_basenames():
    """Same listing+sort key as the evaluator."""
    bns = sorted(
        (os.path.splitext(os.path.basename(p))[0]
         for p in glob.glob(os.path.join(DATA, "imgs", "*"))),
        key=lambda b: (b.split("_")[0], int(b.split("_")[1])),
    )
    return bns


def luma_from_bgr(bgr):
    """0.299R + 0.587G + 0.114B  (float, full res of the given image)."""
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def local_contrast_map(gray, gt_bool, win=WIN, min_bg=8):
    """For every pixel: |mean(gray over GT-wire pixels in win) -
    mean(gray over GT-bg pixels in win)|. Computed densely with box filters.
    Pixels whose window has < min_bg background pixels are NaN (skip).
    `gray` float32, `gt_bool` bool (GT-wire). Returns float32 same shape (NaN where invalid)."""
    g = gray.astype(np.float32)
    w = gt_bool.astype(np.float32)        # 1 at wire
    bg = 1.0 - w                          # 1 at bg
    ksz = (win, win)
    # box filter (normalize=False) => sum over window
    wire_cnt = cv2.boxFilter(w, ddepth=-1, ksize=ksz, normalize=False, borderType=cv2.BORDER_REFLECT)
    bg_cnt = cv2.boxFilter(bg, ddepth=-1, ksize=ksz, normalize=False, borderType=cv2.BORDER_REFLECT)
    wire_sum = cv2.boxFilter(g * w, ddepth=-1, ksize=ksz, normalize=False, borderType=cv2.BORDER_REFLECT)
    bg_sum = cv2.boxFilter(g * bg, ddepth=-1, ksize=ksz, normalize=False, borderType=cv2.BORDER_REFLECT)
    with np.errstate(invalid="ignore", divide="ignore"):
        wire_mean = wire_sum / wire_cnt
        bg_mean = bg_sum / bg_cnt
        contrast = np.abs(wire_mean - bg_mean)
    invalid = (bg_cnt < min_bg) | (wire_cnt < 1)
    contrast[invalid] = np.nan
    return contrast


def sobel_mag(gray):
    """Sobel gradient magnitude, then mean over 3x3 (local sharpness)."""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = cv2.blur(mag, (3, 3), borderType=cv2.BORDER_REFLECT)
    return mag


def stat_block(vals):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return dict(n=0, median=None, q25=None, q75=None, iqr=None, mean=None)
    q25, med, q75 = np.percentile(vals, [25, 50, 75])
    return dict(n=int(vals.size), median=float(med), q25=float(q25),
                q75=float(q75), iqr=float(q75 - q25), mean=float(vals.mean()))


def cliffs_delta(a, b, cap=200000):
    """Cliff's delta of distribution a vs b (P(a>b)-P(a<b)). Subsampled for speed.
    Sign convention: positive => a tends to be LARGER than b."""
    a = np.asarray(a, dtype=np.float64); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=np.float64); b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return None
    rng = np.random.default_rng(0)
    if a.size > cap:
        a = rng.choice(a, cap, replace=False)
    if b.size > cap:
        b = rng.choice(b, cap, replace=False)
    a_sorted = np.sort(b)  # we will rank elements of a within b
    # for each element of a, count b-values strictly less / strictly greater
    less = np.searchsorted(a_sorted, a, side="left")          # # b < a
    greater = b.size - np.searchsorted(a_sorted, a, side="right")  # # b > a
    delta = (less.sum() - greater.sum()) / (a.size * b.size)
    return float(delta)


def main():
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device     : {device}")
    print(f"checkpoint : {CKPT}")
    print(f"data-dir   : {DATA}")
    print(f"infer-size : {INFER_HW[0]}x{INFER_HW[1]}  wire-rule: fg")
    model = load_model_auto(CKPT, device)

    bns = list_basenames()
    print(f"images     : {len(bns)}")

    # ---- pass 1: build pred + GT masks per frame, collect pixel-level props ----
    TP = FP = FN = 0
    per_cam = {}   # cam -> [tp,fp,fn,gt]
    per_frame = []  # for montage selection: (bn, recall, gt_px)

    # pooled property collectors (use lists of arrays, concat at end)
    coll = {k: {"fn": [], "tp": []} for k in
            ("dt", "contrast", "luma", "sharp", "hue")}

    for i, bn in enumerate(bns):
        rgb_path = os.path.join(DATA, "imgs", f"{bn}.jpg")
        mask_path = os.path.join(DATA, "masks", f"{bn}.jpg")
        cam = bn.split("_")[0]

        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError(f"cannot read {rgb_path}")

        # --- EXACT evaluator pipeline ---
        pre = preprocess_size(rgb_bgr, INFER_HW)              # 480x640 BGR uint8
        pred_small = predict_size(model, pre, device, INFER_HW)  # 480x640 argmax
        pred_small = (pred_small >= 1).astype(np.uint8)        # wire-rule "fg"

        gt_bin = load_gt_binary(mask_path)                     # native res {0,1}
        gt_crop = crop_mask_to_aspect(gt_bin, IMAGE_W / IMAGE_H)  # cropped native res
        Hc, Wc = gt_crop.shape
        pred_full = cv2.resize(pred_small, (Wc, Hc), interpolation=cv2.INTER_NEAREST)

        p = pred_full.astype(bool)
        g = gt_crop.astype(bool)
        tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
        TP += tp; FP += fp; FN += fn
        c = per_cam.setdefault(cam, [0, 0, 0, 0]); c[0] += tp; c[1] += fp; c[2] += fn; c[3] += int(g.sum())
        recall_f = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        per_frame.append(dict(bn=bn, cam=cam, recall=recall_f, gt_px=int(g.sum()),
                              tp=tp, fp=fp, fn=fn, iou=(tp / (tp + fp + fn) if (tp + fp + fn) else float("nan"))))

        # --- properties: computed in the SAME GT-crop space the IoU is scored in ---
        # The preprocessed RGB is 480x640; map it to the GT crop the SAME way the
        # pred is mapped (the GT crop is the canonical scoring grid). Use INTER_AREA
        # for the (down/identity) image resize so luma/gradient are sensible; the GT
        # crop is usually a higher res than 640x480 so this is an UPSAMPLE -> LINEAR.
        interp = cv2.INTER_AREA if (pre.shape[1] >= Wc and pre.shape[0] >= Hc) else cv2.INTER_LINEAR
        rgb_crop = cv2.resize(pre, (Wc, Hc), interpolation=interp)   # BGR at GT-crop res
        gray = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        luma = luma_from_bgr(rgb_crop)
        hsv = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0].astype(np.float32)  # OpenCV H in [0,179]

        dt = cv2.distanceTransform(gt_crop.astype(np.uint8), cv2.DIST_L2, 5)  # DT inside wire
        contrast = local_contrast_map(gray, g, win=WIN, min_bg=8)
        sharp = sobel_mag(gray)

        fn_mask = (~p) & g
        tp_mask = p & g
        for key, arr in (("dt", dt), ("contrast", contrast),
                         ("luma", luma), ("sharp", sharp), ("hue", hue)):
            coll[key]["fn"].append(arr[fn_mask])
            coll[key]["tp"].append(arr[tp_mask])

        print(f"  [{i+1:2d}/{len(bns)}] {bn:8s} IoU={per_frame[-1]['iou']:.3f} "
              f"R={recall_f:.3f} gt={int(g.sum()):7d} fn={fn:7d}")

    # ---- SANITY GATE ----
    pooled_iou = TP / (TP + FP + FN) if (TP + FP + FN) else float("nan")
    precision = TP / (TP + FP) if (TP + FP) else float("nan")
    recall = TP / (TP + FN) if (TP + FN) else float("nan")
    print("\n==================== SANITY GATE ====================")
    print(f"pooled IoU = {pooled_iou:.4f}  (target 0.612)")
    print(f"precision  = {precision:.4f}  (target 0.70)")
    print(f"recall     = {recall:.4f}  (target 0.82)")
    gate_ok = (abs(pooled_iou - 0.612) <= 0.01 and abs(precision - 0.70) <= 0.01
               and abs(recall - 0.82) <= 0.01)
    print(f"GATE       : {'PASS' if gate_ok else 'FAIL'}")

    sanity = dict(pooled_iou=pooled_iou, precision=precision, recall=recall,
                  tp=TP, fp=FP, fn=FN,
                  target=dict(pooled_iou=0.612, precision=0.70, recall=0.82),
                  gate_pass=bool(gate_ok))

    if not gate_ok:
        # still dump what we have so the mismatch is inspectable, then STOP.
        with open(os.path.join(OUT, "fn_analysis.json"), "w") as f:
            json.dump(dict(sanity_gate=sanity, ABORTED="sanity gate failed"), f, indent=2)
        print("\nSANITY GATE FAILED -> aborting FN analysis (preprocessing drifted).")
        return

    # ---- pixel-level property stats: FN vs TP ----
    props = {}
    prop_meta = dict(
        dt=("wire thickness (DT, half-width px)", "lower FN => thin wires / edges"),
        contrast=("local contrast (|wire-bg gray|, 11x11)", "lower FN => low-contrast wires"),
        luma=("brightness (luma 0-255)", "lower FN => dark wires"),
        sharp=("edge sharpness (Sobel mag)", "lower FN => soft/blurry wires"),
        hue=("wire hue (HSV H 0-179)", "shift => colour-specific misses"),
    )
    for key, (label, interp) in prop_meta.items():
        fn_vals = np.concatenate(coll[key]["fn"]) if coll[key]["fn"] else np.array([])
        tp_vals = np.concatenate(coll[key]["tp"]) if coll[key]["tp"] else np.array([])
        fn_s = stat_block(fn_vals)
        tp_s = stat_block(tp_vals)
        delta = cliffs_delta(fn_vals, tp_vals)  # +ve => FN larger than TP
        ratio = (fn_s["median"] / tp_s["median"]
                 if (fn_s["median"] is not None and tp_s["median"] not in (None, 0)) else None)
        props[key] = dict(label=label, hint=interp,
                          fn=fn_s, tp=tp_s,
                          fn_over_tp_median_ratio=ratio,
                          cliffs_delta_fn_vs_tp=delta)
        print(f"\n[{key}] {label}")
        print(f"   FN: median={fn_s['median']} IQR=[{fn_s['q25']},{fn_s['q75']}] n={fn_s['n']}")
        print(f"   TP: median={tp_s['median']} IQR=[{tp_s['q25']},{tp_s['q75']}] n={tp_s['n']}")
        print(f"   FN/TP median ratio={ratio}  Cliff's delta(FN vs TP)={delta}")

    # ---- per-camera FN rate ----
    cam_tbl = {}
    for cam, (tp, fp, fn, gtpx) in sorted(per_cam.items()):
        fn_rate = fn / gtpx if gtpx else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        cam_tbl[cam] = dict(fn=fn, gt_wire_px=gtpx, fn_rate=fn_rate, recall=rec,
                            tp=tp, fp=fp,
                            n_frames=sum(1 for fr in per_frame if fr["cam"] == cam))

    print("\n================ PER-CAMERA FN RATE ================")
    print(f"{'cam':4s} {'n':>3s} {'FN_px':>9s} {'GT_wire_px':>11s} {'FN_rate':>8s} {'recall':>7s}")
    for cam, d in cam_tbl.items():
        print(f"{cam:4s} {d['n_frames']:3d} {d['fn']:9d} {d['gt_wire_px']:11d} "
              f"{d['fn_rate']:8.4f} {d['recall']:7.4f}")

    # ---- ranked verdict by |Cliff's delta| (effect size separating FN vs TP) ----
    ranked = sorted(
        [(k, props[k]["cliffs_delta_fn_vs_tp"]) for k in ("dt", "contrast", "luma", "sharp")
         if props[k]["cliffs_delta_fn_vs_tp"] is not None],
        key=lambda kv: abs(kv[1]), reverse=True)
    print("\n================ RANKED VERDICT (|Cliff's delta|) ================")
    for rank, (k, d) in enumerate(ranked, 1):
        direction = "FN LOWER than TP" if d < 0 else "FN HIGHER than TP"
        print(f"  {rank}. {k:9s} |delta|={abs(d):.3f}  ({direction})  -- {props[k]['label']}")

    result = dict(
        checkpoint=CKPT, data_dir=DATA, n_frames=len(bns),
        infer_size_hw=list(INFER_HW), wire_rule="fg", contrast_window=WIN,
        note=("Properties measured in the GT-crop scoring space (the same grid the "
              "evaluator computes IoU on; pred upsampled NEAREST, preprocessed RGB "
              "mapped to GT-crop res). Cliff's delta sign: +ve=FN larger than TP."),
        sanity_gate=sanity,
        properties=props,
        per_camera_fn_rate=cam_tbl,
        ranked_by_abs_cliffs_delta=[dict(rank=i + 1, property=k, cliffs_delta=d)
                                    for i, (k, d) in enumerate(ranked)],
    )
    with open(os.path.join(OUT, "fn_analysis.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {os.path.join(OUT, 'fn_analysis.json')}")

    # ---- montages of the 4 worst-recall frames (weighted: lowest recall, tie-break gt_px) ----
    # Only consider frames with a meaningful amount of GT wire (>=500 px) so a
    # near-empty frame doesn't masquerade as "worst recall".
    cand = [fr for fr in per_frame if fr["gt_px"] >= 500]
    worst = sorted(cand, key=lambda fr: (fr["recall"], -fr["gt_px"]))[:4]
    montage_paths = []
    for fr in worst:
        bn = fr["bn"]
        rgb_bgr = cv2.imread(os.path.join(DATA, "imgs", f"{bn}.jpg"), cv2.IMREAD_COLOR)
        pre = preprocess_size(rgb_bgr, INFER_HW)
        pred_small = (predict_size(model, pre, device, INFER_HW) >= 1).astype(np.uint8)
        gt_bin = load_gt_binary(os.path.join(DATA, "masks", f"{bn}.jpg"))
        gt_crop = crop_mask_to_aspect(gt_bin, IMAGE_W / IMAGE_H)
        Hc, Wc = gt_crop.shape
        pred_full = cv2.resize(pred_small, (Wc, Hc), interpolation=cv2.INTER_NEAREST)
        interp = cv2.INTER_AREA if (pre.shape[1] >= Wc and pre.shape[0] >= Hc) else cv2.INTER_LINEAR
        rgb_crop = cv2.resize(pre, (Wc, Hc), interpolation=interp)

        p = pred_full.astype(bool); g = gt_crop.astype(bool)
        panels = []

        def lab(img, text):
            img = img.copy()
            cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(img, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            return img

        # 1 RGB
        panels.append(lab(rgb_crop, f"{bn} RGB"))
        # 2 GT overlay (green)
        ov = rgb_crop.copy(); lay = np.zeros_like(ov); lay[g] = (0, 255, 0)
        ov = (ov * 0.5 + lay * 0.5).astype(np.uint8)
        # restore non-mask pixels to full RGB
        ov[~g] = rgb_crop[~g]
        panels.append(lab(ov, "GT wire (green)"))
        # 3 pred overlay (green)
        ov2 = rgb_crop.copy(); lay2 = np.zeros_like(ov2); lay2[p] = (0, 255, 0)
        ov2 = (ov2 * 0.5 + lay2 * 0.5).astype(np.uint8); ov2[~p] = rgb_crop[~p]
        panels.append(lab(ov2, f"Pred wire R={fr['recall']:.2f}"))
        # 4 FN highlighted (blue=FN, green=TP)
        err = rgb_crop.copy(); el = np.zeros_like(err)
        el[p & g] = (0, 255, 0); el[(~p) & g] = (255, 0, 0)
        sel = (p | g)
        err[sel] = (rgb_crop[sel] * 0.4 + el[sel] * 0.6).astype(np.uint8)
        panels.append(lab(err, "FN=blue TP=green"))

        sep = np.full((Hc, 4, 3), 200, np.uint8)
        sheet = panels[0]
        for pnl in panels[1:]:
            sheet = np.hstack([sheet, sep, pnl])
        mp = os.path.join(OUT, f"worst_recall_{bn}_R{fr['recall']:.2f}.png")
        cv2.imwrite(mp, sheet)
        montage_paths.append(mp)
        print(f"  montage: {mp}")

    result["worst_recall_montages"] = montage_paths
    with open(os.path.join(OUT, "fn_analysis.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nDONE.")


if __name__ == "__main__":
    main()
