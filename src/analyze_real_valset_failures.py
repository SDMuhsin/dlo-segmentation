#!/usr/bin/env python
"""
Deep failure forensics for the Phase-15 teacher on the partner-team REAL
electric-wires valset (data/real_wires_valset, 62 RGB-D frames, human GT).

Companion to src/eval_real_wires_valset.py -- this script answers WHY pooled
IoU is 0.322 and what the protocol ceiling is, so the realism campaign can
target the synthetic data changes with the largest pooled-IoU upside.

Preprocessing / prediction / GT handling are imported from the SAME modules the
eval script uses (infer_video_rgb_only.preprocess, gen_rgb_only_sota_gifs.predict,
eval_real_wires_valset.crop_mask_to_aspect/load_gt_binary), so every number here
is exactly comparable to the official 0.322 benchmark.

Stages (--stage, default "all"):
  dump      Task 1: run the teacher on all 62 frames, dump per-frame predicted
            masks at model res (480x640) AND at cropped-GT res (PNG 0/255).
            Sanity gate: pooled IoU recomputed from the dumps must reproduce
            the official 0.3220 (+/-0.002) or the script aborts.
  ceiling   Task 2: protocol ceiling -- GT down/up round-trip IoU at simulated
            inference sizes 480x640 / 720x960 / 960x1280 / 1440x1920 (nearest
            and area+0.5-threshold), plus 1px erode/dilate boundary sensitivity
            at 480x640.
  analyze   Tasks 3-6: colour-bin recall, width/contrast/busyness recall curves,
            FP forensics (halo / wire-like / blob), per-frame failure buckets +
            pooled-IoU upside arithmetic.

All outputs -> results/realism_campaign/diag_real/
"""
import os
import sys
import glob
import json
import csv
import argparse

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# infer_video_rgb_only sets HF_HOME/TRANSFORMERS_CACHE -> data/hf_cache on import.
from infer_video_rgb_only import preprocess, center_crop_to_aspect  # noqa: E402,F401
from gen_rgb_only_sota_gifs import load_model, predict  # noqa: E402
from eval_real_wires_valset import crop_mask_to_aspect, load_gt_binary  # noqa: E402

from scipy import ndimage  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data", "real_wires_valset")
DEFAULT_CKPT = os.path.join(
    PROJECT_ROOT,
    "results/segformer_b5_rgb_phase15_wirefree_ft/full_20260605_0409/best_model.pth",
)
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "realism_campaign", "diag_real")
PRED_DIR = os.path.join(OUT_DIR, "preds_p15")

# official benchmark number to reproduce (results/real_wires_valset_eval/summary.json)
OFFICIAL_POOLED_IOU = 0.3219639320468204
IOU_REPRO_TOL = 0.002

MODEL_H, MODEL_W = 480, 640

# ---- colour bins (SHARED with the synthetic-train-set audit -- do not alter) ----
# HSV on the ORIGINAL cropped RGB (float, H in [0,360), S/V in [0,1]).
# chromatic  : S >= 0.25, 8 uniform 45-degree hue bins centred at 0,45,...,315 deg
# achromatic : S < 0.25 -> black V<0.25 | grey 0.25<=V<0.7 | white V>=0.7
COLOUR_BINS = ["red", "orange", "yellow", "green", "cyan", "blue", "purple",
               "magenta", "black", "grey", "white"]
N_CHROMATIC = 8

# Task-2 simulated inference sizes (H, W)
CEILING_SIZES = [(480, 640), (720, 960), (960, 1280), (1440, 1920)]

# Task-4 curve bins
WIDTH_EDGES = [0, 2, 4, 8, 16, np.inf]          # wire width, px at 480x640 scale
DL_EDGES = [0, 5, 10, 20, 40, np.inf]           # |delta L*| wire vs 15px bg ring
BUSY_EDGES = [0, 0.05, 0.10, 0.20, 0.35, 1.01]  # Canny density, 31x31 window

# Task-5 FP categories (at 480x640 model scale)
HALO_DIST_PX = 5          # FP within 5px of GT = halo
WIRELIKE_MIN_ELONG = 3.0  # minAreaRect long/short
WIRELIKE_MAX_WIDTH = 10.0  # CC area / long side (mean width, px)

# Task-6 bucket thresholds (documented in report.md)
BUCKET_GOOD_IOU = 0.50
BUCKET_CAMO_DL = 12.0      # median |dL*| of FN px below this = camouflage
BUCKET_CAMO_DL_SOFT = 18.0
BUCKET_THIN_W = 3.0        # median FN width (px@480) below this = too thin
BUCKET_COLOUR_RECALL = 0.35
BUCKET_HALO_FRAC = 0.50


# --------------------------------------------------------------------------- util
def list_basenames(data_dir):
    """Same ordering as eval_real_wires_valset.main."""
    bns = sorted(
        (os.path.splitext(os.path.basename(p))[0]
         for p in glob.glob(os.path.join(data_dir, "imgs", "*"))),
        key=lambda b: (b.split("_")[0], int(b.split("_")[1])),
    )
    if not bns:
        raise SystemExit(f"no images under {data_dir}/imgs")
    return bns


def iou_from_counts(tp, fp, fn):
    u = tp + fp + fn
    return tp / u if u > 0 else float("nan")


def confusion(pred_bool, gt_bool):
    tp = int((pred_bool & gt_bool).sum())
    fp = int((pred_bool & ~gt_bool).sum())
    fn = int((~pred_bool & gt_bool).sum())
    return tp, fp, fn


def load_pred(bn, which):
    p = os.path.join(PRED_DIR, f"{bn}_{which}.png")
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"missing prediction dump {p} -- run --stage dump first")
    return (m >= 127).astype(np.uint8)


def load_frame(data_dir, bn):
    rgb_bgr = cv2.imread(os.path.join(data_dir, "imgs", f"{bn}.jpg"), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError(f"cannot read image {bn}")
    gt_bin = load_gt_binary(os.path.join(data_dir, "masks", f"{bn}.jpg"))
    gt_crop = crop_mask_to_aspect(gt_bin)
    return rgb_bgr, gt_crop


def disk_kernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


# ------------------------------------------------------------------ stage: dump
def stage_dump(args):
    """Task 1: dump per-pixel predictions; reproduce official pooled IoU or die."""
    os.makedirs(PRED_DIR, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[dump] device={device}  ckpt={args.ckpt}")
    model = load_model(args.ckpt, device)

    bns = list_basenames(args.data_dir)[: args.limit or None]
    TP = FP = FN = 0
    rows = []
    for i, bn in enumerate(bns):
        rgb_bgr, gt_crop = load_frame(args.data_dir, bn)
        pre = preprocess(rgb_bgr)                            # 480x640 BGR uint8
        with torch.no_grad():
            pred_small = predict(model, pre, device)         # 480x640 argmax
        pred_small = (pred_small >= 1).astype(np.uint8)
        Hc, Wc = gt_crop.shape
        pred_full = cv2.resize(pred_small, (Wc, Hc), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(PRED_DIR, f"{bn}_model480.png"), pred_small * 255)
        cv2.imwrite(os.path.join(PRED_DIR, f"{bn}_gtres.png"), pred_full * 255)

        tp, fp, fn = confusion(pred_full.astype(bool), gt_crop.astype(bool))
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        iou = iou_from_counts(tp, fp, fn)
        rows.append(dict(basename=bn, camera=bn.split("_")[0],
                         tp=tp, fp=fp, fn=fn, iou=iou, H=Hc, W=Wc))
        print(f"  [{i+1:2d}/{len(bns)}] {bn:8s} IoU={iou:.3f}")

    pooled = iou_from_counts(TP, FP, FN)
    out = dict(checkpoint=args.ckpt, pooled_iou=pooled, tp=TP, fp=FP, fn=FN,
               official_pooled_iou=OFFICIAL_POOLED_IOU,
               abs_diff=abs(pooled - OFFICIAL_POOLED_IOU), frames=rows)
    with open(os.path.join(OUT_DIR, "dump_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[dump] pooled IoU from dumps = {pooled:.6f} "
          f"(official {OFFICIAL_POOLED_IOU:.6f}, diff {out['abs_diff']:.6f})")
    if len(bns) == 62 and out["abs_diff"] > IOU_REPRO_TOL:
        raise SystemExit("[dump] FAILED to reproduce official pooled IoU within "
                         f"{IOU_REPRO_TOL} -- investigate before continuing")
    print("[dump] sanity gate PASSED")


# --------------------------------------------------------------- stage: ceiling
def downup(gt_crop, ht, wt, method):
    """GT -> (ht,wt) -> back to GT res with nearest. Returns binary uint8."""
    Hc, Wc = gt_crop.shape
    if method == "nearest":
        small = cv2.resize(gt_crop, (wt, ht), interpolation=cv2.INTER_NEAREST)
    elif method == "area0.5":
        small_f = cv2.resize(gt_crop.astype(np.float32), (wt, ht),
                             interpolation=cv2.INTER_AREA)
        small = (small_f >= 0.5).astype(np.uint8)
    else:
        raise ValueError(method)
    return small, cv2.resize(small, (Wc, Hc), interpolation=cv2.INTER_NEAREST)


def stage_ceiling(args):
    """Task 2: protocol ceiling + 1px boundary sensitivity at 480x640."""
    os.makedirs(OUT_DIR, exist_ok=True)
    bns = list_basenames(args.data_dir)[: args.limit or None]
    variants = [(f"{h}x{w}", h, w, m)
                for (h, w) in CEILING_SIZES for m in ("nearest", "area0.5")]
    variants += [("480x640", 480, 640, "nearest_erode1"),
                 ("480x640", 480, 640, "nearest_dilate1")]
    pooled = {(sz, m): [0, 0, 0] for (sz, _, _, m) in variants}
    k3 = np.ones((3, 3), np.uint8)

    rows = []
    for i, bn in enumerate(bns):
        _, gt_crop = load_frame(args.data_dir, bn)
        g = gt_crop.astype(bool)
        Hc, Wc = gt_crop.shape
        for sz, ht, wt, m in variants:
            if m.startswith("nearest_"):
                small, _ = downup(gt_crop, ht, wt, "nearest")
                op = cv2.erode if m.endswith("erode1") else cv2.dilate
                small = op(small, k3, iterations=1)
                up = cv2.resize(small, (Wc, Hc), interpolation=cv2.INTER_NEAREST)
            else:
                _, up = downup(gt_crop, ht, wt, m)
            tp, fp, fn = confusion(up.astype(bool), g)
            acc = pooled[(sz, m)]
            acc[0] += tp; acc[1] += fp; acc[2] += fn
            rows.append(dict(basename=bn, camera=bn.split("_")[0], size=sz,
                             method=m, tp=tp, fp=fp, fn=fn,
                             iou=iou_from_counts(tp, fp, fn)))
        if (i + 1) % 10 == 0:
            print(f"  [ceiling] {i+1}/{len(bns)} frames")

    csv_path = os.path.join(OUT_DIR, "ceiling.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["basename", "camera", "size", "method", "tp", "fp", "fn", "iou"])
        for r in rows:
            w.writerow([r["basename"], r["camera"], r["size"], r["method"],
                        r["tp"], r["fp"], r["fn"], f"{r['iou']:.6f}"])
        for (sz, m), (tp, fp, fn) in pooled.items():
            w.writerow(["POOLED", "all", sz, m, tp, fp, fn,
                        f"{iou_from_counts(tp, fp, fn):.6f}"])
    print(f"[ceiling] wrote {csv_path}")
    print("[ceiling] pooled perfect-model ceilings:")
    summary = {}
    for (sz, m), (tp, fp, fn) in pooled.items():
        iou = iou_from_counts(tp, fp, fn)
        summary[f"{sz}_{m}"] = iou
        print(f"  {sz:10s} {m:16s} pooled IoU = {iou:.4f}")
    with open(os.path.join(OUT_DIR, "ceiling_pooled.json"), "w") as f:
        json.dump(summary, f, indent=2)


# --------------------------------------------------------------- stage: analyze
def colour_binmap(rgb_crop_bgr):
    """Per-pixel colour-bin index (0..10) on the ORIGINAL cropped RGB."""
    hsv = cv2.cvtColor(rgb_crop_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_idx = (((h + 22.5) % 360.0) // 45.0).astype(np.uint8)        # 0..7
    achro = np.where(v < 0.25, 8, np.where(v < 0.7, 9, 10)).astype(np.uint8)
    return np.where(s >= 0.25, hue_idx, achro)


def width_map_480(gt_crop, scale):
    """Per-pixel local wire width (px at 480x640 scale) via medial-axis ridge.

    width(px) = 2 * dist-transform value of the NEAREST ridge (skeleton) pixel,
    computed at full GT-crop res then divided by `scale` (= Hc/480).
    """
    dist = cv2.distanceTransform(gt_crop, cv2.DIST_L2, 5)
    ridge = (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)) - 1e-3) & (gt_crop > 0)
    if not ridge.any():
        return np.zeros_like(dist)
    _, (iy, ix) = ndimage.distance_transform_edt(~ridge, return_indices=True)
    return (2.0 * dist[iy, ix]) / scale


def contrast_and_busyness_480(pre_bgr, gt_small):
    """At model scale: |dL*| vs 15px-radius bg ring, Canny density, local L std."""
    lab = cv2.cvtColor(pre_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    L = lab[..., 0]                                   # 0..100
    bg = (gt_small == 0).astype(np.float32)
    disk = disk_kernel(15).astype(np.float32)
    bg_cnt = cv2.filter2D(bg, -1, disk, borderType=cv2.BORDER_REPLICATE)
    bg_sum = cv2.filter2D(L * bg, -1, disk, borderType=cv2.BORDER_REPLICATE)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_bg = bg_sum / bg_cnt
    dL = np.abs(L - mean_bg)
    dL[bg_cnt < 1] = np.nan                           # no bg within ring

    gray = cv2.cvtColor(pre_bgr, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    busy = cv2.blur(canny, (31, 31))                  # edge density

    mu = cv2.blur(L, (15, 15))
    mu2 = cv2.blur(L * L, (15, 15))
    tex = np.sqrt(np.maximum(mu2 - mu * mu, 0))       # local L* std (texture energy)
    return dL, busy, tex


def classify_fp_ccs(nonhalo_fp):
    """Split non-halo FP pixels into wire-like vs blob via per-CC shape stats."""
    wirelike = np.zeros_like(nonhalo_fp)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(nonhalo_fp, connectivity=8)
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        comp = (labels == k).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        pts = np.vstack([c.reshape(-1, 2) for c in cnts])
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        long_s, short_s = max(rw, rh), max(min(rw, rh), 1.0)
        elong = long_s / short_s
        mean_width = area / max(long_s, 1.0)
        if elong >= WIRELIKE_MIN_ELONG and mean_width <= WIRELIKE_MAX_WIDTH:
            wirelike |= comp
    blob = (nonhalo_fp > 0) & (wirelike == 0)
    return wirelike.astype(bool), blob


def assign_bucket(r):
    """Dominant-failure bucket from the per-frame stats dict (see report.md)."""
    if r["iou"] >= BUCKET_GOOD_IOU:
        return "good"
    if r["fn"] >= r["fp"]:                            # recall-dominant failure
        if not np.isnan(r["median_fn_dL"]) and r["median_fn_dL"] < BUCKET_CAMO_DL:
            return "camouflage-low-contrast"
        if (r["dom_fn_bin"] in COLOUR_BINS[:N_CHROMATIC]
                and r["dom_fn_bin_recall"] < BUCKET_COLOUR_RECALL):
            return "missed-colour"
        if r["median_fn_width"] < BUCKET_THIN_W:
            return "too-thin"
        if not np.isnan(r["median_fn_dL"]) and r["median_fn_dL"] < BUCKET_CAMO_DL_SOFT:
            return "camouflage-low-contrast"
        # high-contrast misses: an appearance gap (any bin, incl. white/grey)
        if r["dom_fn_bin_recall"] < BUCKET_COLOUR_RECALL or \
                r["dom_fn_bin"] in COLOUR_BINS[:N_CHROMATIC]:
            return "missed-colour"
        return "camouflage-low-contrast"
    # FP-dominant failure
    if r["fp_halo_frac"] >= BUCKET_HALO_FRAC:
        return "halo-only"
    return "busy-bg-FP"


def stage_analyze(args):
    """Tasks 3-6 in one pass over the frames (uses the Task-1 dumps)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    bns = list_basenames(args.data_dir)[: args.limit or None]

    nb = len(COLOUR_BINS)
    col_gt = np.zeros(nb, np.int64)         # GT px per colour bin (full res)
    col_tp = np.zeros(nb, np.int64)
    col_dom_frames = {b: [] for b in COLOUR_BINS}
    curve_cnt = {c: np.zeros(5, np.int64) for c in ("width", "contrast", "busyness")}
    curve_tp = {c: np.zeros(5, np.int64) for c in ("width", "contrast", "busyness")}
    fp_cam = {}                              # camera -> dict of FP px sums (480 scale)
    busy_fp_sum = busy_fp_n = busy_bg_sum = busy_bg_n = 0.0
    tex_fp_sum = tex_bg_sum = 0.0
    rows = []

    for i, bn in enumerate(bns):
        rgb_bgr, gt_crop = load_frame(args.data_dir, bn)
        rgb_crop = center_crop_to_aspect(rgb_bgr)
        pre = preprocess(rgb_bgr)
        pred_small = load_pred(bn, "model480")
        pred_full = load_pred(bn, "gtres")
        Hc, Wc = gt_crop.shape
        scale = Hc / float(MODEL_H)
        g = gt_crop.astype(bool)
        p = pred_full.astype(bool)
        tp, fp, fn = confusion(p, g)
        fn_mask = g & ~p

        # ---- Task 3: colour bins at full cropped res -------------------------
        bm = colour_binmap(rgb_crop)
        gt_b = np.bincount(bm[g], minlength=nb)
        tp_b = np.bincount(bm[g & p], minlength=nb)
        col_gt += gt_b
        col_tp += tp_b
        dom = int(np.argmax(gt_b))
        dom_fn = int(np.argmax(gt_b - tp_b))
        col_dom_frames[COLOUR_BINS[dom]].append(bn)
        with np.errstate(invalid="ignore", divide="ignore"):
            dom_fn_recall = (tp_b[dom_fn] / gt_b[dom_fn]) if gt_b[dom_fn] > 0 else np.nan

        # ---- Task 4: width / contrast / busyness -----------------------------
        wmap = width_map_480(gt_crop, scale)          # full res, px@480 units
        gt_small = cv2.resize(gt_crop, (MODEL_W, MODEL_H),
                              interpolation=cv2.INTER_NEAREST)
        dL_s, busy_s, tex_s = contrast_and_busyness_480(pre, gt_small)
        dL = cv2.resize(dL_s, (Wc, Hc), interpolation=cv2.INTER_NEAREST)
        busy = cv2.resize(busy_s, (Wc, Hc), interpolation=cv2.INTER_NEAREST)

        hit = p[g]
        for name, vals, edges in (("width", wmap[g], WIDTH_EDGES),
                                  ("contrast", dL[g], DL_EDGES),
                                  ("busyness", busy[g], BUSY_EDGES)):
            ok = ~np.isnan(vals)
            idx = np.digitize(vals[ok], edges[1:-1])  # 0..4
            curve_cnt[name] += np.bincount(idx, minlength=5)
            curve_tp[name] += np.bincount(idx[hit[ok]], minlength=5)

        w_gt = wmap[g]
        w_fn = wmap[fn_mask]
        dL_gt = dL[g]; dL_gt = dL_gt[~np.isnan(dL_gt)]
        dL_fn = dL[fn_mask]; dL_fn = dL_fn[~np.isnan(dL_fn)]

        # ---- Task 5: FP forensics at 480x640 ---------------------------------
        ps = pred_small.astype(bool)
        gs = gt_small.astype(bool)
        fp_s = ps & ~gs
        d2gt = cv2.distanceTransform((~gs).astype(np.uint8), cv2.DIST_L2, 5)
        halo = fp_s & (d2gt <= HALO_DIST_PX)
        wirelike, blob = classify_fp_ccs((fp_s & ~halo).astype(np.uint8))
        n_fp_s, n_halo = int(fp_s.sum()), int(halo.sum())
        n_wl, n_blob = int(wirelike.sum()), int(blob.sum())
        cam = bn.split("_")[0]
        acc = fp_cam.setdefault(cam, dict(fp=0, halo=0, wirelike=0, blob=0))
        acc["fp"] += n_fp_s; acc["halo"] += n_halo
        acc["wirelike"] += n_wl; acc["blob"] += n_blob
        if n_fp_s:
            busy_fp_sum += float(busy_s[fp_s].sum()); busy_fp_n += n_fp_s
            tex_fp_sum += float(tex_s[fp_s].sum())
        bg_s = ~gs
        busy_bg_sum += float(busy_s[bg_s].sum()); busy_bg_n += int(bg_s.sum())
        tex_bg_sum += float(tex_s[bg_s].sum())

        r = dict(
            basename=bn, camera=cam, H=Hc, W=Wc, tp=tp, fp=fp, fn=fn,
            gt_px=int(g.sum()), pred_px=int(p.sum()),
            iou=iou_from_counts(tp, fp, fn),
            precision=tp / (tp + fp) if tp + fp else float("nan"),
            recall=tp / (tp + fn) if tp + fn else float("nan"),
            dom_colour_bin=COLOUR_BINS[dom],
            dom_colour_share=gt_b[dom] / max(g.sum(), 1),
            dom_fn_bin=COLOUR_BINS[dom_fn],
            dom_fn_bin_recall=float(dom_fn_recall),
            median_gt_width=float(np.median(w_gt)) if w_gt.size else float("nan"),
            median_fn_width=float(np.median(w_fn)) if w_fn.size else float("nan"),
            median_gt_dL=float(np.median(dL_gt)) if dL_gt.size else float("nan"),
            median_fn_dL=float(np.median(dL_fn)) if dL_fn.size else float("nan"),
            mean_wire_busyness=float(busy[g].mean()) if g.any() else float("nan"),
            fp_px_480=n_fp_s,
            fp_halo_frac=n_halo / n_fp_s if n_fp_s else float("nan"),
            fp_wirelike_frac=n_wl / n_fp_s if n_fp_s else float("nan"),
            fp_blob_frac=n_blob / n_fp_s if n_fp_s else float("nan"),
        )
        r["bucket"] = assign_bucket(r)
        rows.append(r)
        print(f"  [{i+1:2d}/{len(bns)}] {bn:8s} IoU={r['iou']:.3f} "
              f"dom={r['dom_colour_bin']:8s} fnW={r['median_fn_width']:5.1f} "
              f"fndL={r['median_fn_dL']:5.1f} halo={r['fp_halo_frac']:.2f} "
              f"-> {r['bucket']}")

    # ---------------- Task 3 outputs: colour recall + IoU-if-fixed ------------
    TP = sum(r["tp"] for r in rows); FP = sum(r["fp"] for r in rows)
    FN = sum(r["fn"] for r in rows)
    P0 = TP / (TP + FP)
    col_rows = []
    for b in range(nb):
        name = COLOUR_BINS[b]
        gtb, tpb = int(col_gt[b]), int(col_tp[b])
        rec = tpb / gtb if gtb else float("nan")
        dtp = max(0.0, 0.95 * gtb - tpb)
        tp2, fn2 = TP + dtp, FN - dtp
        iou_fpfix = tp2 / (tp2 + FP + fn2)
        iou_pfix = tp2 / (tp2 / P0 + fn2)             # FP' = TP2*(1-P0)/P0
        col_rows.append(dict(
            bin=name, gt_px=gtb, gt_share=gtb / max(col_gt.sum(), 1),
            tp=tpb, fn=gtb - tpb, recall=rec,
            n_frames_dominant=len(col_dom_frames[name]),
            frames_dominant=";".join(col_dom_frames[name]),
            iou_if_recall95_fp_fixed=iou_fpfix,
            iou_if_recall95_prec_fixed=iou_pfix,
        ))
    with open(os.path.join(OUT_DIR, "colour_recall.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(col_rows[0].keys()))
        w.writeheader()
        for r in col_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    dtp_all = sum(max(0.0, 0.95 * int(col_gt[b]) - int(col_tp[b])) for b in range(nb))
    tp2 = TP + dtp_all; fn2 = FN - dtp_all
    all95 = dict(fp_fixed=tp2 / (tp2 + FP + fn2), prec_fixed=tp2 / (tp2 / P0 + fn2))

    # ---------------- Task 4 outputs: curves CSV + PNG -------------------------
    curve_path = os.path.join(OUT_DIR, "width_contrast_curves.csv")
    with open(curve_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["curve", "bin_lo", "bin_hi", "gt_px", "tp", "recall"])
        for name, edges in (("width", WIDTH_EDGES), ("contrast", DL_EDGES),
                            ("busyness", BUSY_EDGES)):
            for k in range(5):
                cnt, tpk = int(curve_cnt[name][k]), int(curve_tp[name][k])
                w.writerow([name, edges[k], edges[k + 1], cnt, tpk,
                            f"{tpk / cnt:.6f}" if cnt else "nan"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, (name, edges, xlab) in zip(axes, (
            ("width", WIDTH_EDGES, "wire width (px @480x640)"),
            ("contrast", DL_EDGES, "|dL*| wire vs 15px bg ring"),
            ("busyness", BUSY_EDGES, "Canny edge density (31x31)"))):
        rec = [curve_tp[name][k] / curve_cnt[name][k] if curve_cnt[name][k] else np.nan
               for k in range(5)]
        labels = [f"{edges[k]}-{edges[k+1]}" for k in range(5)]
        ax.bar(range(5), rec, color="#3a7ebf")
        for k, (r_, c_) in enumerate(zip(rec, curve_cnt[name])):
            ax.text(k, (r_ or 0) + 0.02, f"{r_:.2f}\n{c_/1e6:.1f}M",
                    ha="center", fontsize=8)
        ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_xlabel(xlab); ax.set_ylabel("recall (GT-res px)")
        ax.axhline(TP / (TP + FN), color="grey", ls="--", lw=0.8)
        ax.set_title(f"P15 recall vs {name}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "width_contrast_curves.png"), dpi=130)
    plt.close(fig)

    # ---------------- Task 5 outputs: FP forensics ----------------------------
    fp_total = {k: sum(c[k] for c in fp_cam.values())
                for k in ("fp", "halo", "wirelike", "blob")}
    fp_json = dict(
        note="FP categories computed at 480x640 model scale; halo = FP within "
             f"{HALO_DIST_PX}px of GT; wire-like = CC elongation >= "
             f"{WIRELIKE_MIN_ELONG} and mean width <= {WIRELIKE_MAX_WIDTH}px",
        per_camera=fp_cam, total=fp_total,
        mean_canny_density_fp=busy_fp_sum / busy_fp_n if busy_fp_n else float("nan"),
        mean_canny_density_bg=busy_bg_sum / busy_bg_n,
        mean_texstd_fp=tex_fp_sum / busy_fp_n if busy_fp_n else float("nan"),
        mean_texstd_bg=tex_bg_sum / busy_bg_n,
    )
    with open(os.path.join(OUT_DIR, "fp_forensics.json"), "w") as f:
        json.dump(fp_json, f, indent=2)

    # ---------------- Task 6 outputs: buckets + upside arithmetic -------------
    GTPX = TP + FN
    buckets = {}
    for r in rows:
        buckets.setdefault(r["bucket"], []).append(r)
    bucket_rows = []
    for name, rs in sorted(buckets.items()):
        btp = sum(r["tp"] for r in rs); bfp = sum(r["fp"] for r in rs)
        bfn = sum(r["fn"] for r in rs)
        dtp = sum(max(0.0, 0.9 * (r["tp"] + r["fn"]) - r["tp"]) for r in rs)
        # recall->0.9 keeps union constant (FN converts to TP, FP untouched)
        iou_r90 = (TP + dtp) / (TP + FP + FN)
        iou_fp0 = TP / (TP + (FP - bfp) + FN)
        iou_both = (TP + dtp) / (TP + dtp + (FP - bfp) + (FN - dtp))
        bucket_rows.append(dict(
            bucket=name, n_frames=len(rs), frames=";".join(r["basename"] for r in rs),
            gt_px=btp + bfn, gt_share=(btp + bfn) / GTPX,
            fn_px=bfn, fn_share=bfn / FN, fp_px=bfp, fp_share=bfp / FP,
            iou_if_recall90=iou_r90, iou_if_fp0=iou_fp0, iou_if_both=iou_both,
            gain_recall90=iou_r90 - OFFICIAL_POOLED_IOU,
            gain_fp0=iou_fp0 - OFFICIAL_POOLED_IOU,
            gain_both=iou_both - OFFICIAL_POOLED_IOU,
        ))
    with open(os.path.join(OUT_DIR, "buckets.json"), "w") as f:
        json.dump(dict(thresholds=dict(
            good_iou=BUCKET_GOOD_IOU, camo_dL=BUCKET_CAMO_DL,
            camo_dL_soft=BUCKET_CAMO_DL_SOFT, thin_w=BUCKET_THIN_W,
            colour_recall=BUCKET_COLOUR_RECALL, halo_frac=BUCKET_HALO_FRAC),
            pooled=dict(tp=TP, fp=FP, fn=FN, iou=iou_from_counts(TP, FP, FN),
                        precision=P0, recall=TP / (TP + FN)),
            all_bins_recall95=all95, buckets=bucket_rows), f, indent=2)

    # ---------------- per-image diagnostic CSV --------------------------------
    diag_path = os.path.join(OUT_DIR, "per_image_diag.csv")
    fields = ["basename", "camera", "H", "W", "iou", "precision", "recall",
              "tp", "fp", "fn", "gt_px", "pred_px", "dom_colour_bin",
              "dom_colour_share", "dom_fn_bin", "dom_fn_bin_recall",
              "median_gt_width", "median_fn_width", "median_gt_dL",
              "median_fn_dL", "mean_wire_busyness", "fp_px_480",
              "fp_halo_frac", "fp_wirelike_frac", "fp_blob_frac", "bucket"]
    with open(diag_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k])
                        for k in fields})

    print(f"\n[analyze] pooled: IoU={iou_from_counts(TP, FP, FN):.4f} "
          f"P={P0:.4f} R={TP / (TP + FN):.4f}")
    print(f"[analyze] all-bins recall->0.95: IoU {all95['fp_fixed']:.4f} (FP fixed) / "
          f"{all95['prec_fixed']:.4f} (precision fixed)")
    print("[analyze] buckets:")
    for b in sorted(bucket_rows, key=lambda x: -x["gain_both"]):
        print(f"  {b['bucket']:24s} n={b['n_frames']:2d} fn_share={b['fn_share']:.2f} "
              f"fp_share={b['fp_share']:.2f} IoU_if_fixed={b['iou_if_both']:.4f} "
              f"(+{b['gain_both']:.4f})")
    print(f"[analyze] wrote {diag_path}")


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    choices=["dump", "ceiling", "analyze", "all"])
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only first N frames (sanity gate skipped if <62)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.stage in ("dump", "all"):
        stage_dump(args)
    if args.stage in ("ceiling", "all"):
        stage_ceiling(args)
    if args.stage in ("analyze", "all"):
        stage_analyze(args)


if __name__ == "__main__":
    main()
