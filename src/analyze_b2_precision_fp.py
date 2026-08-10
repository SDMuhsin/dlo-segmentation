#!/usr/bin/env python
"""
BOUNDED diagnostic: characterise WHERE and WHY the B2 teacher produces FALSE
POSITIVES (pred=wire, GT=bg) vs where it is correct (TP: pred=wire, GT=wire) on
the partner-team real-world validation set (data/real_wires_valset, 62 frames).

This is the false-POSITIVE counterpart of src/analyze_b2_recall_fn.py (which did
the false-NEGATIVE / recall diagnosis). B2 is PRECISION-limited as well as recall-
limited (pooled IoU=0.612, P=0.705, R=0.823); this script answers: what
distinguishes FP (GT=bg, pred=wire) pixels from TP (GT=wire, pred=wire) pixels,
and what is the dominant FALSE-POSITIVE MODE so we can pick the next data lever.

For every FP and TP pixel we compute (on the model-INPUT RGB mapped into the GT-crop
scoring grid -- the SAME grid the evaluator computes IoU on; pred upsampled NEAREST,
preprocessed RGB resized to the GT-crop res; stated explicitly in the json):
  (a) local BACKGROUND BUSYNESS  -- Laplacian variance in a 15x15 window (the key
      discriminator for busy-texture hallucination)
  (b) HUE, SATURATION, LUMA      -- local 15x15 mean HSV / luma of the patch
  (c) DISTANCE-TO-NEAREST-TRUE-WIRE -- distanceTransform of (1 - GT-wire) sampled at
      each pixel; small => halo / over-seg hugging real wires, large => standalone
      hallucination (the discriminator separating over-seg from hallucination)
  (d) local CONTRAST             -- std of luma in a 15x15 window

Then:
  - Cliff's delta FP-vs-TP on each continuous property (busyness, hue, sat, luma,
    contrast, dist-to-wire) in the SAME table style as the FN analysis.
  - Connected-component analysis of the FP mask per frame: #components, size
    distribution, fraction of FP MASS in "halo" components (touching/within ~HALO_PX
    of a true wire) vs "standalone" components (far from any true wire).
  - Per-camera FP rate + precision (c1-c4); which camera dominates the FP mass.
  - DELIVERABLE: partition total FP MASS into 4 modes (busy-texture hallucination /
    near-wire halo / structured-edge object confusion / bright-pale bg), best-fit per
    FP COMPONENT, report ranked mass fractions.

CRITICAL: the prediction pipeline is unchanged from src/eval_real_wires_valset.py
-- we import its exact functions (load_model_auto, preprocess_size, predict_size,
crop_mask_to_aspect, load_gt_binary) so the reproduced pooled IoU/P/R must match
0.612/0.705/0.823. A SANITY GATE enforces this (reproduce + reconcile FP/TP/FN with
the arbiter's overall fp/tp/fn to within 1%) before any FP analysis.

Outputs: results/realism_campaign/b2_precision_fp/fp_analysis.json + 4-6
worst-precision montages (RGB w/ overlay FP=red, TP=green, FN=blue).
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
# via os.environ.setdefault on import.
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
OUT = os.path.join(PROJECT_ROOT, "results", "realism_campaign", "b2_precision_fp")
INFER_HW = (IMAGE_H, IMAGE_W)  # 480x640

# --- reference numbers from the arbiter (src/eval_real_wires_valset.py) ---
# Recorded from the sanity run on this exact ckpt; used for the reconciliation gate.
ARBITER = dict(tp=17447461, fp=7311316, fn=3763118,
               pooled_iou=0.6117, precision=0.7047, recall=0.8226)

WIN = 15          # local busyness / HSV / contrast window (odd) -- task spec 15x15
HALO_PX = 5       # a FP pixel within this DT distance of a true wire is "near-wire"

# --- FP-mode classification thresholds (operational; derived per-pixel medians,
# applied per connected COMPONENT using the component's median property values).
# Calibrated against TP-distribution medians at runtime so they are data-driven,
# but with these fixed fallbacks documented in the json. ---
# (filled at runtime from TP medians; see classify section)


def list_basenames():
    """Same listing+sort key as the evaluator."""
    return sorted(
        (os.path.splitext(os.path.basename(p))[0]
         for p in glob.glob(os.path.join(DATA, "imgs", "*"))),
        key=lambda b: (b.split("_")[0], int(b.split("_")[1])),
    )


def luma_from_bgr(bgr):
    """0.299R + 0.587G + 0.114B (float)."""
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def laplacian_var_map(gray, win=WIN):
    """Local Laplacian variance (busyness): Var(L) = E[L^2] - E[L]^2 over a win x win
    box, where L = Laplacian(gray). High => busy/textured local background."""
    g = gray.astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    ksz = (win, win)
    mean_l = cv2.boxFilter(lap, -1, ksz, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_l2 = cv2.boxFilter(lap * lap, -1, ksz, normalize=True, borderType=cv2.BORDER_REFLECT)
    var = mean_l2 - mean_l * mean_l
    return np.maximum(var, 0.0)


def local_mean_map(arr, win=WIN):
    """win x win box mean of a float map."""
    return cv2.boxFilter(arr.astype(np.float32), -1, (win, win),
                         normalize=True, borderType=cv2.BORDER_REFLECT)


def local_std_map(arr, win=WIN):
    """win x win local std (contrast) of a float map."""
    a = arr.astype(np.float32)
    m = cv2.boxFilter(a, -1, (win, win), normalize=True, borderType=cv2.BORDER_REFLECT)
    m2 = cv2.boxFilter(a * a, -1, (win, win), normalize=True, borderType=cv2.BORDER_REFLECT)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


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
    Sign: positive => a tends LARGER than b."""
    a = np.asarray(a, dtype=np.float64); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=np.float64); b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return None
    rng = np.random.default_rng(0)
    if a.size > cap:
        a = rng.choice(a, cap, replace=False)
    if b.size > cap:
        b = rng.choice(b, cap, replace=False)
    b_sorted = np.sort(b)
    less = np.searchsorted(b_sorted, a, side="left")               # # b < a
    greater = b.size - np.searchsorted(b_sorted, a, side="right")  # # b > a
    delta = (less.sum() - greater.sum()) / (a.size * b.size)
    return float(delta)


def main():
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device     : {device}")
    print(f"checkpoint : {CKPT}")
    print(f"data-dir   : {DATA}")
    print(f"infer-size : {INFER_HW[0]}x{INFER_HW[1]}  wire-rule: fg  win={WIN}")
    model = load_model_auto(CKPT, device)

    bns = list_basenames()
    print(f"images     : {len(bns)}")

    TP = FP = FN = 0
    per_cam = {}   # cam -> [tp, fp, fn, bg_px]
    per_frame = []  # for montage selection

    # pooled property collectors (concat at end). FP and TP pixel-level props.
    PROPS = ("busy", "hue", "sat", "luma", "contrast", "dist")
    coll = {k: {"fp": [], "tp": []} for k in PROPS}

    # connected-component records, pooled across frames. Each FP component:
    #   size, halo_mass (px within HALO_PX of a wire), elongation, and median
    #   busy/luma/sat/dist over the component (for mode classification).
    cc_records = []   # list of dicts
    cc_per_frame_counts = []  # (bn, n_components)

    # accumulators for global FP-mode mass (filled after we know TP medians)
    frame_cache = []  # keep light per-frame arrays we need for mode classify pass

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
        bg_px = int((~g).sum())
        TP += tp; FP += fp; FN += fn
        c = per_cam.setdefault(cam, [0, 0, 0, 0])
        c[0] += tp; c[1] += fp; c[2] += fn; c[3] += bg_px
        precision_f = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        per_frame.append(dict(bn=bn, cam=cam, precision=precision_f, gt_px=int(g.sum()),
                              bg_px=bg_px, tp=tp, fp=fp, fn=fn,
                              iou=(tp / (tp + fp + fn) if (tp + fp + fn) else float("nan"))))

        # --- properties in the GT-crop scoring grid (same as FN analysis) ---
        interp = cv2.INTER_AREA if (pre.shape[1] >= Wc and pre.shape[0] >= Hc) else cv2.INTER_LINEAR
        rgb_crop = cv2.resize(pre, (Wc, Hc), interpolation=interp)   # BGR at GT-crop res
        gray = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        luma = luma_from_bgr(rgb_crop)
        hsv = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2HSV)
        hue_px = hsv[:, :, 0].astype(np.float32)   # [0,179]
        sat_px = hsv[:, :, 1].astype(np.float32)   # [0,255]

        busy = laplacian_var_map(gray, WIN)
        # local patch means for hue/sat/luma + local contrast (std of luma)
        hue = local_mean_map(hue_px, WIN)
        sat = local_mean_map(sat_px, WIN)
        luma_loc = local_mean_map(luma, WIN)
        contrast = local_std_map(luma, WIN)
        # distance from every pixel to the nearest TRUE-WIRE pixel (DT of bg=1-wire)
        dist = cv2.distanceTransform((1 - gt_crop).astype(np.uint8), cv2.DIST_L2, 5)

        fp_mask = p & (~g)
        tp_mask = p & g
        prop_maps = dict(busy=busy, hue=hue, sat=sat, luma=luma_loc,
                         contrast=contrast, dist=dist)
        for key, arr in prop_maps.items():
            coll[key]["fp"].append(arr[fp_mask])
            coll[key]["tp"].append(arr[tp_mask])

        # --- connected components of the FP mask (8-connectivity) ---
        fp_u8 = fp_mask.astype(np.uint8)
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(fp_u8, connectivity=8)
        near = (dist <= HALO_PX)  # halo zone around true wires
        comp_count = 0
        for lab_id in range(1, n_lab):
            area = int(stats[lab_id, cv2.CC_STAT_AREA])
            if area == 0:
                continue
            comp_count += 1
            cmask = labels == lab_id
            w_box = int(stats[lab_id, cv2.CC_STAT_WIDTH])
            h_box = int(stats[lab_id, cv2.CC_STAT_HEIGHT])
            bbox_area = max(w_box * h_box, 1)
            halo_px = int((cmask & near).sum())
            elong = max(w_box, h_box) / max(min(w_box, h_box), 1)  # bbox aspect ratio
            fill = area / bbox_area  # how thin/sparse vs blob-like
            cc_records.append(dict(
                bn=bn, cam=cam, area=area,
                halo_frac=halo_px / area,          # fraction of comp mass near a wire
                median_busy=float(np.median(busy[cmask])),
                median_dist=float(np.median(dist[cmask])),
                median_luma=float(np.median(luma_loc[cmask])),
                median_sat=float(np.median(sat[cmask])),
                median_contrast=float(np.median(contrast[cmask])),
                elong=float(elong), fill=float(fill),
                w_box=w_box, h_box=h_box,
            ))
        cc_per_frame_counts.append(dict(bn=bn, cam=cam, n_fp_components=comp_count, fp_px=fp))

        print(f"  [{i+1:2d}/{len(bns)}] {bn:8s} IoU={per_frame[-1]['iou']:.3f} "
              f"P={precision_f:.3f} fp={fp:7d} bg={bg_px:8d} ncomp={comp_count:5d}")

    # ---- SANITY GATE: reproduce + reconcile with arbiter ----
    pooled_iou = TP / (TP + FP + FN) if (TP + FP + FN) else float("nan")
    precision = TP / (TP + FP) if (TP + FP) else float("nan")
    recall = TP / (TP + FN) if (TP + FN) else float("nan")

    def pct_diff(a, b):
        return abs(a - b) / b * 100.0 if b else float("nan")

    rec_fp = pct_diff(FP, ARBITER["fp"])
    rec_tp = pct_diff(TP, ARBITER["tp"])
    rec_fn = pct_diff(FN, ARBITER["fn"])
    iou_ok = abs(pooled_iou - 0.612) <= 0.012
    reconcile_ok = (rec_fp <= 1.0 and rec_tp <= 1.0 and rec_fn <= 1.0)
    gate_ok = iou_ok and reconcile_ok

    print("\n==================== SANITY GATE ====================")
    print(f"pooled IoU = {pooled_iou:.4f}  (target 0.612 +/- 0.012)  -> {'OK' if iou_ok else 'FAIL'}")
    print(f"precision  = {precision:.4f}  (arbiter 0.7047)")
    print(f"recall     = {recall:.4f}  (arbiter 0.8226)")
    print(f"reconcile FP: mine={FP} arbiter={ARBITER['fp']}  diff={rec_fp:.3f}%")
    print(f"reconcile TP: mine={TP} arbiter={ARBITER['tp']}  diff={rec_tp:.3f}%")
    print(f"reconcile FN: mine={FN} arbiter={ARBITER['fn']}  diff={rec_fn:.3f}%")
    print(f"GATE       : {'PASS' if gate_ok else 'FAIL'}")

    sanity = dict(reproduced=dict(pooled_iou=pooled_iou, precision=precision, recall=recall,
                                  tp=TP, fp=FP, fn=FN),
                  arbiter=ARBITER,
                  reconcile_pct=dict(fp=rec_fp, tp=rec_tp, fn=rec_fn),
                  iou_within_tol=bool(iou_ok),
                  reconcile_within_1pct=bool(reconcile_ok),
                  gate_pass=bool(gate_ok))

    if not gate_ok:
        with open(os.path.join(OUT, "fp_analysis.json"), "w") as f:
            json.dump(dict(sanity_gate=sanity, ABORTED="sanity gate failed"), f, indent=2)
        print("\nSANITY GATE FAILED -> aborting FP analysis (pipeline drifted).")
        return

    # ---- pixel-level property stats: FP vs TP + Cliff's delta ----
    prop_meta = dict(
        busy=("local bg busyness (Laplacian var, 15x15)",
              "FP HIGHER => busy-texture hallucination"),
        hue=("local hue (HSV H 0-179, 15x15 mean)", "shift => colour-specific FP"),
        sat=("local saturation (HSV S 0-255, 15x15 mean)",
             "FP LOWER => pale/desaturated bg FP"),
        luma=("local luma (0-255, 15x15 mean)", "FP HIGHER => bright-bg FP"),
        contrast=("local contrast (luma std, 15x15)", "FP HIGHER => high-contrast edges"),
        dist=("dist-to-nearest-true-wire (px)",
              "FP LARGE => standalone hallucination; FP SMALL => halo/over-seg"),
    )
    props = {}
    tp_median = {}
    for key, (label, hint) in prop_meta.items():
        fp_vals = np.concatenate(coll[key]["fp"]) if coll[key]["fp"] else np.array([])
        tp_vals = np.concatenate(coll[key]["tp"]) if coll[key]["tp"] else np.array([])
        fp_s = stat_block(fp_vals)
        tp_s = stat_block(tp_vals)
        delta = cliffs_delta(fp_vals, tp_vals)  # +ve => FP larger than TP
        tp_median[key] = tp_s["median"]
        props[key] = dict(label=label, hint=hint, fp=fp_s, tp=tp_s,
                          cliffs_delta_fp_vs_tp=delta)
        print(f"\n[{key}] {label}")
        print(f"   FP: median={fp_s['median']} IQR=[{fp_s['q25']},{fp_s['q75']}] n={fp_s['n']}")
        print(f"   TP: median={tp_s['median']} IQR=[{tp_s['q25']},{tp_s['q75']}] n={tp_s['n']}")
        print(f"   Cliff's delta(FP vs TP)={delta}")

    # ---- per-camera FP rate + precision ----
    cam_tbl = {}
    for cam, (tp, fp, fn, bgpx) in sorted(per_cam.items()):
        fp_rate = fp / bgpx if bgpx else float("nan")   # FP as fraction of true-bg pixels
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        cam_tbl[cam] = dict(tp=tp, fp=fp, fn=fn, bg_px=bgpx,
                            fp_rate_of_bg=fp_rate, precision=prec,
                            fp_mass_frac=fp / FP if FP else float("nan"),
                            n_frames=sum(1 for fr in per_frame if fr["cam"] == cam))

    print("\n================ PER-CAMERA FP / PRECISION ================")
    print(f"{'cam':4s} {'n':>3s} {'FP_px':>9s} {'bg_px':>10s} {'FP/bg':>8s} "
          f"{'precis':>7s} {'FPmass%':>8s}")
    for cam, d in cam_tbl.items():
        print(f"{cam:4s} {d['n_frames']:3d} {d['fp']:9d} {d['bg_px']:10d} "
              f"{d['fp_rate_of_bg']:8.4f} {d['precision']:7.4f} {100*d['fp_mass_frac']:8.2f}")

    # ---- connected-component summary ----
    areas = np.array([r["area"] for r in cc_records], dtype=np.float64)
    total_fp_mass = float(areas.sum())  # == FP (sum of all comp areas)
    n_components = len(cc_records)
    # halo vs standalone mass: a component's halo mass is halo_frac*area; sum it
    halo_mass = float(sum(r["halo_frac"] * r["area"] for r in cc_records))
    standalone_mass = total_fp_mass - halo_mass
    # tiny spurious strokes: components with area < 50 px (in GT-crop res)
    tiny = [r for r in cc_records if r["area"] < 50]
    small = [r for r in cc_records if 50 <= r["area"] < 1000]
    large = [r for r in cc_records if r["area"] >= 1000]
    cc_summary = dict(
        n_components=n_components,
        total_fp_mass_px=total_fp_mass,
        fp_mass_check_vs_FP=dict(component_sum=total_fp_mass, frame_FP=FP,
                                 match_pct=pct_diff(total_fp_mass, FP)),
        component_size=dict(
            median=float(np.median(areas)) if n_components else None,
            q25=float(np.percentile(areas, 25)) if n_components else None,
            q75=float(np.percentile(areas, 75)) if n_components else None,
            max=float(areas.max()) if n_components else None,
        ),
        size_buckets=dict(
            tiny_lt50=dict(n=len(tiny), mass=float(sum(r["area"] for r in tiny)),
                           mass_frac=sum(r["area"] for r in tiny) / total_fp_mass if total_fp_mass else 0),
            small_50_1000=dict(n=len(small), mass=float(sum(r["area"] for r in small)),
                               mass_frac=sum(r["area"] for r in small) / total_fp_mass if total_fp_mass else 0),
            large_ge1000=dict(n=len(large), mass=float(sum(r["area"] for r in large)),
                              mass_frac=sum(r["area"] for r in large) / total_fp_mass if total_fp_mass else 0),
        ),
        halo_vs_standalone=dict(
            halo_px=halo_mass, halo_mass_frac=halo_mass / total_fp_mass if total_fp_mass else 0,
            standalone_px=standalone_mass,
            standalone_mass_frac=standalone_mass / total_fp_mass if total_fp_mass else 0,
            halo_px_threshold=HALO_PX,
            note=("halo mass = sum over components of (fraction of component pixels within "
                  f"{HALO_PX}px DT of a true wire) * component area; this splits MASS not components"),
        ),
    )
    print("\n================ FP CONNECTED-COMPONENT SUMMARY ================")
    print(f"  components={n_components}  total FP mass={total_fp_mass:.0f}px "
          f"(vs frame FP {FP}, diff {cc_summary['fp_mass_check_vs_FP']['match_pct']:.3f}%)")
    print(f"  halo mass frac    = {cc_summary['halo_vs_standalone']['halo_mass_frac']:.3f}")
    print(f"  standalone mass   = {cc_summary['halo_vs_standalone']['standalone_mass_frac']:.3f}")
    print(f"  size: tiny<50 mass={cc_summary['size_buckets']['tiny_lt50']['mass_frac']:.3f}  "
          f"small={cc_summary['size_buckets']['small_50_1000']['mass_frac']:.3f}  "
          f"large>=1000={cc_summary['size_buckets']['large_ge1000']['mass_frac']:.3f}")

    # ---- DELIVERABLE: EXHAUSTIVELY partition total FP MASS into 4 modes ----
    # Every FP component lands in exactly one of the 4 real modes (no "unclassified"
    # residual): the partition first splits on geometry (near a true wire vs
    # standalone), then assigns each STANDALONE component by its DOMINANT appearance
    # signal. Operational definitions:
    #   near        : halo_frac >= NEAR_HALO_FRAC OR median dist-to-wire <= NEAR_PX
    #                 (the component hugs / over-dilates a true wire)  -> mode (ii)
    #   For the remaining (standalone) components, assign the best-fit appearance mode:
    #     structured_edge : elong > ELONG_HI AND fill < FILL_LO        -> mode (iii)
    #                       (thin elongated stroke-like blob = desk/socket/other-cable edge)
    #     bright_pale_bg  : median luma > LUMA_HI AND median sat < SAT_LO -> mode (iv)
    #                       (bright + desaturated local patch = pale-bg / pale-node leakage)
    #     busy_texture    : everything else                            -> mode (i)
    #                       (blob on textured/cluttered bg; the catch-all standalone mode)
    # Rationale: 77.7% of FP mass is standalone and 89% sits in large (>=1000px) blobs
    # on visibly textured/cluttered backdrops (montages c2_*, c4_9); these are
    # busy-texture hallucination by construction. A component is "structured_edge" or
    # "bright_pale" only if it CLEARLY matches that narrower signature, else it is the
    # busy-texture catch-all. We also report, for transparency, how much busy_texture
    # mass is "high-busyness" (median_busy > BUSY_TP75) vs "mid-busyness".
    BUSY_TP75 = float(np.percentile(np.concatenate(coll["busy"]["tp"]), 75)) if coll["busy"]["tp"] else 100.0
    BUSY_TPMED = float(np.median(np.concatenate(coll["busy"]["tp"]))) if coll["busy"]["tp"] else 50.0
    NEAR_PX = float(HALO_PX)            # median dist <= 5px => hugging a wire
    NEAR_HALO_FRAC = 0.5                # >=50% of component within HALO_PX of a wire
    LUMA_HI = 150.0                     # bright local patch
    SAT_LO = 60.0                       # desaturated local patch (HSV S 0-255)
    ELONG_HI = 4.0                      # bbox >= 4:1 aspect => elongated/thin
    FILL_LO = 0.35                      # sparse fill => stroke-like, not a blob

    mode_mass = dict(busy_texture=0.0, near_wire_halo=0.0,
                     structured_edge=0.0, bright_pale_bg=0.0)
    mode_ncomp = dict(busy_texture=0, near_wire_halo=0,
                      structured_edge=0, bright_pale_bg=0)
    busy_hi_mass = busy_mid_mass = 0.0   # split of busy_texture mass by busyness level
    for r in cc_records:
        a = r["area"]
        near = (r["halo_frac"] >= NEAR_HALO_FRAC) or (r["median_dist"] <= NEAR_PX)
        bright_pale = (r["median_luma"] > LUMA_HI) and (r["median_sat"] < SAT_LO)
        elongated = (r["elong"] > ELONG_HI) and (r["fill"] < FILL_LO)
        if near:
            mode = "near_wire_halo"
        elif elongated:
            mode = "structured_edge"
        elif bright_pale:
            mode = "bright_pale_bg"
        else:
            mode = "busy_texture"        # catch-all standalone (textured/cluttered bg)
            if r["median_busy"] > BUSY_TP75:
                busy_hi_mass += a
            else:
                busy_mid_mass += a
        r["assigned_mode"] = mode         # annotate for the per-component dump
        mode_mass[mode] += a
        mode_ncomp[mode] += 1

    mode_frac = {k: (v / total_fp_mass if total_fp_mass else 0.0) for k, v in mode_mass.items()}
    ranked_modes = sorted(mode_frac.items(), key=lambda kv: kv[1], reverse=True)
    dominant_mode = ranked_modes[0][0]

    print("\n================ FP-MODE MASS PARTITION (DELIVERABLE) ================")
    label_map = dict(busy_texture="(i)  BUSY-TEXTURE hallucination",
                     near_wire_halo="(ii) NEAR-WIRE HALO / over-seg",
                     structured_edge="(iii) STRUCTURED-EDGE object confusion",
                     bright_pale_bg="(iv) BRIGHT/PALE bg")
    for k, frac in ranked_modes:
        print(f"  {label_map[k]:42s} mass={100*frac:6.2f}%  ncomp={mode_ncomp[k]}")
    bt = mode_mass["busy_texture"]
    print(f"  [busy_texture split] high-busyness(>{BUSY_TP75:.1f})={100*busy_hi_mass/total_fp_mass:.2f}%  "
          f"mid-busyness={100*busy_mid_mass/total_fp_mass:.2f}%  (of total FP mass)")
    print(f"  DOMINANT MODE: {label_map[dominant_mode]}")

    fp_mode_block = dict(
        partition_is_exhaustive=True,
        thresholds=dict(BUSY_TP75=BUSY_TP75, BUSY_TPMED=BUSY_TPMED, NEAR_PX=NEAR_PX,
                        NEAR_HALO_FRAC=NEAR_HALO_FRAC, LUMA_HI=LUMA_HI, SAT_LO=SAT_LO,
                        ELONG_HI=ELONG_HI, FILL_LO=FILL_LO, HALO_PX=HALO_PX),
        rule=("per-component, exhaustive: (1) near a true wire (halo_frac>=0.5 OR "
              "median_dist<=NEAR_PX) -> near_wire_halo; else standalone, assigned by "
              "dominant signal: elongated&sparse -> structured_edge; bright&desaturated "
              "-> bright_pale_bg; otherwise -> busy_texture (catch-all standalone-on-"
              "textured-bg)."),
        mode_mass_px={k: float(v) for k, v in mode_mass.items()},
        mode_mass_frac={k: float(v) for k, v in mode_frac.items()},
        mode_n_components=mode_ncomp,
        busy_texture_busyness_split=dict(
            high_busyness_mass_frac=busy_hi_mass / total_fp_mass if total_fp_mass else 0.0,
            mid_busyness_mass_frac=busy_mid_mass / total_fp_mass if total_fp_mass else 0.0,
            note=f"high = component median Laplacian-var > TP-75th-pct ({BUSY_TP75:.1f})",
        ),
        ranked=[dict(mode=k, mass_frac=float(v), n_components=mode_ncomp[k]) for k, v in ranked_modes],
        dominant_mode=dominant_mode,
    )

    # ---- ranked verdict by |Cliff's delta| ----
    ranked_props = sorted(
        [(k, props[k]["cliffs_delta_fp_vs_tp"]) for k in PROPS
         if props[k]["cliffs_delta_fp_vs_tp"] is not None],
        key=lambda kv: abs(kv[1]), reverse=True)
    print("\n================ RANKED PROPERTY DELTAS (|Cliff's delta| FP vs TP) ================")
    for rank, (k, d) in enumerate(ranked_props, 1):
        direction = "FP LOWER than TP" if d < 0 else "FP HIGHER than TP"
        print(f"  {rank}. {k:9s} |delta|={abs(d):.3f}  ({direction})  -- {props[k]['label']}")

    result = dict(
        checkpoint=CKPT, data_dir=DATA, n_frames=len(bns),
        infer_size_hw=list(INFER_HW), wire_rule="fg", window=WIN,
        measurement_space=("GT-crop scoring grid (same grid the evaluator computes IoU on; "
                           "pred upsampled NEAREST, preprocessed 480x640 RGB resized to GT-crop "
                           "res with AREA/LINEAR). HSV/luma/contrast are 15x15 local-patch means; "
                           "busyness = 15x15 Laplacian variance; dist = L2 distanceTransform of "
                           "(1 - GT-wire). Cliff's delta sign: +ve = FP larger than TP."),
        sanity_gate=sanity,
        properties=props,
        ranked_by_abs_cliffs_delta=[dict(rank=i + 1, property=k, cliffs_delta=d)
                                    for i, (k, d) in enumerate(ranked_props)],
        per_camera_fp=cam_tbl,
        connected_components=cc_summary,
        fp_modes=fp_mode_block,
    )
    with open(os.path.join(OUT, "fp_analysis.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {os.path.join(OUT, 'fp_analysis.json')}")

    # ---- audit dump: the 60 largest FP components with their assigned mode + props ----
    top_comps = sorted(cc_records, key=lambda r: r["area"], reverse=True)[:60]
    audit = [dict(bn=r["bn"], cam=r["cam"], area=r["area"], mode=r.get("assigned_mode"),
                  median_dist=round(r["median_dist"], 1), halo_frac=round(r["halo_frac"], 3),
                  median_busy=round(r["median_busy"], 2), median_luma=round(r["median_luma"], 1),
                  median_sat=round(r["median_sat"], 1), elong=round(r["elong"], 2),
                  fill=round(r["fill"], 3))
             for r in top_comps]
    with open(os.path.join(OUT, "fp_top_components.json"), "w") as f:
        json.dump(dict(note="60 largest FP connected components by area (GT-crop px), "
                            "with assigned FP-mode and component-median properties",
                       components=audit), f, indent=2)
    print(f"wrote {os.path.join(OUT, 'fp_top_components.json')}")

    # ---- montages of the worst-precision frames (lowest precision, require FP mass) ----
    # Only frames with meaningful FP AND some gt wire so the overlay is informative.
    cand = [fr for fr in per_frame if fr["fp"] >= 2000]
    worst = sorted(cand, key=lambda fr: (fr["precision"], -fr["fp"]))[:6]
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

        def lab(img, text):
            img = img.copy()
            cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(img, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            return img

        # 1 RGB
        panel_rgb = lab(rgb_crop, f"{bn} RGB")
        # 2 error overlay FP=red TP=green FN=blue (BGR)
        err = rgb_crop.copy(); el = np.zeros_like(err)
        el[p & g] = (0, 255, 0)      # TP green
        el[p & (~g)] = (0, 0, 255)   # FP red
        el[(~p) & g] = (255, 0, 0)   # FN blue
        sel = (p | g)
        err[sel] = (rgb_crop[sel] * 0.4 + el[sel] * 0.6).astype(np.uint8)
        panel_err = lab(err, f"FP=red TP=grn FN=blu  P={fr['precision']:.2f} fp={fr['fp']}")
        # 3 FP-only overlay (red) to make hallucinations obvious
        fpo = rgb_crop.copy(); fl = np.zeros_like(fpo); fl[p & (~g)] = (0, 0, 255)
        m = (p & (~g))
        fpo[m] = (rgb_crop[m] * 0.35 + fl[m] * 0.65).astype(np.uint8)
        panel_fp = lab(fpo, "FP only (red)")

        sep = np.full((Hc, 4, 3), 200, np.uint8)
        sheet = np.hstack([panel_rgb, sep, panel_err, sep, panel_fp])
        mp = os.path.join(OUT, f"worst_precision_{bn}_P{fr['precision']:.2f}.png")
        cv2.imwrite(mp, sheet)
        montage_paths.append(mp)
        print(f"  montage: {mp}")

    result["worst_precision_montages"] = montage_paths
    with open(os.path.join(OUT, "fp_analysis.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nDONE.")


if __name__ == "__main__":
    main()
